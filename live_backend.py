
import asyncio
import json
import os
import logging

from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import websockets

from dotenv import load_dotenv
from web3 import Web3

from cnn_inference import CNNWalletRisk


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("ALCHEMY_API_KEY")

if not API_KEY:
    raise RuntimeError("ALCHEMY_API_KEY not found in .env")

HTTP_URL = f"https://eth-mainnet.g.alchemy.com/v2/{API_KEY}"
WS_URL = f"wss://eth-mainnet.g.alchemy.com/v2/{API_KEY}"

w3 = Web3(
    Web3.HTTPProvider(
        HTTP_URL,
        request_kwargs={"timeout": 30}
    )
)

MAX_TRANSACTIONS = 500
MIN_TRANSACTIONS_FOR_PRELIMINARY_SCORE = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("ChainGuard")


# ============================================================
# 2. LOAD CNN
# ============================================================

predictor = CNNWalletRisk()


# ============================================================
# 3. IN-MEMORY STORAGE
# ============================================================

live_transactions = deque(maxlen=MAX_TRANSACTIONS)

wallet_activity = defaultdict(lambda: {
    "sent_values": [],
    "received_values": [],
    "sent_times": [],
    "received_times": [],
    "sent_addresses": set(),
    "received_addresses": set(),
    "sent_contract_values": [],
    "created_contracts": 0,
    "sent_count": 0,
    "received_count": 0,
})

wallet_scores = {}

contract_cache = {}

# ERC-20 token metadata is cached to avoid repeated RPC calls.
token_metadata_cache = {}
MAX_TOKEN_METADATA_CACHE = 1000

# keccak256("Transfer(address,address,uint256)")
TRANSFER_TOPIC = Web3.to_hex(
    Web3.keccak(text="Transfer(address,address,uint256)")
).lower()

ERC20_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function",
    },
]

processed_transactions = set()
MAX_PROCESSED_HASHES = 10000


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def to_int(value):
    """Convert an RPC hexadecimal quantity or integer to int."""

    if isinstance(value, str):
        return int(value, 16) if value.startswith("0x") else int(value)

    return int(value)


def normalize_hash(value):

    if isinstance(value, (bytes, bytearray)):
        return Web3.to_hex(value).lower()

    return str(value).lower()


def mean_interval(times):

    if len(times) < 2:
        return np.nan

    ordered = sorted(times)

    intervals = [
        (ordered[i] - ordered[i - 1]) / 60
        for i in range(1, len(ordered))
    ]

    return float(np.mean(intervals))


def safe_min(values):
    return min(values) if values else np.nan


def safe_max(values):
    return max(values) if values else np.nan


def safe_mean(values):
    return float(np.mean(values)) if values else np.nan


def is_contract(address):

    if not address:
        return False

    address = address.lower()

    if address not in contract_cache:

        code = w3.eth.get_code(
            Web3.to_checksum_address(address)
        )

        contract_cache[address] = (
            len(code) > 0
        )

    return contract_cache[address]


def get_token_metadata(token_address):
    """Read ERC-20 symbol/decimals once per token and cache the result."""

    checksum_address = Web3.to_checksum_address(token_address)
    cache_key = checksum_address.lower()

    if cache_key in token_metadata_cache:
        return token_metadata_cache[cache_key]

    metadata = {
        "symbol": "UNKNOWN",
        "decimals": None,
    }

    contract = w3.eth.contract(
        address=checksum_address,
        abi=ERC20_ABI,
    )

    try:
        decimals = contract.functions.decimals().call()
        if 0 <= int(decimals) <= 255:
            metadata["decimals"] = int(decimals)
    except Exception as error:
        logger.debug(
            "Could not read decimals for token %s: %s",
            checksum_address,
            error,
        )

    try:
        symbol = contract.functions.symbol().call()
        if isinstance(symbol, bytes):
            symbol = symbol.rstrip(b"\\x00").decode(
                "utf-8", errors="replace"
            )
        metadata["symbol"] = str(symbol).strip() or "UNKNOWN"
    except Exception as error:
        logger.debug(
            "Could not read symbol for token %s: %s",
            checksum_address,
            error,
        )

    # Keep the cache bounded during long-running monitoring.
    if len(token_metadata_cache) >= MAX_TOKEN_METADATA_CACHE:
        token_metadata_cache.pop(next(iter(token_metadata_cache)))

    token_metadata_cache[cache_key] = metadata
    return metadata


def decode_erc20_transfers(receipt):
    """Decode standard ERC-20 Transfer events from a transaction receipt.

    amount_raw is retained as a string to preserve the exact integer value.
    amount is a convenience float when token decimals are available.
    """

    transfers = []

    for log in receipt.get("logs", []):
        topics = log.get("topics", [])

        if len(topics) < 3:
            continue

        if normalize_hash(topics[0]) != TRANSFER_TOPIC:
            continue

        try:
            token_address = Web3.to_checksum_address(log["address"])

            from_topic = normalize_hash(topics[1])
            to_topic = normalize_hash(topics[2])

            sender_address = Web3.to_checksum_address(
                "0x" + from_topic[-40:]
            )
            receiver_address = Web3.to_checksum_address(
                "0x" + to_topic[-40:]
            )

            raw_amount = to_int(log.get("data", "0x0"))
            metadata = get_token_metadata(token_address)
            decimals = metadata["decimals"]

            display_amount = (
                raw_amount / (10 ** decimals)
                if decimals is not None
                else None
            )

            transfers.append({
                "token_address": token_address.lower(),
                "token_symbol": metadata["symbol"],
                "token_decimals": decimals,
                "from_address": sender_address,
                "to_address": receiver_address,
                "amount_raw": str(raw_amount),
                "amount": display_amount,
            })

        except Exception as error:
            logger.warning(
                "Could not decode an ERC-20 Transfer log: %s",
                error,
            )

    return transfers


# ============================================================
# 5. UPDATE WALLET ACTIVITY
# ============================================================

def update_wallet_activity(tx, receipt, timestamp):

    sender = tx["from"].lower()

    receiver = (
        tx["to"].lower()
        if tx["to"]
        else None
    )

    value_eth = float(
        w3.from_wei(
            to_int(tx["value"]),
            "ether"
        )
    )

    sender_data = wallet_activity[sender]

    sender_data["sent_count"] += 1
    sender_data["sent_values"].append(value_eth)
    sender_data["sent_times"].append(timestamp)

    if receiver:

        sender_data["sent_addresses"].add(receiver)

        receiver_data = wallet_activity[receiver]

        receiver_data["received_count"] += 1
        receiver_data["received_values"].append(value_eth)
        receiver_data["received_times"].append(timestamp)
        receiver_data["received_addresses"].add(sender)

        if is_contract(receiver):
            sender_data["sent_contract_values"].append(value_eth)

    if receipt.get("contractAddress"):
        sender_data["created_contracts"] += 1

    return sender, receiver


# ============================================================
# 6. BUILD 20 CNN FEATURES
# ============================================================

def build_wallet_features(wallet):

    data = wallet_activity[wallet]

    sent_values = data["sent_values"]
    received_values = data["received_values"]
    contract_values = data["sent_contract_values"]

    all_times = (
        data["sent_times"] +
        data["received_times"]
    )

    time_span = (
        (max(all_times) - min(all_times)) / 60
        if len(all_times) >= 2
        else np.nan
    )

    return {
        "Avg min between sent tnx":
            mean_interval(data["sent_times"]),

        "Avg min between received tnx":
            mean_interval(data["received_times"]),

        "Time Diff between first and last (Mins)":
            time_span,

        "Sent tnx": data["sent_count"],

        "Received Tnx": data["received_count"],

        "Number of Created Contracts":
            data["created_contracts"],

        "Unique Received From Addresses":
            len(data["received_addresses"]),

        "Unique Sent To Addresses":
            len(data["sent_addresses"]),

        "min value received":
            safe_min(received_values),

        "max value received":
            safe_max(received_values),

        "avg val received":
            safe_mean(received_values),

        "min val sent":
            safe_min(sent_values),

        "max val sent":
            safe_max(sent_values),

        "avg val sent":
            safe_mean(sent_values),

        "min value sent to contract":
            safe_min(contract_values),

        "max val sent to contract":
            safe_max(contract_values),

        "avg value sent to contract":
            safe_mean(contract_values),

        "total Ether sent":
            float(sum(sent_values)),

        "total ether received":
            float(sum(received_values)),

        "total ether sent contracts":
            float(sum(contract_values)),
    }


# ============================================================
# 7. CNN WALLET SCORING
# ============================================================

def score_wallet(wallet):

    data = wallet_activity[wallet]

    total_observed = (
        data["sent_count"] +
        data["received_count"]
    )

    if total_observed < MIN_TRANSACTIONS_FOR_PRELIMINARY_SCORE:
        return None

    features = build_wallet_features(wallet)

    result = predictor.predict(features)

    result["wallet_address"] = wallet
    result["observed_transactions"] = total_observed
    result["score_status"] = "PRELIMINARY"
    result["updated_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    wallet_scores[wallet] = result

    return result


# ============================================================
# 8. PROCESS A TRANSACTION
# ============================================================

def process_transaction(tx, receipt, block, timestamp):

    sender, receiver = update_wallet_activity(
        tx,
        receipt,
        timestamp
    )

    transaction_hash = normalize_hash(tx["hash"])

    value_eth = float(
        w3.from_wei(
            to_int(tx["value"]),
            "ether"
        )
    )

    # Convert RPC hexadecimal values into integers.
    gas_used = to_int(receipt["gasUsed"])

    effective_gas_price = to_int(
        receipt.get(
            "effectiveGasPrice",
            tx.get("gasPrice", 0)
        )
    )

    transaction_fee_eth = float(
        w3.from_wei(
            gas_used * effective_gas_price,
            "ether"
        )
    )

    # Token movements are decoded from receipt logs, not tx.value.
    token_transfers = decode_erc20_transfers(receipt)

    transaction = {
        "tx_hash": transaction_hash,

        "block_number": block.number,

        "timestamp": timestamp,

        "datetime_utc": datetime.fromtimestamp(
            timestamp,
            timezone.utc
        ).isoformat(),

        "from_address": sender,

        "to_address": receiver,

        "value_eth": value_eth,

        "token_transfers": token_transfers,

        "gas_used": gas_used,

        "transaction_fee_eth": transaction_fee_eth,

        "status": to_int(receipt["status"]),

        "transaction_type": to_int(
            tx.get("type", 0)
        ),

        "contract_created": (
            receipt.get("contractAddress") is not None
        ),

        "log_count": len(receipt["logs"]),

        "block_transaction_count": len(
            block.transactions
        ),

        "risk_results": {},
    }

    # Score sender and receiver wallets.
    for wallet in {sender, receiver} - {None}:

        result = score_wallet(wallet)

        if result:
            transaction["risk_results"][wallet] = result

            fraud_score = result["wallet_risk_score"]
            prediction = (
                "Fraud" if fraud_score >= 0.70
                else "Not Fraud"
            )

            logger.info(
                "LIVE WALLET RISK | Wallet: %s | "
                "Prediction: %s | Fraud Score: %.4f | "
                "Risk: %s | Status: PRELIMINARY | Transactions: %s",
                wallet,
                prediction,
                fraud_score,
                result["risk_category"],
                result["observed_transactions"]
)

    live_transactions.append(transaction)

    logger.info(
        "TX %s | %.6f ETH | %s -> %s | "
        "Token transfers: %s | Fee: %.8f ETH",
        transaction_hash,
        value_eth,
        sender,
        receiver,
        len(token_transfers),
        transaction_fee_eth,
    )

    for transfer in token_transfers:
        amount_text = (
            f"{transfer['amount']:g}"
            if transfer["amount"] is not None
            else transfer["amount_raw"]
        )
        logger.info(
            "TOKEN | %s %s | %s -> %s",
            amount_text,
            transfer["token_symbol"],
            transfer["from_address"],
            transfer["to_address"],
        )

    return transaction


# ============================================================
# 9. PROCESS NEW BLOCK USING BATCH RECEIPTS
# ============================================================

def process_block(block_number):

    block = w3.eth.get_block(
        block_number,
        full_transactions=True
    )

    timestamp = block.timestamp

    logger.info(
        "Processing block %s | Transactions: %s",
        block_number,
        len(block.transactions)
    )

    # Retrieve all receipts in a single RPC request.
    response = w3.provider.make_request(
        "eth_getBlockReceipts",
        [hex(block_number)]
    )

    if "error" in response:
        raise RuntimeError(
            f"Could not retrieve block receipts: {response['error']}"
        )

    receipts = response.get("result", [])

    # Match receipts to transactions using transaction hashes.
    receipt_map = {
        normalize_hash(receipt["transactionHash"]): receipt
        for receipt in receipts
    }

    processed = 0
    skipped = 0

    for tx in block.transactions:

        try:

            tx_hash = normalize_hash(tx["hash"])

            if tx_hash in processed_transactions:
                skipped += 1
                continue

            receipt = receipt_map.get(tx_hash)

            if receipt is None:

                logger.warning(
                    "Receipt not found for transaction: %s",
                    tx_hash
                )

                continue

            process_transaction(
                tx,
                receipt,
                block,
                timestamp
            )

            processed_transactions.add(tx_hash)
            processed += 1

            if len(processed_transactions) > MAX_PROCESSED_HASHES:
                processed_transactions.pop()

        except Exception as error:

            logger.exception(
                "Could not process transaction %s: %s",
                normalize_hash(tx["hash"]),
                error
            )

    logger.info(
        "Block %s completed | Processed: %s/%s | Skipped duplicates: %s",
        block_number,
        processed,
        len(block.transactions),
        skipped
    )


# ============================================================
# 10. WEBSOCKET MONITOR
# ============================================================

async def listen():

    while True:

        try:

            logger.info("Connecting to Alchemy WebSocket...")

            async with websockets.connect(
                WS_URL,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10
            ) as websocket:

                subscribe = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "eth_subscribe",
                    "params": ["newHeads"]
                }

                await websocket.send(
                    json.dumps(subscribe)
                )

                response = json.loads(
                    await websocket.recv()
                )

                if "error" in response:
                    raise RuntimeError(response["error"])

                subscription_id = response.get("result")

                logger.info(
                    "Subscribed successfully: %s",
                    subscription_id
                )

                logger.info(
                    "ChainGuard is monitoring Ethereum Mainnet."
                )

                while True:

                    message = await websocket.recv()

                    data = json.loads(message)

                    if "params" not in data:
                        continue

                    header = data["params"]["result"]

                    block_number = int(
                        header["number"],
                        16
                    )

                    try:

                        await asyncio.to_thread(
                            process_block,
                            block_number
                        )

                    except Exception:

                        logger.exception(
                            "Block processing failed: %s",
                            block_number
                        )

        except asyncio.CancelledError:
            raise

        except Exception:

            logger.exception(
                "Alchemy connection lost. Reconnecting in 5 seconds..."
            )

            await asyncio.sleep(5)


# ============================================================
# 11. START CHAIN GUARD
# ============================================================

async def main():

    if not w3.is_connected():

        raise ConnectionError(
            "Could not connect to Ethereum through Alchemy HTTP."
        )

    logger.info("Alchemy HTTP connection successful.")

    logger.info("Loading live monitoring engine...")

    await listen()


if __name__ == "__main__":

    try:

        asyncio.run(main())

    except KeyboardInterrupt:

        logger.info("ChainGuard monitoring stopped.")

import os
from pathlib import Path

from dotenv import load_dotenv
from web3 import Web3

load_dotenv(Path(__file__).resolve().parent / ".env")

api_key = os.getenv("ALCHEMY_API_KEY")

if not api_key:
    raise RuntimeError("ALCHEMY_API_KEY not found in .env")

w3 = Web3(
    Web3.HTTPProvider(
        f"https://eth-mainnet.g.alchemy.com/v2/{api_key}"
    )
)

block_number = w3.eth.block_number

print("Connected:", w3.is_connected())
print("Latest block:", block_number)

response = w3.provider.make_request(
    "eth_getBlockReceipts",
    [hex(block_number)]
)

if "error" in response:
    print("RPC error:", response["error"])
else:
    receipts = response.get("result", [])
    print("Receipts retrieved:", len(receipts))
    print("Batch receipt retrieval successful!")
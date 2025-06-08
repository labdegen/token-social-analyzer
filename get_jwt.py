# get_jwt.py
#
# Usage: python get_jwt.py <KEYPAIR_JSON_PATH>
# Example: python get_jwt.py rugcheck-keypair.json

import sys, json, time
import base58
import nacl.signing
import requests

def main():
    if len(sys.argv) != 2:
        print("Usage: python get_jwt.py <KEYPAIR_JSON_PATH>")
        sys.exit(1)

    keypath = sys.argv[1]
    # 1) Load the Solana keypair JSON (64‐byte array of ints)
    with open(keypath, "r") as f:
        raw = json.load(f)
    full_secret = bytes(raw)  # 64 bytes total

    # 2) Extract only the first 32 bytes as the Ed25519 seed
    seed = full_secret[:32]
    signing_key = nacl.signing.SigningKey(seed)

    # 3) Derive the public key (last 32 bytes from Solana keypair should match this)
    verify_key = signing_key.verify_key
    pubkey_bytes = verify_key.encode()
    pubkey_b58 = base58.b58encode(pubkey_bytes).decode()

    # 4) Sign exactly the literal bytes of "Sign-in to Rugcheck.xyz"
    static_msg = b"Sign-in to Rugcheck.xyz"
    signature_bytes = signing_key.sign(static_msg).signature  # 64 raw bytes

    # 5) Build the 'message' object (with timestamp) – this is NOT what we sign, it's just for the payload
    timestamp = int(time.time())
    msg_obj = {
        "message": "Sign-in to Rugcheck.xyz",
        "publicKey": pubkey_b58,
        "timestamp": timestamp
    }

    # 6) Construct the full authentication payload that RugCheck expects
    payload = {
        "message": msg_obj,
        "signature": {
            "data": list(signature_bytes),  # list of 64 ints 0–255
            "type": "ed25519"
        },
        "wallet": pubkey_b58
    }

    # 7) POST to the RugCheck auth endpoint
    url = "https://api.rugcheck.xyz/auth/login/solana"
    resp = requests.post(url, json=payload, timeout=10)
    if not resp.ok:
        print("Failed to get JWT:", resp.status_code, resp.text)
        sys.exit(1)

    token = resp.json().get("token")
    print("Your RugCheck JWT:", token)

if __name__ == "__main__":
    main()

// get_jwt.js
//
// Usage: node get_jwt.js <KEYPAIR_PATH>
// Example: node get_jwt.js ./rugcheck-keypair.json

import fs from "fs";
import fetch from "node-fetch";
import nacl from "tweetnacl";
import bs58 from "bs58";

async function main() {
  const keypairPath = process.argv[2];
  if (!keypairPath) {
    console.error("Usage: node get_jwt.js <KEYPAIR_PATH>");
    process.exit(1);
  }

  // 1) Load your Solana keypair JSON (array of 64 numbers)
  const raw = JSON.parse(fs.readFileSync(keypairPath, "utf8"));
  const secretKey = Uint8Array.from(raw);

  // 2) Extract the public key bytes (last 32 bytes of secretKey)
  const publicKeyBytes = secretKey.slice(32);
  const publicKeyBase58 = bs58.encode(publicKeyBytes);

  // 3) Build and sign the fixed message
  const messageToSign = "Sign-in to Rugcheck.xyz";
  const encodedMessage = new TextEncoder().encode(messageToSign);
  const signatureBytes = nacl.sign.detached(encodedMessage, secretKey);

  // 4) Construct the payload
  const timestamp = Math.floor(Date.now() / 1000);
  const body = {
    message: {
      message: messageToSign,
      publicKey: publicKeyBase58,
      timestamp: timestamp
    },
    signature: {
      data: Array.from(signatureBytes),
      type: "ed25519"
    },
    wallet: publicKeyBase58
  };

  // 5) POST to RugCheck’s auth endpoint
  const resp = await fetch("https://api.rugcheck.xyz/auth/login/solana", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });

  if (!resp.ok) {
    console.error("Failed to get JWT:", resp.status, await resp.text());
    process.exit(1);
  }

  const { token } = await resp.json();
  console.log("Your RugCheck JWT:", token);
}

main().catch((err) => {
  console.error("Error:", err);
  process.exit(1);
});
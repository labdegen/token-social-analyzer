# decode_sig.py

import sys

# Paste your Base58 signature string here (from $SIG_B58 in Step 12):
sig = "2F7JHFMX7CmDXMy8hR3fo8d9eVTUUBFRHCwpv3KR1MEU3TdNrw5hKmsAthS6TSseWSL61SXMMuYptSFLCbAzns4N"

# Base58 alphabet used by Solana
alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

# Decode Base58 into a big integer
n = 0
for c in sig:
    n = n * 58 + alphabet.index(c)

# Convert that integer into raw bytes
# Calculate how many bytes are needed
length = (n.bit_length() + 7) // 8
raw = n.to_bytes(length, "big")

# Account for leading '1's in Base58 (each '1' is a leading zero byte)
num_leading_zeros = len(sig) - len(sig.lstrip("1"))
raw = (b"\x00" * num_leading_zeros) + raw

# Print comma-separated decimal byte values
print(",".join(str(b) for b in raw))

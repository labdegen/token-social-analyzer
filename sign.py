python - <<'PYCODE'
import sys

# Paste your Base58 signature here:
sig = "2F7JHFMX7CmDXMy8hR3fo8d9eVTUUBFRHCwpv3KR1MEU3TdNrw5hKmsAthS6TSseWSL61SXMMuYptSFLCbAzns4N"

# Base58 alphabet
alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

# Decode Base58 to integer
n = 0
for c in sig:
    n = n * 58 + alphabet.index(c)

# Convert integer to bytes
raw = n.to_bytes((n.bit_length() + 7) // 8, "big")

# Account for leading '1' characters (which represent leading zero bytes)
num_leading_zeros = len(sig) - len(sig.lstrip("1"))
raw = (b"\x00" * num_leading_zeros) + raw

# Print comma-separated byte values
print(",".join(str(b) for b in raw))
PYCODE

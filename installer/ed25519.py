"""
Ed25519, in pure Python, so a patch can be signed and an install can check it.

Why this file exists at all: the SHA-256 in the update feed proves a download
arrived intact, and nothing more. It is computed over a file the feed itself
describes, so anyone who can serve the feed can serve a patch and the hash
that matches it. A signature is what makes the check about the publisher
rather than about the transfer — the private key never leaves the machine that
builds patches, and an install with the public key in it will not apply
anything else, whatever the feed says.

Pure Python and no dependency, deliberately. The updater runs before the app
starts, sometimes to repair an install that is already broken, and the one
thing it must not do is need a package to be importable in order to decide
whether it is allowed to run. It verifies one 32-byte signature per update, so
the ~10ms this costs against a C implementation's microseconds is not a number
anybody will ever see.

The maths is the RFC 8032 reference construction. It is not novel and is not
meant to be: this is a transcription, kept short enough to read in one sitting.

    priv, pub = generate()          # bytes, 32 each
    sig = sign(message, priv, pub)  # 64 bytes
    verify(message, sig, pub)       # True / False, never raises
"""

import hashlib
import os

# The curve's field and group orders, and the constants derived from them.
Q = 2 ** 255 - 19
L = 2 ** 252 + 27742317777372353535851937790883648493

_D = (-121665 * pow(121666, Q - 2, Q)) % Q
_I = pow(2, (Q - 1) // 4, Q)

# The base point, recovered below once the helpers exist.
_BY = (4 * pow(5, Q - 2, Q)) % Q


def _sha512(b):
    return hashlib.sha512(b).digest()


def _x_recover(y):
    """The x that goes with a y on the curve, in the sign the encoding implies."""
    xx = (y * y - 1) * pow(_D * y * y + 1, Q - 2, Q)
    x = pow(xx, (Q + 3) // 8, Q)
    if (x * x - xx) % Q != 0:
        x = (x * _I) % Q
    if x % 2 != 0:
        x = Q - x
    return x


# Points are kept in extended coordinates (X, Y, Z, T) — the same point in
# projective form, which is what keeps addition free of a modular inverse per
# step. Only the final encode divides through by Z.
def _point(x, y):
    return (x % Q, y % Q, 1, (x * y) % Q)


_B = _point(_x_recover(_BY), _BY)
_IDENTITY = (0, 1, 1, 0)


def _add(p, q):
    x1, y1, z1, t1 = p
    x2, y2, z2, t2 = q
    a = ((y1 - x1) * (y2 - x2)) % Q
    b = ((y1 + x1) * (y2 + x2)) % Q
    c = (t1 * 2 * _D * t2) % Q
    d = (z1 * 2 * z2) % Q
    e, f, g, h = b - a, d - c, d + c, b + a
    return ((e * f) % Q, (g * h) % Q, (f * g) % Q, (e * h) % Q)


def _mul(p, n):
    """Double-and-add. Not constant time — it only ever runs on public data."""
    out = _IDENTITY
    while n > 0:
        if n & 1:
            out = _add(out, p)
        p = _add(p, p)
        n >>= 1
    return out


def _encode_point(p):
    x, y, z, _ = p
    zi = pow(z, Q - 2, Q)
    x, y = (x * zi) % Q, (y * zi) % Q
    return ((y & ~(1 << 255)) | ((x & 1) << 255)).to_bytes(32, "little")


def _decode_point(data):
    n = int.from_bytes(data, "little")
    y = n & ((1 << 255) - 1)
    x = _x_recover(y)
    if x & 1 != (n >> 255) & 1:
        x = Q - x
    p = _point(x, y)
    if not _on_curve(p):
        raise ValueError("point is not on the curve")
    return p


def _on_curve(p):
    x, y, z, _ = p
    zi = pow(z, Q - 2, Q)
    x, y = (x * zi) % Q, (y * zi) % Q
    return (-x * x + y * y - 1 - _D * x * x * y * y) % Q == 0


def _secret_scalar(priv):
    h = bytearray(_sha512(priv)[:32])
    h[0] &= 248
    h[31] &= 127
    h[31] |= 64
    return int.from_bytes(h, "little")


def public_key(priv: bytes) -> bytes:
    """The 32-byte public key for a 32-byte private key."""
    if len(priv) != 32:
        raise ValueError("an Ed25519 private key is 32 bytes")
    return _encode_point(_mul(_B, _secret_scalar(priv)))


def generate() -> tuple:
    """A fresh (private, public) pair, from the OS entropy source."""
    priv = os.urandom(32)
    return priv, public_key(priv)


def sign(message: bytes, priv: bytes, pub: bytes = None) -> bytes:
    """The 64-byte signature of `message`."""
    if pub is None:
        pub = public_key(priv)
    h = _sha512(priv)
    a = _secret_scalar(priv)
    r = int.from_bytes(_sha512(h[32:] + message), "little") % L
    big_r = _encode_point(_mul(_B, r))
    k = int.from_bytes(_sha512(big_r + pub + message), "little") % L
    s = (r + k * a) % L
    return big_r + s.to_bytes(32, "little")


def verify(message: bytes, signature: bytes, pub: bytes) -> bool:
    """
    True when `signature` is this key's signature of `message`.

    Never raises. Every way a signature can be wrong — truncated, malformed,
    a point off the curve, simply not matching — is one answer, because every
    one of them means the same thing to the caller and a verifier that throws
    on some of them is a verifier somebody will eventually wrap in a bare
    `except` that also swallows the honest False.
    """
    try:
        if len(signature) != 64 or len(pub) != 32:
            return False
        big_r = _decode_point(signature[:32])
        big_a = _decode_point(pub)
        s = int.from_bytes(signature[32:], "little")
        if s >= L:
            return False
        k = int.from_bytes(_sha512(signature[:32] + pub + message), "little") % L
        # sB == R + kA, checked projectively so neither side needs an inverse.
        left = _mul(_B, s)
        right = _add(big_r, _mul(big_a, k))
        return _encode_point(left) == _encode_point(right)
    except (ValueError, OverflowError):
        return False


if __name__ == "__main__":
    # RFC 8032 test vector 1, plus a round trip — enough to catch a
    # transcription error, which is the only kind of bug this file can have.
    priv = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
    pub = bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
    sig = bytes.fromhex(
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
        "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b")
    assert public_key(priv) == pub, "public key derivation"
    assert sign(b"", priv, pub) == sig, "signature"
    assert verify(b"", sig, pub), "verify"
    assert not verify(b"x", sig, pub), "verify rejects a different message"

    p2, k2 = generate()
    m = os.urandom(300)
    assert verify(m, sign(m, p2, k2), k2), "round trip"
    assert not verify(m, sign(m, p2, k2), pub), "round trip rejects the wrong key"
    print("ed25519: all checks passed")

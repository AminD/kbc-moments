"""Minimal demo authentication + authorization.

- Passwords are stored only as salted PBKDF2 hashes in .streamlit/secrets.toml (git-ignored).
- The customer id comes from the authenticated session, NEVER from user input or the URL
  (prevents IDOR: a customer cannot open another customer's data).
- The advisor view requires the 'advisor' role.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time

ITERATIONS = 200_000


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return f"pbkdf2${ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, it, salt_hex, hash_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(it))
        return hmac.compare_digest(dk.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False


_DUMMY = hash_password("dummy-password-for-timing")
_failed: dict[str, list[float]] = {}


def authenticate(users: dict, username: str, password: str):
    """Return the user record (role, customer_id) or None. Basic brute-force throttling."""
    username = (username or "").strip().lower()[:32]
    now = time.time()
    recent = [t for t in _failed.get(username, []) if now - t < 300]
    if len(recent) >= 5:
        return None
    user = users.get(username)
    ok = verify_password(password or "", user["password_hash"] if user else _DUMMY)
    if not (user and ok):
        _failed[username] = recent + [now]
        return None
    _failed.pop(username, None)
    return {"username": username, "role": user["role"], "customer_id": user.get("customer_id")}


def require_role(session_user: dict | None, role: str) -> bool:
    return bool(session_user) and session_user.get("role") == role

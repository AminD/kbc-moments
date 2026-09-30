"""Create .streamlit/secrets.toml with hashed demo passwords (file is git-ignored).

Run once:  python setup_secrets.py
"""
import getpass
from pathlib import Path

from kbc_moments.auth import hash_password

USERS = [
    ("lina", "customer", "C-0001"),
    ("yanis", "customer", "C-0002"),
    ("marc", "customer", "C-0003"),
    ("advisor", "advisor", None),
]

pw = getpass.getpass("Choose a demo password (used for all 4 demo accounts): ")
if len(pw) < 8:
    raise SystemExit("Password must be at least 8 characters.")

lines = []
for name, role, cid in USERS:
    lines.append(f"[users.{name}]")
    lines.append(f'role = "{role}"')
    if cid:
        lines.append(f'customer_id = "{cid}"')
    lines.append(f'password_hash = "{hash_password(pw)}"')
    lines.append("")

path = Path(".streamlit/secrets.toml")
path.parent.mkdir(exist_ok=True)
path.write_text("\n".join(lines), encoding="utf-8")
print(f"Written {path} for users: {', '.join(u[0] for u in USERS)}")

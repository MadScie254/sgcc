"""
Create an API key for one person.

    python scripts/make_api_key.py amina supervisor
    python scripts/make_api_key.py otieno analyst

Prints the key (give it to that person; it is shown once) and the entry to add to
the API_KEYS list, which stores only the key's SHA-256:

    API_KEYS='[{"name": "amina", "role": "supervisor", "sha256": "..."}, ...]'
"""

import argparse
import json
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.dependencies.auth import ROLES, hash_key  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a named API key")
    parser.add_argument("name", help="Who the key is for; recorded on every action they take")
    parser.add_argument("role", choices=ROLES)
    args = parser.parse_args()
    key = secrets.token_urlsafe(32)
    print(f"Key for {args.name} (shown once):\n  {key}\n")
    print("Add this entry to API_KEYS:")
    print("  " + json.dumps({"name": args.name, "role": args.role, "sha256": hash_key(key)}))


if __name__ == "__main__":
    main()

"""
One-time setup: store your FRED API key securely in the OS credential store
(Windows Credential Manager) via `keyring`, so fred_data.py can pick it up
automatically without an environment variable in every session.

Get a free key first: https://fredapi.stlouisfed.org/docs/api/api_key.html

Usage:
    python setup_api_key.py            # prompts and saves a key
    python setup_api_key.py --check    # shows whether a key is currently stored
    python setup_api_key.py --remove   # deletes the stored key
"""

import sys
import getpass
import keyring

KEYRING_SERVICE = "fred_api"
KEYRING_USERNAME = "api_key"


def save_key():
    key = getpass.getpass("Paste your FRED API key (input hidden): ").strip()
    if not key:
        print("No key entered, nothing saved.")
        return
    keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, key)
    print("Saved. fred_data.py will now find it automatically via keyring.")


def check_key() -> bool:
    key = keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
    if key:
        print(f"A key is stored (ends in ...{key[-4:]}).")
    else:
        print("No key stored yet. Run `python setup_api_key.py` to add one.")
    return bool(key)


def remove_key():
    if keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME):
        keyring.delete_password(KEYRING_SERVICE, KEYRING_USERNAME)
        print("Removed stored key.")
    else:
        print("No key was stored.")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if arg == "--check":
        sys.exit(0 if check_key() else 1)
    elif arg == "--remove":
        remove_key()
    else:
        save_key()

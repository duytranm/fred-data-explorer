"""
One-time setup: store your FRED (or BEA) API key securely in the OS credential store
(Windows Credential Manager) via `keyring`, so the app can pick it up automatically
without an environment variable in every session.

Get free keys first:
    FRED:   https://fredapi.stlouisfed.org/docs/api/api_key.html
    BEA:    https://apps.bea.gov/API/signup/
    Census: https://api.census.gov/data/key_signup.html
    BLS:    https://data.bls.gov/registrationEngine/

Usage:
    python setup_api_key.py            # prompts and saves a FRED key
    python setup_api_key.py --check    # shows whether a FRED key is currently stored
    python setup_api_key.py --remove   # deletes the stored FRED key
    Add --bea, --census or --bls to any of these to manage that key instead, e.g. `python setup_api_key.py --bls`.
"""

import sys
import getpass
import keyring

SERVICES = {
    "FRED": ("fred_api", "api_key"),
    "BEA": ("bea_api", "api_key"),
    "Census": ("census_api", "api_key"),
    "BLS": ("bls_api", "api_key"),
}
FLAGS = {"--bea": "BEA", "--census": "Census", "--bls": "BLS"}


def save_key(name: str):
    key = getpass.getpass(f"Paste your {name} API key (input hidden): ").strip()
    if not key:
        print("No key entered, nothing saved.")
        return
    keyring.set_password(*SERVICES[name], key)
    print(f"Saved. The app will now find your {name} key automatically via keyring.")


def check_key(name: str = "FRED") -> bool:
    key = keyring.get_password(*SERVICES[name])
    if key:
        print(f"A {name} key is stored (ends in ...{key[-4:]}).")
    else:
        flag = next((f" {f}" for f, n in FLAGS.items() if n == name), "")
        print(f"No {name} key stored yet. Run `python setup_api_key.py{flag}` to add one.")
    return bool(key)


def remove_key(name: str):
    if keyring.get_password(*SERVICES[name]):
        keyring.delete_password(*SERVICES[name])
        print(f"Removed stored {name} key.")
    else:
        print(f"No {name} key was stored.")


if __name__ == "__main__":
    args = sys.argv[1:]
    name = next((n for f, n in FLAGS.items() if f in args), "FRED")
    if "--check" in args:
        sys.exit(0 if check_key(name) else 1)
    elif "--remove" in args:
        remove_key(name)
    else:
        save_key(name)

import bittensor
import base64
import time
import requests


AUTHENTICATE_ENDPOINT = "https://datasets.dippy-bittensor-subnet.com/authenticate"
FETCH_ENDPOINT = "https://datasets.dippy-bittensor-subnet.com/dataset"

def main():
    auth_flow()

def parse_arguments():
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse wallet and signer arguments")
    parser.add_argument("--wallet-name", type=str, default="default", help="Name of the wallet")
    parser.add_argument("--wallet-hotkey", type=str, default="default", help="Hotkey of the wallet")
    parser.add_argument("--signer", type=str, choices=["coldkey", "hotkey"], required=True, help="Signer type")
    
    return parser.parse_args()

"""
Usage:
This script authenticates a user using their Bittensor wallet and fetches dataset information.

To run the script, use the following command:

python token.py --wallet-name <wallet_name> --wallet-hotkey <wallet_hotkey> --signer <signer_type>

Arguments:
--wallet-name: Name of the Bittensor wallet (default: "default")
--wallet-hotkey: Hotkey of the Bittensor wallet (default: "default")
--signer: Type of signer to use, either "coldkey" or "hotkey" (required)

Example:
python token_check.py --wallet-name my_wallet --wallet-hotkey my_hotkey --signer hotkey

This will run the authentication flow using the specified wallet and signer information.
"""

def get_dataset(jwt: str):
    # Set up the headers with the JWT
    headers = {
        "Authorization": f"Bearer {jwt}"
    }

    # Make the GET request
    try:
        url = f"{FETCH_ENDPOINT}?epoch_date=20241001&current_date=20241030"
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        print(f"GET request successful. Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"GET request failed: {e}")

def auth_flow():
    args = parse_arguments()

    print(f"Wallet Name: {args.wallet_name}")
    print(f"Wallet Hotkey: {args.wallet_hotkey}")
    print(f"Signer Type: {args.signer}")
    
    wallet = bittensor.wallet(name=args.wallet_name, hotkey=args.wallet_hotkey)
    
    signer = wallet.hotkey
    address = wallet.hotkey.ss58_address
    if args.signer == "coldkey":
        signer = wallet.coldkey
        address = wallet.coldkey.ss58_address

    
    message = str(int(time.time()))
    
    signature = signer.sign(message)
    signature_base64 = base64.b64encode(signature).decode('utf-8')
    print(f"Signed message using {args.signer}: {signature_base64}")

    import requests
    import json

    payload = {
        "key": address,
        "timestamp": message,
        "signature": signature_base64
    }
    access_token = "x"
    try:
        response = requests.post(AUTHENTICATE_ENDPOINT, json=payload)
        response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx
        response_json = response.json()
        access_token = response_json.get("access_token")
        print(f"POST request successful. Access token: {access_token}")
        print(f"Use the token to set the env variable DATASET_API_KEY in this format: DATASET_API_KEY=\"Bearer {access_token}\"")
    except requests.exceptions.RequestException as e:
        print(f"POST request failed: {e}")

    get_dataset(access_token)


if __name__ == "__main__":
    main()

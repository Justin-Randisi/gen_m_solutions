from xrpl.clients import JsonRpcClient
from xrpl.wallet import Wallet, XRPLRequestFailureException
from xrpl.models.transactions import Payment

client = JsonRpcClient("https://s1.ripple.com")

wallet = Wallet.generate()

account_info = client.request_account_info(wallet.classic_address)
balance = account_info["account_data"]["Balance"]
print(f"Account balance: {balance} XRP")

# Destination address
destination_address = "rXXXXXXXXXXXXXXXXXXXXXXXXXX"
# replace rXXXXXXXXXXXXXXXXXXXXXXXXXX with actual address

# Send XRP
tx_result = wallet.send_xrp(
    amount="10",
    destination=destination_address,
    xrpl_client=client,
)

payment = Payment(
    account=wallet.classic_address,
    destination="<destination_address>",
    amount="10",
)

signed_payment = payment.sign(wallet)
response = client.submit(signed_payment)

print(f"Transaction result: {tx_result}")

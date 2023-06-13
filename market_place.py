import requests
from binance.client import Client

def convert_cryptocurrency(client: Client, from_symbol: str, to_symbol: str, quantity: float):
    order = client.create_test_order(
        symbol=from_symbol + to_symbol,
        side=Client.SIDE_SELL,
        type=Client.ORDER_TYPE_MARKET,
        quantity=quantity)
    return order

def get_conversion_rate(from_currency, to_currency):
    response = requests.get(f"https://api.exchangeratesapi.io/latest?base={from_currency}&symbols={to_currency}")
    return response.json()['rates'][to_currency]

usdt_to_usdc = get_conversion_rate("USDT", "USDC")
usdt_to_xrp = get_conversion_rate("USDT", "XRP")

def distribute_with_royalty(total: float, num_partners: int):
    royalty = total * 0.10
    total_after_royalty = total - royalty
    per_partner = total_after_royalty / num_partners
    return per_partner, royalty

total = 1000.0
num_partners = 5

per_partner, royalty = distribute_with_royalty(total, num_partners)
print(f"Each partner gets: {per_partner}")
print(f"Royalty is: {royalty}")


class User:
    def __init__(self, name):
        self.name = name
        self.balance = 0
        self.tokens = 0

    def add_balance(self, amount):
        self.balance += amount

    def subtract_balance(self, amount):
        if self.balance < amount:
            raise Exception("Not enough balance")
        self.balance -= amount

    def add_tokens(self, amount):
        self.tokens += amount

    def remove_tokens(self, amount):
        if self.tokens < amount:
            raise Exception("Not enough tokens")
        self.tokens -= amount

class Marketplace:
    def __init__(self):
        self.users = {}
        self.token_prices = {}

    def buy_token(self, user_id, token_name, amount):
        user = self.users[user_id]
        token_price = self.token_prices[token_name]
        
        if user.balance >= token_price * amount:
            user.balance -= token_price * amount
            user.tokens[token_name] = user.tokens.get(token_name, 0) + amount

    def sell_token(self, user_id, token_name, amount):
        user = self.users[user_id]
        token_price = self.token_prices[token_name]
        
        if user.tokens.get(token_name, 0) >= amount:
            user.balance += token_price * amount
            user.tokens[token_name] -= amount

class ReferralSystem:
    def __init__(self):
        self.referral_codes = {}
        self.bonus_percentage = 0.05

    def add_referral_code(self, username, referral_code):
        self.referral_codes[username] = referral_code

    def remove_referral_code(self, username):
        if username not in self.referral_codes:
            raise Exception("Referral code not found")
        del self.referral_codes[username]

    def adjust_bonus_percentage(self, new_percentage):
        self.bonus_percentage = new_percentage

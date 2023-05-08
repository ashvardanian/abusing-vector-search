import os

from usearch import Index

import pandas as pd
import numpy as np

directory: str = 'stocks'
last_days: int = 365
tickets: list[str] = []
ticket_to_prices: dict[str, np.array] = {}
index = Index(ndim=last_days)

for filename in os.listdir(directory):
    path = os.path.join(directory, filename)
    ticket = path.split('.')[0]
    df = pd.read_csv(path)

    prices_list = df['Close'][-last_days:].to_list()
    prices = np.zeros(last_days, dtype=np.float32)
    prices[-len(prices_list):] = prices_list

    # Normalize for future Covariance estimates
    prices -= np.mean(prices)

    tickets.append(ticket)
    index.add(len(index), prices)
    ticket_to_prices[ticket] = prices

# Reshaping it for batch search:
favorite_ticker = 'AAPL'
matches, _, _ = index.search(ticket_to_prices[favorite_ticker], 10)
print([tickets[match] for match in matches])

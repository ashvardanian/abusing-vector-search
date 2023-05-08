import os
from statistics import covariance
import time

from usearch import Index

import pandas as pd
import numpy as np

directory: str = 'stocks'
last_days: int = 30
tickets: list[str] = []
ticket_to_prices: dict[str, np.array] = {}
index = Index(ndim=last_days)

for filename in os.listdir(directory):
    path = os.path.join(directory, filename)
    ticket = filename.split('.')[0]
    df = pd.read_csv(path)

    prices_list = df['Close'][-last_days:].to_list()
    prices = np.zeros(last_days, dtype=np.float32)
    prices[-len(prices_list):] = prices_list

    tickets.append(ticket)
    index.add(len(index), prices - np.mean(prices))
    ticket_to_prices[ticket] = prices

selected_ticker = 'AAPL'

tic = time.perf_counter()
selected_prices = ticket_to_prices[selected_ticker]
approx_matches, _, _ = index.search(
    selected_prices - np.mean(selected_prices), 10)
approx_tickets = [tickets[match] for match in approx_matches]
toc = time.perf_counter()
print('Approximate matches:', ','.join(approx_tickets))
print(f'- Measurement took {toc - tic:0.4f} seconds')


tic = time.perf_counter()
covariances = [
    (ticket, covariance(selected_prices, prices))
    for ticket, prices in ticket_to_prices.items()]
covariances = sorted(covariances, key=lambda x: x[1], reverse=True)
exact_tickets = [ticket for ticket, _ in covariances[:10]]
toc = time.perf_counter()
print('Exact matches:', ','.join(exact_tickets))
print(f'- Measurement took {toc - tic:0.4f} seconds')

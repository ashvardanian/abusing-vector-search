from transformers import BertTokenizer
from usearch import SetsIndex, HashIndex
from tqdm import tqdm
import numpy as np
import pandas as pd


sets_index = SetsIndex()
hash_index = HashIndex(bits=2048)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

df = pd.read_csv(
    'posts.csv',
    usecols=['text', 'by', 'id'],
    dtype={'text': 'str', 'by': 'str', 'id': 'int'},
    nrows=100000,
)
print(f'Loaded {len(df)} lines')


def text2set(sample: str) -> np.array:
    encoding = tokenizer.encode(sample, truncation=True)
    encoding = encoding[1:-1]
    encoding = sorted(set(encoding))
    encoding = np.array(encoding, dtype=np.int32)
    return encoding


def text2hashes(sample: str) -> np.array:
    words = sample.lower().split()
    return np.array([hash(word) for word in words], dtype=np.int64)


for idx, sample in tqdm(enumerate(df['text'])):
    if not isinstance(sample, str):
        continue
    sets_index.add(idx, text2set(sample))
    hash_index.add(idx, text2hashes(sample))

print(f'Added {len(sets_index)} titles to indices')


def print_matches(matches: np.array):
    for idx, match in enumerate(matches):
        row = df.iloc[match]
        title = row['text'].replace('\n', ' ')
        id = row['id']
        print(f'- {idx+1}. {title}')
        print(f'    https://news.ycombinator.com/item?id={id}')


try:
    while True:
        query: str = input('Please enter a search query: ')
        print('You entered:', query)

        print('For SetIndex the results are:')
        matches = sets_index.search(text2set(query), 3)
        print_matches(matches)

        print('For HashIndex the results are:')
        matches = hash_index.search(text2hashes(query), 3)
        print_matches(matches)

except KeyboardInterrupt:
    exit(0)

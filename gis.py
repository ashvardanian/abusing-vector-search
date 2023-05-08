from usearch import Index

import pandas as pd
import numpy as np
import geocoder

my_coordinates = np.array(geocoder.ip('me').latlng, dtype=np.float32)

df = pd.read_csv('cities.csv')
coordinates = np.zeros((df.shape[0], 2), dtype=np.float32)
coordinates[:, 0] = df['latitude'].to_numpy(dtype=np.float32)
coordinates[:, 1] = df['longitude'].to_numpy(dtype=np.float32)
labels = np.array(range(df.shape[0]), dtype=np.longlong)

index = Index(metric='haversine')
index.add(labels, coordinates)

# Reshaping it for batch search:
matches, _, _ = index.search(my_coordinates, 10)
print(df.iloc[matches])

"""Generate track coefficients for testing."""

import json
import numpy as np
import pandas as pd

df = pd.read_csv('testing/data/track_limits_coeffs.csv')

left = np.polyfit(df['LeftBnd_X'], df['LeftBnd_Y'], 3)
right = np.polyfit(df['RightBnd_X'], df['RightBnd_Y'], 3)

data = {}
data['left'] = left[::-1]
data['right'] = right[::-1]

with open('testing/data/track_limits_coeffs.json', 'w') as fp:
    fp.write(json.dumps(data, indent=4))

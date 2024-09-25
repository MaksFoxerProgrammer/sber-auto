import pandas as pd

import dill
import json

model = None
with open('data/ga_hits-002.pkl', 'rb') as file:
   model = dill.load(file)

print(model)

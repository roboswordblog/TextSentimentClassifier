import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn
import torch.functional.F
df = pd.read_csv('chat_dataset.csv')

def encodeSentiment(x):
    if x == "neutral":
        return 0
    elif x == "negative":
        return 1
    else:
        return 2

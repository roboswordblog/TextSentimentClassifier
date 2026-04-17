import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn
import torch.functional.F

df = pd.read_csv('chat_dataset.csv')
def encodeSentiment(x):
    if x == "neutral":
        return 0.0
    elif x == "negative":
        return 1.0
    else:
        return 2.0

def encodeLetters(x):
    alphabet_dict = {
        ' ': '0', 'a': '1', 'b': '2', 'c': '3', 'd': '4', 'e': '5',
        'f': '6', 'g': '7', 'h': '8', 'i': '9', 'j': '10', 'k': '11',
        'l': '12', 'm': '13', 'n': '14', 'o': '15', 'p': '16', 'q': '17',
        'r': '18', 's': '19', 't': '20', 'u': '21', 'v': '22', 'w': '23',
        'x': '24', 'y': '25', 'z': '26'
    }
    thing = ""
    for letter in x:
        thing += alphabet_dict[letter]
    return float(thing)
    
    

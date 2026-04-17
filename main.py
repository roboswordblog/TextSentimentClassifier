import torch
import pandas as pd
import numpy as np
from jinja2.compiler import F
from sklearn.model_selection import train_test_split
import torch.nn as nn

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
        'x': '24', 'y': '25', 'z': '26', '!': '27', '"': '28', '#': '29',
        '$': '30', '%': '31', '&': '32', "'": '33', '(': '34', ')': '35',
        '*': '36', '+': '37', ',': '38', '-': '39', '.': '40', '/': '41',
        ':': '42', ';': '43', '<': '44', '=': '45', '>': '46', '?': '47',
        '@': '48', '[': '49', '\\': '50', ']': '51', '^': '52', '_': '53',
        '`': '54', '{': '55', '|': '56', '}': '57', '~': '58'
    }

    thing = ""
    for letter in x.lower():
        thing += alphabet_dict[letter]
    return float(thing)

# do some goofya data handling
X = df["message"].apply(encodeLetters)
y = df["sentiment"].apply(encodeSentiment)
print(X)
print(y)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 1)

    def forward(self, x):
        x = nn.ReLU(self.fc1(x))
        x = nn.ReLU(self.fc2(x))
        x = nn.ReLU(self.fc3(x))
        x = self.out(x)
        return x
X = torch.FloatTensor(X)
y = torch.FloatTensor(y).reshape(-1, 1)
# splitting them
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
torch.manual_seed(41)
model = Model()
criterion = nn.BCELoss() # Standard for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train it
epochs = 1500

for i in range(epochs):
    # get result
    y_pred = model(X_train)
    # get loss
    loss = criterion(y_pred, y_train)
    # reset gradient
    optimizer.zero_grad()
    # go backwards and fix everything
    loss.backward()
    optimizer.step()
    # print it out every 10 epochs
    if i % 10 == 0:
        print(f"Epoch {i}, Loss: {loss.item()}")
# get the test results
with torch.no_grad():
    model.eval()
    test_outputs = model(X_test)
    predictions = (test_outputs >= 0.5).float()
    accuracy = (predictions == y_test).sum() / y_test.shape[0]
    print(f"Test Accuracy: {accuracy.item():.4f}")

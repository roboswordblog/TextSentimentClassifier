import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('chat_dataset.csv')
def encodeSentiment(x):
    if x == "neutral":
        return 0.0
    elif x == "negative":
        return 1.0
    else:
        return 2.0


vectorizer = TfidfVectorizer(max_features=500)

X = vectorizer.fit_transform(df["message"].astype(str)).toarray()
y = df["sentiment"].apply(encodeSentiment)

print(X)
print(y)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(300, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.out(x)
        return x
X = torch.FloatTensor(X)
y = torch.LongTensor(df["sentiment"].apply(encodeSentiment).values)
# splitting them
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
torch.manual_seed(41)
model = Model()
criterion = nn.CrossEntropyLoss() 
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
        predictions = torch.argmax(y_pred, dim=1)
        accuracy = (predictions == y_train).float().mean()
        print(accuracy)
# get the test results
with torch.no_grad():
    model.eval()
    test_outputs = model(X_test)
    predictions = torch.argmax(test_outputs, dim=1)
    accuracy = (predictions == y_test).float().mean()
    print(f"Test Accuracy: {accuracy.item():.4f}")
with torch.no_grad():
    model.eval()

sentiment_map = {0: "Neutral", 1: "Negative", 2: "Positive"}

user_input = input("\nEnter a message (or type 'quit' to stop): ")
input_vector = vectorizer.transform([user_input]).toarray()
input_tensor = torch.FloatTensor(input_vector)
    
with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()
    
    print(f"Predicted Sentiment: {sentiment_map[prediction]}")

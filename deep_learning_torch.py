import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

class Model(nn.Module):
    """
    Normalization-Linear in PyTorch
    """
    def __init__(self, seq_len, pred_len):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :]  # Extract the last time step
        x = x - seq_last  # Normalize by the last time step
        x = x.permute(0, 2, 1)  # [Batch, Channel, Input length]
        x = self.linear(x)  # Apply linear layer
        x = x.permute(0, 2, 1)  # [Batch, Output length, Channel]
        x = x + seq_last  # Add back the normalization term
        return x  # [Batch, Output length, Channel]

# Parameters
seq_len = 12
pred_len = 1
input_channels = 3
epochs = 50

# Load and sort data
file_path = "./data/apple_quarterly_balance_sheet.csv"
data = pd.read_csv(file_path)
data.sort_values(by='fiscalDateEnding', inplace=True)

# Prepare data
columns_to_use = ['totalShareholderEquity', 'totalAssets', 'totalLiabilities']
data[columns_to_use] = data[columns_to_use].apply(pd.to_numeric, errors='coerce')
data = data.dropna(subset=columns_to_use)
data_values = data[columns_to_use].values

# Generate sequential data
def generate_data(data, seq_len, pred_len):
    X, Y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(Y)

X, Y = generate_data(data_values, seq_len, pred_len)

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32)

# Instantiate the model
model = Model(seq_len, pred_len)
criterion = nn.L1Loss()  # Mean Absolute Error
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Recursive prediction
model.eval()
recursive_predictions = []
current_input = X_test[:, :seq_len, :]  # Initial input from test data
for _ in range(len(X_test)):
    pred = model(current_input)
    recursive_predictions.append(pred[:, -1, :].detach().numpy())
    current_input = torch.cat([current_input[:, 1:, :], pred[:, -1:, :]], dim=1)  # Slide window

# Convert predictions to numpy array
recursive_predictions = np.array(recursive_predictions).transpose((1, 0, 2))  # [Batch, Time, Channel]

# Adjust predictions to satisfy the accounting equation
def adjust_predictions(predictions):
    for batch in range(predictions.shape[0]):
        for t in range(predictions.shape[1]):
            assets = predictions[batch, t, 1]  # totalAssets
            liabilities = predictions[batch, t, 2]  # totalLiabilities
            equity = predictions[batch, t, 0]  # totalShareholderEquity
            discrepancy = assets - (liabilities + equity)
            if discrepancy != 0:
                total = abs(liabilities) + abs(equity)
                if total > 0:
                    predictions[batch, t, 2] += discrepancy * (abs(liabilities) / total)  # Adjust liabilities
                    predictions[batch, t, 0] += discrepancy * (abs(equity) / total)  # Adjust equity
    return predictions

adjusted_predictions = adjust_predictions(recursive_predictions)

# Calculate mean absolute error for each target
mae_results = {}
Y_test_np = Y_test.numpy()
for i, col in enumerate(columns_to_use):
    mae_results[col] = np.mean(np.abs(adjusted_predictions[:, :, i] - Y_test_np[:, :, i]))

# Print MAE results
print("Mean Absolute Error for each target:")
for col, mae in mae_results.items():
    print(f"{col}: {mae}")

# Calculate balance check errors
balance_check = adjusted_predictions[:, :, 1] - adjusted_predictions[:, :, 2] - adjusted_predictions[:, :, 0]  # totalAssets - totalLiabilities - totalShareholderEquity
balance_check_mean = np.mean(np.abs(balance_check))
print(f"Mean Balance Check Error: {balance_check_mean}")

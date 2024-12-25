import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load and sort data
file_path = "/Users/mixian/Downloads/Q2/microsoft_quarterly_balance_sheet.csv"
data = pd.read_csv(file_path)
data.sort_values(by='fiscalDateEnding', inplace=True)

# Ensure numeric columns
columns_to_use = ['totalShareholderEquity', 'totalAssets', 'totalLiabilities']
data[columns_to_use] = data[columns_to_use].apply(pd.to_numeric, errors='coerce')
data = data.dropna(subset=columns_to_use)

# Train-test split
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# Linear regression helper
def train_and_predict(train_data, test_data, target_column):
    X_train = np.arange(len(train_data)).reshape(-1, 1)
    y_train = train_data[target_column].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    X_test = np.arange(len(train_data), len(train_data) + len(test_data)).reshape(-1, 1)
    predictions = model.predict(X_test)
    return predictions

# Predict directly for totalAssets, totalLiabilities, totalShareholderEquity
predictions = {}
for column in columns_to_use:
    predictions[column] = train_and_predict(train_data, test_data, column)

# Add predictions to test_data
test_data['totalAssets_pred'] = predictions['totalAssets']
test_data['totalLiabilities_pred'] = predictions['totalLiabilities']
test_data['totalShareholderEquity_pred'] = predictions['totalShareholderEquity']

# Adjust predictions to match the accounting equation
def adjust_accounting_equation(test_data):
    for i in range(len(test_data)):
        assets = test_data.loc[i, 'totalAssets_pred']
        liabilities = test_data.loc[i, 'totalLiabilities_pred']
        equity = test_data.loc[i, 'totalShareholderEquity_pred']

        discrepancy = assets - (liabilities + equity)
        if discrepancy != 0:
            # Proportionally adjust liabilities and equity
            total = abs(liabilities) + abs(equity)
            if total > 0:
                adj_liabilities = (abs(liabilities) / total) * discrepancy
                adj_equity = (abs(equity) / total) * discrepancy

                test_data.loc[i, 'totalLiabilities_pred'] += adj_liabilities
                test_data.loc[i, 'totalShareholderEquity_pred'] += adj_equity
                test_data.loc[i, 'totalAssets_pred'] -= discrepancy

    return test_data

test_data = adjust_accounting_equation(test_data)

# Validate accounting equation consistency
test_data['accounting_equation_check'] = (
    test_data['totalAssets_pred'] -
    (test_data['totalLiabilities_pred'] + test_data['totalShareholderEquity_pred'])
)
mean_balance_check = test_data['accounting_equation_check'].abs().mean()

# Evaluate predictions
def calculate_errors(test_data, columns):
    errors = {}
    for col in columns:
        error = np.mean(np.abs(test_data[col] - test_data[f'{col}_pred']))
        errors[col] = error
    return errors

errors = calculate_errors(test_data, columns_to_use)

# Print errors and balance check
print("Prediction Errors:")
for col, error in errors.items():
    print(f"{col}: {error:.4f}")

print(f"Mean Accounting Equation Error after adjustment: {mean_balance_check:.4f}")

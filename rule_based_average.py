import pandas as pd
import numpy as np

# Load and sort data
file_path = "/Users/mixian/Downloads/Q2/microsoft_quarterly_balance_sheet.csv"
data = pd.read_csv(file_path)
data.sort_values(by='fiscalDateEnding', inplace=True)

# Ensure numeric columns
columns_to_process = [
    'totalCurrentAssets', 'propertyPlantEquipment', 'intangibleAssets',
    'longTermInvestments', 'otherNonCurrentAssets', 'totalCurrentLiabilities',
    'totalNonCurrentLiabilities', 'retainedEarnings', 'commonStock',
    'totalAssets', 'totalLiabilities', 'totalShareholderEquity',
    'accumulatedDepreciationAmortizationPPE'
]
data[columns_to_process] = data[columns_to_process].apply(pd.to_numeric, errors='coerce').fillna(0)

# Historical average helper function using the last three data points
def predict_with_recent_average(train_data, test_data, column_name):
    recent_values = train_data[column_name].tail(3).dropna()
    if len(recent_values) < 3:
        historical_avg = recent_values.mean()  # Use available data if less than 3 points
    else:
        historical_avg = recent_values.mean()
    test_data[f'{column_name}_pred'] = historical_avg
    return test_data

# Predict components of totalCurrentAssets and totalNonCurrentAssets
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# Predict totalCurrentAssets using recent average
test_data = predict_with_recent_average(train_data, test_data, 'totalCurrentAssets')

# Predict propertyPlantEquipment with CapEx and Depreciation
train_data['CapEx'] = train_data['propertyPlantEquipment'].diff().abs()
train_data['Depreciation'] = train_data['accumulatedDepreciationAmortizationPPE'].diff().abs()

average_capex = train_data['CapEx'].tail(3).mean()
average_depreciation = train_data['Depreciation'].tail(3).mean()

last_pp_e = train_data.iloc[-1]['propertyPlantEquipment']
test_data['propertyPlantEquipment_pred'] = (
    last_pp_e + average_capex - average_depreciation
)

# Predict other non-current asset components using recent averages
for col in ['intangibleAssets', 'longTermInvestments', 'otherNonCurrentAssets']:
    test_data = predict_with_recent_average(train_data, test_data, col)

# Compute totalNonCurrentAssets
test_data['totalNonCurrentAssets_pred'] = (
    test_data['propertyPlantEquipment_pred'] +
    test_data['intangibleAssets_pred'] +
    test_data['longTermInvestments_pred'] +
    test_data['otherNonCurrentAssets_pred']
)

# Compute totalAssets
test_data['totalAssets_pred'] = (
    test_data['totalCurrentAssets_pred'] + test_data['totalNonCurrentAssets_pred']
)

# Predict totalLiabilities
for col in ['totalCurrentLiabilities', 'totalNonCurrentLiabilities']:
    test_data = predict_with_recent_average(train_data, test_data, col)

test_data['totalLiabilities_pred'] = (
    test_data['totalCurrentLiabilities_pred'] + test_data['totalNonCurrentLiabilities_pred']
)

# Predict totalShareholderEquity
for col in ['retainedEarnings', 'commonStock']:
    test_data = predict_with_recent_average(train_data, test_data, col)

test_data['totalShareholderEquity_pred'] = (
    test_data['retainedEarnings_pred'] + test_data['commonStock_pred']
)

# Evaluate errors
def calculate_errors(test_data, columns):
    errors = {}
    for col in columns:
        error = np.mean(np.abs(test_data[col] - test_data[f'{col}_pred']))
        errors[col] = error
    return errors

columns_to_evaluate = ['totalAssets', 'totalLiabilities', 'totalShareholderEquity']
errors = calculate_errors(test_data, columns_to_evaluate)

# Print results
print("Prediction Errors:")
for col, error in errors.items():
    print(f"{col}: {error:.4f}")

# Validate accounting equation consistency
test_data['accounting_equation_check'] = (
    test_data['totalAssets_pred'] -
    (test_data['totalLiabilities_pred'] + test_data['totalShareholderEquity_pred'])
)
mean_balance_check = test_data['accounting_equation_check'].abs().mean()
print(f"Mean Accounting Equation Error: {mean_balance_check:.4f}")

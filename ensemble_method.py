import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import json
from LLM.gpt import GPT_agent
from LLM.methods import LLM_Method
import ast

agent = GPT_agent()
methods = LLM_Method()
# Load and sort data
file_path = "./data/amazon_quarterly_balance_sheet.csv"
news_path = "LLM/news/amazon.json"
data = pd.read_csv(file_path)
data.sort_values(by='fiscalDateEnding', inplace=True)
news = json.load(open(news_path))


# Ensure numeric columns
columns_to_use = ['totalShareholderEquity', 'totalAssets', 'totalLiabilities']
data[columns_to_use] = data[columns_to_use].apply(pd.to_numeric, errors='coerce')
data = data.dropna(subset=columns_to_use)
times = list(data.fiscalDateEnding)
news = [news[time]["news"][0]["summary"] for time in times]

# Train-test split
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]
train_news = news[:train_size]
test_news = news[train_size:]

history_data = {
    "news": train_news,
    "totalShareholderEquity": train_data.totalShareholderEquity.tolist(),
    "totalAssets": train_data.totalAssets.tolist(),
    "totalLiabilities": train_data.totalLiabilities.tolist()
}

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
   
print("predicted assets:",predictions['totalAssets'])
print("predicted liability:",predictions['totalLiabilities'])
print("predicted equity:",predictions['totalShareholderEquity'])

# Adjust predictions to match the accounting equation
def adjust_accounting_equation(predictions):
    for i in range(len(predictions)):
        assets = predictions['totalAssets'][i]
        liabilities = predictions['totalLiabilities'][i]
        equity = predictions['totalShareholderEquity'][i]

        discrepancy = assets - (liabilities + equity)
        if discrepancy != 0:
            # Proportionally adjust liabilities and equity
            total = abs(liabilities) + abs(equity)
            if total > 0:
                adj_liabilities = (abs(liabilities) / total) * discrepancy
                adj_equity = (abs(equity) / total) * discrepancy

                predictions['totalLiabilities'][i] += adj_liabilities
                predictions['totalShareholderEquity'][i] += adj_equity
                predictions['totalAssets'][i] -= discrepancy

    return predictions

def response_to_json(response):
    try:
        start_idx = response.index('{')
        end_idx = response.rindex('}')
        python_dict = ast.literal_eval(response[start_idx : end_idx + 1])
        response = json.dumps(python_dict)
        response = json.loads(response)
        return response
    except Exception as e:
        print(e)
        return {}
    
predictions = adjust_accounting_equation(test_data)
print(predictions)
prompt = methods.ensemble(history_data, predictions, test_news)
# Query the LLM agent with the generated prompt.
prediction_str = agent.ask_text(prompt)
print(prediction_str)
prediction = response_to_json(prediction_str)

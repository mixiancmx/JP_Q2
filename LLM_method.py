from LLM.gpt import GPT_agent
from LLM.methods import LLM_Method
import pandas as pd
import json
import numpy as np
import ast

agent = GPT_agent()
methods = LLM_Method()

# Load and sort data
file_path = "data/amazon_quarterly_balance_sheet.csv"
news_path = "LLM/news/amazon.json"

data = pd.read_csv(file_path)
data.sort_values(by='fiscalDateEnding', inplace=True)
news = json.load(open(news_path))

# Prepare data
columns_to_use = ['totalShareholderEquity', 'totalAssets', 'totalLiabilities']
data[columns_to_use] = data[columns_to_use].apply(pd.to_numeric, errors='coerce')
data = data.dropna(subset=columns_to_use)
times = list(data.fiscalDateEnding)
news = [news[time]["news"][0]["summary"] for time in times]
history_size = int(len(data) * 0.8)
history_df = data.iloc[:history_size]
test_df = data.iloc[history_size:]
history_news = news[:history_size]
test_news = news[history_size:]

# Convert data frames to dictionaries with lists.
history_data = {
    "news": history_news,
    "totalShareholderEquity": history_df.totalShareholderEquity.tolist(),
    "totalAssets": history_df.totalAssets.tolist(),
    "totalLiabilities": history_df.totalLiabilities.tolist()
}
test_data = {
    "news": test_news,
    "totalShareholderEquity": test_df.totalShareholderEquity.tolist(),
    "totalAssets": test_df.totalAssets.tolist(),
    "totalLiabilities": test_df.totalLiabilities.tolist()
}

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
    
predictions = {"totalShareholderEquity":[],"totalAssets":[],"totalLiabilities":[]}
# Rolling prediction: for each test instance, use the current history and the test news update.
num_test = len(test_data["news"])
print(num_test)

for i in range(num_test):
    # Get ground truth values (for logging purposes).
    prediction = {}
    while prediction == {}:
        gt_equity = test_data["totalShareholderEquity"][i]
        gt_assets = test_data["totalAssets"][i]
        gt_liability = test_data["totalLiabilities"][i]
        
        # Get the corresponding news update for the test quarter.
        current_news = test_data["news"][i]
        
        # Generate prompt using zero-shot pure numerical method.
        prompt = methods.zero_shot_with_news(history_data, current_news)
        # Query the LLM agent with the generated prompt.
        prediction_str = agent.ask_text(prompt)
        prediction = response_to_json(prediction_str)

    # Append the prediction to the test data predictions.
    test_data["totalShareholderEquity_pred"].append(prediction["totalShareholderEquity"])
    test_data["totalAssets_pred"].append(prediction["totalAssets"])
    test_data["totalLiabilities_pred"].append(prediction["totalLiabilities"])
    
    # Update the historical data with the prediction (simulate a rolling forecast).
    history_data["totalShareholderEquity"].append(prediction["totalShareholderEquity"])
    history_data["totalAssets"].append(prediction["totalAssets"])
    history_data["totalLiabilities"].append(prediction["totalLiabilities"])
    history_data["news"].append(current_news)

    predictions["totalShareholderEquity"].append(prediction["totalShareholderEquity"])
    predictions["totalAssets"].append(prediction["totalAssets"])
    predictions["totalLiabilities"].append(prediction["totalLiabilities"])
    

# Calculate and print the prediction errors.
for name in ["totalShareholderEquity", "totalAssets", "totalLiabilities"]:
    errors = np.mean(np.abs(test_data[name] - predictions[name]))
    print("Prediction Errors:", name, errors)

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
    
predictions = {'totalShareholderEquity': [74929569897.95918,
  76766105428.57143,
  78602640959.18367,
  80439176489.79591,
  82275712020.40816,
  84112247551.02042,
  85948783081.63264,
  87785318612.24489,
  89621854142.85715,
  91458389673.46939,
  93294925204.08162,
  95131460734.69388,
  96967996265.30612],
 'totalAssets': [271584285714.28574,
  278233514971.42865,
  284882744228.5715,
  291531973485.71436,
  298181202742.8572,
  304830432000.00006,
  311479661257.14294,
  318128890514.2857,
  324778119771.42865,
  331427349028.57153,
  338076578285.71436,
  344725807542.85724,
  351375036800.0],
 'totalLiabilities': [188317632857.14288,
  192926245665.30615,
  197534858473.46942,
  202143471281.63266,
  206752084089.79596,
  211360696897.95923,
  215969309706.1225,
  220577922514.28577,
  225186535322.449,
  229795148130.61227,
  234403760938.77554,
  239012373746.9388,
  243620986555.1021]}

# Calculate and print the prediction errors.
for name in ["totalShareholderEquity", "totalAssets", "totalLiabilities"]:
    errors = np.mean(np.abs(np.array(test_data[name]) - np.array(predictions[name])))
    print("Prediction Errors:", name, errors)
bias = []
for i in range(len(test_data["totalAssets"])):
    bias.append(predictions["totalAssets"][i] -  predictions["totalLiabilities"][i] - predictions["totalShareholderEquity"][i])
print(np.mean(np.abs(bias)))
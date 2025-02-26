
def compute_anchor_list(series, threshold=0.02):
    """
    Computes a list of anchors ("rising", "stable", "falling") for a numerical series.
    The first entry is always "stable". For subsequent values, if the percentage change
    from the previous value is below the threshold, the anchor is "stable"; otherwise,
    it is "rising" if the value increased, or "falling" if it decreased.
    
    Parameters:
      series: list of numerical values.
      threshold: float, minimum relative change to be considered significant (default 0.02).
      
    Returns:
      List of strings of the same length as series.
    """
    anchors = []
    n = len(series)
    for i in range(n):
        if i == 0:
            anchors.append("stable")
        else:
            prev = series[i-1]
            curr = series[i]
            if prev == 0:
                change = 0
            else:
                change = (curr - prev) / prev
            if abs(change) < threshold:
                anchors.append("stable")
            elif change > 0:
                anchors.append("rising")
            else:
                anchors.append("falling")
    return anchors

class LLM_Method:
    def __init__(self):
        pass

    def zero_shot_pure_numerical(self, historical_data, future_news):
        """
        Zero-Shot (Pure Numerical):
        This method uses only the historical numerical data to prompt the LLM.
        Note: It does not incorporate any historical or new news context.
        
        Parameters:
        historical_data: dict with keys 'totalShareholderEquity', 'totalAssets', 'totalLiabilities', 'news'.
                        (Only the numerical lists are used.)
        future_news: string (ignored in this method).
                        
        Returns:
        A prompt string instructing the LLM to predict the next quarter's financial data using only the historical numbers.
        """
        tse = ", ".join(map(str, historical_data.get("totalShareholderEquity", [])))
        ta  = ", ".join(map(str, historical_data.get("totalAssets", [])))
        tl  = ", ".join(map(str, historical_data.get("totalLiabilities", [])))
        
        prompt = (
            f"Quarterly Financial Data:\n"
            f"- Historical Values of Shareholder Equity: [{tse}]\n"
            f"- Historical Values of Assets: [{ta}]\n"
            f"- Historical Values of Liabilities: [{tl}]\n\n"
            "Without considering any external news or contextual information, "
            "Output a dictionary with keys 'totalShareholderEquity', 'totalAssets', and 'totalLiabilities' corresponding to the predicted values for the next quarter."
        )
        return prompt

    def zero_shot_with_anchors(self, historical_data, future_news):
        """
        Zero-Shot with Anchors:
        This method analyzes historical numerical data for each series and derives a corresponding
        semantic anchor list ("rising", "stable", "falling") based on percentage changes. Each series gets
        its own anchor list. These anchor lists, along with the new news update, are then incorporated into
        the prompt to guide the LLM in predicting the next quarter's data.
        
        Parameters:
        historical_data: dict with keys 'totalShareholderEquity', 'totalAssets', 'totalLiabilities', 'news'.
                        (Only the numerical lists are used for analysis.)
        future_news: string representing the new news update.
        threshold: float representing the minimum relative change to be considered significant (default 0.02).
                            
        Returns:
        A prompt string that includes the numerical data and the derived anchor lists for each series,
        followed by the new news update.
        """
        equity = historical_data.get("totalShareholderEquity", [])
        assets = historical_data.get("totalAssets", [])
        liabilities = historical_data.get("totalLiabilities", [])
        
        anchors_equity = compute_anchor_list(equity, 0.05)
        anchors_assets = compute_anchor_list(assets, 0.05)
        anchors_liabilities = compute_anchor_list(liabilities, 0.05)
        
        # Format numerical data and anchor lists as comma-separated strings.
        tse = ", ".join(map(str, equity))
        ta  = ", ".join(map(str, assets))
        tl  = ", ".join(map(str, liabilities))
        
        anchor_equity_str = ", ".join(anchors_equity)
        anchor_assets_str = ", ".join(anchors_assets)
        anchor_liabilities_str = ", ".join(anchors_liabilities)
        
        prompt = (
            f"Quarterly Financial Data:\n"
            f"- Historical Values of Shareholder Equity: [{tse}]\n"
            f"- Historical Values of Assets: [{ta}]\n"
            f"- Historical Values of Liabilities: [{tl}]\n\n"
            f"Derived Semantic Anchors per Series:\n"
            f"  Historical Equity Trend: [{anchor_equity_str}]\n"
            f"  Historical Assets Trend: [{anchor_assets_str}]\n"
            f"  Historical Liabilities Trend: [{anchor_liabilities_str}]\n\n"
            "Based on the above numerical trends and the derived semantic cues for each series, "
            "Output a dictionary with keys 'totalShareholderEquity', 'totalAssets', and 'totalLiabilities' corresponding to the predicted values for the next quarter."
        )
        return prompt


    def zero_shot_with_news(self, historical_data, future_news):
        """
        Zero-Shot with News and Relevant Information as Input:
        This method incorporates historical numerical data and the new news update,
        but does not include historical news context.
        
        Parameters:
        historical_data: dict with keys 'totalShareholderEquity', 'totalAssets', 'totalLiabilities', 'news'.
                        (Only numerical data is used.)
        future_news: string representing the new news update.
                        
        Returns:
        A prompt string that includes the historical numerical data and the new news update,
        asking the LLM to predict the next quarter's financial data.
        """
        tse = ", ".join(map(str, historical_data.get("totalShareholderEquity", [])))
        ta  = ", ".join(map(str, historical_data.get("totalAssets", [])))
        tl  = ", ".join(map(str, historical_data.get("totalLiabilities", [])))
        
        prompt = (
            f"Quarterly Financial Data:\n"
            f"- Historical Values of Shareholder Equity: [{tse}]\n"
            f"- Historical Values of Assets: [{ta}]\n"
            f"- Historical Values of Liabilities: [{tl}]\n\n"
            f"New News Update: \"{future_news}\"\n\n"
            "Based solely on the numerical trends and the external news update provided, "
            "Output a dictionary with keys 'totalShareholderEquity', 'totalAssets', and 'totalLiabilities' corresponding to the predicted values for the next quarter."
        )
        return prompt


    def few_shot_with_news(self, historical_data, future_news):
        """
        Few-Shot with News and Relevant Information as Input:
        This approach provides a few examples, historical numerical data, and historical news context.
        The goal is to help the LLM learn the correlation between historical numbers and historical news,
        and then predict the next quarter's data given the new news update.
        
        Parameters:
        historical_data: dict with keys 'totalShareholderEquity', 'totalAssets', 'totalLiabilities', 'news'.
                        Uses both numerical data and historical news.
        future_news: string representing the new news update.
                        
        Returns:
        A prompt string that includes a few-shot example section, the current historical data and historical news context,
        and the new news update to guide prediction.
        """
        tse = ", ".join(map(str, historical_data.get("totalShareholderEquity", [])))
        ta  = ", ".join(map(str, historical_data.get("totalAssets", [])))
        tl  = ", ".join(map(str, historical_data.get("totalLiabilities", [])))
        hist_news = " ".join(historical_data.get("news", []))
        
        examples = (
            "Example 1:\n"
            "Historical Data: Equity: [100, 110, 115], Assets: [200, 210, 220], Liabilities: [100, 100, 105].\n"
            "News: \"Steady economic growth observed.\"\n"
            "Prediction: Next quarter - Equity: 118, Assets: 225, Liabilities: 107.\n\n"
            "Example 2:\n"
            "Historical Data: Equity: [150, 155, 160], Assets: [300, 310, 320], Liabilities: [150, 155, 160].\n"
            "News: \"Market volatility due to policy changes.\"\n"
            "Prediction: Next quarter - Equity: 162, Assets: 325, Liabilities: 163.\n\n"
        )
        
        prompt = (
            f"Few-Shot Examples:\n{examples}\n"
            f"Current Historical Data:\n"
            f"- Historical Values of Shareholder Equity: [{tse}]\n"
            f"- Historical Values of Assets: [{ta}]\n"
            f"- Historical Values of Liabilities: [{tl}]\n\n"
            f"Historical News Context: \"{hist_news}\"\n\n"
            f"New News Update: \"{future_news}\"\n\n"
            "Based on the above examples and contextual information, "
            "Output a dictionary with keys 'totalShareholderEquity', 'totalAssets', and 'totalLiabilities' corresponding to the predicted values for the next quarter."
        )
        return prompt


    def few_shot_with_cot_and_news(self, historical_data, future_news):
        """
        Few-Shot with Chain of Thought (CoT) and News:
        This method builds on the few-shot approach by additionally providing a chain-of-thought
        reasoning process example that outlines the steps typically followed in an economic balance sheet analysis.
        
        Parameters:
        historical_data: dict with keys 'totalShareholderEquity', 'totalAssets', 'totalLiabilities', 'news'.
                        Uses both numerical data and historical news.
        future_news: string representing the new news update.
                        
        Returns:
        A prompt string that includes few-shot examples, historical data and news context,
        as well as a detailed chain-of-thought example describing the reasoning process.
        """
        tse = ", ".join(map(str, historical_data.get("totalShareholderEquity", [])))
        ta  = ", ".join(map(str, historical_data.get("totalAssets", [])))
        tl  = ", ".join(map(str, historical_data.get("totalLiabilities", [])))
        hist_news = " ".join(historical_data.get("news", []))
        
        cot_example = (
            "Chain-of-Thought Example:\n"
            "Step 1: Analyze the trend in historical equity, assets, and liabilities. Note whether the values are increasing, decreasing, or stable.\n"
            "Step 2: Consider the impact of news on these trends. For instance, positive economic news may support an increase in equity and assets, while negative news may indicate caution.\n"
            "Step 3: Combine these insights to forecast the next quarter's figures. For example, if equity and assets have been gradually increasing and recent news is positive, the forecast should reflect a moderate rise.\n"
        )
        
        examples = (
            "Example 1:\n"
            "Historical Data: Equity: [100, 110, 115], Assets: [200, 210, 220], Liabilities: [100, 100, 105].\n"
            "News: \"Steady economic growth observed.\"\n"
            "Prediction: Next quarter - Equity: 118, Assets: 225, Liabilities: 107.\n\n"
            "Example 2:\n"
            "Historical Data: Equity: [150, 155, 160], Assets: [300, 310, 320], Liabilities: [150, 155, 160].\n"
            "News: \"Market volatility due to policy changes.\"\n"
            "Prediction: Next quarter - Equity: 162, Assets: 325, Liabilities: 163.\n\n"
        )
        
        prompt = (
            f"Few-Shot Examples:\n{examples}\n"
            f"Historical Data:\n"
            f"- Historical Values of Shareholder Equity: [{tse}]\n"
            f"- Historical Values of Assets: [{ta}]\n"
            f"- Historical Values of Liabilities: [{tl}]\n\n"
            f"Historical News Context: \"{hist_news}\"\n\n"
            f"New News Update: \"{future_news}\"\n\n"
            f"{cot_example}\n"
            "Using the above reasoning process, provide a detailed step-by-step analysis and Output a dictionary with keys 'totalShareholderEquity', 'totalAssets', and 'totalLiabilities' corresponding to the predicted values for the next quarter."
        )
        return prompt

    def ensemble(self, historical_data, predicted_value, future_news):
        tse = ", ".join(map(str, historical_data.get("totalShareholderEquity", [])))
        ta  = ", ".join(map(str, historical_data.get("totalAssets", [])))
        tl  = ", ".join(map(str, historical_data.get("totalLiabilities", [])))
        pse = ", ".join(map(str, predicted_value.get("totalShareholderEquity", [])))
        pa  = ", ".join(map(str, predicted_value.get("totalAssets", [])))
        pl  = ", ".join(map(str, predicted_value.get("totalLiabilities", [])))

        prompt = (
            f"Quarterly Financial Data:\n"
            f"- Historical Values of Shareholder Equity: [{tse}]\n"
            f"- Historical Values of Assets: [{ta}]\n"
            f"- Historical Values of Liabilities: [{tl}]\n\n"
            f"Recent Predicted Value:\n"
            f"- Shareholder Equity: [{pse}]\n"
            f"- Assets: [{pa}]\n"
            f"- Liabilities: [{pl}]\n\n"
            "Your job is to adjust the predicted value based on the news related to the company"
            f"News (each correspond to a prediction value): [{future_news}]\n\n"
            "Predicted value of Equity, Assets, Liabilities should match the accounting equilty constraint. Assets = Equity + Liabilities"
            "Output a dictionary with keys 'totalShareholderEquity', 'totalAssets', and 'totalLiabilities' corresponding to the predicted values after adjustment."
        )
        return prompt
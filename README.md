For details, please see report_part1.pdf and report_part2.pdf.

## Part 1 Overview
In this project, I explore several distinct approaches for forecasting key balance sheet components, including total assets, total liabilities, and shareholder equity.

Firstly, inspired by the methods of Velez-Pareja (from the paper provided by the project), I develop a rule-based average method that utilizes historical averages to predict future values, providing a simple yet interpretable baseline. Secondly, I implement a machine learning method, employing linear regression to directly forecast key balance sheet items without decomposition. Lastly, I explore a deep learning approach, incorporating a sequence-to-sequence neural network for time-series predictions.  

These methods are evaluated on real-world quarterly balance sheet data, comparing their prediction accuracy, adherence to the accounting equation, and computational efficiency. My findings provide a comprehensive assessment of the performance of different approaches in financial forecasting, highlighting the limitations of each method and offering insights into building more accurate and reliable predictive models.

## Part 2 Overview
Large language models (LLMs) have recently demonstrated significant potential in time series forecasting by effectively integrating textual information. Building on this evolving research, I leverage GPT-4's robust capabilities for financial statement analysis. GPT-4's ease of integration, scalable batch processing, and high accuracy make it particularly well-suited for various forecasting scenarios—from numerical predictions to advanced multi-modal methods. In this report, I study several strategies of LLMs on balance sheet forecasting, including few-shot examples, chain-of-thought reasoning, and anchor point techniques. Additionally, I propose an ensemble strategy that combines GPT-4’s insights with traditional linear regression models, offering a comprehensive forecasting solution and achieve best results among all methods I tested. Finally, I discuss the advantages of this approach, presenting a recommendation for potential users.
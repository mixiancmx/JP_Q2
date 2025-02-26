import json
import base64
import requests
import os
from openai import OpenAI

class GPT_agent:
    def __init__(self):
        self.client = OpenAI(
            api_key="sk-QpHUrsblHgB7kAzcpwLmrFz3yKKTiFVlFOW2vgVc7ARfqsXR",
            base_url="https://a.fe8.cn/v1",
        )

    def ask_text(self, question):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ]
        response = self.client.chat.completions.create(messages=messages, model="gpt-4o")

        try:
            return response.choices[0].message.content
        except:
            print('ERROR:', response)
            return json.dumps({})
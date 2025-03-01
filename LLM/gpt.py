import json
import base64
import requests
import os
from openai import OpenAI

class GPT_agent:
    def __init__(self):
        self.client = OpenAI(
            api_key="", #enter yours
            base_url="",
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

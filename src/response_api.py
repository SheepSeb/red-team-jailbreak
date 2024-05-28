import requests
import json


class ResponseAPI:
    def __init__(self):
        self.url = 'http://localhost:11434/api/generate'

    def get_response_text(self, model: str, prompt: str) -> str:
        data = {
            "model": model,
            "prompt": prompt
        }

        response = requests.post(self.url, json=data)

        if response.status_code == 200:
            return self.parse_response(response)
        else:
            return f"Request failed with status code: {response.status_code}"

    @staticmethod
    def parse_response(response: requests.Response) -> str:
        full_response = ""
        for item in response.iter_lines():
            json_response = item.decode('utf-8')
            if json_response.strip():  # skip empty lines
                response_data = json.loads(json_response)
                full_response += response_data.get('response', '')

        # Remove any trailing whitespace and newline characters
        full_response = full_response.strip()
        full_response = full_response.replace('\n', ' ')

        return full_response
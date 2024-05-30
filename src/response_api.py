import json

import requests


class ResponseAPI:
    def __init__(self):
        self.url = 'http://localhost:11434/api/generate'

    @staticmethod
    def create_system_prompt(model: str, system_prompt: str, query: str) -> str:
        if model == 'llama2' or model == 'llava':
            return f'[INST]<<SYS>>{system_prompt}<</SYS>>{query}[/INST]'
        elif model == 'llama3:instruct':
            return f'<|start_header_id|>{system_prompt}<|end_header_id|><|eot_id|><|start_header_id|>{query}<|end_header_id|><|eot_id|><|start_header_id|><|end_header_id|><|eot_id|>'
        elif model == 'llama2-uncensored':
            return f'{system_prompt}### HUMAN: {query} ### RESPONSE:'
        else:
            return f'system_prompt {system_prompt} query {query}'

    def get_response_text(self, model: str, prompt: str) -> str:
        data = {
            "model": model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
        }

        response = requests.post(self.url, json=data)

        if response.status_code == 200:
            return self.parse_response(response)
        else:
            return f"Request failed with status code: {response.status_code}"

    def get_response_text_with_temp(self, model: str, prompt: str, temperature: float) -> str:
        data = {
            "model": model,
            "prompt": prompt,
            "raw": True,
            "stream": False,
            "options": {
                "temperature": temperature
            }
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
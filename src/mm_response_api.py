import requests
import json
import base64
import os

class MMResponseAPI:

    def __init__(self):
        self.url = 'http://localhost:11434/api/generate'
        
    def get_image_data(self, image_path):
        if not os.path.isabs(image_path):
            image_path = os.path.abspath(image_path)

        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        return image_data


    def get_response_multimodal(self, model: str, prompt: str, image: str) -> str:

        data = {
            "model": model,
            "prompt": prompt,
            "image": image
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

import base64
import os

import requests

from src.response_api import ResponseAPI


class MMResponseAPI(ResponseAPI):

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_image_data(image_path):
        if not os.path.isabs(image_path):
            image_path = os.path.abspath(image_path)

        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        return image_data

    def get_response_multimodal(self, model: str, prompt: str, image: str) -> str:

        data = {
            "model": model,
            "prompt": prompt,
            "image": image,
            "raw": True,
            "stream": False,
        }

        response = requests.post(self.url, json=data)

        if response.status_code == 200:
            return self.parse_response(response)
        else:
            return f"Request failed with status code: {response.status_code}"
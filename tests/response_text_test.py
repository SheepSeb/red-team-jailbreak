from src.response_api import ResponseAPI
import unittest


class TestResponseAPI(unittest.TestCase):
    def test_get_response_text(self):
        response_api = ResponseAPI()
        response = response_api.get_response_text('llama2', 'Why the sky is blue?')
        self.assertNotEqual(response, '')

    def test_is_str(self):
        response_api = ResponseAPI()
        response = response_api.get_response_text('llama2', 'Why the sky is blue?')
        self.assertIsInstance(response, str)

    def test_newline(self):
        response_api = ResponseAPI()
        response = response_api.get_response_text('llama2', 'Why the sky is blue?')
        print(response)
        self.assertNotIn('\n', response)
import unittest
import base64
from src.response_api import ResponseAPI

with open('test.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')
    
class mm_TestResponseAPI(unittest.TestCase):
    def test_get_response_with_image(self):
        response_api = mm_ResponseAPI()

        response = response_api.get_response('llava', 'What is in the image?', image_data)
        self.assertNotEqual(response, '')

    def test_get_response_with_image_is_str(self):
        response_api = mm_ResponseAPI()

        response = response_api.get_response('llava', 'What is in the image?', image_data)
        self.assertIsInstance(response, str)

    def test_get_response_with_image_no_newline(self):
        response_api = mm_ResponseAPI()

        response = response_api.get_response('llava', 'What is in the image?', image_data)
        self.assertNotIn('\n', response)

if __name__ == '__main__':
    unittest.main()

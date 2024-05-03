import unittest
import base64
from src.mm_response_api import MM_ResponseAPI


    
class MM_TestResponseAPI(unittest.TestCase):

    with open('test.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')
    
    def test_get_response_with_image(self):
        response_api = MM_ResponseAPI()

        response = response_api.get_response('llava', 'What is in the image?', image_data)
        self.assertNotEqual(response, '')

    def test_get_response_with_image_is_str(self):
        response_api = MM_ResponseAPI()

        response = response_api.get_response('llava', 'What is in the image?', image_data)
        self.assertIsInstance(response, str)

    def test_get_response_with_image_no_newline(self):
        response_api = MM_ResponseAPI()

        response = response_api.get_response('llava', 'What is in the image?', image_data)
        self.assertNotIn('\n', response)

if __name__ == '__main__':
    unittest.main()

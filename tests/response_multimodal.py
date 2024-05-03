import unittest
import base64
from src.mm_response_api import MMResponseAPI


    
class TestMMResponseAPI(unittest.TestCase):

    
    
    def test_get_response_with_image(self):
        response_api = MMResponseAPI()
        image_path = 'test_images/test.jpg'  # Relative path to the image
        image = self.get_image_data(image_path)
        response = response_api.get_response_multimodal('llava', 'What is in the image?', image)
        self.assertNotEqual(response, '')

    def test_get_response_with_image_is_str(self):
        response_api = MMResponseAPI()
        image_path = 'test_images/test.jpg'  # Relative path to the image
        image = self.get_image_data(image_path)
        response = response_api.get_response_multimodal('llava', 'What is in the image?', image)
        self.assertIsInstance(response, str)

    def test_get_response_with_image_no_newline(self):
        response_api = MMResponseAPI()
        image_path = 'test_images/test.jpg'  # Relative path to the image
        image = self.get_image_data(image_path)
        response = response_api.get_response_multimodal('llava', 'What is in the image?', image)
        self.assertNotIn('\n', response)

if __name__ == '__main__':
    unittest.main()

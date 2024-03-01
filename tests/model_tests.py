import requests
import unittest


class TestClassifier(unittest.TestCase):
    def setUp(self) -> None:
        self.API_URL = "http://localhost:5000/classify"
        self.dialects = ['Egypt', 'Iraq', 'Saudi_Arabia', 'Mauritania', 'Algeria', 'Syria',
        'Oman', 'Tunisia', 'Lebanon', 'Morocco', 'Djibouti','United_Arab_Emirates','Kuwait', 
        'Libya', 'Bahrain', 'Qatar', 'Yemen', 'Palestine', 'Jordan', 'Somalia', 'Sudan']

    def test_response(self):
        """Test if the response of the /classify API endpoint is correct"""
        request_data = {"text": "حاجة حلوة اكيد"}
        response = requests.post(self.API_URL, json=request_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("class", response.json())
        self.assertIn(response.json()["class"], self.dialects)


if __name__ == "__main__":
    unittest.main()
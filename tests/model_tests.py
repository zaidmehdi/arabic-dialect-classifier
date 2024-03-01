import requests
import unittest


class TestClassifier(unittest.TestCase):
    def setUp(self) -> None:
        self.API_URL = "http://localhost:5000/classify"
        self.dialects = ['Egypt', 'Iraq', 'Saudi_Arabia', 'Mauritania', 'Algeria', 'Syria',
        'Oman', 'Tunisia', 'Lebanon', 'Morocco', 'Djibouti','United_Arab_Emirates','Kuwait', 
        'Libya', 'Bahrain', 'Qatar', 'Yemen', 'Palestine', 'Jordan', 'Somalia', 'Sudan']
        self.test_set = {
            "Egypt": "حضرتك بروح زي كدا؟ على طول النهار ده",
            "Iraq": "همين: شلون، زين، خوش، هواية، كلش، شقد",
            "Algeria": "بصح راك فاهم لازم الزيت",
            "Morocco": "واش نتا خدام ولا لا"
        }

    def test_response(self):
        """Test if the response of the /classify API endpoint is correct"""
        request_data = {"text": "حاجة حلوة اكيد"}
        response = requests.post(self.API_URL, json=request_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("class", response.json())
        self.assertIn(response.json()["class"], self.dialects)

    def test_model_output(self):
        """Test that the model correctly classifies obvious dialects"""
        for country, text, in self.test_set.items():
            request_data = {"text": text}
            response = requests.post(self.API_URL, json=request_data)
            self.assertEqual(response.json()["class"], country)

    
if __name__ == "__main__":
    unittest.main()
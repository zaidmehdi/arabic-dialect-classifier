import unittest

from src.main import classify_arabic_dialect


class TestClassifier(unittest.TestCase):
    def setUp(self) -> None:
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
        """Test if the response of the main function is correct"""
        text = "حاجة حلوة اكيد"
        predictions, probabilities = classify_arabic_dialect(text)
        self.assertEqual(len(predictions), 3)
        self.assertEqual(len(probabilities), 3)
        for i in range(3):
            self.assertIn(predictions[i], self.dialects)
            self.assertGreaterEqual(probabilities[i], 0)
            self.assertLessEqual(probabilities[i], 1)

    def test_model_output(self):
        """Test that the model correctly classifies obvious dialects"""
        for country, text, in self.test_set.items():
            predictions, _ = classify_arabic_dialect(text)
            self.assertEqual(predictions[0], country)

    
if __name__ == "__main__":
    unittest.main()
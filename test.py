import unittest
import pandas as pd
from sklearn.linear_model import LinearRegression
from main import load_and_preprocess_data, select_features, train_model

class TestMain(unittest.TestCase):
    def setUp(self):
        self.file_path = 'CarPrice_Assignment.csv'

    def test_load_and_preprocess_data(self):
        # Test loading and preprocessing data
        DataCar = load_and_preprocess_data(self.file_path)
        self.assertIsInstance(DataCar, pd.DataFrame)
        self.assertIn('price', DataCar.columns)
        # Add more assertions as needed to verify the data is correctly loaded and preprocessed

    def test_select_features(self):
        # Test selecting features
        features = select_features()
        self.assertIsInstance(features, list)
        self.assertEqual(len(features), 5)  # Assuming 5 features are selected
        # Add more assertions as needed to verify the correct features are selected

    def test_train_model(self):
        # Test training the model
        model = train_model(self.file_path)
        self.assertIsInstance(model, LinearRegression)
        # Add more assertions as needed to verify the model is correctly trained

if __name__ == '__main__':
    unittest.main()

import unittest
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Assuming the provided code is in a file named main.py
from main import vectorizer, model

class TestSpamModel(unittest.TestCase):

    def test_model_accuracy(self):
        # Load test data
        test_data = pd.read_csv('spam.csv', encoding='ISO-8859-1')
        test_data = test_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
        test_data['v1'] = test_data['v1'].map({'ham': 0, 'spam': 1})

        x_test = test_data['v2']
        y_test = test_data['v1']

        # Vectorize test data
        x_test_vectorized = vectorizer.transform(x_test)

        # Make predictions
        y_pred = model.predict(x_test_vectorized)

        # Evaluate model accuracy
        accuracy = accuracy_score(y_test, y_pred)
        self.assertGreater(accuracy, 0.9)  # Assuming a decent accuracy

if __name__ == '__main__':
    unittest.main()

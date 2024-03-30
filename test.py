import unittest
import pandas as pd
from sklearn.metrics import accuracy_score

from main import vectorizer, model


class TestSpamModel(unittest.TestCase):

    def test_model_accuracy(self):
        test_data = pd.read_csv('spam.csv', encoding='ISO-8859-1')
        test_data = test_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
        test_data['v1'] = test_data['v1'].map({'ham': 0, 'spam': 1})

        x_test = test_data['v2']
        y_test = test_data['v1']

        x_test_vectorized = vectorizer.transform(x_test)

        y_pred = model.predict(x_test_vectorized)

        accuracy = accuracy_score(y_test, y_pred)
        self.assertGreater(accuracy, 0.9)  


if __name__ == '__main__':
    unittest.main()

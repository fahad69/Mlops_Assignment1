import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load data
data = pd.read_csv('spam.csv', encoding='ISO-8859-1')
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1)
data['v1'] = data['v1'].map({'ham': 0, 'spam': 1})
# Split data into features and labels
x = data['v2']
y = data['v1']
vectorizer = TfidfVectorizer()
x_vectorized = vectorizer.fit_transform(x)
model = MultinomialNB()
model.fit(x_vectorized, y)
pickle.dump(model, open('model.pkl', 'wb'))
print('Model saved as model.pkl')

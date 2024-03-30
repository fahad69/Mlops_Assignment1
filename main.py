from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the model and prepare it for prediction
def load_and_preprocess_data(file_path):
    # Load the model and prepare it for prediction
    DataCar = pd.read_csv(file_path)
    DataCar.dropna(inplace=True)
    DataCar.reset_index(drop=True, inplace=True)
    DataCar.drop_duplicates(inplace=True)

    DataCar = pd.get_dummies(DataCar, columns=['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem'])
    
    return DataCar

selected_features = ['horsepower', 'enginesize', 'curbweight', 'carwidth', 'highwaympg']

def select_features():
    return selected_features

def train_model():
    DataCar = load_and_preprocess_data('CarPrice_Assignment.csv')
    X = DataCar[selected_features]
    y = DataCar['price']

    model = LinearRegression()
    model.fit(X, y)
    
    return model

if __name__ == "__main__":
    app.run(debug=True)

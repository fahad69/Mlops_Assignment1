from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the model and prepare it for prediction
DataCar = pd.read_csv('CarPrice_Assignment.csv')
DataCar.dropna(inplace=True)
DataCar.reset_index(drop=True, inplace=True)
DataCar.drop_duplicates(inplace=True)

DataCar = pd.get_dummies(DataCar, columns=['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem'])

selected_features = ['horsepower', 'enginesize', 'curbweight', 'carwidth', 'highwaympg']

X = DataCar[selected_features]
y = DataCar['price']

model = LinearRegression()
model.fit(X, y)

if __name__ == "__main__":
    app.run(debug=True)

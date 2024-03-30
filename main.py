import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the model and prepare it for prediction
def load_and_preprocess_data(file_path):
    # Load the model and prepare it for prediction
    DataCar = pd.read_csv(file_path)
    DataCar.dropna(inplace=True)
    DataCar.reset_index(drop=True, inplace=True)
    DataCar.drop_duplicates(inplace=True)

    DataCar = pd.get_dummies(DataCar, columns=['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 'fuelsystem'])
    
    return DataCar


def select_features():
    return ['horsepower', 'enginesize', 'curbweight', 'carwidth', 'highwaympg']


def train_model():
    DataCar = load_and_preprocess_data('CarPrice_Assignment.csv')
    X = DataCar[select_features()]
    y = DataCar['price']

    model = LinearRegression()
    model.fit(X, y)
    
    return model


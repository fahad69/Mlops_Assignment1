#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Read the CSV file into a Pandas DataFrame
data = pd.read_csv('Death_rates_for_suicide__by_sex__race__Hispanic_origin__and_age__United_States.csv')
# Clean data
data = data.drop(['INDICATOR', 'UNIT', 'UNIT_NUM','STUB_NAME_NUM','STUB_LABEL_NUM'
                 ,'YEAR_NUM','AGE_NUM','FLAG'], axis=1)

# Rename columns
data.columns 
data = data.dropna()




# In[3]:


# Rename the columns
# data.columns = ['Indicator', 'Age', 'Year', 'Sex', 'Race', 'DeathRate']
# data
print("Before")
print(data.isnull().sum())
data = data.dropna()
print("After")
print(data.isnull().sum())


# In[4]:


data


# In[5]:


# Show the first few rows of the DataFrame
print("few columns      ",data.head())
print(" ")

# Get summary statistics for the columns
print("Summary statistics      ",data.describe())
print(" ")

# Check for missing values
print("Number of missing values     ",data.isna().sum())
print(" ")

# Check the data types of the columns
print("Data types of columns      ",data.dtypes)
print(" ")


# # Exploratory Data Analysis

# In[6]:


data['STUB_NAME'].value_counts()


# In[7]:


data['STUB_LABEL'].value_counts()


# In[8]:


data['YEAR'].value_counts()


# In[9]:


data['AGE'].value_counts()


# In[10]:


#Total estimates according to stub name 

grouped_df = data.groupby(['STUB_NAME'])[['ESTIMATE']].sum()
grouped_df


# In[11]:


sns.set(style="darkgrid") 

# create the plot
sns.catplot(x='STUB_NAME', y='ESTIMATE', kind='bar', data=data.groupby(['STUB_NAME'])[['ESTIMATE']].sum().reset_index(), height=8, aspect=3)

# set the x and y labels
plt.xlabel('Group1')
plt.ylabel('Suicide Estimate')

# show the plot
plt.show()


# In[12]:


#Total estimates according to stub label 

grouped_df = data.groupby('STUB_LABEL')['ESTIMATE'].sum()
grouped_df


# In[13]:


data.groupby('AGE')['ESTIMATE'].sum()


# In[14]:


#sns.set(style="darkgrid") 

# create the plot
sns.catplot(x='AGE', y='ESTIMATE', kind='bar', data=data.groupby(['AGE'])[['ESTIMATE']].sum().reset_index(), height=8, aspect=2)

# set the x and y labels
plt.xlabel('AGE')
plt.ylabel('Suicide Estimate')

# show the plot
plt.show()



# In[15]:


data.groupby('YEAR')['ESTIMATE'].sum()


# In[16]:


sns.scatterplot(data=data.groupby('YEAR')['ESTIMATE'].sum().reset_index(),
                x='YEAR', y='ESTIMATE')
plt.show()


# In[17]:


# Check for outliers and handle them accordingly

plt.figure(figsize=(15, 8))

sns.boxplot(x='STUB_NAME', y='ESTIMATE', data=data)

plt.show()


# In[18]:


# Visualize the distribution of each variable

plt.figure(figsize=(15, 8))


sns.histplot(data['AGE'])

plt.show()


# In[19]:


plt.figure(figsize=(15, 8))


sns.kdeplot(data['YEAR'])

plt.show()


# In[20]:


# Calculate correlation coefficients between variables

data.corr()



# In[21]:


# Visualize the relationships between variables

plt.figure(figsize=(15, 8))

sns.scatterplot(x='AGE', y='ESTIMATE', data=data)

plt.show()


# In[22]:


plt.figure(figsize=(15, 8))

sns.heatmap(data.corr(), annot=True)

plt.show()


# In[23]:


plt.figure(figsize=(15, 8))

# visualize the relationships between variables
sns.pairplot(data=data, vars=['AGE', 'ESTIMATE'])

plt.show() 
    


# In[24]:


plt.figure(figsize=(15, 8))

sns.lineplot(x='YEAR', y='ESTIMATE', hue='STUB_NAME', data=data)

plt.show()



# In[25]:


data.columns


# In[26]:


plt.figure(figsize=(15, 8))

sns.lineplot(x='YEAR', y='ESTIMATE', hue='STUB_LABEL', data=data)

plt.show()



# In[27]:


# Importing the adfuller function
import pandas as pd 
from statsmodels.tsa.stattools import adfuller
# Extracting the 'ESTIMATE' column from the 'data' DataFrame 
data1 = data['ESTIMATE'] 
#Performing the Augmented Dickey-Fuller (ADF) test
result = adfuller(data1)
p_value = result[1] 
print("ADF p-value:", p_value) 

if p_value < 0.05:
   print("The data is stationary.") 
else:
   print("The data is non-stationary.") 


# In[28]:


from sklearn.model_selection import train_test_split 
import pmdarima as pm 
from statsmodels.tsa.arima.model import ARIMA 
# Extracting features from 'data' DataFrame
X = data[['STUB_NAME', 'STUB_LABEL', 'YEAR', 'AGE']]  
# Extracting target variable 'ESTIMATE' from 'data' DataFrame 
y = data['ESTIMATE'] 
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# Creating an ARIMA model with the extracted order
model = pm.auto_arima(y_train, seasonal=False, suppress_warnings=True) 
p, d, q = model.order 
arima_model = ARIMA(y_train, order=(p, d, q)) 
trained_model = arima_model.fit() 


# In[29]:


import pmdarima as pm 
from sklearn.metrics import mean_squared_error 
import warnings 
warnings.filterwarnings("ignore") 
 # Variable to store the best parameters
p_values = range(0, 3) 
d_values = range(0, 3)
q_values = range(0, 3) 
best_params = None
lowest_mse = float('inf') 
for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                 # Creating an ARIMA model
                model = ARIMA(y_train, order=(p, d, q))
                # Fitting the ARIMA model
                trained_model = model.fit() 
                y_pred = trained_model.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)  
                mse = mean_squared_error(y_test, y_pred) 
                print("ARIMA Parameters:", (p, d, q))
                print("Mean Squared Error (MSE):", mse)
                # Checking if the current MSE is the lowest
                if mse < lowest_mse:  
                    best_params = (p, d, q)  
                    lowest_mse = mse 
            except:
                continue

        arima_model = ARIMA(y_train, order=best_params) 
        trained_model = arima_model.fit()

        print("Best ARIMA Parameters:", best_params)
        print("Lowest Mean Squared Error (MSE):", lowest_mse)


# In[30]:


from sklearn.metrics import mean_squared_error
y_pred = trained_model.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)


# In[ ]:


from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/estimate', methods=['POST'])
def get_estimate():
    # Get user input from the form
    stub_name = request.form['stub_name']
    stub_label = request.form['stub_label']
    year = int(request.form['year'])
    age = request.form['age']
    df = pd.DataFrame({
        'STUB_NAME': [stub_name],
        'STUB_LABEL': [stub_label],
        'YEAR': [year],
        'AGE': [age]
    })
    predicted_estimate = trained_model.predict(start=len(y_train), end=len(y_train) + len(df) - 1, exog=df[['STUB_NAME', 'STUB_LABEL', 'YEAR', 'AGE']])
    predicted_value = predicted_estimate.values[0]
    
    return render_template('estimate.html', estimate=predicted_value)

if __name__ == '__main__':
    app.run(host='192.168.0.102', port=8000)


# In[32]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the data
data = pd.read_csv("Death_rates_for_suicide__by_sex__race__Hispanic_origin__and_age__United_States.csv")
data = data.drop(['INDICATOR', 'UNIT', 'UNIT_NUM','STUB_NAME_NUM','STUB_LABEL_NUM'
                 ,'YEAR_NUM','AGE_NUM','FLAG'], axis=1)

data=data.dropna()
categorical_cols = ['STUB_NAME', 'STUB_LABEL', 'YEAR', 'AGE']

# Encode categorical variables
data_encoded = pd.get_dummies(data, columns=categorical_cols)

# Separate the features and target variable
X = data_encoded.drop('ESTIMATE', axis=1)
y = data_encoded['ESTIMATE']

# Split the data into training and evaluation sets
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)

# Define new input data
new_data = pd.DataFrame({
    'STUB_NAME': ['Sex, age and race and Hispanic origin (Single race)'],
    'STUB_LABEL': ['Female: Not Hispanic or Latino: Asian: 65 years and over'],
    'YEAR': [3000],
    'AGE': ['65-74 years']
})

# Concatenate the training and evaluation sets with new data
combined_data = pd.concat([X_train, X_eval, new_data])

# Encode the combined dataset
combined_data_encoded = pd.get_dummies(combined_data, columns=categorical_cols)

# Separate the combined dataset back into training, evaluation, and new data
X_train_combined = combined_data_encoded[:X_train.shape[0]]
X_eval_combined = combined_data_encoded[X_train.shape[0]:X_train.shape[0] + X_eval.shape[0]]
new_data_combined = combined_data_encoded[-new_data.shape[0]:]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_combined), columns=X_train_combined.columns)
X_eval_imputed = pd.DataFrame(imputer.transform(X_eval_combined), columns=X_eval_combined.columns)
new_data_imputed = pd.DataFrame(imputer.transform(new_data_combined), columns=new_data_combined.columns)

# Scale the data
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train_imputed.columns)
X_eval_scaled = pd.DataFrame(scaler.transform(X_eval_imputed), columns=X_eval_imputed.columns)
new_data_scaled = pd.DataFrame(scaler.transform(new_data_imputed), columns=new_data_imputed.columns)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the evaluation set
y_pred_eval = model.predict(X_eval_scaled)

# Calculate mean squared error for the evaluation set
mse_eval = mean_squared_error(y_eval, y_pred_eval)
print("Mean Squared Error (Evaluation set):", mse_eval)

# Make predictions on the new input data
y_pred_new = model.predict(new_data_scaled)

# Print the predicted values for the new input data
print("Estimated values for new data:")
print(y_pred_new)
# [13.79888548]


# In[ ]:





# In[ ]:





# In[ ]:





# Linear Regression

READ ALL THE INSTRUCTIONS CAREFULLY.

## The Challenge

We are providing you with a dataset that contains data about "Area of a piece of land (in sq. feet) vs price of the land (in Rs./sq. foot)". Implement Linear Regression over the provided data. You can find the dataset at the following URLs -

- Training data - https://storage.googleapis.com/kubric-hiring/linreg_train.csv
- Test data - https://storage.googleapis.com/kubric-hiring/linreg_test.csv

The data has already been split into training and testing sets. DO NOT split it further. Use ALL of the training data for fitting the linear regression model.

Inspect the data once 

## Getting Started

You can use numpy, pandas, scipy & requests. We suggest creating a new virtual environment and installing these libraries there using -

`pip install -r requirements.txt`

You must add your code in the regression.py file in the `predict_price` function. Don't change anything else in the file.

Once you are ready, evaluate your code before submitting it by running it using -

`python3 regression.py`

Your root mean squared error (RMSE) must be under 170 for the submission to succeed. The above command will measure that and throw an error if the RMSE is higher.
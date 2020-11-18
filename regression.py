import requests
import pandas
import scipy
import numpy
import sys
import csv
from scipy import stats


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE
    # For extracting training data
    decoded_content = response.content.decode('utf-8')
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    my_list = list(cr)
    x_train = []
    y_train = []
    for index, area in enumerate(my_list[0]):
    	if index == 0:
    		pass
    	else:
    		x_train.append(float(area))

    for index, price in enumerate(my_list[1]):
    	if index == 0:
    		pass
    	else:
    		y_train.append(float(price))

    x_train = numpy.array(x_train)
    y_train = numpy.array(y_train)

    # The linear regression model fitting process
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_train, y_train)
    predictions = slope * areas + intercept

    return predictions


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")

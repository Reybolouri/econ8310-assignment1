#1
import pandas as pd
from prophet import Prophet




train_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
training_data = pd.read_csv(train_url)


# Convert Timestamp column to datetime objects
training_data["Timestamp"] = pd.to_datetime(training_data["Timestamp"])

training_data.rename(columns={"Timestamp": "ds", "trips": "y"}, inplace=True)
training_data.sort_values(by="ds", inplace=True)

training_data.head()



model = Prophet(daily_seasonality=True, weekly_seasonality=True)

# add hourly seasonality(period=24) 
model.add_seasonality(name='hourly', period=24, fourier_order=5)

modelFit = model.fit(training_data)



# test data
test_url = "https://raw.githubusercontent.com/dustywhite7/econ8310-assignment1/main/assignment_data_test.csv"

test_df = pd.read_csv(test_url)

test_df["Timestamp"] = pd.to_datetime(test_df["Timestamp"])


test_df.rename(columns={"Timestamp": "ds"}, inplace=True)
test_df.sort_values(by="ds", inplace=True)


test_df.head()



# Forecasts on the Test Data
forecast = model.predict(test_df)

pred = forecast['yhat'].values

print("Forecasts for the test period (744 hours):")
print(pred[:10])  




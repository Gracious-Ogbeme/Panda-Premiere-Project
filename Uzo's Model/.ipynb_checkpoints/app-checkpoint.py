import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

# importing needed variables from the variables.py file 
from variables import *




# Load your model
model = joblib.load('my_model.pkl')

def get_country(country):
    data = df_clean_4.loc[country]
    train_data = data.loc['1999':'2019']
    test_data =  data.loc['2017':'2021']
    return data, train_data, test_data, country

def generate_testing_data(dataframe, window_size = 3):
    data = dataframe.values
    tf_data = tf.data.Dataset.from_tensor_slices(data)
    tf_data = tf_data.window(size = window_size, shift = 1, drop_remainder = True)
    tf_data = tf_data.flat_map(lambda x: x.batch(window_size))
    tf_data = tf_data.batch(1).prefetch(1)
    return tf_data, dataframe.index

def forecast(model, periods, ts):
    data_series = ts.squeeze().iloc[-3:]
    data = list(np.squeeze(data_series.values))
    for _ in range(periods):
        input = np.expand_dims(data[-3:], axis = 0)
        output = float(np.squeeze(model.predict(input)))
        data.append(output)
    data = data[2:]
    data = pd.Series(data, index = pd.period_range('2021', freq = '1Y', periods = periods + 1))
    return data

def visualize_timeseries(data, title, style='-', label=None):
    fig, ax = plt.subplots()

    # Ensure the data index is in a proper datetime format
    try:
        data.index = pd.to_datetime(data.index)
    except Exception as e:
        print(f"Error converting index to datetime: {e}")
        return

    # Convert data values to floats
    try:
        data_values = data.astype(float)
    except ValueError as e:
        print(f"Error converting data values to float: {e}")
        return

    ax.plot(data.index, data_values, style, label=label)

    if label is not None:
        ax.legend()

    ax.set_xlabel('Time')
    ax.set_ylabel('% of GDP')
    ax.set_title(title)
    ax.grid()

    return fig

def plot_forecast(country, periods):
    data, train_data, test_data, country_name = get_country(country)
    forecast_data = forecast(model, periods, data).to_frame().rename(columns={0: 'Forecasts'})

    # Ensure the indices are in datetime format
    if isinstance(train_data.index, pd.PeriodIndex):
        train_data.index = train_data.index.to_timestamp()
    if isinstance(test_data.index, pd.PeriodIndex):
        test_data.index = test_data.index.to_timestamp()
    if isinstance(forecast_data.index, pd.PeriodIndex):
        forecast_data.index = forecast_data.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_data.index, train_data, label='Train Data', linestyle='-')
    ax.plot(test_data.index, test_data, label='Test Data', linestyle='-')
    ax.plot(forecast_data.index, forecast_data, label=f'{periods}-Year Forecast', linestyle='-')
    ax.set_title(f'{periods}-Year Forecast for {country}')
    ax.legend()
    ax.grid(True)

    return forecast_data, fig





def app():
    st.title("Forecasting Application")

    country = st.selectbox('Select a country', ["Argentina" , "Austria" , "Switzerland" , "Cote d'Ivoire", "Cyprus" , "Denmark", "Spain", "Finland", "United Kingdom", "Greece", "Italy", "Japan", "Madagascar", "Mauritius", "Netherlands", "Norway", "Portugal", "Sweden", "Thailand"])
    periods = st.number_input('Enter the number of years to predict', min_value=1, max_value=100, value=5)
    
    if st.button('Forecast'):
        forecast_data, fig = plot_forecast(country, periods)
        
        st.pyplot(fig)
        
        st.write("Forecast Data")
        st.dataframe(forecast_data)


if __name__ == "__main__":
    app()


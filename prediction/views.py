from re import X
from typing import Optional
from django.shortcuts import render, HttpResponse
from tensorflow.python.eager.context import context
import pandas as pd
import numpy as np
from datetime import datetime
import datetime as dt
import yfinance as yf
import math, random
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
matplotlib.use('Agg')

def index(request):
    return render(request, 'index.html')

def pred(request):
    return render(request, 'prediction.html')

def result(request):
    quote = request.POST['nm']
    
    def getHistoricalData(quote):
        end = datetime.now()
        start = datetime(end.year-6,end.month,end.day)
        ticker = yf.Ticker(quote)
        data = ticker.history(start=start,end=end)
        df = pd.DataFrame(data=data)
        return df

    def LSTM_ALGO(df):
        #Create a new dataframe with only the 'Close column
        data = df.filter(['Close'])
        #Convert the dataframe to a numpy array
        dataset = data.values
        #Get the number of rows to train the model on
        training_data_len = math.ceil( len(dataset) * .9 )
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)
        train_data = scaled_data[0:training_data_len , :]

        #Split the data into x_train and y_train data sets
        x_train = []
        y_train =   []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])

        #Convert the x_train and y_train to numpy arrays 
        x_train, y_train = np.array(x_train), np.array(y_train)

        #Reshape the data
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        #Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences= False))
        model.add(Dense(25))
        model.add(Dense(1))
    
        # we use the optimiser as adam and loss as mean squared error
        #Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        #Train the model
        model.fit(x_train, y_train, batch_size=1, epochs=1)

        #Create the testing data set
        #Create a new array containing scaled values from index 1543 to 2002
        test_data = scaled_data[training_data_len - 60: , :] 
        #Create the data sets x_test and y_test
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])

        x_test = np.array(x_test)
        
        #Reshape the data
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
        print(x_test.shape)

        #Get the models predicted price values 
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        #Get the root mean squared error (RMSE)
        rmse = round(math.sqrt(mean_squared_error(y_test, predictions)), 2)
        print("RMSE: ",rmse)

        #Tomorrow's Predicted price
        last_60_days = data[-60:].values
        last_60_days_scaled = scaler.transform(last_60_days)
        Xtest = []
        Xtest.append(last_60_days_scaled)
        Xtest = np.array(Xtest)
        Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], 1))
        pred = model.predict(Xtest)
        pred = scaler.inverse_transform(pred)

        #Plot the data
        train = data[:training_data_len]
        test = data[training_data_len:]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions

        fig1 = plt.figure(figsize=(7.2,4.8),dpi=65)
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price', fontsize=18)
        plt.plot(test['Close'])
        plt.legend(['Value'], loc='lower right')
        plt.savefig('static/img/Trends.png')
        plt.close(fig1)

        fig2 = plt.figure(figsize=(7.2,4.8),dpi=65)
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price', fontsize=18)
        plt.plot(test['Close'])
        plt.plot(valid['Predictions'])
        plt.legend(['Value', 'Predictions'], loc='lower right')
        plt.savefig('static/img/LSTM.png')
        plt.close(fig2)
        print(valid)
        return rmse,pred

    df = getHistoricalData(quote)
    
    if(df.empty):
        return render(request, 'prediction.html', context={'not_found':True})
        
    else:
        print("##############################################################################")
        print("Today's",quote,"Stock Data: ")
        open = round((df.tail(1)['Open'].iloc[0]), 2)
        close = round((df.tail(1)['Close'].iloc[0]), 2)
        low = round((df.tail(1)['Low'].iloc[0]), 2)
        high = round((df.tail(1)['High'].iloc[0]), 2)
        volume = round((df.tail(1)['Volume'].iloc[0]), 2)
        rmse,pred = LSTM_ALGO(df)
        print("##############################################################################")
        
        return render(request, 'result.html', context={'quote':quote,'open':open, 'close':close, 'low':low, 'high':high,
                        'volume':volume, 'rmse':rmse, 'pred':round(float(pred), 2)})
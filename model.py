# -*- coding: utf-8 -*-
"""
Created on Sun May 30 16:13:39 2021

@author: saniya
"""


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
import pickle

def load_data():
    return pd.read_csv('C:\\Users\\saniya\\Desktop\\Sales Forecasting Final\\arima_df.csv').set_index('date')

ts_data = load_data()

ts_data.index = pd.to_datetime(ts_data.index)

def get_scores(data):
    
    model_scores = {}
    
    rmse = np.sqrt(mean_squared_error(data.sales_diff[-12:], data.forecast[-12:]))
    mae = mean_absolute_error(data.sales_diff[-12:], data.forecast[-12:])
    r2 = r2_score(data.sales_diff[-12:], data.forecast[-12:])
    model_scores['ARIMA'] = [rmse, mae, r2]
    
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")
    
    pickle.dump(model_scores, open( "arima_model_scores.p", "wb" ))
    
def sarimax_model(data):
    
    # Model
    sar = sm.tsa.statespace.SARIMAX(ts_data.sales_diff, order=(12,0,0), seasonal_order=(0,1,0,12), trend='c').fit()

    # Predictions
    start, end, dynamic = 40, 100, 7
    data['forecast'] = sar.predict(start=start, end=end, dynamic=dynamic) 
    pred_df = data.forecast[start+dynamic:end]
    
    data[['sales_diff', 'forecast']].plot(color=['mediumblue', 'Red'])
    
    get_scores(data)

    return sar, data, pred_df

sar, ts_data, predictions = sarimax_model(ts_data)

sar.plot_diagnostics(figsize=(10, 8));

def predict_df(prediction_df):
    
    #load in original dataframe without scaling applied
    original_df = pd.read_csv('C:\\Users\\saniya\\Desktop\\Sales Forecasting Final\\train.csv')
    original_df.date = original_df.date.apply(lambda x: str(x)[:-3])
    original_df = original_df.groupby('date')['sales'].sum().reset_index()
    original_df.date = pd.to_datetime(original_df.date)
    
    #create dataframe that shows the predicted sales
    result_list = []
    sales_dates = list(original_df[-13:].date)
    act_sales = list(original_df[-13:].sales)
    
    for index in range(0,len(prediction_df)):
        result_dict = {}
        result_dict['pred_value'] = int(prediction_df[index] + act_sales[index])
        result_dict['date'] = sales_dates[index+1]
        result_list.append(result_dict)
        
    df_result = pd.DataFrame(result_list)
    print(df_result, original_df)
    
    return df_result, original_df

def plot_results(results, original_df, model_name):

    fig, ax = plt.subplots(figsize=(15,5))
    sns.lineplot(original_df.date, original_df.sales, data=original_df, ax=ax, 
                label='Original', color='mediumblue')
    sns.lineplot(results.date, results.pred_value, data=results, ax=ax, 
                 label='Predicted', color='Red')
    
    ax.set(xlabel = "Date",
           ylabel = "Sales",
           title = f"{model_name} Sales Forecasting Prediction")
    
    ax.legend()
    
    sns.despine()
prediction_df, original_df = predict_df(predictions)
plot_results(prediction_df, original_df, 'arima')

def __getnewargs__(self):
    return((self.endog), (self.k_lags),(self.k_diff), (self.k_ma))
ARIMA.__getnewargs__= __getnewargs__

series = pd.read_csv('train.csv', header=0, index_col=0)
X= series.values
X=X.astype('float32')

#fit model
model= ARIMA(X,order=(1,1,1),exog=None)
model_fit = model.fit()
model_fit.save('model.pkl')

loaded = ARIMAResults('model.pkl')




















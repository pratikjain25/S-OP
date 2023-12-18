#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[ ]:


import pandas as pd
import numpy as np
import datetime
from datetime import timedelta

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display

from scipy.stats import rankdata

import statsmodels.api as sm
import statsmodels.tsa.api as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

import pmdarima as pm
from pmdarima import auto_arima

import xgboost as xgb
from xgboost import XGBRegressor

from prophet import Prophet

import streamlit as st

import warnings
warnings.filterwarnings("ignore")


# # Dataframe

# ## Data

# In[ ]:


file_path_actual = r'C:\Users\jainp\OneDrive - SGS&CO\S&OP\data\ActualOrders.csv'
file_path_forecast = r'C:\Users\jainp\OneDrive - SGS&CO\S&OP\data\ForecastedOrders.csv'
file_path_Actual_SOP = r'C:\Users\jainp\OneDrive - SGS&CO\S&OP\data\S&OP and Actual Data Combined.csv'

df_actual = pd.read_csv(file_path_actual)
df_forecast = pd.read_csv(file_path_forecast)
df_combined = pd.read_csv(file_path_Actual_SOP)


# ### Streamlit App

# In[ ]:


st.set_page_config(layout="wide")
st.sidebar.header('Customer Name')
customer_names = list(df_combined['Customer Name'].unique())
customer_name = st.sidebar.selectbox('Select a Customer', customer_names) 
# holiday_effect = ["Yes","No"]
# holiday_effect = st.sidebar.selectbox('Holiday Effect', holiday_effect)
start_date = st.sidebar.date_input("Start date", datetime.date(2022,7,1))
end_date = st.sidebar.date_input("End date")


# ### Actual Data

# In[ ]:


if customer_name:
    df = df_actual
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S')
    df = df.sort_values(by=['Date'])
    df = df[df['Customer Name'] == customer_name]
    df = df.groupby('Date')['Orders'].sum().reset_index()
    df.set_index('Date', inplace=True)
    df.sort_index()
else:
    df = df_actual
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S')
    df = df.sort_values(by=['Date'])
    df = df.groupby('Date')['Orders'].sum().reset_index()
    df.set_index('Date', inplace=True)
    df.sort_index()
    
if 'Orders' in df.columns:
    first_non_zero_index = (df['Orders'] != 0).idxmax()
    df['Orders'] = df['Orders'][first_non_zero_index:]

df = df.dropna()


# ### S&OP Data

# In[ ]:


if customer_name:
    df_for = df_forecast
    df_for['Date'] = pd.to_datetime(df_for['Date']).dt.date
    df_for = df_for.sort_values(by=['Date'])
    df_for = df_for[df_for['Customer Name'] == customer_name]
    df_for = df_for.groupby('Date')['S&OP Forecast'].sum().reset_index()
    df_for.set_index('Date', inplace=True)
    df_for.index = pd.to_datetime(df_for.index) + timedelta(days=1)
    df_for.sort_index()
else:
    df_for = df_forecast
    df_for['Date'] = pd.to_datetime(df_for['Date']).dt.date
    df_for = df_for.sort_values(by=['Date'])
    df_for = df_for.groupby('Date')['S&OP Forecast'].sum().reset_index()
    df_for.set_index('Date', inplace=True)
    df_for.index = pd.to_datetime(df_for.index) + timedelta(days=1)
    df_for.sort_index()


# ### Combine Data

# In[ ]:


if customer_name:
    df_com = df_combined
    df_com['Date'] = pd.to_datetime(df_com['Date'], format='%Y-%m-%d %H:%M:%S')
    df_com = df_com.sort_values(by=['Date'])
    df_com = df_com[df_com['Customer Name'] == customer_name]
    df_com = df_com[(df_com['Date'] >= pd.to_datetime(start_date)) & (df_com['Date'] <= pd.to_datetime(end_date))]
    df_com.set_index('Date', inplace=True)
    df_com = df_com.sort_index()
else:
    df_com = df_combined
    df_com['Date'] = pd.to_datetime(df_com['Date'], format='%Y-%m-%d %H:%M:%S')
    df_com = df_com[(df_com['Date'] >= pd.to_datetime(start_date)) & (df_com['Date'] <= pd.to_datetime(end_date))]
    df_com = df_com.sort_values(by=['Date'])
    df_com.set_index('Date', inplace=True)
    df_com = df_com.sort_index()

df_com = df_com.dropna()


# In[ ]:


mean_actual_orders = df_com['Actual Orders'].mean()
total_actual_orders = df_com['Actual Orders'].sum()
total_forecasted_orders = df_com['Forecast Orders'].sum()
mae_kpi = mean_absolute_error(df_com['Actual Orders'], df_com['Forecast Orders'])

def mean_absolute_percentage_error_data(actual_orders, forecast_orders):
    actual_orders_masked = actual_orders.copy()
    forecast_orders_masked = forecast_orders.copy()

    mask = actual_orders != 0
    actual_orders_masked[~mask] = 1
    forecast_orders_masked[~mask] = 1
    
    return np.mean(np.abs((actual_orders_masked - forecast_orders_masked) / actual_orders_masked)) * 100

mape_kpi = mean_absolute_percentage_error_data(df_com['Actual Orders'], df_com['Forecast Orders'])

mean_actual_orders = f"{mean_actual_orders:.2f}"
mae_kpi = f"{mae_kpi:.2f}"
mape_kpi = f"{mape_kpi:.2f}%"

st.title(f'📉 Key Performance Indicators (KPIs) for {customer_name}')
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Average Actual Orders", mean_actual_orders)

with col2:
    st.metric("Total Actual Orders", int(total_actual_orders))

with col3:
    st.metric("Total S&OP Forecasted Orders", int(total_forecasted_orders))

col4, col5, col6 = st.columns(3)

with col4:
    st.metric("MAE (Actual vs. S&OP)", mae_kpi)
    st.write("MAE (Mean Absolute Error) is the average of the absolute differences between actual and predicted values.")

with col5:
    st.metric("MAPE (Actual vs. S&OP)", mape_kpi)
    st.write("MAPE (Mean Absolute Percentage Error) measures the accuracy of a forecasting method by calculating the percentage difference between predicted and actual values.")

with col6:
    pass


# In[ ]:


st.title("📊 Visualizing Actual Orders vs S&OP Forecast")
st.markdown("---")

fig = go.Figure()

fig.add_trace(go.Scatter(x=df_com.index, y=df_com['Actual Orders'],mode='lines+markers', name='Actual Orders'))

fig.add_trace(go.Scatter(x=df_com.index, y=df_com['Forecast Orders'],mode='lines+markers', name='S&OP Forecast'))

fig.update_layout(title=f'Actual Orders vs. S&OP Forecast for {customer_name}',
                  xaxis_title='Date',
                  yaxis_title='Orders',
                  width=1000,
                  legend_title='Legend')

st.plotly_chart(fig)


# ## Holiday Dataframe

# In[ ]:


holiday_dates = [
    "01-Jan", "06-Jan", "29-Mar", "01-Apr", "01-May", "06-May", "08-May", "09-May", "19-May",
    "20-May", "27-May", "14-Jul", "15-Aug", "26-Aug", "12-Oct", "01-Nov", "11-Nov", "06-Dec",
    "25-Dec", "26-Dec"
]

years = range(2019, 2025)

holiday_names = [
    "New Year's Day", "Epiphany", "Good Friday", "Easter Monday", "Labour Day",
    "Early May Bank Holiday", "Fête de la Victoire 1945", "Ascension Day", "Pentecost",
    "Whit Monday", "Spring Bank Holiday", "Fête Nationale de la France", "Assumption Day",
    "Summer Bank Holiday", "Hispanic Day", "All Saints' Day", "Armistice Day",
    "Constitution Day", "Christmas Day", "Boxing Day"
]

holidays_df = pd.DataFrame({
    'holiday': [holiday_name for _ in years for holiday_name in holiday_names],
    'ds': pd.to_datetime([f"{year}-{date}" for year in years for date in holiday_dates])
})


# # Classical Time Series Analysis

# ## ARIMA

# In[ ]:


df_am = df.copy()


# In[ ]:


# def ad_test(dataset):
#      dftest = adfuller(dataset, autolag = 'AIC')
#      print("1. ADF : ",dftest[0])
#      print("2. P-Value : ", dftest[1])
#      print("3. Num Of Lags : ", dftest[2])
#      print("4. Num Of Observations Used For ADF Regression:", dftest[3])
#      print("5. Critical Values :")
#      for key, val in dftest[4].items():
#          print("\t",key, ": ", val)
# ad_test(df_am['Orders'])


# In[ ]:


# def find_best_differencing_order(dataset):
#     dftest = adfuller(dataset, autolag='AIC')
#     p_value = dftest[1]
#     d = 1 if p_value > 0.05 else 0
#     return d
# d = find_best_differencing_order(df_am['Orders'])


# In[ ]:


# def find_best_pq_order(dataset):
#     acf_vals = acf(dataset)
#     pacf_vals = pacf(dataset)

#     p = 0
#     for lag in range(1, len(pacf_vals)):
#         if pacf_vals[lag] < 0.05:
#             p = lag
#             break
#     q = 0
#     for lag in range(1, len(acf_vals)):
#         if acf_vals[lag] < 0.05:
#             q = lag
#             break

#     return p, q

# p, q = find_best_pq_order(df_am['Orders'])


# In[ ]:


# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
# acf_plot = plot_acf(df_am.Orders, lags=60, ax=ax1, alpha=0.05)
# pacf_plot = plot_pacf(df_am.Orders, lags=60, ax=ax2, alpha=0.05)
# plt.tight_layout()
# plt.show()


# In[ ]:


# print(df_am.shape)
train_am=df_am.iloc[:-8]
test_am=df_am.iloc[-8:]
# print(train_am.shape,test_am.shape)


# In[ ]:


# model_am=sm.tsa.arima.ARIMA(train_am['Orders'],order=(p,d,q))
model_am = pm.auto_arima(train_am['Orders'], seasonal=True, stepwise=False, approximation=False, trace=True)
model_am=model_am.fit(train_am['Orders'])
# model_am.summary()


# In[ ]:


pred_am= model_am.predict(n_periods=len(test_am))
pred_am = pred_am.round()
# pred_am


# In[ ]:


mae_am = mean_absolute_error(test_am['Orders'], pred_am)
rmse_am = np.sqrt(mean_squared_error(test_am['Orders'], pred_am))
mape_am = mean_absolute_percentage_error(test_am['Orders'], pred_am)

# print(f"Mean Absolute Error (MAE): {mae_am:.2f}")
# print(f"Root Mean Square Error (RMSE): {rmse_am:.2f}")
# print(f"Mean Absolute Percentage Error (MAPE): {mape_am:.2f}")


# In[ ]:


# plt.figure(figsize=(12, 6))
# plt.plot(test_am.index, test_am['Orders'], label='Actual Data', marker='o') 
# plt.plot(test_am.index, pred_am, label='ARIMA Prediction', linestyle='--', marker='o', markersize=5)

# plt.xlabel('Date')
# plt.ylabel('Orders')
# plt.title('ARIMA Prediction vs. Actual Data')
# plt.legend()

# for i, row in test_am.iterrows():
#     plt.text(i, row['Orders'], f"{row['Orders']:.2f}", ha='left', va='bottom')

# for i, row in test_am.iterrows():
#     plt.text(i, pred_am.loc[i], f"{pred_am.loc[i]:.2f}", ha='right', va='bottom')

# plt.xticks(rotation=45)
# plt.show()


# In[ ]:


results_am = pd.DataFrame({'Date': test_am.index, 'Actual Orders': test_am['Orders'], 'ARIMA Predictions': pred_am})
# print(results_am)


# In[ ]:


model_am = pm.auto_arima(df_am['Orders'], seasonal=True, stepwise=False, approximation=False, trace=True)
model_am=model_am.fit(df_am['Orders'])
# model_am.summary()


# In[ ]:


future_start_date_am = df_am.index.max() + pd.DateOffset(days=7)
future_end_date_am = future_start_date_am + pd.DateOffset(weeks=7)
future_dates_am = pd.date_range(start=future_start_date_am, end=future_end_date_am, freq='7D')
# future_dates_am


# In[ ]:


forecast_am= model_am.predict(n_periods=len(future_dates_am))
forecast_am = forecast_am.round()
# forecast_am


# In[ ]:


future_am = pd.DataFrame({'ARIMA Predictions': forecast_am})
future_am['ARIMA Predictions'] = future_am['ARIMA Predictions'].apply(lambda x: f"{x} (±{mae_am:.0f})")
future_am.index.name = 'Date'
# future_am


# ## Exponential Smoothing

# In[ ]:


df_exp =df.copy()
df_exp['Orders'] = df_exp['Orders'].replace(0, 1)


# In[ ]:


train_exp = df_exp.iloc[:-8]
test_exp = df_exp.iloc[-8:]


# In[ ]:


best_model_exp = None
best_aic = float('inf')

smoothing_methods = ['add', 'additive', 'multiplicative']
trend_methods = ['add', 'additive', 'multiplicative']
seasonality_methods = [None, 'add', 'additive', 'multiplicative']

for smoothing in smoothing_methods: 
    for trend in trend_methods:
        for seasonal in seasonality_methods:
            model_exp = ExponentialSmoothing(train_exp['Orders'], trend=trend, seasonal=seasonal, seasonal_periods=12, use_boxcox=(smoothing == 'multiplicative'))
            model_fit_exp = model_exp.fit()
            aic_exp = model_fit_exp.aic

            if aic_exp < best_aic:
                best_aic = aic_exp
                best_model_exp = model_fit_exp

# print("Selected Exponential Smoothing Method:", best_model_exp.model)


# In[ ]:


y_pred_exp = best_model_exp.forecast(steps=8)
y_pred_exp=y_pred_exp.round()
# y_pred_exp


# In[ ]:


mae_exp = mean_absolute_error(test_exp['Orders'], y_pred_exp)
rmse_exp = np.sqrt(mean_squared_error(test_exp['Orders'], y_pred_exp))
mape_exp = mean_absolute_percentage_error(test_exp['Orders'], y_pred_exp)

# print(f"Mean Absolute Error (MAE): {mae_exp:.2f}")
# print(f"Root Mean Squared Error (RMSE): {rmse_exp:.2f}")
# print(f"Mean Absolute Percentage Error (MAPE): {mape_exp:.2f}")


# In[ ]:


# plt.figure(figsize=(12, 6))
# plt.plot(test_exp.index, test_exp, label='Actual Orders')
# plt.plot(test_exp.index, y_pred_exp, label='Predicted Orders')
# plt.legend()
# plt.title('Actual vs.Predicted Orders')
# plt.xlabel('Date')
# plt.ylabel('Orders')
# for i, row in test_exp.iterrows():
#     position = test_exp.index.get_loc(i)
#     plt.text(i, row['Orders'], f"{row['Orders']:.2f}", ha='left', va='bottom')
#     plt.text(i, y_pred_exp[position], f"{y_pred_exp[position]:.2f}", ha='right', va='bottom')

# plt.xticks(rotation=45)
# plt.show()
# plt.figure(figsize=(12, 6))


# In[ ]:


results_exp = pd.DataFrame({'Date': test_exp.index, 'Actual Orders': test_exp['Orders'], 'Exponential Smoothing Predictions':y_pred_exp})
# results_exp


# In[ ]:


best_model_exp = None
best_aic = float('inf')

smoothing_methods = ['add', 'additive', 'multiplicative']
trend_methods = ['add', 'additive', 'multiplicative']
seasonality_methods = [None, 'add', 'additive', 'multiplicative']

for smoothing in smoothing_methods: 
    for trend in trend_methods:
        for seasonal in seasonality_methods:
            model_exp = ExponentialSmoothing(df_exp['Orders'], trend=trend, seasonal=seasonal, seasonal_periods=12, use_boxcox=(smoothing == 'multiplicative'))
            model_fit_exp = model_exp.fit()
            aic_exp = model_fit_exp.aic

            if aic_exp < best_aic:
                best_aic = aic_exp
                best_model_exp = model_fit_exp

# print("Selected Exponential Smoothing Method:", best_model_exp.model)


# In[ ]:


future_exp = best_model_exp.forecast(steps=8)
future_exp=future_exp.round()
future_exp = future_exp.values
# future_exp


# In[ ]:


future_start_date_exp = df_exp.index.max() + pd.DateOffset(days=7)
future_end_date_exp = future_start_date_exp + pd.DateOffset(weeks=7)
future_dates_exp = pd.date_range(start=future_start_date_exp, end=future_end_date_exp, freq='7D')
# future_dates_exp


# In[ ]:


future_exp = pd.DataFrame({'Date': future_dates_exp, 'Exponential Smoothing Predictions': future_exp})
future_exp['Exponential Smoothing Predictions'] = future_exp['Exponential Smoothing Predictions'].apply(lambda x: f"{x} (±{mae_exp:.0f})")
future_exp.set_index('Date', inplace=True)
# future_exp


# ## Prophet

# In[ ]:


df_fb = df.copy()
df_fb.reset_index(inplace=True)
# df_fb


# In[ ]:


df_fb.columns = ['ds', 'y']
# df_fb


# In[ ]:


train_fb = df_fb.iloc[:-8]
test_fb = df_fb.iloc[-8:]


# In[ ]:


m = Prophet(
    interval_width=0.95,
    holidays=holidays_df
)
m.fit(train_fb)
y_pred_fb = m.predict(test_fb)
# y_pred_fb


# In[ ]:


y_pred_fb = y_pred_fb[['ds','yhat']]
y_pred_fb = y_pred_fb.round()


# In[ ]:


mae_fb = mean_absolute_error(test_fb['y'], y_pred_fb['yhat'])
rmse_fb = np.sqrt(mean_squared_error(test_fb['y'], y_pred_fb['yhat']))
mape_fb = mean_absolute_percentage_error(test_fb['y'], y_pred_fb['yhat'])

# print(f"Mean Absolute Error (MAE): {mae_fb:.2f}")
# print(f"Root Mean Squared Error (RMSE): {rmse_fb:.2f}")
# print(f"Mean Absolute Percentage Error (MAPE): {mape_fb:.2f}")


# In[ ]:


# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 6))
# plt.plot(test_fb['ds'], test_fb['y'], label='Actual Orders')
# plt.plot(y_pred_fb['ds'], y_pred_fb['yhat'], label='Predicted Orders')
# plt.legend()
# plt.title('Actual vs. Predicted Orders')
# plt.xlabel('Date')
# plt.ylabel('Orders')

# for i, row in test_fb.iterrows():
#     position = y_pred_fb[y_pred_fb['ds'] == row['ds']].index[0]
#     plt.text(row['ds'], row['y'], f"{row['y']:.2f}", ha='left', va='bottom')
#     plt.text(row['ds'], y_pred_fb.loc[position, 'yhat'], f"{y_pred_fb.loc[position, 'yhat']:.2f}", ha='right', va='bottom')

# plt.xticks(rotation=45)
# plt.show()


# In[ ]:


results_fb = pd.DataFrame({'Date': test_fb['ds'].values, 'Actual Orders': test_fb['y'].values, 'Prophet Predictions': y_pred_fb['yhat'].values})
results_fb.set_index('Date', inplace=True)
# results_fb


# In[ ]:


future_start_date_fb = df_fb['ds'].max() + pd.DateOffset(days=7)
future_end_date_fb = future_start_date_fb + pd.DateOffset(weeks=7)
future_fb = pd.DataFrame({'ds': pd.date_range(start=future_start_date_fb, end=future_end_date_fb, freq='7D')})
# future_fb


# In[ ]:


m = Prophet(
    interval_width=0.95,
    holidays=holidays_df
)
m.fit(df_fb)
forecast_fb = m.predict(future_fb)


# In[ ]:


forecast_fb = forecast_fb[['ds','yhat']]
forecast_fb = forecast_fb.round()
# forecast_fb


# In[ ]:


future_fb = pd.DataFrame({'Date': forecast_fb['ds'].values, 'Prophet Predictions': forecast_fb['yhat'].values})
future_fb['Prophet Predictions'] = future_fb['Prophet Predictions'].apply(lambda x: f"{x} (±{mae_fb:.0f})")
future_fb.set_index('Date', inplace=True)
# future_fb


# # Ensemble Learning

# ## XGBOOST

# In[ ]:


df_xg =df.copy()
train_xg=df_xg.iloc[:-9]
test_xg=df_xg.iloc[-9:]


# In[ ]:


# tss = TimeSeriesSplit(n_splits=5, test_size=9)

# fig, axs = plt.subplots(5, 1, figsize=(15, 15), sharex=True)

# fold = 0
# for train_idx, val_idx in tss.split(df_xg):
#     train_xg = df_xg.iloc[train_idx]
#     test_xg = df_xg.iloc[val_idx]
    
#     axs[fold].plot(train_xg.index, train_xg['Orders'], label='Training Set')
#     axs[fold].plot(test_xg.index, test_xg['Orders'], label='Test Set')
    
#     axs[fold].set_title(f'Data Train/Test Split Fold {fold}')
#     axs[fold].legend()
    
#     fold += 1

# plt.tight_layout()
# plt.show()


# In[ ]:


tss = TimeSeriesSplit(n_splits=5, test_size=9)

reg = xgb.XGBRegressor(
    base_score=0.5,
    booster='gbtree',
    n_estimators=1000,
    early_stopping_rounds=50,
    objective='reg:linear',
    max_depth=3,
    learning_rate=0.01
)

fold = 0
preds = []
scores = []

for train_idx, val_idx in tss.split(df_xg):
    train_xg = df_xg.iloc[train_idx]
    test_xg = df_xg.iloc[val_idx]
    
    train_xg['week'] = train_xg.index.week
    train_xg['month'] = train_xg.index.month
    train_xg['quarter'] = train_xg.index.quarter
    train_xg['year'] = train_xg.index.year

    test_xg['week'] = test_xg.index.week
    test_xg['month'] = test_xg.index.month
    test_xg['quarter'] = test_xg.index.quarter
    test_xg['year'] = test_xg.index.year

    rolling_window = 2
    train_xg['rolling_mean'] = train_xg['Orders'].rolling(window=rolling_window).mean()
    test_xg['rolling_mean'] = test_xg['Orders'].rolling(window=rolling_window).mean()
    train_xg['rolling_std'] = train_xg['Orders'].rolling(window=rolling_window).std()
    test_xg['rolling_std'] = test_xg['Orders'].rolling(window=rolling_window).std()

    FEATURES = ['week','month', 'quarter', 'year','rolling_mean', 'rolling_std']
    TARGET = ['Orders']

    X_train_xg = train_xg[FEATURES].dropna()
    y_train_xg = train_xg[TARGET].loc[X_train_xg.index]

    X_test_xg = test_xg[FEATURES].dropna()
    y_test_xg = test_xg[TARGET].loc[X_test_xg.index]

    reg.fit(X_train_xg, y_train_xg,
            eval_set=[(X_train_xg, y_train_xg), (X_test_xg, y_test_xg)],
            verbose=100)

    pred_xg = reg.predict(X_test_xg)
    pred_xg = pred_xg.round()
    preds.append(pred_xg)

    score = np.sqrt(mean_squared_error(y_test_xg, pred_xg))
    scores.append(score)

    fold += 1

print("Average RMSE:", np.mean(scores))


# In[ ]:


# print(f'Score across folds {np.mean(scores):0.4f}')
# print(f'Fold scores:{scores}')


# In[ ]:


test_xg = test_xg.tail(8)


# In[ ]:


mae_xg = mean_absolute_error(y_test_xg, pred_xg)
rmse_xg = np.sqrt(mean_squared_error(y_test_xg, pred_xg))
mape_xg = mean_absolute_percentage_error(y_test_xg, pred_xg)

# print(f"Mean Absolute Error (MAE): {mae_xg:.2f}")
# print(f"Root Mean Square Error (RMSE): {rmse_xg:.2f}")
# print(f"Mean Absolute Percentage Error (MAPE): {mape_xg:.2f}")


# In[ ]:


# plt.figure(figsize=(12, 6))
# plt.plot(test_xg.index, y_test_xg, label='Actual Data', marker='o') 
# plt.plot(test_xg.index, pred_xg, label='XG Prediction', linestyle='--', marker='o', markersize=5)

# plt.xlabel('Date')
# plt.ylabel('Orders')
# plt.title('XG Prediction vs. Actual Data')
# plt.legend()

# for i, row in test_xg.iterrows():
#     position = test_xg.index.get_loc(i)
#     plt.text(i, row['Orders'], f"{row['Orders']:.2f}", ha='left', va='bottom')
#     plt.text(i, pred_xg[position], f"{pred_xg[position]:.2f}", ha='right', va='bottom')

# plt.xticks(rotation=45)
# plt.show()


# In[ ]:


results_xg = pd.DataFrame({'Date': test_xg.index, 'Actual Orders':test_xg['Orders'], 'XG Boost Predictions':pred_xg})
# results_xg


# In[ ]:


reg = xgb.XGBRegressor(
    base_score=0.5,
    booster='gbtree',
    n_estimators=1000,
    objective='reg:linear',
    max_depth=3,
    learning_rate=0.01
)

df_xg['week'] = df_xg.index.week
df_xg['month'] = df_xg.index.month
df_xg['quarter'] = df_xg.index.quarter
df_xg['year'] = df_xg.index.year

rolling_window = 2
df_xg['rolling_mean'] = df_xg['Orders'].rolling(window=rolling_window).mean()
df_xg['rolling_std'] = df_xg['Orders'].rolling(window=rolling_window).std()

FEATURES = ['week','month', 'quarter', 'year', 'rolling_mean', 'rolling_std']
TARGET = ['Orders']

X = df_xg[FEATURES].dropna()
y = df_xg[TARGET].loc[X.index]

reg.fit(X, y, verbose=100)


# In[ ]:


future_start_date_xg = df_xg.index.max() + pd.DateOffset(days=7)
future_end_date_xg = future_start_date_xg + pd.DateOffset(weeks=7)
future_dates_xg = pd.date_range(start=future_start_date_xg, end=future_end_date_xg, freq='7D')
future_dates_xg = pd.DataFrame(index=future_dates_xg)
# future_dates_xg


# In[ ]:


future_dates_xg['week'] = future_dates_xg.index.week
future_dates_xg['month'] = future_dates_xg.index.month
future_dates_xg['quarter'] = future_dates_xg.index.quarter
future_dates_xg['year'] = future_dates_xg.index.year
future_dates_xg['rolling_mean'] = df_xg['rolling_mean'].iloc[-1]
future_dates_xg['rolling_std'] = df_xg['rolling_std'].iloc[-1]
# future_dates_xg


# In[ ]:


features_for_prediction = future_dates_xg[['week','month', 'quarter', 'year', 'rolling_mean', 'rolling_std']]
predictions = reg.predict(features_for_prediction)
predictions = predictions.round()
future_dates_xg['Predictions'] = predictions
# future_dates_xg


# In[ ]:


future_xg = pd.DataFrame({'Date': future_dates_xg.index, 'XG Boost Predictions': future_dates_xg['Predictions']})
future_xg['XG Boost Predictions'] = future_xg['XG Boost Predictions'].apply(lambda x: f"{x} (±{mae_xg:.0f})")
future_xg.set_index('Date', inplace=True)
# future_xg


# # Past 8 weeks Predicted Summary of all models

# In[ ]:


rmse_scores = [rmse_am, rmse_exp, rmse_fb, rmse_xg]
prediction_names = ['ARIMA Predictions', 'Exponential Smoothing Predictions', 'Prophet Predictions', 'XG Boost Predictions']
rmse_rank = {score: rank for score, rank in zip(rmse_scores, rankdata(rmse_scores))}
weights = [1 / rmse_rank[score] for score in rmse_scores]
weight_sum = sum(weights)
normalized_weights_rmse = [weight / weight_sum for weight in weights]

# for i, weight in enumerate(normalized_weights_rmse):
#     print(f"Weight for {prediction_names[i]}: {weight}")


# In[ ]:


prediction_columns = ['ARIMA Predictions', 'Exponential Smoothing Predictions', 'Prophet Predictions', 'XG Boost Predictions']

combined_results = pd.concat([results_am, results_exp, results_fb, results_xg], axis=1)
combined_results = combined_results.loc[:, ~combined_results.columns.duplicated()]

combined_results['Average'] = combined_results[prediction_columns].mean(axis=1)
combined_results['Average'] =combined_results['Average'].round()
combined_results = combined_results.set_index('Date')

combined_results_formatted = combined_results.copy()

for i, col in enumerate(prediction_columns):
    combined_results_formatted[col] = combined_results_formatted[col] * normalized_weights_rmse[i]

combined_results_formatted['Weighted Average'] = combined_results_formatted[prediction_columns].sum(axis=1)
combined_results_formatted['Weighted Average'] = combined_results_formatted['Weighted Average'].round()

combined_results['Weighted Average'] = combined_results_formatted['Weighted Average']
combined_results['Week'] = combined_results.index.strftime('(%W)')
# display(combined_results)


# In[ ]:


combined_results = combined_results.merge(df_for, how='left', left_index=True, right_index=True)
# display(combined_results)


# In[ ]:


st.title("⌛ Visualizing Actual Orders vs. AI Prediction vs. S&OP Forecast - Past 8 Weeks")
st.markdown("---")

min_rmse_index = rmse_scores.index(min(rmse_scores))
best_prediction = prediction_columns[min_rmse_index]

fig = go.Figure()

fig.add_trace(go.Scatter(x=combined_results['Week'], y=combined_results['Actual Orders'], name='Actual Orders'))
fig.add_trace(go.Scatter(x=combined_results['Week'], y=combined_results[best_prediction], name='AI Prediction'))
fig.add_trace(go.Scatter(x=combined_results['Week'], y=combined_results['S&OP Forecast'], name='S&OP Forecast'))

fig.update_layout(
    title=f'Actual Orders vs. AI Prediction vs. S&OP Forecast for {customer_name}',
    xaxis_title='Date',
    yaxis_title='Orders',
    legend_title='Legend',
    width=1000
)
st.plotly_chart(fig)


# # Future 8 weeks Predicted Summary of all models

# In[ ]:


future_combined_results = pd.concat([future_am, future_exp, future_fb, future_xg], axis=1)
future_combined_results = future_combined_results.loc[:, ~future_combined_results.columns.duplicated()]
future_combined_results_formatted = future_combined_results.copy()

prediction_order = ['ARIMA Predictions', 'Exponential Smoothing Predictions', 'Prophet Predictions', 'XG Boost Predictions']
future_combined_results = future_combined_results[prediction_order]

for col in prediction_order:
    future_combined_results[col] = future_combined_results[col].str.split(' ', expand=True)[0].astype(float)

future_combined_results['Average'] = future_combined_results.iloc[:, 0:4].mean(axis=1)
future_combined_results_formatted['Average'] = future_combined_results['Average'].round()

for i, col in enumerate(prediction_order):
    future_combined_results[col] = normalized_weights_rmse[i] * future_combined_results[col]

future_combined_results['Weighted Average'] = future_combined_results.iloc[:, 0:4].sum(axis=1)
future_combined_results_formatted['Weighted Average'] = future_combined_results['Weighted Average'].round()

future_combined_results_formatted['Week'] = future_combined_results.index.strftime('(%W)')
# display(future_combined_results_formatted)


# In[ ]:


st.title("🤖 Visualizing AI Prediction vs. S&OP Forecast - Future 8 Weeks")
st.markdown("---")

merged_data = pd.merge(future_combined_results_formatted, df_for, left_index=True, right_index=True)

fig = go.Figure()

fig.add_trace(go.Scatter(x=merged_data['Week'], y=merged_data['Average'],mode='lines+markers', name='AI Prediction'))

fig.add_trace(go.Scatter(x=merged_data['Week'], y=merged_data['S&OP Forecast'],mode='lines+markers', name='S&OP Forecast'))

fig.update_layout(title=f'AI Prediction vs. S&OP Forecast for {customer_name}',
                  xaxis_title='Week',
                  yaxis_title='Orders',
                  width=1000,
                  legend_title='Legend')

st.plotly_chart(fig)


# In[ ]:


# if holiday_effect == "Yes":
#     st.title("🤖 Visualizing AI Prediction vs. S&OP Forecast - Future 8 Weeks (Holiday Effect)")
#     st.markdown("---")

#     merged_data = pd.merge(future_combined_results_formatted, df_for, left_index=True, right_index=True)
#     merged_data['Prophet Predictions'] = merged_data['Prophet Predictions'].str.split('(').str[0].astype(float)

#     fig = go.Figure()

#     fig.add_trace(go.Scatter(x=merged_data['Week'], y=merged_data['Prophet Predictions'], mode='lines+markers', name='AI Prediction'))
#     fig.add_trace(go.Scatter(x=merged_data['Week'], y=merged_data['S&OP Forecast'], mode='lines+markers', name='S&OP Forecast'))

#     fig.update_layout(
#         title=f'AI Prediction vs. S&OP Forecast for {customer_name}',
#         xaxis_title='Week',
#         yaxis_title='Orders',
#         width=1000,
#         legend_title='Legend'
#     )

#     st.plotly_chart(fig)


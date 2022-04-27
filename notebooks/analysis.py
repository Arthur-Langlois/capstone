import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from functools import reduce
from fbprophet import Prophet

#function to check stationarity with Adfuller test
def get_adfuller(series, transformation = False):
    #calculate the 2 years rolling means (average of t-2 for each t)
    moving_avg = series.rolling(window = 2).mean()
    moving_std = series.rolling(window = 2).std()
    orig = plt.plot(series, color = 'black', label = 'Original')
    mean = plt.plot(moving_avg, color = 'red', label = 'Mean')
    std = plt.plot(moving_std, color = 'blue', label = 'Std')
    plt.legend(loc = 'best')
    if transformation == False:
        plt.title('Before Transformation')
    else:
        plt.title('After Transformation')
    plt.show(block = False)
    result = adfuller(series, autolag = 'AIC')
    print(f'ADF Statistic:{round(result[0],3)} and p-value: {round(result[1],3)}')

#function to make data stationary by taking logarithm of data inputed
def get_stationary(data):
    return np.log(data)

#function to get prediction for inputed model
def get_prediction(model, model_string):
    
    if model_string == 'ARIMA' or model_string == 'SARIMAX' or model_string == 'VARMAX' or model_string == 'SES' or model_string == 'ES':
        result = model.fit()
        return list(result.forecast(steps=4))
    elif model_string == 'Prophet':
        temp = df.reset_index().rename(columns={'Year': 'ds', 'Deaths':'y'})
        result = model.fit(temp)
        future = result.make_future_dataframe(periods=4)
        return list(result.predict(future))

def get_plot_predictions(predictions):
    b = data[:-4]
    a = b.append(pd.Series(predictions, index = [2017,2018,2019,2020]))
    plt.plot(a)
    plt.plot(data)
    plt.xlim([2010,2020])
    
#return dataframe with difference between prediction and actual data
def get_diff(actual,predicted, model_string):
    
    diff = pd.DataFrame({'Year': actual.index[-4:],
        f'Treatment {model_string}': actual[-4:] - predicted
                        })
    return diff

def get_plot_predictions(predictions):
    a = data[:-4]
    a = a.append(pd.Series(predictions[0], index = [2017,2018,2019,2020]))
    b = data[:-4]
    b = b.append(pd.Series(predictions[1], index = [2017,2018,2019,2020]))
    c = data[:-4]
    c = c.append(pd.Series(predictions[2], index = [2017,2018,2019,2020]))
    
    plt.figure(figsize = (10,5))
    plt.plot(a, color = 'red', linestyle = 'dotted')
    plt.plot(b, color = 'blue', linestyle = 'dotted')
    plt.plot(c, color = 'orange', linestyle = 'dotted')
    plt.plot(data, color = 'black')
    plt.legend(['ARIMA', 'ARIMAX', 'ES', 'Actual'])
    plt.axvline(x = 2017, color = 'grey', linestyle = 'dashdot')
    plt.xlim([2010,2020])
    plt.title('Model Predictions After Treatment')
    plt.xlabel('Years')
    plt.ylabel('Suicide Deaths')
    
df = pd.read_excel('../data/cdc_suicide.xlsx', index_col=4)
#Only keep ages between 5-24 years old
df = df[(df['Ten-Year Age Groups'] == '5-14 years') | (df['Ten-Year Age Groups'] == '15-24 years')]
#group data by average yearly deaths
df = df.groupby('Year')['Deaths'].mean()
#convert the series back to a dataframe
df = pd.DataFrame(df)

#get adfuller tests with untransformed data
get_adfuller(df['Deaths'])

data = df['Deaths']

#transform data with get_stationary function above
data_transformed = get_stationary(df['Deaths'])

#insert string of model used and its model parameters within the same tuple
models = [
        ('ARIMA', ARIMA(data, order = (1,1,2))),
       ('SARIMAX', sm.tsa.statespace.SARIMAX(data, order=(1, 1, 1),seasonal_order=(1,1,1,12))),
    ('SES', SimpleExpSmoothing(data)),
    ('Prophet',Prophet())
         ]

#iterate through the models variables to get a prediction for 2019,2019,2020 for each of those models
#then append result to the preds variable
preds = []
for i in models:
    preds.append(tuple([get_prediction(i[1], i[0]), i[0]]))
    
#iterate over predicted results and call function to get dataframe for all models that we tried and append the result to the list_df
ind = [2017,2018,2019,2020]
list_df = []
for i in preds[:-1]:
    df = get_diff(data, i[0], i[1])
    df = df.iloc[: , 1:]
    list_df.append(df)

final = reduce(lambda x, y: pd.merge(x, y, on = 'Year'), list_df)

#call graph predictions
for i in preds[:-1]:
    p = [i[0] for i in preds[:-1]]
    get_plot_predictions(p)


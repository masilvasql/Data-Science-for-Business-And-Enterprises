# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 10:45:48 2020

@author: marcelo.silva
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

#dados das vendas 
sales_train_df= pd.read_csv('train.csv')

shapeTrain = sales_train_df.shape
headeTrain = sales_train_df.head()

daysOfWeek = sales_train_df['DayOfWeek'].unique();
isOpen = sales_train_df['Open'].unique()
isPromo = sales_train_df['Promo'].unique()
stateHoliday = sales_train_df['StateHoliday'].unique()
schoolHoliday = sales_train_df['SchoolHoliday'].unique()

cincoUltimos = sales_train_df.tail()

sales_train_df.info()

sales_train_df.describe()

#dados das lojas
store_info_df = pd.read_csv('store.csv')

shapeStore = store_info_df.shape

headStore = store_info_df.head()

infoStore = store_info_df.info()

store_info_df.describe()

#DADOS DAS VENDAS
"Verificando se existem dados faltantes"
sns.heatmap(sales_train_df.isnull())
sales_train_df.isnull().sum()

#histogramas
sales_train_df.hist(bins = 30, figsize = [20,20], color='b')
sales_train_df['Customers'].max()

closed_train_df = sales_train_df[sales_train_df['Open'] == 0]
open_train_df = sales_train_df[sales_train_df['Open'] == 1]

print('Total', len(sales_train_df))
print('Número de lojas/dias fechado', len(closed_train_df))
print('Número de lojas/dias aberto', len(open_train_df))

closed_train_df.head()
open_train_df.head()

sales_train_df = sales_train_df[sales_train_df['Open'] == 1]
sales_train_df.shape

sales_train_df.drop(['Open'], axis = 1, inplace = True)

describeSales = sales_train_df.describe()

#DADOS DAS LOJAS
sns.heatmap(store_info_df.isnull())

store_info_df[store_info_df['CompetitionDistance'].isnull()]
store_info_df[store_info_df['CompetitionOpenSinceMonth'].isnull()]
store_info_df[store_info_df['CompetitionOpenSinceYear'].isnull()]

naoEstaNaPromocao = store_info_df[store_info_df['Promo2'] == 0]

str_cols = ['Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']

for str in str_cols:
    store_info_df[str].fillna(0, inplace = True)
    
sns.heatmap(store_info_df.isnull())

#preenchendo valores faltantes com a média
store_info_df['CompetitionDistance'].fillna(store_info_df['CompetitionDistance'].mean(), inplace = True)
sns.heatmap(store_info_df.isnull())

store_info_df.hist(bins = 30, figsize=[20,20], color = 'g')

#DADOS UNIDOS - combinando dataframes Lojas x Vendas

sales_train_df.head()
store_info_df.head()
#inner join com o atributo store que contem nas duas tabelas
sales_train_all_df = pd.merge(sales_train_df, store_info_df, how = 'inner', on = 'Store')

headAll = sales_train_all_df.head()
tailAll = sales_train_all_df.tail()

correlations = sales_train_all_df.corr()
f, ax = plt.subplots(figsize = [20,20])
sns.heatmap(correlations, annot = True)

#hanking
correlations = sales_train_all_df.corr()['Sales'].sort_values()

sales_train_all_df['Year'] = pd.DatetimeIndex(sales_train_all_df['Date']).year
sales_train_all_df['Month'] = pd.DatetimeIndex(sales_train_all_df['Date']).month
sales_train_all_df['Day'] = pd.DatetimeIndex(sales_train_all_df['Date']).day
headAll = sales_train_all_df.head()

#grafico para gerar a média de venda e numero de clientes por mês

axis = sales_train_all_df.groupby('Month')[['Sales']].mean().plot(figsize = [10,5], marker = 'o', color ='y')
axis.set_title('Média de vendas por mês')

axis = sales_train_all_df.groupby('Month')[['Customers']].mean().plot(figsize = [10,5], marker = '^', color ='b')
axis.set_title('Média de clientes por mês')

axis = sales_train_all_df.groupby('Day')[['Sales']].mean().plot(figsize = [10,5], marker = '^', color ='b')
axis.set_title('Média de vendas por dia')

axis = sales_train_all_df.groupby('Day')[['Customers']].mean().plot(figsize = [10,5], marker = '^', color ='b')
axis.set_title('Média de clientes por dia')

axis = sales_train_all_df.groupby('DayOfWeek')[['Sales']].mean().plot(figsize = [10,5], marker = '^', color ='b')
axis.set_title('Média de vendas por dia da Semana')

axis = sales_train_all_df.groupby('DayOfWeek')[['Customers']].mean().plot(figsize = [10,5], marker = '^', color ='r')
axis.set_title('Média de clientes por dia da Semana')

fix, ax = plt.subplots(figsize = [20,10])
sales_train_all_df.groupby(['Date', 'StoreType']).mean()['Sales'].unstack().plot(ax = ax)

sns.barplot(x = 'Promo', y = 'Sales', data = sales_train_all_df)

sns.barplot(x = 'Promo', y = 'Customers', data = sales_train_all_df)

#Treinamento

from fbprophet import Prophet

#nomenclatura que a biblioteca trabalha
# Date: ds
# Sales : y
def sales_prediction(store_id, sales_df, periods):
    sales_df = sales_df[sales_df['Store'] == store_id]
    sales_df = sales_df[['Date', 'Sales']].rename(columns = {'Date' : 'ds', 'Sales': 'y'})
    sales_df = sales_df.sort_values(by = 'ds')
    model = Prophet()
    model.fit(sales_df)
    
    future = model.make_future_dataframe(periods = periods)
    forecast = model.predict(future) #aqui é a previsão
    figure = model.plot(forecast, xlabel = 'Data', ylabel = 'Vendas')
    figure2 = model.plot_components(forecast)
    return sales_df, forecast

df_origin, df_prediction = sales_prediction(10, sales_train_all_df, 60)

df_origin.shape

df_prediction.shape

headPreditcion = df_prediction.head()

df_origin.tail(60)
df_prediction.tail(60) #dez ultimos registros

df_prediction.tail(60).to_csv('previsores_vendas.csv') #dez ultimos registros

#df = sales_prediction(10, sales_train_all_df, 60)

"""FERIADOOOOOOOOOOOOOOOOOOOOOOO"""
#nomenclatura que a biblioteca trabalha
# Date: ds
# Sales : y
def sales_prediction_with_holiday(store_id, sales_df, holidays , periods):
    sales_df = sales_df[sales_df['Store'] == store_id]
    sales_df = sales_df[['Date', 'Sales']].rename(columns = {'Date' : 'ds', 'Sales': 'y'})
    sales_df = sales_df.sort_values(by = 'ds')
    model = Prophet(holidays = holidays)
    model.fit(sales_df)
    
    future = model.make_future_dataframe(periods = periods)
    forecast = model.predict(future) #aqui é a previsão
    figure = model.plot(forecast, xlabel = 'Data', ylabel = 'Vendas')
    figure2 = model.plot_components(forecast)
    return sales_df, forecast

school_holidays = sales_train_all_df[sales_train_all_df['SchoolHoliday'] == 1].loc[:,'Date'].values

school_holidays.shape

len(np.unique(school_holidays))

state_holidays = sales_train_all_df[(sales_train_all_df['StateHoliday'] == 'a') | 
                                    (sales_train_all_df['StateHoliday'] == 'b') |
                                    (sales_train_all_df['StateHoliday'] == 'c')].loc[:,'Date'].values

state_holidays.shape

len(np.unique(state_holidays))

state_holidays = pd.DataFrame({'ds': pd.to_datetime(state_holidays),
                               'holiday': 'state_holiday'})

school_holidays = pd.DataFrame({'ds': pd.to_datetime(school_holidays),
                               'holiday': 'school_holidays'})

school_state_holidays = pd.concat((state_holidays, school_holidays))

school_state_holidays



df_origin2, df_prediction2 = sales_prediction_with_holiday(10, sales_train_all_df, school_state_holidays ,30)

df_origin2.shape

df_prediction2.shape

headPreditcion2 = df_prediction.head()

df_origin2.tail(60)
df_prediction2.tail(60) #dez ultimos registros

df_prediction2.tail(60).to_csv('previsores_vendas_with_holiday.csv') #dez ultimos registros


    


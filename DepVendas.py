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

sales_train_df.shape
sales_train_df.head()

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

store_info_df.shape
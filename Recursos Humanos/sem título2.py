# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 14:10:37 2020

@author: marcelo.silva
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

employee_df = pd.read_csv('Human_Resources.csv')

linhasEcolunas = employee_df.shape

"Retorna os 5 primeiros registros"
cabecalho = employee_df.head() 

informacoesAdicionais = employee_df.info()

estatisicasAtributos = employee_df.describe()

">>>>>>>>>>>>> Visualização de DADOS <<<<<<<<<<<<<"

employee_df['Attrition'] = employee_df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
employee_df['Over18'] = employee_df['Over18'].apply(lambda x: 1 if x == 'Y' else 0)
employee_df['OverTime'] = employee_df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)

"Gráfico de calor"
sns.heatmap(employee_df.isnull())

"Gráfico para os atributos"
"""
    hist = histograma    
    bins = 30 -> 30 faixa de valores
    
"""
employee_df.hist(bins = 30, figsize = (20,20), color = 'r')

"""
Removendo dados desnecessários

axis = COLUNAS
inplace = faz alterações diretamente na variável employee_df, caso não utilize, tem que atribuir em uma variável
"""
employee_df.drop(['EmployeeCount', "StandardHours", "Over18", "EmployeeNumber"], axis = 1, inplace = True)

left_df = employee_df[employee_df['Attrition'] == 1 ]; "Dataframe das pessoas que saíram"
stayed_df = employee_df[employee_df['Attrition'] == 0 ]; "Dataframe das pessoas que ficaram"

print('TOTAL = ', len(employee_df))
print("Número de funcionários que saíram da empresa " , len(left_df))
print("Porcentagem de funcionários que saíram da empresa", (len(left_df) / len(employee_df)) *100 )
print("----------------------------------------------------------------------------------------------")
print("Número de funcionários que ficaram da empresa " , len(stayed_df))
print("Porcentagem de funcionários que ficaram da empresa", (len(stayed_df) / len(employee_df)) *100 )

estatisticasSairamEmpresa = left_df.describe()

estatisticasFicaramEmpresa = stayed_df.describe()

"corelações"
"""corr = calculo estátistico que compara todos os atributos com todos
    cálculo para avaliar o quanto um atributo pode afetar o outro
    quanto mais um atributo estiver perto de 1, mas correlacionado com o outro estará
"""
correlations = employee_df.corr()
f, ax = plt.subplots(figsize = (20,20))
sns.heatmap(correlations, annot = True)

"Gráficos de contagem"
"""
x= atributo do eixo x
hue = "agrupador"
data = seu dataFrame
"""
plt.figure(figsize=[25,12])
sns.countplot(x= 'Age', hue = 'Attrition', data = employee_df)

"""
    plt.subplot(411)
    
    411 é lido como 
    4 = qtd de linhas
    1 = coluna
    1 = idDoGrafico
"""

plt.figure(figsize=[20,20])
plt.subplot(411)
sns.countplot(x= 'JobRole', hue = 'Attrition', data = employee_df)

plt.subplot(412)
sns.countplot(x= 'MaritalStatus', hue = 'Attrition', data = employee_df)

plt.subplot(413)
sns.countplot(x= 'JobInvolvement', hue = 'Attrition', data = employee_df)

plt.subplot(414)
sns.countplot(x= 'JobLevel', hue = 'Attrition', data = employee_df)

#KDE (Kernel Density Estimate)
"Desnsidade da probabilidade de uma variável numérica"
plt.figure(figsize = [12,7])
sns.kdeplot(left_df['DistanceFromHome'], label = "Funcionários que sáiram da Empresa", shade = True ,color = 'r')
sns.kdeplot(stayed_df['DistanceFromHome'], label = "Funcionários que ficaram na Empresa", shade = True ,color = 'b')

plt.figure(figsize = [12,7])
sns.kdeplot(left_df['TotalWorkingYears'], label = "Funcionários que sáiram da Empresa", shade = True ,color = 'r')
sns.kdeplot(stayed_df['TotalWorkingYears'], label = "Funcionários que ficaram na Empresa", shade = True ,color = 'b')

plt.figure(figsize = [15,10])
sns.boxplot(x = "MonthlyIncome", y = "Gender", data = employee_df)

plt.figure(figsize = [15,10])
sns.boxplot(x = 'MonthlyIncome', y ='JobRole', data = employee_df)


""" >>>>> Pré processamento <<<<< """

cabecalho = employee_df.head()

"Conversão de atributos categóricos 'String'"
X_cat = employee_df[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]

'Variáveis dummies'
from sklearn.preprocessing import OneHotEncoder
oneHotenCoder = OneHotEncoder()
X_cat = oneHotenCoder.fit_transform(X_cat).toarray()
X_cat.shape
type(X_cat)
X_cat = pd.DataFrame(X_cat)
type(X_cat)

employee_df['BusinessTravel'].unique()

X_numerical = employee_df[['Age',	'DailyRate',	'DistanceFromHome',	'Education',		
                           'EnvironmentSatisfaction',	'HourlyRate',	'JobInvolvement',	
                           'JobLevel',	'JobSatisfaction',	'MonthlyIncome',	'MonthlyRate',	
                           'NumCompaniesWorked',	'PercentSalaryHike',	'PerformanceRating',	
                           'RelationshipSatisfaction',		'StockOptionLevel',	'TotalWorkingYears',	
                           'TrainingTimesLastYear',	'WorkLifeBalance',	'YearsAtCompany',	'YearsInCurrentRole',	
                           'YearsSinceLastPromotion',	'YearsWithCurrManager']]

X_all = pd.concat([X_cat, X_numerical], axis = 1)
"NORMALIZAÇÃO DOS DADOS para não considerar um dado mais importante que o outro"

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X_all) #atributos previsores 

y = employee_df['Attrition']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) # 25% dos dados para testar e o restante para o algoritmo aprender
# X_TRAIN contem os dados que serão utilizados como base para a previsão
# Y_TRAIN contem os dados que contem os dados de entrada para o algoritmo prever
X_train.shape, y_train

#X_test -> Atributos previsores 
#y_test -> Classe

X_test.shape, y_test
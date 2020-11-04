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

"""REGRESSÃO LOGÍSTICA"""
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
logistic.fit(X_train, y_train) #treinamento

"Iniciando as previsões"
y_predLogistic = logistic.predict(X_test)

#y_test são os dados reais que estão na base que serão comparados com os valores que o algoritmo previu
y_test

#calculando a acurácia
from sklearn.metrics import accuracy_score
acuracia = accuracy_score(y_test, y_predLogistic)

#matriz de confusão
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predLogistic)
sns.heatmap(cm, annot = True)

#precision
"""
 TP / (TP + FP)
 17 /( 17 + 4)
"""
precision = 17 / (17 +4)

#RECALL
"""
TP / (TP + FP)
"""
recall = 17 / (17 + 49)

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

precision = precision_score(y_test, y_predLogistic)

recall = recall_score(y_test, y_predLogistic)

f1Score = f1_score(y_test, y_predLogistic, average='macro')

clReport = classification_report(y_test, y_predLogistic)

"""RANDOM FOREST"""
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier()
forest.fit(X_train, y_train)

y_predForest = forest.predict(X_test)

accuracyScoreForest = accuracy_score(y_test, y_predForest)
cmForest = confusion_matrix(y_test, y_predLogistic)
sns.heatmap(cmForest, annot = True)
clReportForest = classification_report(y_test, y_predForest)

"""REDES NEURAIS"""
import tensorflow as tf
#sequência de camadas
X_train.shape
"""
numero de entradas + o número de saidas / 2
(49 + 1) / 2  = 25
"""
#relu se for > 0 retorna o mesmo valor, caso contrário = 0
rede_neural = tf.keras.models.Sequential()
rede_neural.add(tf.keras.layers.Dense(units = 25, activation = 'relu', input_shape = (49,))) #entrada
rede_neural.add(tf.keras.layers.Dense(units = 25, activation = 'relu')) #oculta
rede_neural.add(tf.keras.layers.Dense(units = 25, activation = 'relu')) #oculta
rede_neural.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid')) # saída

rede_neural.summary()

rede_neural.compile(optimizer='Adam', loss='binary_crossentropy', metrics = ['accuracy'])

rede_neural.fit(X_train, y_train, epochs=200)

y_predNeural = rede_neural.predict(X_test)

y_predNeural = (y_predNeural >=0.5)

cm = confusion_matrix(y_test, y_predNeural)
sns.heatmap(cm, annot = True)

clReportNeural = classification_report(y_test, y_predNeural)

"SALVAR CLASSIFICADOR"
import pickle 

with open('variaveis_modelo.pkl', 'wb') as f:
    pickle.dump([scaler, oneHotenCoder ,logistic], f)
    
with open('variaveis_modelo.pkl', 'rb') as f:
    min_max, encoder, model = pickle.load(f)
    
min_max, encoder, model

X_novo = employee_df.iloc[0:1]

X_cat_novo = X_novo[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]
X_cat_novo = encoder.transform(X_cat_novo).toarray()

X_cat_novo = pd.DataFrame(X_cat_novo)

X_numerical_novo = X_novo[['Age',	'DailyRate',	'DistanceFromHome',	'Education',		
                           'EnvironmentSatisfaction',	'HourlyRate',	'JobInvolvement',	
                           'JobLevel',	'JobSatisfaction',	'MonthlyIncome',	'MonthlyRate',	
                           'NumCompaniesWorked',	'PercentSalaryHike',	'PerformanceRating',	
                           'RelationshipSatisfaction',		'StockOptionLevel',	'TotalWorkingYears',	
                           'TrainingTimesLastYear',	'WorkLifeBalance',	'YearsAtCompany',	'YearsInCurrentRole',	
                           'YearsSinceLastPromotion',	'YearsWithCurrManager']]

X_all_novo = pd.concat([X_cat_novo, X_numerical_novo], axis = 1)

X_novo = min_max.transform(X_all_novo)

resultPredict = model.predict(X_novo)

modelo.predict_proba(X_novo)
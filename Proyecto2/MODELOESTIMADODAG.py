#»»————-　★　————-«« LIBRERIAS »»————-　★　————-««
import pandas as pd
import statistics as st
from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import seaborn as sns
import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import numpy as np
import sklearn as sk
from sklearn.metrics import confusion_matrix
from pgmpy.estimators import BayesianEstimator
from sklearn.model_selection import train_test_split
##Nuevas librerias
from pgmpy . estimators import HillClimbSearch
from pgmpy . estimators import K2Score
from pgmpy . estimators import PC
from pgmpy . models import BayesianNetwork
from pgmpy . estimators import MaximumLikelihoodEstimator

#»»————-　★　————-««»»————-　★　————-«« CODIGO »»————-　★　————-««»»————-　★　————-««

#Vamos a importar los datos
# fetch dataset 
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697) 
  
# data (as pandas dataframes) 
X = predict_students_dropout_and_academic_success.data.features 
y = predict_students_dropout_and_academic_success.data.targets 

dfx=pd.DataFrame(X)
dfy=pd.DataFrame(y)

df = pd.concat([dfx, dfy], axis=1)
data_og =df
data_top = df.head()

#»»————-　★　————-««»»————-　★　————-«« DROP »»————-　★　————-««»»————-　★　————-««
#Application mode 
#Application order 
#Previous qualification (grade) 
#Mother's qualification 
#Father's qualification 
#Mother's occupation 
#Father's occupation  
#Curricular units 1st sem (credited)
#Curricular units 1st sem (evaluations)
#Curricular units 1st sem (grade)
#Curricular units 1st sem (without evaluations)
#Curricular units 2nd sem (credited)
#Curricular units 2nd sem (evaluations)
#Curricular units 2nd sem (grade)
#Curricular units 2nd sem (without evaluations)
#Unemployment rate
#Inflation rate
#GDP
#»»————-　★　————-««»»————-　★　————-««»»————-　★　————-««»»————-　★　————-««»»————-　★　————-««

df=df.drop(["Marital Status","Application mode","Application order","Previous qualification (grade)",
           "Mother's qualification","Father's qualification","Mother's occupation","Father's occupation","Curricular units 1st sem (credited)",
           "Curricular units 1st sem (grade)","Curricular units 1st sem (without evaluations)","Curricular units 2nd sem (credited)","Curricular units 2nd sem (evaluations)",
           "Curricular units 2nd sem (grade)","Curricular units 2nd sem (without evaluations)","Curricular units 1st sem (evaluations)","Unemployment rate","Inflation rate","GDP"],axis=1)

#print(df.head())

#for col in df.columns:
#    print(col)

nombres_df=("cor","day","prevqua","nac","adgr","dis","spn","deb","tui","g","sch","age","inter","enr1","apr1","enr2","apr2","Target")
df.columns=nombres_df
df['pass1'] = df['apr1']/df['enr1']
df['pass2'] = df['apr2']/df['enr2']
df= df.fillna(0)
df=df.drop(['enr1','apr1','enr2','apr2'], axis=1)

print(df.head())

#Usando pgmpy estime la estructura del modelo usando el método de restricciones.
#est = PC(data =df)
#estimated_model = est.estimate ( variant ="stable",ci_test='chi_square', max_cond_vars=15, return_type='dag', significance_level=0.01)
#print ( estimated_model )
#print ( estimated_model.nodes ())
#print ( estimated_model.edges ())

#DAG with 13 nodes and 17 edges
#NODES (['dis', 'age', 'day', 'cor', 'pass2', 'prevqua', 'g', 'Target', 'pass1', 'tui', 'nac', 'inter', 'deb'])
#DAG [('dis', 'age'), ('age', 'prevqua'), ('age', 'cor'), ('age', 'pass1'), ('day', 'age'), ('day', 'cor'), ('pass2', 'age'), ('pass2', 'cor'), ('prevqua', 'cor'), ('g', 'cor'), ('Target', 'pass2'), ('Target', 'pass1'), ('pass1', 'cor'), ('pass1', 'pass2'), ('tui', 'Target'), ('tui', 'deb'), ('inter', 'nac')]
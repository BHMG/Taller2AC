#»»————-　★　————-««»»————-　★　————-««Proyecto y tratamiento de datos Parte 2»»————-　★　————-««»»————-　★　————-««
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
from pgmpy . estimators import BicScore

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
print(df.head())
df=df.drop(['enr1','apr1','enr2','apr2'], axis=1)
print(df.head())

#print(df["enr1"].mean())

#Usando pgmpy estime la estructura del modelo usando el método de restricciones.
#est = PC(data =df)
#estimated_model = est.estimate ( variant ="stable", max_cond_vars =18)
#print ( estimated_model )
#print ( estimated_model.nodes ())
#print ( estimated_model.edges ())

#»»————-　★　————-««»»————-　★　————-«« MODELOS ESTIMADOS »»————-　★　————-««»»————-　★　————-««
#[('apr1', 'Target'), ('apr1', 'age'), ('apr1', 'enr1'), ('Target', 'apr2'), ('prevqua', 'cor'), ('enr1', 'cor'), ('enr1', 'apr2'), ('enr2', 'apr2'), ('enr2', 'cor'), ('enr2', 'enr1'), ('apr2', 'cor'), ('apr2', 'apr1'), ('age', 'cor'), ('age', 'day'), ('age', 'prevqua'), ('age', 'dis'), ('age', 'apr2'), ('age', 'Target'), ('day', 'cor'), ('g', 'cor'), ('inter', 'nac'), ('tui', 'deb')]
#[('enr2', 'cor'), ('enr2', 'enr1'), ('enr2', 'apr2'), ('age', 'dis'), ('age', 'day'), ('age', 'prevqua'), ('age', 'apr2'), ('age', 'cor'), ('age', 'Target'), ('day', 'cor'), ('Target', 'apr2'), ('apr2', 'cor'), ('apr2', 'apr1'), ('g', 'cor'), ('prevqua', 'cor'), ('apr1', 'age'), ('apr1', 'Target'), ('apr1', 'enr1'), ('enr1', 'apr2'), ('enr1', 'cor'), ('tui', 'deb'), ('nac', 'inter')]

##DAG with 13 nodes and 17 edges----hecho con ci-square
#NODES (['dis', 'age', 'day', 'cor', 'pass2', 'prevqua', 'g', 'Target', 'pass1', 'tui', 'nac', 'inter', 'deb'])
#DAG [('dis', 'age'), ('age', 'prevqua'), ('age', 'cor'), ('age', 'pass1'), ('day', 'age'), ('day', 'cor'), ('pass2', 'age'), ('pass2', 'cor'), ('prevqua', 'cor'), ('g', 'cor'), ('Target', 'pass2'), ('Target', 'pass1'), ('pass1', 'cor'), ('pass1', 'pass2'), ('tui', 'Target'), ('tui', 'deb'), ('inter', 'nac')]


#Convierta el objeto DAG obtenido con el anterior procedimiento a una red bayesiana y use el estimador de máxima verosimilitud para estimar los par´ametros de la red.

#estimated_model_1 = BayesianNetwork([('enr2', 'cor'), ('enr2', 'enr1'), 
#                                    ('enr2', 'apr2'), ('age', 'dis'), 
#                                    ('age', 'day'), ('age', 'prevqua'), 
#                                    ('age', 'apr2'), ('age', 'cor'), 
#                                    ('age', 'Target'), ('day', 'cor'),(Target', 'apr2'), ('apr2', 'cor'), 
#                                    ('apr2', 'apr1'), ('g', 'cor'), 
#                                    ('prevqua', 'cor'), ('apr1', 'age'), 
#                                    ('apr1', 'Target'), ('apr1', 'enr1'), 
#                                    ('enr1', 'apr2'), ('enr1', 'cor'), 
#                                    ('tui', 'deb'), ('nac', 'inter')])

#Estos causan problemas

#[('enr2', 'enr1'), ('apr2', 'cor'), ('apr2', 'apr1'), ('age', 'cor'), ('age', 'day'), ('age', 'prevqua'), ('age', 'dis'), ('age', 'apr2'), ('age', 'Target'), ('day', 'cor'), ('g', 'cor'), ('inter', 'nac'), ('tui', 'deb')]
#estimated_model_2 = BayesianNetwork([('apr1', 'Target'),('apr1', 'age'),('apr1', 'enr1'),
#                                     ('Target', 'apr2'),('prevqua', 'cor'),('enr1', 'cor'),
#                                     ('enr1', 'apr2'),('enr2', 'apr2'),('enr2', 'cor'),
#                                     ('enr2', 'enr1'),('apr2', 'cor')]) #hasta acá sirve

estimated_model_3=BayesianNetwork()
estimated_model_3.add_nodes_from(['dis', 'age', 'day', 'cor', 'pass2', 'prevqua', 'g', 'Target', 'pass1', 'tui', 'nac', 'inter', 'deb'])
estimated_model_3.add_edges_from(ebunch=[('dis', 'age'), ('age', 'prevqua'), ('age', 'cor'), ('age', 'pass1'), ('day', 'age'), ('day', 'cor'), ('pass2', 'age'), ('pass2', 'cor'), ('prevqua', 'cor'), ('g', 'cor'), ('Target', 'pass2'), ('Target', 'pass1'), ('pass1', 'cor'), ('tui', 'Target'), ('tui', 'deb'), ('inter', 'nac')])

# ('pass1', 'pass2') SE QUITA PORQUE HACE LOOP

print("Im here my liege, the model 1 is over")

estimated_model_p1=BayesianNetwork()
estimated_model_p1.add_nodes_from(['dis', 'age', 'day', 'cor','prevqua', 'g', 'Target','tui', 'nac', 'inter', 'deb'])
estimated_model_p1.add_edges_from(ebunch=[('dis','Target'),('age','Target'),('day','Target'),('cor','Target'),('prevqua','Target'),
                                          ('g','Target'),('tui','Target'),('nac','Target'),('inter','Target'),('deb','Target')])

print("Im here my liege, the model 2 is over")

#estimated_model_2.fit( data =df , estimator = MaximumLikelihoodEstimator)

scoring_method = BicScore ( data =df)
print('El puntaje BIC resultante del modelo estimado DAG es: ')
print ( scoring_method.score( estimated_model_3))

print('El puntaje BIC resultante del modelo de p1 es: ')
print ( scoring_method.score( estimated_model_p1))
#CONCLUSION EL MODELO DAG ES MEJOR

scoring_method2 = K2Score ( data =df)
print('El puntaje resultante K2 del modelo estimado DAG es: ')
print ( scoring_method2.score( estimated_model_3))

print('El puntaje resultante K2 del modelo de p1 es: ')
print ( scoring_method2.score( estimated_model_p1))
#CONCLUSION ES MEJOR EL MODELO DAG

train_data,predict_data= train_test_split(df, test_size=0.33, random_state=42)

estimated_model_3.fit( train_data , estimator = MaximumLikelihoodEstimator)
print("Im here my liege, the fit is over")

#predict_data = predict_data.copy()
#predict_data.drop(['Target','adgr','spn','sch'], axis=1, inplace=True)
#print(predict_data.head())
#y_pred = estimated_model_3.predict(predict_data)
#y_pred




#for i in estimated_model_1.nodes():
#    print(estimated_model_1.get_cpds(i))


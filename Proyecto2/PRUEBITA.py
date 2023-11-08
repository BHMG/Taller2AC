import dash
from dash import dcc  # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output
import psycopg2
# pip install python-dotenv
import os
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator,BayesianEstimator
from pgmpy.inference import VariableElimination

#━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡#｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡
#━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡#｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡


#variables
USER='postgres'
PASSWORD='contrasena'
HOST='database-2.cx2pwtnkiilc.us-east-1.rds.amazonaws.com'
PORT='5432'
DBNAME='postgres'
    
#connect to DB
engine = psycopg2.connect(
    host=HOST,
    port=PORT,
    user=USER,    
    password=PASSWORD,
    dbname=DBNAME
)


df=pd.read_sql_query('SELECT * FROM sheet1', engine)
engine.close()
datos1=df.drop(columns=['adgr','spn','sch','pass1','pass2'],)
#convertimos a numeros la variable de interes 'target'
mapping = {'Dropout': 0, 'Graduate': 1, 'Enrolled': 2}

# Use the map function to apply the mapping to the 'target' column
datos1['target'] = datos1['target'].map(mapping)

# If you want to change the column data type to integer
datos1['target'] = datos1['target'].astype(int)
datos1['day'] = datos1['day'].astype(int)
datos1['prevqua'] = datos1['prevqua'].astype(int)
datos1['nac'] = datos1['nac'].astype(int)
datos1['dis'] = datos1['dis'].astype(int)
datos1['deb'] = datos1['deb'].astype(int)
datos1['tui'] = datos1['tui'].astype(int)
datos1['g'] = datos1['g'].astype(int)
datos1['age'] = datos1['age'].astype(int)
datos1['inter'] = datos1['inter'].astype(int)

print('ya va a empezar la funcion')
#-----------------------------TRATAMIENTO DE DATOS-----------------------
def bayesian_inference_1(cor,day,prevqua,nac,dis,deb,tui,g,age,inter):
    print('nohe hecho nada')
    #cero el modelo bayesiano
    model = BayesianNetwork([('dis','target'),('age','target'),('day','target'),('cor','target'),('prevqua','target'),
                                          ('g','target'),('tui','target'),('nac','target'),('inter','target'),('deb','target')])
    print('ya se creo el modelo')
    #estimamos las distribuciones de probabilidad utilizando MLE y BayesianEstimator
    model.fit(datos1,estimator=MaximumLikelihoodEstimator)
    model.fit(datos1, estimator=BayesianEstimator, prior_type='BDeu',equivalent_sample_size = 100 )
    #Hacemos inferencia en el modelo bayesiano
    infer = VariableElimination(model)

    q=infer.query(['target'],evidence={'cor':cor,'day':day,'prevqua':prevqua,'nac':nac,'dis':dis,'deb':deb,'tui':tui,'g':g,'age':age,'inter':inter})
    print('im here  my liege')
    return q



a = bayesian_inference_1(33,1,1,1,1,1,1,1,21,1)
print(a)
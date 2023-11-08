import pandas as pd
import dash
from dash import dcc  # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import pgmpy
from ucimlrepo import fetch_ucirepo


predict_students_dropout_and_academic_success = fetch_ucirepo(id=697) 
  
# data (as pandas dataframes) 
X = predict_students_dropout_and_academic_success.data.features 
y = predict_students_dropout_and_academic_success.data.targets 

dfx=pd.DataFrame(X)
dfy=pd.DataFrame(y)

df = pd.concat([dfx, dfy], axis=1)
data_og =df
data_top = df.head()
df=df.drop(["Marital Status","Application mode","Application order","Previous qualification (grade)",
           "Mother's qualification","Father's qualification","Mother's occupation","Father's occupation","Curricular units 1st sem (credited)",
           "Curricular units 1st sem (grade)","Curricular units 1st sem (without evaluations)","Curricular units 2nd sem (credited)","Curricular units 2nd sem (evaluations)",
           "Curricular units 2nd sem (grade)","Curricular units 2nd sem (without evaluations)","Curricular units 1st sem (evaluations)","Unemployment rate","Inflation rate","GDP"],axis=1)

nombres_df=("cor","day","prevqua","nac","adgr","dis","spn","deb","tui","g","sch","age","inter","enr1","apr1","enr2","apr2","Target")
df.columns=nombres_df
df['pass1'] = df['apr1']/df['enr1']
df['pass2'] = df['apr2']/df['enr2']

df= df.fillna(0)
#print(df.head())
df=df.drop(['enr1','apr1','enr2','apr2'], axis=1)

# Especificar la ruta y el nombre del archivo
ruta_archivo = 'C:/Users/BEATRIZ/Documents/2023-2/Analiticacomputacional/Proyecto2/archivo.csv'

# Guardar el DataFrame en un archivo CSV
#df.to_csv(ruta_archivo, index=False)

from sqlalchemy import create_engine
import pandas as pd

disk_engine = create_engine('sqlite:///awesome.db')

import csvtosql as cs

cs.makeFileIntoSQL('file1.csv', 'data', disk_engine)
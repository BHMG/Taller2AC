#visualizaciones

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
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt

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


top_cor_values = df['cor'].value_counts().nlargest(3).index
df_top_cor = df[df['cor'].isin(top_cor_values)]
# Agrupamos por 'cor' y 'target' y contamos el número de ocurrencias
grouped_df = df_top_cor.groupby(['cor', 'Target']).size().reset_index(name='count')
pivot_table = grouped_df.pivot(index='Target', columns='cor', values='count').fillna(0)
figure1, ax = plt.subplots()
# Colores para cada valor de 'cor'
colors = ['lightblue', 'lightgreen', 'lightcoral']
# Crear barras apiladas
pivot_table.plot(kind='bar', stacked=True, ax=ax, color=colors)
# Añadimos leyendas personalizadas para 'cor'
custom_labels = {
    9147: 'Management',
    9238: 'Social Service',
    9500: 'Nursing'
}
plt.legend(title='Cursos', labels=[f'{key}:{value}' for key, value in custom_labels.items()])

# Añadimos etiquetas
plt.xlabel('Estado académico')
plt.ylabel('Cantidad de estudiantes')
plt.title('Cursos más comunes y el estado académico de sus estudiantes')

# Mostrar la gráfica

grouped_df = df.groupby(['g', 'Target']).size().unstack(fill_value=0)
# Creamos una figura y un eje para la gráfica de barras apiladas
figure2, ax = plt.subplots()
# Colores para cada valor de 'Target'
colors = ['lightblue', 'lightgreen', 'lightcoral']
grouped_df.plot(kind='bar', stacked=True, ax=ax, color=colors)
# Cambiamos las etiquetas del eje x
new_labels = ['Mujer', 'Hombre']
plt.xticks(range(len(new_labels)), new_labels)
plt.xlabel('Género')
plt.ylabel('Cantidad de estudiantes')
plt.title('Cantidad de Dropout, Enrolled y Graduate según Género')
plt.legend(title='Target', loc='upper right')

plt.show()

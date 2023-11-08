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
#-----------------------------TRATAMIENTO DE DATOS-----------------------
def bayesian_inference_tr(cor,day,prevqua,nac,dis,deb,tui,g,age,inter):
    #cero el modelo bayesiano
    model = BayesianNetwork([('dis','Target'),('age','Target'),('day','Target'),('cor','Target'),('prevqua','Target'),
                                          ('g','Target'),('tui','Target'),('nac','Target'),('inter','Target'),('deb','Target')])
    
    #estimamos las distribuciones de probabilidad utilizando MLE y BayesianEstimator
    model.fit(df,estimator=MaximumLikelihoodEstimator)
    model.fit(df, estimator=BayesianEstimator, prior_type='BDeu',equivalent_sample_size = 10 )
    #Hacemos inferencia en el modelo bayesiano
    infer = VariableElimination(model)

    q=infer.query(['Target'],evidence={'cor':cor,'day':day,'prevqua':prevqua,'nac':nac,'dis':dis,'deb':deb,'tui':tui,'g':g,'age':age,'inter':inter})
    return q
a = bayesian_inference_tr()
print(a)



#━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡#｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡
#━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡#｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app_p2 = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app_p2.server

app_p2.layout = html.Div(children=[
    html.Div(children=[
        html.H1("¿Qué probabilidad tengo de graduarme?", style={'text-align': 'left', 'font-family': 'Arial', 'color': 'gray', 'font-weight': 'bold'}),
        html.Div(children=[
            html.H3("Nuestro objetivo"),
            dcc.Textarea(
                placeholder="Escribe tu párrafo aquí...",
                style={'width': '100%', 'height': '50px', 'font-family': 'Arial'}
            )
        ], style={'margin': '20px'}),
    ], style={'flex': '1'}),
    html.Div(children=[
        html.Div([
            html.H1("49%", style={'font-size': '8em', 'text-align': 'left', 'margin-bottom': '0', 'font-family': 'Arial', 'font-weight': 'bold', 'margin-left': '15px'}),
            dcc.Textarea(
                placeholder="Cuadro de texto",
                style={'width': '200px', 'height': '50px', 'font-family': 'Arial', 'margin-top': '10px'}
            ),
            html.P("Texto descriptivo a la derecha del número", style={'text-align': 'left', 'font-family': 'Arial'}),
        ], style={'text-align': 'left'}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    html.Div([
        html.Div([
            html.H4("Dropdown 1"),
            dcc.Dropdown(
                options=[
                    {'label': 'Opción 1', 'value': 'opcion1'},
                    {'label': 'Opción 2', 'value': 'opcion2'}
                ],
                value='opcion1'
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H4("Dropdown 2"),
            dcc.Dropdown(
                options=[
                    {'label': 'Opción A', 'value': 'opcionA'},
                    {'label': 'Opción B', 'value': 'opcionB'}
                ],
                value='opcionA'
            )
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        html.Div([
            html.H4("Dropdown 3"),
            dcc.Dropdown(
                options=[
                    {'label': 'Opción X', 'value': 'opcionX'},
                    {'label': 'Opción Y', 'value': 'opcionY'}
                ],
                value='opcionX'
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H4("Dropdown 4"),
            dcc.Dropdown(
                options=[
                    {'label': 'Opción M', 'value': 'opcionM'},
                    {'label': 'Opción N', 'value': 'opcionN'}
                ],
                value='opcionM'
            )
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        html.Div([
            html.H4("Dropdown 5"),
            dcc.Dropdown(
                options=[
                    {'label': 'Opción A1', 'value': 'opcionA1'},
                    {'label': 'Opción B1', 'value': 'opcionB1'}
                ],
                value='opcionA1'
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H4("Dropdown 6"),
            dcc.Dropdown(
                options=[
                    {'label': 'Opción X1', 'value': 'opcionX1'},
                    {'label': 'Opción Y1', 'value': 'opcionY1'}
                ],
                value='opcionX1'
            )
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        html.Div([
            html.H4("Dropdown 7"),
            dcc.Dropdown(
                options=[
                    {'label': 'Opción M1', 'value': 'opcionM1'},
                    {'label': 'Opción N1', 'value': 'opcionN1'}
                ],
                value='opcionM1'
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H4("Dropdown 8"),
            dcc.Dropdown(
                options=[
                    {'label': 'Opción A2', 'value': 'opcionA2'},
                    {'label': 'Opción B2', 'value': 'opcionB2'}
                ],
                value='opcionA2'
            )
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        html.Div([
            html.H4("Dropdown 9"),
            dcc.Dropdown(
                options=[
                    {'label': 'Opción X2', 'value': 'opcionX2'},
                    {'label': 'Opción Y2', 'value': 'opcionY2'}
                ],
                value='opcionX2'
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H4("Dropdown 10"),
            dcc.Dropdown(
                options=[
                    {'label': 'Opción M2', 'value': 'opcionM2'},
                    {'label': 'Opción N2', 'value': 'opcionN2'}
                ],
                value='opcionM2'
            )
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        html.Div([
            html.H4("Dropdown 11"),
            dcc.Dropdown(
                options=[
                    {'label': 'Opción A3', 'value': 'opcionA3'},
                    {'label': 'Opción B3', 'value': 'opcionB3'}
                ],
                value='opcionA3'
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H4("Dropdown 12"),
            dcc.Dropdown(
                options=[
                    {'label': 'Opción X3', 'value': 'opcionX3'},
                    {'label': 'Opción Y3', 'value': 'opcionY3'}
                ],
                value='opcionX3'
            )
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
    ]),
])



if __name__ == '__main__':
    app_p2.run_server(debug=True, port=8040)
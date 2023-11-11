import dash
from dash import dcc  # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output, State
import psycopg2
# pip install python-dotenv
import os
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator,BayesianEstimator
from pgmpy.inference import VariableElimination
import numpy as np
import matplotlib.pyplot as plt

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
def bayesian_inference_1(cor,day,prevqua,nac,dis,deb,tui,g,age,inter):
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


datos2=df.drop(columns=['adgr','spn','sch'],)
#convertimos a numeros la variable de interes 'target'
mapping = {'Dropout': 0, 'Graduate': 1, 'Enrolled': 2}

# Use the map function to apply the mapping to the 'target' column
datos2['target'] = datos2['target'].map(mapping)

# If you want to change the column data type to integer
datos2['target'] = datos2['target'].astype(int)
datos2['cor'] = datos2['cor'].astype(int)
datos2['day'] = datos2['day'].astype(int)
datos2['prevqua'] = datos2['prevqua'].astype(int)
datos2['nac'] = datos2['nac'].astype(int)
datos2['dis'] = datos2['dis'].astype(int)
datos2['deb'] = datos2['deb'].astype(int)
datos2['tui'] = datos2['tui'].astype(int)
datos2['g'] = datos2['g'].astype(int)
datos2['age'] = datos2['age'].astype(int)
datos2['inter'] = datos2['inter'].astype(int)

#Number of bins
num_bins = 10

# Discretize the values into 10 equally spaced intervals and convert to integer values from 1 to 10
datos2['pass1'] = np.digitize(datos2['pass1'], np.linspace(0, 1, num_bins + 1))  # Convert 'pass1'
datos2['pass2'] = np.digitize(datos2['pass2'], np.linspace(0, 1, num_bins + 1))  # Convert 'pass2'

# Adjust values from 1-11 to 1-10
datos2['pass1'] = np.clip(datos2['pass1'], 1, 10)
datos2['pass2'] = np.clip(datos2['pass2'], 1, 10)


datos2 = datos2.sample(frac=0.2)

unique_values_cor = datos2['cor'].unique().tolist()
unique_values_day = datos2['day'].unique().tolist()
unique_values_prevqua = datos2['prevqua'].unique().tolist()
unique_values_nac = datos2['nac'].unique().tolist()
unique_values_dis = datos2['dis'].unique().tolist()
unique_values_deb = datos2['deb'].unique().tolist()
unique_values_tui = datos2['tui'].unique().tolist()
unique_values_g = datos2['g'].unique().tolist()
unique_values_age = datos2['age'].unique().tolist()
unique_values_inter = datos2['inter'].unique().tolist()
unique_values_pass1 = datos2['pass1'].unique().tolist()
unique_values_pass2 = datos2['pass2'].unique().tolist()
unique_values_target = datos2['target'].unique().tolist()


modelHill = BayesianNetwork([('dis', 'age'), ('age', 'prevqua'), ('age', 'cor'), ('age', 'pass1'), ('day', 'age'), ('day', 'cor'), ('pass2', 'cor'), ('prevqua', 'cor'), ('g', 'cor'), ('target', 'pass2'), ('target', 'pass1'), ('pass1', 'cor'), ('tui', 'target'), ('tui', 'deb'), ('inter', 'nac')])
    
modelHill.fit(datos2,estimator=MaximumLikelihoodEstimator)
modelHill.fit(datos2, estimator=BayesianEstimator, prior_type='BDeu',equivalent_sample_size = 100 )
    #print(modelHill)
    #print(modelHill.getModel())

#LA UNICA INFERENCIA QUE NOS INTERESA
def infierelo(cor,day,prevqua,nac,dis,deb,tui,g,age,inter,pass1,pass2):
    infer = VariableElimination(modelHill)

    q=infer.query(['target'],evidence={'cor':cor,'day':day,'prevqua':prevqua,'nac':nac,'dis':dis,'deb':deb,'tui':tui,'g':g,'age':age,'inter':inter,'pass1':pass1,'pass2':pass2})
    print(q)
    return q


#━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡#｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡
#━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡#｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app_p2 = dash.Dash(__name__, external_stylesheets=external_stylesheets,suppress_callback_exceptions=True)
server = app_p2.server

tab1_content = html.Div()

tab3_content = html.Div(children=[
    html.Div(children=[
        html.H1("Realiza una evaluacion", style={'text-align': 'left', 'font-family': 'Arial', 'color': 'gray', 'font-weight': 'bold'}),
        html.Div(children=[
            html.H3("Nuestro objetivo"),
            dcc.Textarea(
                placeholder="¡Bienvenido a nuestro Dash ! Tenemos como objetivo brindar una herrameinta de claridad a estudiantes universitarios apra que estos sepan que tan duro será graduarse de su carrera. Se utilizó el conjunto de datos de la universidad de Irvine para la base de datos “Predict students’ dropout and academic succes” del machine learning repository como base para realizar un modelo bayesiano que permita calcular a los estudiantes su probabilidad de graduarse a partir de variables bajo su control como su curso y si sus clases son diurnas o nocturnas, y variables fuera de su control cómo el género y su nacionalidad.",
                style={'width': '100%', 'height': '50px', 'font-family': 'Arial'}
            )
        ], style={'margin': '20px'}),
    ], style={'flex': '1'}),
    html.Div(children=[
        html.Div([
            html.H1("49%", style={'font-size': '8em', 'text-align': 'left', 'margin-bottom': '0', 'font-family': 'Arial', 'font-weight': 'bold', 'margin-left': '15px'}),
            dcc.Textarea(
                placeholder="Es el porcentaje de personas graduadas en su totalidad a partir de los datos",
                style={'width': '200px', 'height': '50px', 'font-family': 'Arial', 'margin-top': '10px'}
            ),
            html.P("", style={'text-align': 'left', 'font-family': 'Arial'}),
        ], style={'text-align': 'left'}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    html.Div([
        html.Div([
            html.H4("Codigo del curso"),
            dcc.Dropdown(id='cor',
                options=[{'label': value, 'value': value} for value in unique_values_cor],
                value='opcion1'
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H4("Asistencia diurna/nocturna"),
            dcc.Dropdown(id='day',
                options=[{'label': value, 'value': value} for value in unique_values_day],
                value='opcionA'
            )
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        html.Div([
            html.H4("Ya esta graduado del colegio?"),
            dcc.Dropdown(id='prevqua',
                options=[{'label': value, 'value': value} for value in unique_values_prevqua],
                value='opcionX'
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H4("Codigo de nacionalidad"),
            dcc.Dropdown(id='nac',
                options=[{'label': value, 'value': value} for value in unique_values_nac],
                value='opcionM'
            )
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        html.Div([
            html.H4("Es desplazado?"),
            dcc.Dropdown(id='dis',
                options=[{'label': value, 'value': value} for value in unique_values_dis],
                value='opcionA1'
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H4("Tiene deuda?"),
            dcc.Dropdown(id='deb',
                options=[{'label': value, 'value': value} for value in unique_values_deb],
                value='opcionX1'
            )
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        html.Div([
            html.H4("Está a paz y salvo?"),
            dcc.Dropdown(id='tui',
                options=[{'label': value, 'value': value} for value in unique_values_tui],
                value='opcionM1'
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H4("Género"),
            dcc.Dropdown(id='g',
                options=[{'label': value, 'value': value} for value in unique_values_g],
                value='opcionpA2'
            )
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        html.Div([
            html.H4("Edad"),
            dcc.Dropdown(id='age',
                options=[{'label': value, 'value': value} for value in unique_values_age],
                value='opcionX2'
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H4("Es estudiante internacional?"),
            dcc.Dropdown(id='inter',
                options=[{'label': value, 'value': value} for value in unique_values_inter],
                value='opcionM2'
            )
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        html.Div([
            html.H4("porcentaje de aprobacion de materias en el primer semestre(1-10)"),
            dcc.Dropdown(id='pass1',
                options=[{'label': value, 'value': value} for value in unique_values_pass1],
                value='opcionA3'
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H4("porcentaje de aprobacion de materias en el segundo semestre(1-10)"),
            dcc.Dropdown(id='pass2',
                options=[{'label': value, 'value': value} for value in unique_values_pass2],
                value='opcionX3'
            )
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
    ]),

    html.Br(),
    html.Div(id='output2'),
    html.Button('Calcular', id='submit2', n_clicks=0),
    html.Br(),
    html.Br()
])



tab2_content = html.Div(
    html.Div([
    html.H1("¿Si me podré graduar?: El condicionamiento sobre la graduación estudiantil",
            style={'text-align': 'center', 'color': '#FEE7DB', 'font-weight': 'bold', 'background-color': '#550a84'}),
    
    dcc.Store(id='clikos',data=0),

    html.Div([
        html.P('Este es un juego social, escoge las opciones categoricas presentadas a continuación y revisa que tan probable es que te puedas graduar.',
               style={'text-align': 'center', 'color': '#550a84', 'font-weight': 'bold', 'background-color': '#f2f2f2'}),
               html.Br(),
        html.P('Los datos utilizados para este modelo son de UCE Irvine Machine learning repositorie. Puedes jugar con los datos en nuestro creador de grafos', 
               style={'text-align': 'center', 'color': '#550a84', 'font-weight': 'bold', 'background-color': '#f2f2f2'})]),
    
    # Dropdown for selecting options
    html.Div([
            html.Div([
            html.Label('¿Eres estudiante internacional?'),
            dcc.Dropdown(id='inter', options=[{'label': 'Si', 'value': 1}, {'label': 'No', 'value': 0}], placeholder='Internacional'),
        ], className='six columns', style={'margin-top': '10px'}),
    ], className='row'),

    html.Div([
            html.Div([
            html.Label('¿En qué jornada estudias?'),
            dcc.Dropdown(id='dt', options=[{'label': 'En el día', 'value': 1}, {'label': 'En la noche', 'value': 0}], placeholder='Daytime/evening attendance'),
        ], className='six columns', style={'margin-top': '10px'}),
    ], className='row'),

    html.Div([
            html.Div([
            html.Label('¿Eres desplazado de la violencia?'),
            dcc.Dropdown(id='dis', options=[{'label': 'Si', 'value': 1}, {'label': 'No', 'value': 0}], placeholder='Displaced'),
        ], className='six columns', style={'margin-top': '10px'}),
    ], className='row'),

    html.Div([
            html.Div([
            html.Label('¿Tienes necesidades educativas que deban ser tomadas en cuenta?'),
            dcc.Dropdown(id='ed', options=[{'label': 'Si', 'value': 1}, {'label': 'No', 'value': 0}], placeholder='Educational special needs'),
        ], className='six columns', style={'margin-top': '10px'}),
    ], className='row'),

    html.Div([
            html.Div([
            html.Label('¿Te endeudaste para pagar la universidad?'),
            dcc.Dropdown(id='deb', options=[{'label': 'Si', 'value': 1}, {'label': 'No', 'value': 0}], placeholder='Debtor'),
        ], className='six columns', style={'margin-top': '10px'}),
    ], className='row'),
    
    html.Div([
            html.Div([
            html.Label('¿Estas al día con los pagos de la universidad?'),
            dcc.Dropdown(id='tui', options=[{'label': 'Si', 'value': 1}, {'label': 'No', 'value': 0}], placeholder='Tuition fees up to date'),
        ], className='six columns', style={'margin-top': '10px'}),
    ], className='row'),

    html.Div([
            html.Div([
            html.Label('¿Con qué genero te presentas más seguido?'),
            dcc.Dropdown(id='gen', options=[{'label': 'Masculino', 'value': 1}, {'label': 'Femenino', 'value': 0}], placeholder='Gender'),
        ], className='six columns', style={'margin-top': '10px'}),
    ], className='row'),

    html.Div([
            html.Div([
            html.Label('¿Eres beneficiario de beca?'),
            dcc.Dropdown(id='scho', options=[{'label': 'Si', 'value': 1}, {'label': 'No', 'value': 0}], placeholder='Scholarship holder'),
        ], className='six columns', style={'margin-top': '10px'}),
    ], className='row'),

    html.Div([
            html.Div([
            html.Label('¿A que nacionalidad perteneces?'),
            dcc.Dropdown(id='n', options=[
                {'label': 'Portugues', 'value': 1}, 
                {'label': 'Aleman', 'value': 2},
                {'label': 'Español', 'value': 6},
                {'label': 'Italiano', 'value': 11},
                {'label': 'Dutch', 'value': 13},
                {'label': 'Inglés', 'value': 14},
                {'label': 'Lituano', 'value': 17},
                {'label': 'Angoleño', 'value': 21},
                {'label': 'Cape verdeano', 'value': 22},
                {'label': 'Guineo', 'value': 24},
                {'label': 'Mozambiqueño', 'value': 25},
                {'label': 'Santotomense', 'value': 26},
                {'label': 'Turco', 'value': 32},
                {'label': 'Brasileño', 'value': 41},
                {'label': 'Rumano', 'value': 62},
                {'label': 'Moldoveño', 'value': 100},
                {'label': 'Mexicano', 'value': 101},
                {'label': 'Ucraniano', 'value': 103},
                {'label': 'Ruso', 'value': 105},
                {'label': 'Cubano', 'value': 108},
                {'label': 'Colombiano', 'value': 109}
                ], 
                placeholder='Nacionality'),
        ], className='six columns', style={'margin-top': '10px'}),
    ], className='row'),


    html.Div([
            html.Div([
            html.Label('¿En qué curso estas inscrito?'),
            dcc.Dropdown(id='c', options=[
                {'label': 'Biofuel Production Technologies', 'value': 33}, 
                {'label': 'Animation and Multimedia Design', 'value': 171},
                {'label': 'Social Service (evening attendance)', 'value': 8014},
                {'label': 'Agronomy', 'value': 9003},
                {'label': 'Communication Design', 'value': 9070},
                {'label': 'Veterinary Nursing', 'value': 9085},
                {'label': 'Informatics Engineering', 'value': 9119},
                {'label': 'Equinculture', 'value': 9130},
                {'label': 'Management', 'value': 9147},
                {'label': 'Social Service', 'value': 9238},
                {'label': 'Tourism', 'value': 254},
                {'label': 'Nursing', 'value': 9500},
                {'label': 'Oral Hygiene', 'value': 9556},
                {'label': 'Advertising and Marketing Management', 'value': 9670},
                {'label': 'Journalism and Communication', 'value': 9773},
                {'label': 'Basic Education', 'value': 9853},
                {'label': 'Management (evening attendance)', 'value': 9991},
                ], 
                placeholder='Course'),
        ], className='six columns', style={'margin-top': '10px'}),
    ], className='row'),

   
    # Placeholder for the graph
    #dcc.Graph(id='probability-graph'),

    html.Br(),
    html.Button('Calcular', id='submit', n_clicks=0),
    html.Br(),
    html.Br(),
    html.Div(id='output'),

], className='container', style={'font-family': 'Cambria', 'background-color': '#f2f2f2'})

)
app_p2.layout = html.Div([
    html.H1('¿Qué probabilidad tengo de graduarme?'),
    dcc.Tabs(id='tabs', value='tab1', children=[
        dcc.Tab(label='Datos relacionados', value='tab1'),
        dcc.Tab(label='Modelo basico', value='tab2'),
        dcc.Tab(label='Modelo mejorado', value='tab3'),
    ]),
    html.Div(id='tab-content')
])
#━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡#｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡
# CALLBACKS
#━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡#｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡


#✿ஜீ۞ஜீ✿•.¸¸.•*`*•.•ஜீ☼۞☼ஜீ•.•*`*•.¸¸.•✿ஜீ۞ஜீ✿✿ஜீ۞ஜீ✿•.¸¸.•*`*•.•ஜீ☼۞☼ஜீ•.•*`*•.¸¸.•✿ஜீ۞ஜீ✿✿ஜீ۞ஜீ✿•.¸¸.•*`*•.•ஜீ☼۞☼ஜீ•.•*`*•.¸¸.•✿ஜீ۞ஜீ✿

#✿ஜீ۞ஜீ✿•.¸¸.•*`*•.•ஜீ☼۞☼ஜீ•.•*`*•.¸¸.•✿ஜீ۞ஜீ✿✿ஜீ۞ஜீ✿•.¸¸.•*`*•.•ஜீ☼۞☼ஜீ•.•*`*•.¸¸.•✿ஜீ۞ஜீ✿✿ஜீ۞ஜீ✿•.¸¸.•*`*•.•ஜீ☼۞☼ஜீ•.•*`*•.¸¸.•✿ஜீ۞ஜீ✿


#━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡#｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡
#━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡#｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡

#realizar la inferencia
@app_p2.callback(Output('output2', 'children'),[Input('submit2', 'n_clicks')],[State('cor', 'value'),State('day', 'value'),State('prevqua', 'value'),State('nac', 'value'),State('dis', 'value'),State('deb', 'value'),State('tui', 'value'),State('g', 'value'),State('age', 'value'),State('inter', 'value'),State('pass1', 'value'),State('pass2', 'value')])
def callback_test_rapido(n_clicks,cor,day,prevqua,nac,dis,deb,tui,g,age,inter,pass1,pass2):
    if not n_clicks:
        return ''
    else:
        result = infierelo(cor,day,prevqua,nac,dis,deb,tui,g,age,inter,pass1,pass2)
        probability = round(result.values[0], 2)

        return html.Div([
            f'   La probabilidad de que logre graduarse es de {probability*100}%',
            html.Br(),])
        #creamos recomendaciones para cada signo vital





# Callback para mostrar el contenido de la pestaña seleccionada
@app_p2.callback(Output('tab-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab1':
        return tab1_content
    elif tab == 'tab2':
        return tab2_content
    elif tab == 'tab3':
        return tab3_content

if __name__ == '__main__':
    app_p2.run_server(debug=True, port=8040)
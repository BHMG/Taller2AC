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

modelHill = BayesianNetwork([('dis', 'age'), ('age', 'prevqua'), ('age', 'cor'), ('age', 'pass1'), ('day', 'age'), ('day', 'cor') , ('pass2', 'cor'), ('prevqua', 'cor'), ('g', 'cor'), ('target', 'pass2'), ('target', 'pass1'), ('pass1', 'cor'), ('tui', 'target'), ('tui', 'deb'), ('inter', 'nac')])
    
modelHill.fit(datos2,estimator=MaximumLikelihoodEstimator)
modelHill.fit(datos2, estimator=BayesianEstimator, prior_type='BDeu',equivalent_sample_size = 100 )
    #print(modelHill)
    #print(modelHill.getModel())

#━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡#｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡
# VISUALIZACIONES
#━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡#｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡


#✿ஜீ۞ஜீ✿•.¸¸.•*`*•.•ஜீ☼۞☼ஜீ•.•*`*•.¸¸.•✿ஜீ۞ஜீ✿✿ஜீ۞ஜீ✿•.¸¸.•*`*•.•ஜீ☼۞☼ஜீ•.•*`*•.¸¸.•✿ஜீ۞ஜீ✿✿ஜீ۞ஜீ✿•.¸¸.•*`*•.•ஜீ☼۞☼ஜீ•.•*`*•.¸¸.•✿ஜீ۞ஜீ✿
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

#✿ஜீ۞ஜீ✿•.¸¸.•*`*•.•ஜீ☼۞☼ஜீ•.•*`*•.¸¸.•✿ஜீ۞ஜீ✿✿ஜீ۞ஜீ✿•.¸¸.•*`*•.•ஜீ☼۞☼ஜீ•.•*`*•.¸¸.•✿ஜீ۞ஜீ✿✿ஜீ۞ஜீ✿•.¸¸.•*`*•.•ஜீ☼۞☼ஜீ•.•*`*•.¸¸.•✿ஜீ۞ஜீ✿


#━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡#｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡
#━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡#｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app_p2 = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app_p2.server

tab1_content = html.Div(
    children=[
        html.Div(children=[
            html.H1("Visualizaciones", style={'text-align': 'center', 'font-family': 'Arial', 'color': 'gray', 'font-weight': 'bold'}),
            html.Div(children=[
                dcc.Graph(id='Viz1', figure=figure1),
                dcc.Graph(id='Viz2', figure=figure2),
                html.Div(children=[
                    html.Div([
                        html.H1("49,93%", style={'font-size': '8em', 'text-align': 'left', 'margin-bottom': '0', 'font-family': 'Arial', 'font-weight': 'bold', 'margin-left': '15px'}),
                        dcc.Textarea(
                            placeholder="Es el porcentaje de personas graduadas en su totalidad a partir de los datos",
                            style={'width': '200px', 'height': '50px', 'font-family': 'Arial', 'margin-top': '10px'}
                        ),
                    ]),
                ]),
            ]),
        ]),
    ]
)

tab2_content = html.Div(children=[
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
            html.H4("COR"),
            dcc.Dropdown(
                options=[
                    {'label': 'Opción 1', 'value': 'opcion1'},
                    {'label': 'Opción 2', 'value': 'opcion2'}
                ],
                value='opcion1'
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H4("DAY"),
            dcc.Dropdown(
                options=[
                    {'label': 'Opción A', 'value': 'opcionA'},
                    {'label': 'Opción B', 'value': 'opcionB'}
                ],
                value='opcionA'
            )
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        html.Div([
            html.H4("PREVQUA"),
            dcc.Dropdown(
                options=[
                    {'label': 'Opción X', 'value': 'opcionX'},
                    {'label': 'Opción Y', 'value': 'opcionY'}
                ],
                value='opcionX'
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H4("NAC"),
            dcc.Dropdown(
                options=[
                    {'label': 'Opción M', 'value': 'opcionM'},
                    {'label': 'Opción N', 'value': 'opcionN'}
                ],
                value='opcionM'
            )
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        html.Div([
            html.H4("DIS"),
            dcc.Dropdown(
                options=[
                    {'label': 'Opción A1', 'value': 'opcionA1'},
                    {'label': 'Opción B1', 'value': 'opcionB1'}
                ],
                value='opcionA1'
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H4("DEB"),
            dcc.Dropdown(
                options=[
                    {'label': 'Opción X1', 'value': 'opcionX1'},
                    {'label': 'Opción Y1', 'value': 'opcionY1'}
                ],
                value='opcionX1'
            )
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        html.Div([
            html.H4("TUI"),
            dcc.Dropdown(
                options=[
                    {'label': 'Opción M1', 'value': 'opcionM1'},
                    {'label': 'Opción N1', 'value': 'opcionN1'}
                ],
                value='opcionM1'
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H4("G"),
            dcc.Dropdown(
                options=[
                    {'label': 'Opción A2', 'value': 'opcionA2'},
                    {'label': 'Opción B2', 'value': 'opcionB2'}
                ],
                value='opcionpA2'
            )
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        html.Div([
            html.H4("AGE"),
            dcc.Dropdown(
                options=[
                    {'label': 'Opción X2', 'value': 'opcionX2'},
                    {'label': 'Opción Y2', 'value': 'opcionY2'}
                ],
                value='opcionX2'
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H4("INTER"),
            dcc.Dropdown(
                options=[
                    {'label': 'Opción M2', 'value': 'opcionM2'},
                    {'label': 'Opción N2', 'value': 'opcionN2'}
                ],
                value='opcionM2'
            )
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        html.Div([
            html.H4("PASS1"),
            dcc.Dropdown(
                options=[
                    {'label': 'Opción A3', 'value': 'opcionA3'},
                    {'label': 'Opción B3', 'value': 'opcionB3'}
                ],
                value='opcionA3'
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.H4("PASS2"),
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

tab3_content = html.Div(
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
#Dashborad-P2
#⍣ ೋ⍣ ೋ｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡
#Librerias⍣ ೋ⍣ ೋ⍣ ೋ
#｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡

import pandas as pd
import dash
from dash import dcc  # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import pgmpy
from ucimlrepo import fetch_ucirepo


#｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡#｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡

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
#✿ஜீ۞ஜீ✿•.¸¸.•*`*•.•ஜீ☼۞☼ஜீ•.•*`*•.¸¸.•✿ஜீ۞ஜீ✿✿ஜீ۞ஜீ✿•.¸¸.•*`*•.•ஜீ☼۞☼ஜீ•.•*`*•.¸¸.•✿ஜீ۞ஜீ✿✿ஜீ۞ஜீ✿•.¸¸.•*`*•.•ஜீ☼۞☼ஜீ•.•*`*•.¸¸.•✿ஜீ۞ஜீ✿✿ஜீ۞ஜீ✿•.¸¸.•*`*•.•ஜீ☼۞☼ஜீ•.•*`*•.¸¸.•✿ஜீ۞ஜீ✿


#print(df[df['Target']=='Graduate'].count()) ##Num 2209
#print(df['Target'].count()) ## den 4424

gente_graduada= round((2209/4424)*100,2)
#print(gente_graduada)

#✿ஜீ۞ஜீ✿•.¸¸.•*`*•.•ஜீ☼۞☼ஜீ•.•*`*•.¸¸.•✿ஜீ۞ஜீ✿✿ஜீ۞ஜீ✿•.¸¸.•*`*•.•ஜீ☼۞☼ஜீ•.•*`*•.¸¸.•✿ஜீ۞ஜீ✿✿ஜீ۞ஜீ✿•.¸¸.•*`*•.•ஜீ☼۞☼ஜீ•.•*`*•.¸¸.•✿ஜீ۞ஜீ✿
# 'Course':
            # 33 - Biofuel Production Technologies 
            # 171 - Animation and Multimedia Design 
            # 8014 - Social Service (evening attendance) 
            # 9003 - Agronomy 
            # 9070 - Communication Design 
            # 9085 - Veterinary Nursing 
            # 9119 - Informatics Engineering 
            # 9130 - Equinculture 
            # 9147 - Management 
            # 9238 - Social Service 
            # 254 - Tourism 
            # 9500 - Nursing 
            # 9556 - Oral Hygiene 
            # 9670 - Advertising and Marketing Management 
            # 9773 - Journalism and Communication 
            # 9853 - Basic Education 
            # 9991 - Management (evening attendance)

#Nacionality:
            # 1 - Portuguese 
            # 2 - German
            # 6 - Spanish
            # 11 - Italian
            # 13 - Dutch
            # 14 - English
            # 17 - Lithuanian
            # 21 - Angolan
            # 22 - Cape Verdean
            # 24 - Guinean
            # 25 - Mozambican
            # 26 - Santomean
            # 32 - Turkish
            # 41 - Brazilian
            # 62 - Romanian
            # 100 - Moldova (Republic of)
            # 101 - Mexican
            # 103 - Ukrainian
            # 105 - Russian
            # 108 - Cuban
            # 109 - Colombian
#✿ஜீ۞ஜீ✿•.¸¸.•*`*•.•ஜீ☼۞☼ஜீ•.•*`*•.¸¸.•✿ஜீ۞ஜீ✿✿ஜீ۞ஜீ✿•.¸¸.•*`*•.•ஜீ☼۞☼ஜீ•.•*`*•.¸¸.•✿ஜீ۞ஜீ✿✿ஜீ۞ஜீ✿•.¸¸.•*`*•.•ஜீ☼۞☼ஜீ•.•*`*•.¸¸.•✿ஜீ۞ஜீ✿
#☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡ KEYS - Diccionarios  ☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡
cursos = {33:['Biofuel Production Technologies'],171:['Animation and Multimedia Design'],8014:['Social Service (evening attendance)'],
          9003: ['Agronomy'], 9070:['Communication Design'],9085:['Veterinary Nursing'],9119:['Informatics Engineering'],9130: ['Equinculture'],
          9147:['Management'], 9238:['Social Service'], 254:['Tourism'], 9500: ['Nursing'], 9556:['Oral Hygiene'],9670:['Advertising and Marketing Management'],
          9773:['Journalism and Communication'], 9853:['Basic Education'],9991:['Management (evening attendance)']}


cursos2 = {'Biofuel Production Technologies':[33],'Animation and Multimedia Design':[171],'Social Service (evening attendance)':[8014],
          'Agronomy':[9003],'Communication Design':[9070],'Veterinary Nursing':[9085],'Informatics Engineering':[9119],'Equinculture':[9130],
          'Management':[9147], 'Social Service':[9238], 'Tourism':[254], 'Nursing':[9500], 'Oral Hygiene':[9556],'Advertising and Marketing Management':[9670],
          'Journalism and Communication':[9773], 'Basic Education':[9853],'Management (evening attendance)':[9991]}

naciones ={'Portuguese':[1],'German':[2],'Spanish':[6], 'Italian':[11],'Ducth':[13],'English':[14],'Lithuanian':[17],'Angolan':[21],
           'Cape Verdean':[22],'Guinean':[24],'Mozambican':[25],'Santomean':[26],'Turkish':[32],'Brazilian':[41],'Romanian':[62],
           'Moldova (Republic of)':[100],'Mexican':[101],'Ukrainian':[103],'Russian':[105],'Cuban':[108],'Colombian':[109]}


#print(naciones.keys())
#boolean_label=['si','no']

#｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡
##｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼    EMPIEZA EL DASH   ｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★
#｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡｡☆✼★━━━━━━━━━━━━★✼☆｡

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
            html.H1("49,93%", style={'font-size': '8em', 'text-align': 'left', 'margin-bottom': '0', 'font-family': 'Arial', 'font-weight': 'bold', 'margin-left': '15px'}),
            dcc.Textarea(
                placeholder="Cuadro de texto",
                style={'width': '200px', 'height': '50px', 'font-family': 'Arial', 'margin-top': '10px'}
            ),
            html.P("Texto descriptivo a la derecha del número", style={'text-align': 'left', 'font-family': 'Arial'}),
        ], style={'text-align': 'left'}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    html.Div([
        html.Div([
            html.H4("Curso"),
            dcc.Dropdown(
                options=[
                    {'label'='Biofuel Production Technologies', 'value':33},
                    {'label'='Animation and Multimedia Design', 'value':171},{'label'='Social Service (evening attendance)', 'value' = 8014},
                    {'label'='Agronomy', 'value':9003},{'label' = 'Communication Design', 'value'=9070},{'label'='Veterinary Nursing', 'value'=9085},
                    {'label'='Informatics Engineering','value'=9119},{'label'='Equinculture', 'value'=9130}, {'label'='Management', 'value'=9147}, 
                    {'label'='Social Service','value'=9238}, {'label'='Tourism', 'value'=254}, {'label'='Nursing', 'value' =9500},
                    {'label'='Oral Hygiene', 'value'=9556},{'label'='Advertising and Marketing Management', 'value' =9670},
                    {'label'='Journalism and Communication', 'value'=9773}, {'label'='Basic Education', 'value'=9853},
                    {'label'='Management (evening attendance)', 'value'= 9991}
                ],
                value='Animation and Multimedia Design'
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
    app_p2.run_server(debug=True)

# -*- coding: utf-8 -*-

import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
from PIL import Image
import os

import requests
from io import BytesIO

from blueJay.utils import get_image_decode, predict_specie

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

index_img = get_image_decode("blueJay.jpeg")


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(children=[html.Div(children=[html.Img(src='data:image/png;base64,{}'.format(index_img), style={"width": "30%"})],
                                style={"display": "block",
                                       "textAlign": "center",
                                       "backgroundColor": "#b5d5d9"}),
                       html.Div(children=["Welcome to BlueJay"],
                                style={"display": "block",
                                       "textAlign": "center",
                                       "fontSize": "25pt",
                                       "lineHeight": "75px"}),
                       html.Div(children=["enter the url of the bird whose species you want to know"],
                                style={"display": "block",
                                       "textAlign": "center",
                                       "fontSize": "12pt",
                                       "lineHeight": "75px"}),
                       html.Div(children=[dcc.Input(id="input_image",
                                                    type="text",
                                                    placeholder="Insert image url here",
                                                    style={'height': '25px',
                                                           'width': "50%"})],
                                style={"display": "block",
                                       "textAlign": "center",
                                       "fontSize": "25pt"}),
                       html.Div(children=[html.Button('Submit', id='submit_image',
                                                                n_clicks=0)],
                                style={"display": "block",
                                       "textAlign": "center",
                                       "fontSize": "25pt"}),
                       html.Div(children=[html.Div(id='render_image',
                                                   style={"width": "50%",
                                                          "display": "block",
                                                          "float": "left"}),
                                          html.Div(id='render_prediction',
                                                   style={"width": "40%",
                                                          "display": "block",
                                                          "float": "right",
                                                          "fontFamily": "Courier",
                                                          "padding": "10px"})],
                                style={"display": "block",
                                       "textAlign": "center",
                                       "fontSize": "25pt",
                                       "width": "50%",
                                       "margin": "auto",
                                       "marginTop": "50px"}),
                       html.Div(children=[html.P("This application can predict the species of 315 different birds."),
                                          html.P("The model is made using transfert learning from VGG16 CNN model, and the images come from Kaggle dataset.")],
                                style={"display": "inline-block",
                                       "width": "100%",
                                       "textAlign": "center",
                                       "fontSize": "10pt",
                                       "backgroundColor": "white",
                                       "marginTop": "50px"}),
                       html.Div(children=[html.Span("The Kaggle dataset is here: "),
                                          dcc.Link('315 Bird Species - Classification', href='https://www.kaggle.com/gpiosenka/100-bird-species', target="_blank"),
                                          html.Br(),
                                          html.Span("The notebook explaining the model training is here: "),
                                          dcc.Link('Notebook', href='https://www.kaggle.com/victorbnnt/95-accuracy-on-315-birds-species', target="_blank"),
                                          html.Br(),
                                          html.Br(),
                                          html.Span("victor.bonnet.mg@gmail.com")],
                                style={"display": "inline-block",
                                       "width": "100%",
                                       "textAlign": "center",
                                       "fontSize": "10pt",
                                       "backgroundColor": "white",
                                       "marginTop": "50px",
                                       "paddingTop": "10px",
                                       "paddingBottom": "10px"})
                       ],
             id='page-content')], style={"backgroundColor": "#b5d5d9", "height": "100%"})

@app.callback(
    Output('render_image', 'children'),
    Input('submit_image', 'n_clicks'),
    State('input_image', 'value')
)
def update_output(n_clicks, value):
    try:
        response = requests.get(value)
        img = Image.open(BytesIO(response.content))
        return html.Img(src=img.resize((224, 224)), style={"width": "100%", "borderRadius": "5px", "border": "1px solid"})
    except:
        pass

@app.callback(
    Output('render_prediction', 'children'),
    Input('submit_image', 'n_clicks'),
    State('input_image', 'value')
)
def update_output(n_clicks, value):
    try:
        response = requests.get(value)
        img = Image.open(BytesIO(response.content))
        pred = predict_specie(img)
        return [html.Div(pred[0], style={"marginBottom": "20px"}), html.Div(pred[1] + "\n" + pred[2], style={"fontSize": "25px"})]
    except:
        pass



if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

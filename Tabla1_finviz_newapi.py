#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import requests #Para conectarnos a la API
from urllib.request import urlopen
import pandas as pd # Manipular datos
import json
from datetime import datetime 
import os
# Especifica la ruta del nuevo directorio de trabajo
new_directory = r'C:\Users\plata\Desktop\UMH\TFM\Stock sentiment\Tablas nuevas\datos_diarios'
# Cambia al nuevo directorio de trabajo
os.chdir(new_directory)
import nltk
nltk.download('vader_lexicon')  # Descargar el lexicon necesario para el análisis de sentimientos
nltk.download('stopwords')
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
# In[] Importamos librerias para aplicar BERT
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
from tqdm.auto import tqdm

MODEL = f"mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
#---------------------------
#Web scraping
from bs4 import BeautifulSoup
# Data de Yahoo finance
import yfinance as yf


###
# In[ ]:
def vader_sentiment(input):
    score = sia.polarity_scores(input)['compound']
    score = 5 - (score + 1) * 2
    return score


#funcion que aplica nuestro algoritmo NLP de BERT ###################
def roberta_function(value):
    encoded_text = tokenizer(value, return_tensors = 'pt')
    output = model(**encoded_text)
    score = output[0][0].detach().numpy()
    score = softmax(score)
    return score

#funcion para escalar la salida de roberta function ##################
def scale_value(value, index):
    if index == 0:
        min_desired = 3.5
        max_desired = 5

    if index == 1:
        min_desired = 3.5
        max_desired = 2.5

    if index == 2:
        min_desired = 2.5
        max_desired = 1
    
    scaled_value = (value * (max_desired - min_desired)) + min_desired
    return scaled_value

def map_to_scale(score):
    # Obtenemos el índice de la clase con la probabilidad más alta ##########
    predicted_class = int(np.argmax(score))

    # Mapeamos el índice de la clase a la escala requerida
    if predicted_class == 0:  # Clase positiva
        out = scale_value(score[0], predicted_class)
        
        return out
    elif predicted_class == 1:  # Clase neutra
        out = scale_value(score[1], predicted_class)
        
        return out
    else:  # Clase negativa
        out = scale_value(score[2], predicted_class)
        
        return out

# In[] -------------------------------------------------------------------
def ticker_news(tickers):
    # Configurar la URL base y la clave de la API
    base_url = 'https://newsapi.org/v2/everything'
    api_key = 'd27180091aac42d983e83ad3595a3892'  # clave de Api
    data = {}
    title=[]
    date =[] 
    stock=[]
    vader=[]
    bert =[]
    for ticker in tickers:    
        # Parámetros de la solicitud
        parameters = {
            'q': ticker + " stock",
            'apiKey': api_key
        }

        # Realizar la solicitud GET a la API
        response = requests.get(base_url, params=parameters)

        # Verificar si la solicitud fue exitosa (código de estado 200)
        if response.status_code == 200:
            json_data = response.json()
            articles = json_data["articles"]
            
            for article in articles:
                stock.append(ticker)
                title.append(article['title'])
                fecha = article['publishedAt']
                fecha = datetime.strptime(fecha, '%Y-%m-%dT%H:%M:%SZ')
                date.append(fecha.strftime('%Y-%m-%d'))
                vader.append(vader_sentiment(article['title'])) 
                bert.append(map_to_scale(roberta_function(article['title'])))            
                # Agregar los datos al diccionario
                          
        else:
            print("Error al hacer la solicitud para", ticker, ":", response.status_code)

    # Crear un DataFrame de pandas a partir de los datos
    
    data={
        'Ticker': stock,
        'Date': date,
        'Title': title,
        'News_Api_sentiment_vader': vader,
        'News_Api_sentiment_BERT' : bert
        }
    
    df = pd.DataFrame(data)

    # Obtener la fecha actual

    df = df.groupby(['Ticker','Date'], as_index=False)[['News_Api_sentiment_vader','News_Api_sentiment_BERT']].mean()

    
    return df


# In[]  FINVIZ

def ticker_news_finviz(tickers):
    # Contiene los encabezados de cada ticker
    news_tables = {}  
    for ticker in tickers:
        url = f'https://finviz.com/quote.ashx?t={ticker}'
        req = requests.get(url=url, headers={'user-agent': 'news'})
        response = requests.get(url=url, headers={'user-agent': 'news'})
        html = BeautifulSoup(response.text, 'html.parser')
        news_table = html.find(id='news-table')
        news_tables[ticker] = news_table
               
    parsed = []
    # Obtiene la fecha actual para la fecha anterior por defecto
    fecha_anterior = datetime.now().strftime('%Y-%m-%d')
    for ticker, news_table in news_tables.items():
        for row in news_table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) == 2:
                time = cells[0].text.strip()
                title = cells[1].text.strip()
                vader = vader_sentiment(title)
                bert = map_to_scale(roberta_function(title))
                expert_recommendation = yf.Ticker(ticker).info['recommendationMean']
                # Si el artículo fue publicado hoy, establece la fecha como la fecha actual
                if "Today" in time:
                    time = datetime.now().strftime('%Y-%m-%d')
                # Si la longitud de la fecha es mayor que 7, analiza y formatea la fecha correctamente
                elif len(time) > 7:
                    fecha_anterior = datetime.strptime(time[:9], '%b-%d-%y').strftime('%Y-%m-%d')
                    time = fecha_anterior
                # Si no, usa la fecha anterior
                else:
                    time = fecha_anterior
                parsed.append([ticker, time, title, vader, bert, expert_recommendation])

    # Crea un DataFrame con los datos analizados
    df = pd.DataFrame(parsed, columns=['Ticker', 'Date', 'Title', 'Finviz_sentiment_vader', 'Finviz_sentiment_BERT','yahoo_expert_recommendation'])
    # Agrupa los datos por fecha y calcula el promedio de los sentimientos de Finviz
    df = df.groupby(['Ticker','Date'], as_index=False)[['Finviz_sentiment_vader', 'Finviz_sentiment_BERT','yahoo_expert_recommendation']].mean()

    df2 = ticker_news(tickers)
    merged_df = pd.merge(df2, df,  on=['Date', 'Ticker'], how='outer')
    merged_df = merged_df.sort_values(by='Date')
    

    



    return merged_df



# %% finviz
def exportar_datos(acciones):
    #Fecha de hoy
    today_date = datetime.today().strftime('%Y-%m-%d')

    data = ticker_news_finviz(acciones)
    datos = yf.download(acciones, start=data['Date'].iloc[0], end=data['Date'].iloc[-1])
    datos.reset_index(inplace=True)
    datos['Date'] = datos['Date'].dt.date
    data['Date'] = pd.to_datetime(data['Date'])
    datos['Date'] = pd.to_datetime(datos['Date'])

    #recomendacion de los expertos



    result = pd.concat([data.set_index('Date'),datos.set_index('Date')], axis=1)
    
    carpeta_accion = acciones[0]
    os.makedirs(carpeta_accion, exist_ok=True) 
    
    # Si la carpeta ya existe, no se crea de nuevo
    
    # Guardar el archivo Excel dentro de la carpeta
    
    nombre_archivo = f'datos_{acciones[0]}_{today_date}.xlsx'
    ruta_archivo = os.path.join(carpeta_accion, nombre_archivo)
    result.to_excel(ruta_archivo)
    return result


# %%
stocks =  [
    "MSFT", "AAPL", "NVDA", "AMZN", "GOOG", "META", "BRK-B", "LLY", "AVGO", "V", "TSLA",
    "JPM", "WMT", "MA", "UNH", "XOM", "JNJ", "HD", "PG", "COST", "AMD", "ABBV", "MRK",
    "ORCL", "CRM", "BAC", "CVX", "NFLX", "KO", "ADBE", "TMO", "PEP", "MCD", "ABT", "WFC",
    "DIS", "CSCO", "TMUS", "INTC", "QCOM", "DHR", "INTU", "IBM", "AMAT", "GE", "VZ", "CAT",
    "CMCSA", "UBER", "AXP", "TXN", "PFE", "UNP", "NOW", "AMGN", "NKE", "PM", "MS", "LOW",
    "ISRG", "SYK", "SPGI", "COP", "HON", "UPS", "LRCX", "GS", "SCHW", "PLD", "BLK", "T",
    "BA", "RTX", "ELV", "BKNG", "PGR", "NEE", "TJX", "C", "BMY", "REGN", "VRTX", "MU", "LMT",
    "ABNB", "SBUX", "DE", "ADP", "MMC", "CI", "BSX", "KLAC", "MDLZ", "AMT", "ADI", "CVS",
    "GILD", "PANW", "BX", "ANET"
]
for stock in stocks:
    exportar_datos([stock])


# %%

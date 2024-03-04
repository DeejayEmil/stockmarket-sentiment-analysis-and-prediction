#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import requests
import pandas as pd
import json
from datetime import datetime 
import os
###
# In[ ]:
def ticker_news(tickers):
    # Configurar la URL base y la clave de la API
    base_url = 'https://newsapi.org/v2/everything'
    api_key = 'd27180091aac42d983e83ad3595a3892'  # Asegúrate de reemplazar 'Tu clave de API' con tu clave real
    data = {}
    title=[]
    date =[] 
    stock=[]
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
                date.append(fecha)             
                # Agregar los datos al diccionario
                

            
        else:
            print("Error al hacer la solicitud para", ticker, ":", response.status_code)

    # Crear un DataFrame de pandas a partir de los datos
    
    data={
        'Ticker': stock,
        'Title': title,
        'Date': date
        }
    
    df = pd.DataFrame(data)

    # Obtener la fecha actual
    today_date = datetime.today().strftime('%Y-%m-%d')

    # Definir la ruta del archivo Excel
    directory = "data_stocks\\data_{:02d}".format(datetime.today().month)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, 'data_' + today_date + '.xlsx')

    # Escribir el archivo Excel
    df.to_excel(file_path)

    return df


# In[ ]:

acciones = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA','JPM','GS','V','MA','BAC','AMZN','WMT','KO','PG','MCD','JNJ','PFE','MRK','ABT','BMY']

datos = ticker_news(acciones)


# %%

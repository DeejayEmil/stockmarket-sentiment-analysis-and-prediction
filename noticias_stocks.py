#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests

def ticker_news(tickers):
    # Configurar la URL base y la clave de la API
    base_url = 'https://newsapi.org/v2/everything'
    api_key = 'Td27180091aac42d983e83ad3595a3892'
    data = {}
    for ticker in tickers:    
        # Parámetros de la solicitud
        parameters = {
            'q': ticker,  # Palabra clave para buscar noticias relacionadas con Amazon
            'apiKey': 'd27180091aac42d983e83ad3595a3892'  # Tu clave de API
        }

        # Realizar la solicitud GET a la API
        response = requests.get(base_url, params=parameters)

        # Verificar si la solicitud fue exitosa (código de estado 200)
        if response.status_code == 200:
            # Convertir la respuesta a formato JSON
            json_data = response.json()

            # Agregar los resultados al diccionario data usando el ticker como clave
            data[ticker] = json_data
        else:
            print("Error al hacer la solicitud:", response.status_code)


    return data


# In[20]:

acciones = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

datos = ticker_news(acciones)
print(datos['GOOGL'])


# In[ ]:





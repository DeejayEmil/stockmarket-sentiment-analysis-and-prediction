'''*****************************************************************************
Purpose: To analyze the news headline of a specific stock.
This program uses Vader SentimentIntensityAnalyzer to calculate the news headline
compound value of a stock for a given day. 
You can analyze multiple stocks at the same time. Ex: 'AAPL, MSFT, F, TSLA' separate
each input by a comma.
You can also analyze all news or a specific date of news.
You can also ignore source: Ex: ignore_source = ['Motley Fool', 'TheStreet.com'] 
Limitations:
This program only analyzes headlines and only for the dates that have available news
on finviz.
-------------------------------------------------------------------
****************************************************************************'''
# In[ ]:
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
from datetime import datetime 
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA','JPM','GS','V','MA','BAC','AMZN','WMT','KO','PG','MCD','JNJ','PFE','MRK','ABT','BMY']

# In[ ]:  Getting Finviz Data
news_tables = {}        # contains each ticker headlines
for ticker in tickers:
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    req = Request(url=url, headers={'user-agent': 'news'})
    response = urlopen(req)     # taking out html response
            
    html = BeautifulSoup(response, features = 'html.parser')
    news_table = html.find(id = 'news-table') # gets the html object of entire table
    news_tables[ticker] = news_table

#ignore_source = ['Motley Fool', 'TheStreet.com'] # sources to exclude

print(news_table)

# In[]

# getting date
'''date_allowed = []
start = input("Enter the date/press enter for today's news (Ex: Dec-27-20) or 'All' for all the available news: ")
if len(start) == 0:
    start = datetime.today().strftime("%b-%d-%y")   
    date_allowed.append(start) '''
        
    
# Parsing and Manipulating
parsed = []    
for ticker, news_table in news_tables.items():
    for row in news_table.find_all('tr'):  # Busca todas las filas en la tabla
        cells = row.find_all('td')  # Busca todas las celdas en la fila
        if len(cells) == 2:  # Asegúrate de que haya dos celdas en la fila
            time = cells[0].text.strip()  # Extrae la fecha y la hora y elimina los espacios en blanco
            title = cells[1].text.strip()  # Extrae el título de la noticia y elimina los espacios en blanco
            source = cells[1].find('span').text.strip()  # Extrae la fuente de la noticia y elimina los espacios en blanco

            # Agrega los datos a la lista de análisis
            parsed.append([ticker, time, title, source])

# In[]
fecha_anterior = datetime.now().strftime("%b-%d-%y ")
for parse in parsed:
    if "Today" in parse[1]:
        parse[1]= datetime.now().strftime("%b-%d-%y %I:%M%p")
        
    else:
        if len(parse[1]) > 7:
            fecha_anterior = parse[1][:9] + " "
        else:
            parse[1] = fecha_anterior + parse[1]

# In[]
# Applying Sentiment Analysis
df = pd.DataFrame(parsed, columns=['Ticker', 'date', 'Title', 'Source'])
vader = SentimentIntensityAnalyzer()

today_date = datetime.today().strftime('%Y-%m-%d')
directory = "data_stocks_finviz\\data_{:02d}".format(datetime.today().month)
if not os.path.exists(directory):
    os.makedirs(directory)
    
file_path = os.path.join(directory, 'data_finviz_' + today_date + '.xlsx')

    # Escribir el archivo Excel
df.to_excel(file_path)

print(df)


# In[]
# Visualization of Sentiment Analysis
df['date'] = pd.to_datetime(df.date).dt.date # takes date comlumn convert it to date/time format

plt.figure(figsize=(6,6))      # figure size
# unstack() allows us to have dates as x-axis
mean_df = df.groupby(['date', 'Ticker']).mean() # avg compund score for each date
mean_df = mean_df.unstack() 

# xs (cross section of compund) get rids of compund label
mean_df = mean_df.xs('compound', axis="columns")
mean_df.plot(kind='bar')
plt.show()
# %%

# In[]
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import praw
import matplotlib.pyplot as plt
import math
import datetime as dt
import pandas as pd
import numpy as np
import os
# Especifica la ruta del nuevo directorio de trabajo
new_directory = r'C:\Users\plata\Desktop\UMH\TFM\Stock sentiment\Tablas nuevas\datos_reddit'
# Cambia al nuevo directorio de trabajo
os.chdir(new_directory)
# 
nltk.download('vader_lexicon')
nltk.download('stopwords')
# Importamos librerias para aplicar BERT
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
from tqdm.auto import tqdm
import yfinance as yf

MODEL = f"mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
tokenizer.model_max_length
# In[]
reddit = praw.Reddit(
    client_id = "Kl-YW8cHgy1DD1EeP_Jk8w",
    client_secret="FYvlM5w79IAYAyYtCFzDgi3fI06cEg",
    user_agent = "emil's stock sentiment analysis by Various_Self_6812"
)


sub_reddits = reddit.subreddit('wallstreetbets')
#stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA','JPM','GS','V','MA','BAC','AMZN','WMT','KO','PG','MCD','JNJ','PFE','MRK','ABT','BMY']
stocks = ['AAPL']

# In[5]: Funciones del analisis de sentimientos

# Mapeo para VADER Lexicon
def map_sentiment_score(score):
    # Mapear el rango (-1, 1) a (5, 1)
    return 5 - (score + 1) * 2

#funcion que aplica nuestro algoritmo NLP de BERT ###################
def roberta_function(value):
    encoded_text = tokenizer(value, return_tensors = 'pt',padding=True, truncation=True,max_length=512, add_special_tokens = True)
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
    
# %%
def commentSentimentbert(ticker, urlT):
    subComments = []
    bodyComment = []

    try:
        check = reddit.submission(url=urlT)
        subComments = check.comments
    except:
        return 0
    
    for comment in subComments:
        try: 
            bodyComment.append(comment.body)
        except:
            return 0
    results = []
    for line in bodyComment: 
        scores = map_to_scale(roberta_function(line))
        
        results.append(scores)

    if len(results) > 0:
        averageScore = sum(results) / len(results)
    else:
        averageScore = 0    
    return averageScore

# %%
def commentSentiment(ticker, urlT):
    subComments = []
    bodyComment = []
    try:
        check = reddit.submission(url=urlT)
        subComments = check.comments
    except:
        return 0
    
    for comment in subComments:
        try: 
            bodyComment.append(comment.body)
        except:
            return 0
    
    sia = SIA()
    results = []
    for line in bodyComment:
        scores = sia.polarity_scores(line)
        scores['headline'] = line
        results.append(scores)


    
    df =pd.DataFrame.from_records(results)
    df.head()
    df['label'] = 0
    
    try:
        df.loc[df['compound'] > 0.1, 'label'] = 1
        df.loc[df['compound'] < -0.1, 'label'] = -1
    except:
        return 0
    
    averageScore = 0
    position = 0
    while position < len(df.label)-1:
        averageScore = averageScore + df.label[position]
        position += 1
    averageScore = averageScore/len(df.label) 
    
    return(averageScore)

# %%



# In[6]:


def latestComment(ticker, urlT):
    subComments = []
    updateDates = []
    try:
        check = reddit.submission(url=urlT)
        subComments = check.comments
    except:
        return 0
    
    for comment in subComments:
        try: 
            updateDates.append(comment.created_utc)
        except:
            return 0
    
    updateDates.sort()
    return(updateDates[-1])

# In[7]:
def get_date(date):
    return dt.datetime.fromtimestamp(date).strftime('%Y-%m-%d')


# In[8]:


submission_statistics = []
d = {}
start_date = dt.datetime(2024, 1, 1).timestamp()
end_date = dt.datetime(2024, 3, 1).timestamp()

subreddits = ['wallstreetbets', 'personalfinance', 'MemeEconomy','investing', 'Daytrading','stocks', 'Trading', 'StockMarket', 'binance', 'investors', 'etoro','iama','finance', 'stocks', 'FinanceNews']  # Lista de subreddits a buscar
#subreddits = ['wallstreetbets']

for ticker in stocks:
    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        for submission in subreddit.search(ticker, limit=200):
            if submission.domain != "self." + subreddit_name:
                continue
            if not (start_date <= submission.created_utc <= end_date):
                continue  # Salta si la fecha de creación del post no está en el rango
            d = {}
            d['ticker'] = ticker
            if commentSentiment(ticker, submission.url) == 0.000000:
                continue
            d['Reddit_API_sentiment_vader'] = map_sentiment_score(commentSentiment(ticker, submission.url))
            d['Reddit_API_sentiment_bert'] = commentSentimentbert(ticker, submission.url)
            d['date'] = submission.created_utc

            submission_statistics.append(d)
    
dfSentimentStocks = pd.DataFrame(submission_statistics)

_timestampcreated = dfSentimentStocks["date"].apply(get_date)
dfSentimentStocks = dfSentimentStocks.assign(timestamp = _timestampcreated)
dfSentimentStocks = dfSentimentStocks.rename(columns={'timestamp': 'Date'})
dfSentimentStocks = dfSentimentStocks.sort_values(by='Date')
dfSentimentStocks = dfSentimentStocks.groupby(['ticker','Date',], as_index=False)[['Reddit_API_sentiment_vader','Reddit_API_sentiment_bert']].mean()



dfSentimentStocks
# In[]


def exportar_datos(dfSentimentStocks):
    #Fecha de hoy
    today_date = dt.datetime.today().strftime('%Y-%m-%d')
    datos = yf.download(dfSentimentStocks['ticker'].iloc[0], start=dfSentimentStocks['Date'].iloc[0], end=dfSentimentStocks['Date'].iloc[-1])
    datos.reset_index(inplace=True)
    datos['Date'] = datos['Date'].dt.strftime('%Y-%m-%d')
    dfSentimentStocks['Date'] = pd.to_datetime(dfSentimentStocks['Date'])
    datos['Date'] = pd.to_datetime(datos['Date'])
    result = pd.concat([dfSentimentStocks.set_index('Date'),datos.set_index('Date')], axis=1)

    return result


resultados = exportar_datos(dfSentimentStocks)
resultados

# %%

resultados.to_excel('Reddit_Sentiment_Equity.xlsx') 
# %%

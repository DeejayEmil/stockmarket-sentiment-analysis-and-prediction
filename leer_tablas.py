# In[]
import pandas as pd
import os
import UMHTrading as tr
import todos_indicadores as ti

new_directory = r'C:\\Users\\plata\\Desktop\\UMH\\TFM\\Stock sentiment\\Tablas nuevas\\datos_diarios'
# Lista para almacenar los DataFrames de cada archivo
os.chdir(new_directory)
dfs = []

# Tickers
stocks =  [
    "GOOG"
    ]

os.getcwd()

# In[]

# Iterar sobre cada stock
for stock in stocks:
    carpeta = stock
    if os.path.exists(carpeta):  # Verificar si la carpeta del stock existe
        for archivo in os.listdir(carpeta):
            if archivo.endswith('.xlsx') or archivo.endswith('.xls'):  # Verificar que el archivo sea un Excel
                ruta_archivo = os.path.join(carpeta, archivo)
                # Leer el archivo Excel y agregarlo al DataFrame
                df = pd.read_excel(ruta_archivo)
                dfs.append(df)
    else:
        print(f"La carpeta para {stock} no existe.")

# Unir todos los DataFrames en uno solo
df_final = pd.concat(dfs, ignore_index=True)

df_final  = df_final.sort_values(by='Date')
df_final.reset_index(inplace=True)
#Forward fill
df_final = df_final.fillna(method='ffill')
#Backward fill a los primeros valores vacios
df_final = df_final.fillna(method='bfill')
# Mostrar el DataFrame final, agrupamos y promediamos por fechas
df_final = df_final.groupby(['Ticker','Date'], as_index=False)[['News_Api_sentiment_vader','News_Api_sentiment_BERT','Finviz_sentiment_vader', 'Finviz_sentiment_BERT','yahoo_expert_recommendation','Open','High','Low','Close','Adj Close','Volume']].mean()
df_final 

# %% Gap 
df_final['Gaps'] = df_final ['Open'] - df_final ['Close'].shift(1)
# Rendimiento
df_final ['Rendimiento_diario'] = df_final['Close'].pct_change() * 100
df_final



# %% Media Movil
df_final = tr.calcular_MA(df_final)
# Bandas de Bollinger
df_final = tr.calcular_BB(df_final, periodo = 20)
# MACD
df_final = tr.calcular_MACD(df_final)
# Trix 
df_final = ti.calcular_trix(df_final)
# Estocastico
df_final = tr.calcular_estocastico(df_final)
df_final 

ti.calcular_trix(df_final)


# %%

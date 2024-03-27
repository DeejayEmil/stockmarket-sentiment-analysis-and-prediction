import UMHTrading as tr
import yfinance as yf
import pandas as pd
import math
import numpy as np

import warnings
warnings.filterwarnings("ignore")

PERCENTIL = 75



import matplotlib.pyplot as plt

def calcular_estocastico(df, ventana_K=14, ventana_D=3):
    # Calcular el mínimo y máximo de la ventana K
    df['Min_K'] = df['Low'].rolling(window=ventana_K).min()
    df['Max_K'] = df['High'].rolling(window=ventana_K).max()
    
    # Calcular %K
    df['K'] = ((df['Close'] - df['Min_K']) / (df['Max_K'] - df['Min_K'])) * 100
    
    # Calcular %D (promedio móvil de %K)
    df['D'] = df['K'].rolling(window=ventana_D).mean()
    
    return df

def calcular_trix(df, ventana=15):
    # Calcular el EMA de cierre
    df['EMA'] = df['Close'].ewm(span=ventana, min_periods=ventana).mean()
    
    # Calcular el primer EMA suavizado
    df['EMA1'] = df['EMA'].ewm(span=ventana, min_periods=ventana).mean()
    
    # Calcular el segundo EMA suavizado
    df['EMA2'] = df['EMA1'].ewm(span=ventana, min_periods=ventana).mean()
    
    # Calcular TRIX
    df['TRIX'] = (df['EMA2'] - df['EMA2'].shift(1)) / df['EMA2'].shift(1) * 100
    
    #df = df.drop('EMA', axis=1)
    #df = df.drop('EMA1', axis=1)
    #df = df.drop('EMA2', axis=1)

    return df


def generar_senales_BB(df):
    """
    Genera señales de compra y venta en función de los cruces de la cotización con las bandas de Bollinger.
    
    Parámetros:
        - df: DataFrame que contiene los precios y las bandas de Bollinger.
        
    Retorna:
        El DataFrame original con una columna adicional llamada 'Señal', que contiene las señales de compra y venta.
    """
    df['Señal'] = 0
    
    # Se invierten los criterios de señal y se aplica la tolerancia directamente en las condiciones
    #df.loc[ (df['Close'] > df['Upper_Band']) & (df['gaps'] > tolerancia_gaps), 'Señal'] = 1  # Cruce con banda superior
    #df.loc[ (df['Close'] < df['Lower_Band']) & (df['gaps'] < -tolerancia_gaps), 'Señal'] = -1 # Cruce con la banda inferior

    df.loc[ (df['Close'] >= df['Upper_Band']) , 'Señal'] = -2  # Cruce con banda superior
    df.loc[ (df['Close'] <= df['Lower_Band']) , 'Señal'] = 2 # Cruce con la banda inferior

    # Ponemos a cero valores consecutivos repetidos y localizamos cruce de banda    
    anteriorS =  df['Señal'].iloc[0]
    df['Señal'].iloc[0] = 0 
    df['Señal'].iloc[1] = 0 

    estado = 0
    for i in range(2, len(df['Señal'])):
        if (df['Close'][i-1]- df['MA'][i-1]>=0 and df['Close'][i]- df['MA'][i]<0):
            df['Señal'].iloc[i] = -1
        elif (df['Close'][i-1]- df['MA'][i-1]<=0 and df['Close'][i]- df['MA'][i]>0):
            df['Señal'].iloc[i] = 1

        if np.percentile(df['Volume'][0:i-1], PERCENTIL)>df['Volume'][i] and abs(df['Señal'].iloc[i])==1:
           df['Señal'].iloc[i] = 0  # Si no hay volumen significativo, no la tenemos en cuenta

        if df['Señal'].iloc[i] == anteriorS:
            df['Señal'].iloc[i] = 0
        
        if df['Señal'].iloc[i] == 1:
            estado = 1
        elif df['Señal'].iloc[i] == -1:
            estado = -1

        '''
        if (df['Close'].iloc[i] < df['Upper_Band'].iloc[i]) and estado == 1:
            df['Señal'].iloc[i]==-2
            estado = 0
        elif (df['Close'].iloc[i] > df['Lower_Band'].iloc[i]) and estado == -1:
            df['Señal'].iloc[i]==2
            estado = 0

        if(df['Señal'].iloc[i]==1):
            estado = 1
        elif(df['Señal'].iloc[i]==-1):
            estado = -1
        '''

        anteriorS =  df['Señal'].iloc[i]
        anterior = df['Close'].iloc[i]

    return df

def generar_senales_MACD(df):
    """
    Genera señales de compra y venta en función de los cruces de la cotización con las bandas de Bollinger.
    
    Parámetros:
        - df: DataFrame que contiene los precios y las bandas de Bollinger.
        
    Retorna:
        El DataFrame original con una columna adicional llamada 'Señal', que contiene las señales de compra y venta.
    """
    df['Señal'] = 0

    TOL = 0.1
    for i in range(2, len(df['Señal'])):
        if (df['MACD'][i] - df['MACD_Señal'][i] > 0):
            df['Señal'][i] = 1
        elif (df['MACD'][i] - df['MACD_Señal'][i] <0):
            df['Señal'][i] = -1

        if np.percentile(df['Volume'][0:i-1], PERCENTIL)>df['Volume'][i] and abs(df['Señal'].iloc[i])==1:
           df['Señal'].iloc[i] = 0  # Si no hay volumen significativo, no la tenemos en cuenta

        if df['Señal'].iloc[i] == df['Señal'].iloc[i-1]:
            df['Señal'].iloc[i] = 0

        anteriorS =  df['Señal'].iloc[i]
        anterior = df['Close'].iloc[i]

    return df

def generar_senales_estocastico(df):
    """
    Genera señales de compra y venta en función de los cruces de la cotización con las bandas de Bollinger.
    
    Parámetros:
        - df: DataFrame que contiene los precios y las bandas de Bollinger.
        
    Retorna:
        El DataFrame original con una columna adicional llamada 'Señal', que contiene las señales de compra y venta.
    """
    df['Señal'] = 0

    for i in range(2, len(df['Señal'])):
        if (df['D'][i] <= 20):
            df['Señal'][i] = 1
        elif ((df['D'][i] >= 80)):
            df['Señal'][i] = -1

        if np.percentile(df['Volume'][0:i-1], PERCENTIL)>df['Volume'][i] and abs(df['Señal'].iloc[i])==1:
           df['Señal'].iloc[i] = 0  # Si no hay volumen significativo, no la tenemos en cuenta

        if df['Señal'].iloc[i] == df['Señal'].iloc[i-1]:
            df['Señal'].iloc[i] = 0

        anteriorS =  df['Señal'].iloc[i]
        anterior = df['Close'].iloc[i]

    return df

def generar_senales_trix(df):
    """
    Genera señales de compra y venta en función de los cruces de la cotización con las bandas de Bollinger.
    
    Parámetros:
        - df: DataFrame que contiene los precios y las bandas de Bollinger.
        
    Retorna:
        El DataFrame original con una columna adicional llamada 'Señal', que contiene las señales de compra y venta.
    """
    df['Señal'] = 0
    TOL = 0.1

    for i in range(2, len(df['Señal'])):
        if (df['TRIX'][i] >= TOL):
            df['Señal'][i] = 1
        elif ((df['TRIX'][i] <= -TOL)):
            df['Señal'][i] = -1

        if np.percentile(df['Volume'][0:i-1], PERCENTIL)>df['Volume'][i] and abs(df['Señal'].iloc[i])==1:
           df['Señal'].iloc[i] = 0  # Si no hay volumen significativo, no la tenemos en cuenta

        if df['Señal'].iloc[i] == df['Señal'].iloc[i-1]:
            df['Señal'].iloc[i] = 0

        anteriorS =  df['Señal'].iloc[i]
        anterior = df['Close'].iloc[i]

    return df



def calcular_rentabilidad(df, ini=0, fin=-1):
    """
    Calcula la rentabilidad acumulada de un activo según los precios de cierre en el DataFrame.
    
    Parámetros:
        - df: DataFrame que contiene los precios de cierre del activo.
        
    Retorna:
        La rentabilidad acumulada del activo como un valor decimal.
    """
    precio_inicial = df['Close'].iloc[ini]
    precio_final = df['Close'].iloc[fin]
    
    rentabilidad = (precio_final - precio_inicial) / precio_inicial
    
    return rentabilidad

# Uso de las funciones
fecha_inicio = '2023-01-01'
fecha_fin = '2024-12-31'

p='Spain'
extension='.MC'
simbolos=tr.simbolos(tipo='acciones', pais=p)

simbolos = [
    'SAN.MC', 'TEF.MC', 'BBVA.MC', 'ITX.MC', 'IBE.MC', 'REP.MC', 'ACS.MC', 'GRF.MC', 'AMS.MC', 'NTGY.MC',
    'CLNX.MC', 'ENG.MC', 'IDR.MC', 'MAP.MC', 'VIS.MC', 'REE.MC', 'IAG.MC', 'SGRE.MC', 'MTS.MC', 'FER.MC',
    'AENA.MC', 'COL.MC', 'MEL.MC', 'ABE.MC', 'CABK.MC', 'FCC.MC', 'LITE.MC', 'A3M.MC', 'SAB.MC', 'ELE.MC',
    'ACX.MC', 'CIE.MC', 'BKT.MC', 'IBG.MC', 'ALM.MC', 'LOG.MC', 'EKT.MC', 'BKIA.MC', 'FAE.MC'
]

extension=''

'''
p='United States'
extension=''
simbolos=tr.simbolos(tipo='acciones', pais=p)
'''

tTicker = []
tGaps = []
tMA = []
tBB = []
tMACD = []
tEsto = []
tTRIX = []
tRdto = []

'''

for j in range(0,len(simbolos)):
#for j in range(0,10):
    ticker = simbolos[j] + extension
    try:
        i = 0
        df = tr.cotizaciones(ticker, fecha_inicio, fecha_fin)
        df = tr.calcular_gaps(df)
        df = tr.calcular_BB(df)
        df = tr.calcular_MACD(df)
        df = calcular_estocastico(df)
        df = calcular_trix(df)
        tTicker.append(ticker)        

        for i in range(0, 6):
            if i==0:
                df = tr.generar_señales_gaps(df)
                indicador ='Gaps'
            elif i==1:
                df = tr.generar_señales_MA(df)
                indicador = 'MA'
            elif i==2:
                df = generar_senales_BB(df)
                indicador = 'BB'
            elif i==3:
                df = generar_senales_MACD(df)
                indicador = 'MACD'
            elif i==4:
                df = generar_senales_estocastico(df)
                indicador = 'Estocástico'
            elif i==5:
                df = generar_senales_trix(df)
                indicador = 'TRIX'

            # Uso de la función para simular la estrategia de inversión
            importe_final, importe_final_activo = tr.simular_estrategia_inversion(df, tipo_operacion = 0)
            print(indicador, "Rent. (%): ", (importe_final-1000)/10, " Rent. act. (%): ", (importe_final_activo-1000)/10)

            rentabilidad = (importe_final-1000)/10
            rentabilidadAct = (importe_final_activo-1000)/10

            if i==0:
                tGaps.append(rentabilidad)
            elif i==1:
                tMA.append(rentabilidad)
            elif i==2:
                tBB.append(rentabilidad)
            elif i==3:
                tMACD.append(rentabilidad)
            elif i==4:
                tEsto.append(rentabilidad)
            elif i==5:
                tTRIX.append(rentabilidad)

        rentabilidad = (df['Close'][-1]-df['Close'][0])/df['Close'][0]
        tRdto.append(rentabilidad*100)
    except:
        continue

        
    # Crear un DataFrame con todas las listas

# Crear un DataFrame con todas las listas y las columnas xxxMejora
df = pd.DataFrame({
    'Ticker': tTicker,
    'Rdto': tRdto,
    'Gaps': tGaps,
    'GapsMejora': [1 if g > ga else 0 for g, ga in zip(tGaps, tRdto)],
    'MA': tMA,
    'MAMejora': [1 if m > ma else 0 for m, ma in zip(tMA, tRdto)],
    'BB': tBB,
    'BBMejora': [1 if b > ba else 0 for b, ba in zip(tBB, tRdto)],
    'MACD': tMACD,
    'MACDMejora': [1 if m > ma else 0 for m, ma in zip(tMACD, tRdto)],
    'Esto': tEsto,
    'EstoMejora': [1 if e > ea else 0 for e, ea in zip(tEsto, tRdto)],
    'TRIX': tTRIX,
    'TRIXMejora': [1 if t > ta else 0 for t, ta in zip(tTRIX, tRdto)]
})

# Imprimir el DataFrame
print(df)

# Calcular el promedio de todas las columnas numéricas
promedio_numerico = df.select_dtypes(include=['number']).mean()

# Crear una fila en blanco
df.loc[len(df)] = ''

# Añadir la fila del promedio solo para las columnas numéricas
df.loc['Promedio'] = promedio_numerico


# Guardar el DataFrame en un archivo CSV
df.to_excel('indicadores.xlsx', index=False)

'''
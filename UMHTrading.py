# UMHTrading contiene distintas funciones
# con la finalidad de simplificar la obtención y análisis de datos bursátiles.
# Para usarla se precisa la instalación de las librerías yfinance e investpy

#python.exe -m pip install yfinance
#python.exe -m pip install investpy

import investpy 
import yfinance as yf
import statistics
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def simbolos(tipo, pais='', categoria=''):
  # devuelve el listado de productos o símbolos que cotizan.
  if tipo == 'acciones':
    return investpy.get_stocks_list(country=pais)
  elif tipo == 'fondos':
    return investpy.get_funds_list(country=pais)
  elif tipo == 'etfs':
    return investpy.get_etfs_list(country=pais)
  elif tipo == 'bonos':
    return investpy.get_bonds_list(country=pais)
  elif tipo == 'indices':
    return investpy.indices.get_indices_list(country=pais)
  elif tipo == 'commodities':
    if categoria=='':
      return investpy.get_commodities_list()  
    else:
      return investpy.get_commodities_list(group=categoria)  
  elif tipo == 'divisas':
    return investpy.currency_crosses.get_available_currencies()
  elif tipo == 'derivados':
    return investpy.certificates.get_certificates_list()
  elif tipo == 'criptomonedas':
    return investpy.crypto.get_cryptos_list()
  elif tipo == 'commodities':
    return investpy.commodities.get_commodity_groups()

def cotizacion_actual(ticker):
  # devuelve la cotización actual de un ticker
  activo = yf.Ticker(ticker)
  a=activo.info

  return a['currentPrice']

def sectores_bolsa(pais, extension_bolsa):
  # devuelve el conjunto de los sectores de una bolsa de un país

  s=simbolos(tipo='acciones', pais=pais)
  sectores=[]
  for item in s:
    if extension_bolsa!=None:
      ticker=item + '.' + extension_bolsa
    else:
      ticker=item

    activo = yf.Ticker(ticker)
    a=activo.info
    if('sector' in a):
      sectores.append(a['sector'])
      # tr.analisis_fundamental_sector('Technology', 'Spain', 'MC')

  sectores=set(sectores)

  return sectores

############################ FUNCIONES DE ANALISIS FUNDAMENTAL ##############################
def calcula_cuartiles_fundamentales(fichero_salida, pais, extension_bolsa=None):
  # graba en fichero_sal los cuartiles de indicadores fundamentales
  # para cada sector

  # Bajamos los símbolos del país
  s=simbolos(tipo='acciones', pais=pais)

  # Definimos la lista de indicadores que nos interesan
  indicadores=['dividendYield', 'payoutRatio', 'beta', 'enterpriseToEbitda',
  'returnOnEquity', 'priceToBook', 'marketCap', 'freeCashflow']

  valores={}          # Diccionario que va a contener los valores que nos interesan.
  for simbolo in s:   # Recorremos los símbolos.
    print(simbolo)
    if extension_bolsa!=None:
      ticker=simbolo + '.' + extension_bolsa
    else:
      ticker=simbolo
    
    try:              # Descargamos información sobre el activo, tratando los posibles errores
      activo = yf.Ticker(ticker)
      a=activo.info
    except:
      a=None

    if a!=None and 'sector' in a: # Si se ha podido descargar algo
      if not (a['sector'] in valores): # El sector es nuevo
        valores[a['sector']]={}   # Creamos un diccionario anidado para dicho sector        

      for indicador in indicadores:
        if not (indicador in valores[a['sector']]): # Si en los valores no está el indicador, lo añadimos.
          valores[a['sector']][indicador]=[]
        if indicador in a:                          # Si el indicador está en la información, lo añadimos
          valores[a['sector']][indicador].append(a[indicador])
        else:
          valores[a['sector']][indicador].append(0) # En caso contrario, entendemos que es 0

  # PER y PFCF los calculamos mediante otros indicadores.
  indicadores.append('PER')
  indicadores.append('PFCF')
  sectores=[]
  for sector in valores:
    sectores.append(sector)
    valores[sector]['PER']=[]
    valores[sector]['PFCF']=[]
    for i in range(0,len(valores[sector]['dividendYield'])):
      if valores[sector]['dividendYield'][i]!=0:
        valores[sector]['PER'].append((1/valores[sector]['dividendYield'][i])*valores[sector]['payoutRatio'][i])
      else:
        valores[sector]['PER'].append(0)
    for i in range(0,len(valores[sector]['freeCashflow'])):
      if valores[sector]['freeCashflow'][i]!=0:
        valores[sector]['PFCF'].append(valores[sector]['marketCap'][i]/valores[sector]['freeCashflow'][i])
      else:
        valores[sector]['PFCF'].append(0)

  # Grabamos en un dataframe los cuartiles de cada sector-indicador.
  sectores=[]
  inds=[]
  q1=[]
  q2=[]
  q3=[]
  for sector in valores:    
    for indicador in indicadores:
      if indicador in valores[sector]:
        sectores.append(sector)
        inds.append(indicador)
        if(len(valores[sector][indicador])>2):     
          cuartiles=statistics.quantiles(valores[sector][indicador])     
          q1.append(cuartiles[0])
          q2.append(cuartiles[1])
          q3.append(cuartiles[2])
        else:
          q1.append(0)
          q2.append(0)
          q3.append(0)
      else:
        q1.append(0)
        q2.append(0)
        q3.append(0)

  df=pd.DataFrame()
  df['sector']=sectores
  df['indicadores']=inds
  df['q1']=q1
  df['q2']=q2
  df['q3']=q3

  # Guardamos los cuartiles por sector en el fichero excel
  df.to_excel(fichero_salida)

def fundamentales_sector(sector, fichero_fundamentales):
  # Devuelve un diccionario con los cuartiles de los indicadores fundamentales de un sector
  # a raíz del fichero excel fichero_fundamentales indicado y creado previamente mediante la función
  # calcula_cuartiles_fundamentales

  # Leo fichero_fundamentales
  df=pd.read_excel(fichero_fundamentales)

  # Creo un diccionario con los cuartiles de cada indicador para un sector dado
  fundamentales={}

  for i in range(0,df.shape[0]):
    if df.iloc[i,1]==sector:
      indicador=df.iloc[i,2]
      if indicador!='marketCap' and indicador!='freeCashflow': # marketCap y freeCashflow no los usamos.
        if not indicador in fundamentales:
          fundamentales[indicador]=[]
        fundamentales[indicador].append(df.iloc[i,3])
        fundamentales[indicador].append(df.iloc[i,4])
        fundamentales[indicador].append(df.iloc[i,5])

  return fundamentales

def analisis_fundamental(ticker_lista, fichero_fundamentales, fichero_salida):  
  # Crea el fichero Excel fichero_salida con las advertencias y recomendaciones positivas y negativas 
  # de cada activo de ticker_lista que presenta alguna.

  # Lista de los indicadores que utilizaremos
  indicadores=['dividendYield', 'payoutRatio', 'PER', 'beta', 'enterpriseToEbitda', 'returnOnEquity', 'priceToBook', 'PFCF']

  # Diccionario que utilizaremos para crear posteriormente el dataframe a grabar
  dic={'ticker':[], 'sector':[], 'dividendYield':[], 'payoutRatio':[], 'PER':[], 'beta':[], 'enterpriseToEbitda':[], 'returnOnEquity':[], 'priceToBook':[], 'PFCF': []}
    
  numEmpresas=0 # Indicaremos además el número de empresas finalmente analizadas
  for ticker in ticker_lista:
    try:    # Trato los posibles errores
      activo = yf.Ticker(ticker)
      print(ticker)
      a=activo.info
      fund=fundamentales_sector(a['sector'], fichero_fundamentales) # Obtengo los cuartiles de un sector

      numEmpresas+=1
      dic['ticker'].append(ticker)
      dic['sector'].append(a['sector'])
      eliminar=True
      for i in range(0,len(indicadores)):        
        indicador=indicadores[i]
        valor=''      # En principio, no tenemos ninguna observación a realizar para dicho activo-indicador
        if indicador in ['dividendYield', 'payoutRatio'] and indicador in a:
          # Si dividendYield o payoutRatio están por debajo de Q1 -> negativo
          if a[indicador]<fund[indicador][1]:
            valor='NEGATIVO'
        elif indicador=='beta' and indicador in a:
          # Si beta está por encima de Q2 -> negativo (muy arriesgado)
          if a[indicador]>fund[indicador][2]:
              valor='NEGATIVO'
          elif a[indicador]<fund[indicador][1] and a[indicador]<=1:
              # si está por debajo de Q1 y además es inferior a 1 -> poco riesgo
              valor='POSITIVO'
        elif indicador in ['enterpriseToEbitda', 'returnOnEquity', 'priceToBook'] and indicador in a:
          # Si enterpriseToEbitda, returnOnEquity o priceTobook están por encima de Q2 -> advertencia
          if a[indicador]>fund[indicador][2]:
            valor='Advertencia'
          elif a[indicador]<0:
            # Si enterpriseToEbitda, returnOnEquity o priceTobook están por debajo de 0 -> negativo
            valor='NEGATIVO'
        elif indicador=='PER' and 'dividendYield' in a and 'payoutRatio' in a:
          per=(1/a['dividendYield'])*a['payoutRatio']
          if(per>15):   # per>15 -> negativo
            valor='NEGATIVO'
          elif (per<fund[indicador][1]):  # per<Q1 -> positivo
            valor='POSITIVO'
        elif indicador=='PFCF'and 'marketCap' in a and 'freeCashflow' in a:
          pfcf=a['marketCap']/a['freeCashflow']   
          if(pfcf>fund[indicador][2]):  # PFCF>Q2 -> Advertencia
            valor='Advertencia'
                     
        dic[indicador].append(valor) # Añadimos la observación
        if valor!='':    # Si hay observación, no eliminaremos el activo de la tabla
          eliminar=False
          
      if eliminar:  # Si no se realizó ninguna observación, eliminamos el activo de la tabla
        for key in dic:
          dic[key].pop()
    except:
      continue  # Continuamos con el siguiente activo
  
  # Creamos el dataframe y lo guardamos en fichero_salida
  df=pd.DataFrame(dic)
  df.to_excel(fichero_salida)
  
  print('Tras analizar', numEmpresas,'empresas, el análisis fundamental fue grabado con éxito en', fichero_salida)
  return

def analisis_fundamental_accion(ticker, fichero_fundamentales):
  
  indicadores=['dividendYield', 'payoutRatio', 'PER', 'beta', 'enterpriseToEbitda', 'returnOnEquity', 'priceToBook', 'PFCF']
    
  #try:    
  if True:
    activo = yf.Ticker(ticker)
    a=activo.info
    fund=fundamentales_sector(a['sector'], fichero_fundamentales)
    print('Sector:', a['sector'])
    for i in range(0,len(indicadores)):        
      indicador=indicadores[i]
      if indicador in ['dividendYield', 'payoutRatio'] and indicador in a:
        if a[indicador]<fund[indicador][1]:
            print('NEGATIVO:', indicador, '. Valor:', a[indicador], '. Cuartiles sector:', fund[indicador])
      elif indicador=='beta' and indicador in a:
        if a[indicador]>fund[indicador][2]:
          print('NEGATIVO:', indicador, '. Valor:', a[indicador], '. Cuartiles sector:', fund[indicador])
        elif a[indicador]<fund[indicador][1] and a[indicador]<=1:
          print('POSITIVO:', indicador, '. Valor:', a[indicador], '. Cuartiles sector:', fund[indicador])
      elif indicador in ['enterpriseToEbitda', 'returnOnEquity', 'priceToBook'] and indicador in a:
        if a[indicador]>fund[indicador][2]:
          print('Advertencia:', indicador, '. Valor:', a[indicador], '. Cuartiles sector:', fund[indicador])
        if indicador in ['returnOnEquity', 'enterpriseToEbitda'] and a[indicador]<0:
          print('NEGATIVO:', indicador, '. Valor:', a[indicador], '(<0)')
      elif indicador=='PER' and 'dividendYield' in a and 'payoutRatio' in a:
        per=(1/a['dividendYield'])*a['payoutRatio']
        if(per>15):
          print('NEGATIVO:', indicador, '. Valor:', per, ' (>15)')
        elif (per<fund[indicador][1]):
          print('POSITIVO:', indicador, '. Valor:', per, '. Cuartiles sector:', fund[indicador])
      elif indicador=='PFCF'and 'marketCap' in a and 'freeCashflow' in a:
        pfcf=a['marketCap']/a['freeCashflow']          
        if(pfcf>fund[indicador][2]):
          print(indicador, 'advertencia. Valor:', pfcf, '. Cuartiles sector:', fund[indicador])                     
  #except:
   # print('Error al descargar el ticker', ticker)

  return

############################ FUNCIONES DE ANALISIS FUNDAMENTAL ##############################

def cotizaciones(ticker, fIni, fFin=datetime.now().strftime('%Y-%m-%d'), periodo='1d', intervalo = '1d'):
  # devuelve los distintos datos de cotización entre dos fechas de un ticker
  # periodo= 1d, 5d, 1mo, 1y
  # intervalo= 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk y 1mo

  df = yf.download(ticker, start = fIni, end=fFin, period=periodo)
  return df

def calcular_gaps(df):

  # Calcula los gaps existentes en un DataFrame que contenga precios
  # y le añade las bandas y la media móvil central

    gaps=[]
    gaps.append(None)
    high=df['High'].iloc[0]
    low=df['Low'].iloc[0]


    for i in range(1, len(df)):
        prev_high = high
        prev_low = low
        high = df['High'].iloc[i]
        low = df['Low'].iloc[i]

        if low > prev_high:
            gaps.append( (low-prev_high)*100/prev_high)
        elif prev_low > high: 
            gaps.append( (high-prev_low)*100/prev_low)
        else: 
            gaps.append(0)

    df['gaps']=gaps

    return df

def calcular_MA(df, periodo = 20):
  df['MA'] = df['Close'].rolling(window=periodo).mean()
  return df

def calcular_BB(df, periodo = 20):
    
    # Calcula las bandas de Bollinger a partir de un DataFrame que contenga precios.
    # y le añade las bandas y la media móvil central
    
    df = calcular_MA(df, periodo)
    df['Upper_Band'] = df['MA'] + 2 * df['Close'].rolling(window=periodo).std()
    df['Lower_Band'] = df['MA'] - 2 * df['Close'].rolling(window=periodo).std()

    return df


def calcular_MACD(df, periodo_corto = 12, periodo_largo = 26, periodo_señal = 9):
    
     # Calcula lel MACD y sus señal
    # en un dataframe que contenga precios y se los añade

    # Calcular MACD
    df['ema_corto'] = df['Close'].ewm(span=periodo_corto, adjust=False).mean()
    df['ema_largo'] = df['Close'].ewm(span=periodo_largo, adjust=False).mean()
    df['MACD'] = df['ema_corto'] - df['ema_largo']
    
    # Calcular la señal del MACD
    df['MACD_Señal'] = df['MACD'].ewm(span=periodo_señal, adjust=False).mean()

    df['MACD_Diferencia'] = df['MACD'] - df['MACD_Señal']
    
    df = df.drop('ema_corto', axis=1)
    df = df.drop('ema_largo', axis=1)

    return df

def calcular_estocastico(df, periodo_K=14, periodo_D=3):
    # Calcular el mínimo y máximo de la ventana K
    df['Min_K'] = df['Low'].rolling(window=periodo_K).min()
    df['Max_K'] = df['High'].rolling(window=periodo_K).max()
    
    # Calcular %K
    df['K'] = ((df['Close'] - df['Min_K']) / (df['Max_K'] - df['Min_K'])) * 100
    
    # Calcular %D (promedio móvil de %K)
    df['D'] = df['K'].rolling(window=periodo_D).mean()
    
    return df

def generar_grafico_cotizaciones(df, ticker, MA=False, BB=False, MACD=False, MACD_Diferencia = False, señales=True):
   
    """
    Genera un gráfico con las cotizaciones de ticker según los datos de df del activo ticker
    MA = true -> Añade media móvil
    BB = true -> Añade bandas de Bollinger (y media móvil también)
    MACD = true -> Añade el MACD
    señales = true -> Añade las señales marcadas en df
    """
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    ax1.plot(df.index, df['Close'], label=ticker + ' cotización', color='blue')

    if MA or BB:
        ax1.plot(df.index, df['MA'], label='Media móvil', color='orange', linestyle='--')

    if BB:
        ax1.plot(df.index, df['Upper_Band'], label='Banda superior', color='green', linestyle='--')
        ax1.plot(df.index, df['Lower_Band'], label='Banda inferior', color='gray', linestyle='--')

    if MACD:
        ax1.plot(df.index, df['MACD'], label='MACD', color='green')
        ax1.plot(df.index, df['MACD_Señal'], label='Señal MACD', color='gray')
    
    if MACD_Diferencia:
        ax1.plot(df.index, df['MACD_Diferencia'], label='MACD', color='gray')
        ax1.fill_between(df.index, df['MACD_Diferencia'], 0, where=(df['MACD_Diferencia'] > 0), color='green', alpha=0.3)
        ax1.fill_between(df.index, df['MACD_Diferencia'], 0, where=(df['MACD_Diferencia'] < 0), color='red', alpha=0.3)

    # Identificar y graficar los puntos específicos
    if ('Señal') in df and señales:
      for i, señal in enumerate(df['Señal']):
          if señal == -2:
              ax1.scatter(df.index[i], df['Close'][i], color='yellow', zorder=5)
          elif señal == 2:
              ax1.scatter(df.index[i], df['Close'][i], color='yellow', zorder=5)
          elif señal == -1:
              ax1.scatter(df.index[i], df['Close'][i], color='red', zorder=5)
          elif señal == 1:
              ax1.scatter(df.index[i], df['Close'][i], color='blue', zorder=5)
        
    ax2 = ax1.twinx()
    try:
      ax2.fill_between(df.index, y1=0, y2=df['Volume'], color='gray', alpha=0.2, label='Volume')
      ax2.set_ylabel('Volume', color='gray')
    except:
      pass

    ax1.set_title(ticker)
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Precio')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.show()

    return

def generar_señales_gaps(df, percentil_volumen=75, tolerancia=0.1):
    """
    Genera señales de compra y venta según los gaps en función de una tolerancia
    y el percentil de volumen deseado
    """
    señal =[0, 0]
    valorAnt = 0
    
    for i in range(2, len(df['Close'])):
        valor = 0
        if (df['gaps'].iloc[i] >= tolerancia):
            valor = 1            
        elif (df['gaps'].iloc[i] <= -tolerancia):
            valor = -1            
        elif df['gaps'].iloc[i] > 0:
            valor = 2            
        elif df['gaps'].iloc[i] < 0:
            valor = -2

        ini = i-60
        if ini < 0:
          ini = 0
        if np.percentile(df['Volume'][ini:i-1], percentil_volumen)>df['Volume'].iloc[i] and abs(valor)==1:
           valor = 0

        if valor == valorAnt:
            valor = 0
        
        valorAnt = valor
        señal.append(valor)

    df['Señal'] = señal

    return df


def generar_señales_MA(df, percentil_volumen = 75):
    """
    Genera señales de compra y venta según se atraviesa la Media Móvil
    y en función del percentil de volumen deseado
    """
    señal = [0, 0]

    for i in range(2, len(df['Close'])):
        valor = 0

        ini = i-60
        if ini < 0:
          ini = 0
        volumen_alto = np.percentile(df['Volume'][ini:i-1], percentil_volumen)

        if df['Volume'].iloc[i]>volumen_alto:
          if (df['Close'].iloc[i-1]- df['MA'].iloc[i-1]>=0 and df['Close'].iloc[i]- df['MA'].iloc[i]<0):
              # Se atraviesa la media descendentemente
              valor = -1
          elif (df['Close'].iloc[i-1]- df['MA'].iloc[i-1]<=0 and df['Close'].iloc[i]- df['MA'].iloc[i]>0):
              # Se atraviesa la media ascendentemente
              valor = 1

        señal.append(valor)

    df['Señal'] = señal

    return df

def generar_señales_BB(df, percentil_volumen_alto = 75, percentil_volumen_bajo = 25):
    """
    Genera señales de compra y venta según se atraviesa la Media Móvil
    y en función del percentil de volumen deseado
    """
    señal = [0, 0]

    for i in range(2, len(df['Close'])):
      valor = 0

      ini = i-60
      if ini < 0:
        ini = 0
      volumen_alto = np.percentile(df['Volume'][ini:i-1], percentil_volumen_alto)
      volumen_bajo = np.percentile(df['Volume'][ini:i-1], percentil_volumen_bajo)
      volumen_bajo = volumen_alto
      if (df['Close'].iloc[i-1]-df['MA'].iloc[i-1]>=0 and df['Close'].iloc[i]-df['MA'].iloc[i]<0 and df['Volume'].iloc[i]>volumen_alto):
        # La cotización atraviesa la media móvil de forma descendente
        valor = -1
      elif (df['Close'].iloc[i-1]-df['MA'].iloc[i-1]<=0 and df['Close'].iloc[i]-df['MA'].iloc[i]>0 and df['Volume'].iloc[i]>volumen_alto):
        # La cotización atraviesa la media móvil de forma descendente
        valor = 1
      elif (df['Close'].iloc[i-1]-df['Upper_Band'].iloc[i-1]<=0 and df['Close'].iloc[i]-df['Upper_Band'].iloc[i]>0 and df['Volume'].iloc[i]>volumen_alto):
        # La cotización atraviesa la banda superior con fuerza
        valor = 1
      elif (df['Close'].iloc[i-1]-df['Lower_Band'].iloc[i-1]>=0 and df['Close'].iloc[i]-df['Lower_Band'].iloc[i]<0 and df['Volume'].iloc[i]>volumen_alto):
        # La cotización atraviesa la banda inferior con fuerza
        valor = -1
      elif (df['Close'].iloc[i-1]-df['Upper_Band'].iloc[i-1]<=0 and df['Close'].iloc[i]-df['Upper_Band'].iloc[i]>0 and df['Volume'].iloc[i]<volumen_bajo):
        # La cotización toca la banda superior sin fuerza
        valor = -1
      elif (df['Close'].iloc[i-1]-df['Lower_Band'].iloc[i-1]>=0 and df['Close'].iloc[i]-df['Lower_Band'].iloc[i]<0 and df['Volume'].iloc[i]<volumen_bajo):
        # La cotización toca la banda inferior sin fuerza
        valor = 1
      elif (df['Close'].iloc[i-1]-df['Upper_Band'].iloc[i-1]<=0 and df['Close'].iloc[i]-df['Upper_Band'].iloc[i]>0):
        # La cotización toca la banda superior con fuerza intermedia (agotamiento)
        valor = -2
      elif (df['Close'].iloc[i-1]-df['Lower_Band'].iloc[i-1]>=0 and df['Close'].iloc[i]-df['Lower_Band'].iloc[i]<0):
        # La cotización toca la banda superior con fuerza intermedia (agotamiento)
        valor = 2
      
      señal.append(valor)

    df['Señal'] = señal

    return df

def generar_señales_MACD(df, percentil_volumen = 75):
    """
    Genera señales de compra y venta en función de los cruces del MACD  con su señal
    Se tienen en cuenta sólo cuando hay un volumen relevante y una tolerancia
    """

    señal = [0, 0]

    for i in range(2, len(df['Close'])):
      valor = 0

      ini = i-60
      if ini < 0:
        ini = 0
      volumen_alto = np.percentile(df['Volume'][ini:i-1], percentil_volumen)

      if (df['MACD'].iloc[i-1] - df['MACD_Señal'].iloc[i-1] <= 0 and df['MACD'].iloc[i] - df['MACD_Señal'].iloc[i] > 0 and df['Volume'].iloc[i] > volumen_alto):
          valor = 1
      elif (df['MACD'].iloc[i-1] - df['MACD_Señal'].iloc[i-1] >= 0 and df['MACD'].iloc[i] - df['MACD_Señal'].iloc[i] < 0 and df['Volume'].iloc[i] > volumen_alto ):
          valor = -1

      señal.append(valor)

    df['Señal'] = señal

    return df

def generar_señales_estocastico(df, percentil_volumen = 75):
    """
    Genera señales de compra y venta en función del indicador estocástico
    """
    
    señal = [0, 0]

    for i in range(2, len(df['Close'])):
        
      valor = 0
      ini = i-60
      if ini < 0:
        ini = 0
      volumen_alto = np.percentile(df['Volume'][ini:i-1], percentil_volumen)
        
      if (df['D'].iloc[i] < 20 and df['D'].iloc[i-1]>=20 and df['Volume'].iloc[i]>percentil_volumen):
        valor = 1
      elif (df['D'].iloc[i] > 80 and df['D'].iloc[i-1]<=80  and df['Volume'].iloc[i]>percentil_volumen):
        valor = -1

      señal.append(valor)

    df['Señal'] = señal

    return df

def deshaz_operacion(dfHistorico, capital, acciones, precio_operacion, precio_cierre, fecha):
    """
    Deshace la última operación realizada en base a la señal y actualiza el DataFrame de operaciones históricas.    
    """
    beneficio = acciones * (precio_cierre - precio_operacion)
    #print('Capital:', capital,'. Beneficio: ', beneficio, 'p.oper:', precio_operacion, 'p.cier:', precio_cierre)
    capital += beneficio
    dfHistorico.loc[len(dfHistorico)] = [fecha, precio_cierre, beneficio, capital, 'Venta' if acciones>0 else 'Compra']
    acciones = 0    
    #print(capital, acciones, precio_cierre, señal)
    return capital, acciones, dfHistorico


def simular_estrategia_inversion(df, tipo_operacion=0, grabar=False, stop = 0.1):
    """
    Simula una estrategia de inversión en función de las señales generadas en df
    tipo_operacion = 0, genera compras y ventas en función de las señales
    tipo_operacion = 1, sólo compras
    tipo_operacion = -1, sólo ventas

    Se establece en dicha estrategia tanto stop_loss como trailing del mismo

    Devuelve:
        rentabilidad, el retorno final que se tendría después de aplicar la estrategia de inversión.
        rentabilidad_activo, el retorno final que se tendría si en los períodos en los que se invirtió, se hubiera comprado el activo.

    Además, graba en operaciones.xlsx el histórico de operaciones realizadas
    """

    capital = 1000  # Capital inicial
    capital_activo = 1000
    acciones = 0     # Cantidad de acciones en cartera
    precio_stop = 0

    # Listas para cada columna del DataFrame historico
    fechas = []
    precios = []
    beneficios = []
    operaciones = []
    num_acciones = []  # Lista para el número de acciones compradas o vendidas
    precio_operacion = 0    
    dfHistorico = pd.DataFrame(columns=['Fecha', 'Precio', 'Beneficio', 'Capital', 'Operación'])
    iInversion = -1

    for i in range(len(df)-1):
        señal = df['Señal'].iloc[i]
        precio_cierre = df['Close'].iloc[i]
        fecha = df.index[i]
        
        # Trailing
        if acciones>0 and precio_cierre*(1-stop)>precio_stop:
            precio_stop = precio_cierre*(1-stop)
        elif acciones<0 and precio_cierre*(1+stop)<precio_stop:
            precio_stop = precio_cierre*(1+stop)

        # Deshacer la operación si hay señal de debilitamiento o contraria a la actual
        if ( ( señal >=1 or precio_cierre >= precio_stop) and acciones < 0) or ( ( señal <=-1 or precio_cierre <= precio_stop) and acciones > 0) :  
            capital_activo += (precio_cierre - precio_operacion) * abs(acciones)
            capital, acciones, dfHistorico = deshaz_operacion(dfHistorico, capital, acciones, precio_operacion, precio_cierre, fecha)    

        # Realizar la compra o venta si se dan las condiciones
        if señal == 1 and acciones == 0 and capital > 0  and (tipo_operacion == 1 or tipo_operacion == 0) :
            # Señal de compra y no hay acciones en la cartera -> Compramos
            acciones = capital / precio_cierre   
            precio_operacion = precio_cierre
            precio_stop = precio_operacion * (1-stop)
            fechas.append(fecha)
            precios.append(precio_cierre)
            beneficios.append(0)
            operaciones.append('Compra')
            num_acciones.append(acciones)
            dfHistorico.loc[len(dfHistorico)] = [fecha, precio_cierre, 0, capital, 'Compra']
            if iInversion ==-1:
                iInversion = i

        elif señal == -1 and acciones == 0 and capital > 0  and (tipo_operacion == -1 or tipo_operacion == 0):  
            # Señal de venta y no hay acciones en cartera -> Vendemos (el mismo número de acciones que podríamos comprar)
            acciones = -capital / precio_cierre
            precio_operacion = precio_cierre
            precio_stop = precio_operacion * (1+stop)  
            fechas.append(fecha)
            precios.append(precio_cierre)
            beneficios.append(0)
            operaciones.append('Venta')
            num_acciones.append(acciones)
            dfHistorico.loc[len(dfHistorico)] = [fecha, precio_cierre, 0, capital, 'Venta']
            if iInversion ==-1:
                iInversion = i

    if acciones!=0: # Si al final del período tengo acciones, he de deshacer la operación sí o sí
        i = len(df)-1
        precio_cierre = df['Close'].iloc[i]
        fecha = df.index[i]
        capital_activo += (precio_cierre - precio_operacion) * abs(acciones)
        capital, acciones, dfHistorico = deshaz_operacion(dfHistorico, capital, acciones, precio_operacion, precio_cierre, fecha)

    # Calcular el importe final (considerando el valor de las acciones en cartera)
    importe_final = capital if capital > 0 else acciones * df['Close'].iloc[-1]
    
    # Guardar el historial de operaciones en un archivo Excel
    dfHistorico.to_excel('operaciones.xlsx', index=False)
    
    rentabilidad = (importe_final-1000)/10
    rentabilidad_activo = (capital_activo-1000)/10

    return rentabilidad, rentabilidad_activo
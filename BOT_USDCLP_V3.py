# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 21:59:27 2023

@author: jfcog
"""

import talib
import numpy as np
import pandas as pd
from tqdm import tqdm
import MetaTrader5 as mt5
from datetime import datetime
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import data_prep_combined_lstm_V3_CLP as dpcl
import res_analysis_V3_CLP as resan
import models_V3_CLP
from fredapi import Fred
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
from IPython import get_ipython
import time
import math


#MT5 config
is_connected = mt5.initialize()
login = 41490
password = '*nYw6qBc'
server= 'MercadosG-Server'

mt5.login(login, password, server)

delta = timedelta(days = 40)
start_date = datetime(2023,12,27)
timeframe = mt5.TIMEFRAME_H1
n = 50
fut = 8
umbral = 2
symbol = 'USDCLP'

scaler = StandardScaler()
flag_out = 0
margin = 0
margin_level = 0
min_margin = 100
volume = 0.02
min_vol = 0.01
positions_df = pd.DataFrame(columns = ['ticket', 'time', 'time_msc', 'time_update', 'time_update_msc', 'type', 'magic', 'identifier', 'reason', 'volume', 'price_open', 'sl', 'tp', 'price_current', 'swap', 'profit', 'symbol', 'comment', 'external_id','max','low','new_sl','new_tp'])

start_tp = 1.5
start_sl = 1.5

new_max = 0
new_min = 0

def modify_sltp(position, symbol, sl, tp):
    
    ticket = position._asdict()['ticket']
    price_open = position._asdict()['price_open']
    tipo = position._asdict()['type']
    volume = position._asdict()['volume']
    comment = position._asdict()['comment']
    magic = position._asdict()['magic']
    if tipo == 0:
        tipo = mt5.ORDER_TYPE_BUY
    else:
        tipo = mt5.ORDER_TYPE_SELL
    
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "type": tipo,
        "position": ticket,
        "volume": volume,
        "price_open": price_open,
        "tp": tp,
        "sl": sl,
        "type_time": mt5.ORDER_TIME_GTC,
        "deviation": 5,  # Desviación permitida en el precio
        "magic": magic,  # Número mágico para identificar la orden
        "comment": comment,
        "type_filling": mt5.ORDER_FILLING_FOK,
        }
    # Envía la orden de compra
    result = mt5.order_send(request)
    print("Se modifica SL/TP ticket: " + str(ticket))
    
while True:
    
    total_profit = 0
    positions_cant = 0
    end_date = datetime.now()
    account_info = mt5.account_info()
    margin = account_info.margin
    margin_level = account_info.margin_level
    positions = mt5.positions_get()
    # Itera sobre todas las posiciones y suma el beneficio de cada una
    for position in positions:
        positions_cant +=1
        total_profit += position.profit
        pos_tp = position.tp
        pos_sl = position.sl
        ticket = position._asdict()['ticket']
        ticket_index = positions_df.index[positions_df['ticket'] == ticket]
        current = position._asdict()['price_current']
        price_open = position._asdict()['price_open']
        positions_df.loc[ticket_index, 'profit'] = position.profit
        # Verifica si el ticket ya está en el DataFrame
        if positions_df.empty:
            positions_df = pd.concat([positions_df,pd.DataFrame([position._asdict()])], ignore_index=True)
            ticket_index = positions_df.index[positions_df['ticket'] == ticket]
            positions_df.loc[ticket_index,'max'] = price_open
            positions_df.loc[ticket_index,'low'] = price_open
            positions_df.loc[ticket_index,'new_sl'] = pos_sl
            positions_df.loc[ticket_index,'new_tp'] = pos_tp
            print('a')
            
        if ticket not in positions_df['ticket'].values:
            positions_df = pd.concat([positions_df,pd.DataFrame([position._asdict()])], ignore_index=True)
            ticket_index = positions_df.index[positions_df['ticket'] == ticket]
            positions_df.loc[ticket_index,'max'] = price_open
            positions_df.loc[ticket_index,'low'] = price_open
            positions_df.loc[ticket_index,'new_sl'] = pos_sl
            positions_df.loc[ticket_index,'new_tp'] = pos_tp
            print('b')
        else:
            # Si el ticket ya está en el DataFrame, actualiza 'max' y 'low'
            max_value = positions_df.loc[ticket_index, 'max'].values[0]  # Obtener el valor escalar
            if math.isnan(max_value):
                max_value = price_open
                positions_df.loc[ticket_index, 'max'] = max_value
            low_value = positions_df.loc[ticket_index, 'low'].values[0]  # Obtener el valor escalar
            if math.isnan(low_value):
                low_value = price_open
                positions_df.loc[ticket_index, 'low'] = low_value
            if current > max_value:
                positions_df.loc[ticket_index, 'max'] = current
                new_max = 1
            if current < low_value:
                positions_df.loc[ticket_index, 'low'] = current
                new_min = 1
            #positions_df.loc[ticket_index,'new_sl'] = pos_sl
            #positions_df.loc[ticket_index,'new_tp'] = pos_tp
        
        # Estrategia SL/TP A Favor
        order_type = position._asdict()['type']

        #Posción es Compra
        if order_type == 0:
            if new_max == 1:
                max_value = positions_df.loc[ticket_index, 'max'].values[0]
                sl = price_open - positions_df.loc[ticket_index, 'new_sl'].values[0]
                tp = positions_df.loc[ticket_index, 'new_tp'].values[0] - price_open
                profit = positions_df.loc[ticket_index, 'profit'].values[0]
                pos_volume = positions_df.loc[ticket_index, 'volume'].values[0]
                aux = 0.5*tp + price_open
                
                if max_value > 0.5*tp + price_open or profit >= 400*pos_volume*100:
                    
                    if profit > 1500*pos_volume*100:
                        #SL se lleva a 61,8%
                        print("Se alcanza el 76.4% de tp para ticket: " + str(ticket))
                        new_sl = 0.8*(current - price_open)
                        new_tp = current*1.004 - price_open
                        modify_sltp(position, symbol, price_open + new_sl, price_open + new_tp)
                        positions_df.loc[ticket_index, 'new_sl'].value = price_open - new_sl
                        positions_df.loc[ticket_index, 'new_tp'].value = price_open + new_tp
                        print('c2')
                    
                    elif max_value > 0.764*tp + price_open or profit >= 1300*pos_volume*100:
                        #SL se lleva a 61,8%
                        print("Se alcanza el 76.4% de tp para ticket: " + str(ticket))
                        new_sl = 0.618*(current - price_open)
                        new_tp = tp*1.5
                        modify_sltp(position, symbol, price_open + new_sl, price_open + new_tp)
                        positions_df.loc[ticket_index, 'new_sl'].value = price_open - new_sl
                        positions_df.loc[ticket_index, 'new_tp'].value = price_open + new_tp
                        print('c')
                        
                    elif max_value > 0.618*tp + price_open or profit >= 1000*pos_volume*100:
                        #SL se lleva a 38,2%
                        print("Se alcanza el 61.8% de tp para ticket: " + str(ticket))
                        new_sl = 0.382*(current - price_open)
                        new_tp = tp
                        modify_sltp(position, symbol, price_open + new_sl, price_open + new_tp)
                        positions_df.loc[ticket_index, 'new_sl'].value = price_open - new_sl
                        positions_df.loc[ticket_index, 'new_tp'].value = price_open + new_tp
                        print('d')
                        
                    elif profit >= 600*pos_volume*100:
                        #SL se lleva a 38,2%
                        print("Se alcanzan 600 de profit para ticket: " + str(ticket))
                        new_sl = 0.1*(current - price_open)
                        new_tp = tp
                        modify_sltp(position, symbol, price_open + new_sl, price_open + new_tp)
                        positions_df.loc[ticket_index, 'new_sl'].value = price_open + new_sl
                        positions_df.loc[ticket_index, 'new_tp'].value = price_open - new_tp
                        print('e')
                        
                    elif profit >= 400*pos_volume*100:
                        #SL se lleva a 38,2%
                        print("Se alcanzan 400 de profit para ticket: " + str(ticket))
                        new_sl = 0.5*sl
                        new_tp = tp
                        modify_sltp(position, symbol, price_open - new_sl, price_open + new_tp)
                        positions_df.loc[ticket_index, 'new_sl'].value = price_open + new_sl
                        positions_df.loc[ticket_index, 'new_tp'].value = price_open - new_tp
                        print('f')
                        
                    else:
                        #Se asegura al menos un sl  = 0
                        print("Se alcanza el 50% de tp para ticket: " + str(ticket))
                        new_sl = 0.1*(current - price_open)
                        new_tp = tp
                        modify_sltp(position, symbol, price_open + new_sl, price_open + new_tp)
                        positions_df.loc[ticket_index, 'new_sl'].value = price_open - new_sl
                        positions_df.loc[ticket_index, 'new_tp'].value = price_open + new_tp
                        print('g')
            
            #Estrategia de salida
            if new_min == 1:
                min_value = positions_df.loc[ticket_index, 'low'].values[0]
                sl = price_open - positions_df.loc[ticket_index, 'new_sl'].values[0]
                tp = positions_df.loc[ticket_index, 'new_tp'].values[0] - price_open
                aux = 0.5*tp + price_open
                
                if min_value < price_open - 0.618*tp:
                    
                    #Se disminuye el SL a 76.4%
                    print("Se alcanza el -61,8% de sl para ticket: " + str(ticket))
                    new_sl = 1*sl
                    new_tp = tp
                    modify_sltp(position, symbol, price_open - new_sl, price_open + new_tp)
                    positions_df.loc[ticket_index, 'new_sl'].value = price_open - new_sl
                    positions_df.loc[ticket_index, 'new_tp'].value = price_open + new_tp
                    print('h')
                    
        #Posición es Venta
        if order_type == 1:
            if new_min == 1:
                min_value = positions_df.loc[ticket_index, 'low'].values[0]
                sl = positions_df.loc[ticket_index, 'new_sl'].values[0] - price_open
                tp = price_open - positions_df.loc[ticket_index, 'new_tp'].values[0]
                profit = positions_df.loc[ticket_index, 'profit'].values[0]
                pos_volume = positions_df.loc[ticket_index, 'volume'].values[0]
                aux = price_open - 0.5*tp
                
                if min_value <  price_open - 0.5*tp or profit >= 400*pos_volume*100:
                    
                    if profit > 1500*pos_volume*100:
                        #SL se lleva a 61,8%
                        print("Se alcanza el 76,4% de tp para ticket: " + str(ticket))
                        new_sl = 0.8*(price_open - current)
                        new_tp = price_open - current*0.997
                        modify_sltp(position, symbol, price_open - new_sl, price_open - new_tp)
                        positions_df.loc[ticket_index, 'new_sl'].value = price_open + new_sl
                        positions_df.loc[ticket_index, 'new_tp'].value = price_open - new_tp
                        print('i2')
                    
                    elif min_value < price_open - 0.764*tp or profit >= 1300*pos_volume*100:
                        #SL se lleva a 61,8%
                        print("Se alcanza el 76,4% de tp para ticket: " + str(ticket))
                        new_sl = 0.618*(price_open - current)
                        new_tp = tp*1.5
                        modify_sltp(position, symbol, price_open - new_sl, price_open - new_tp)
                        positions_df.loc[ticket_index, 'new_sl'].value = price_open + new_sl
                        positions_df.loc[ticket_index, 'new_tp'].value = price_open - new_tp
                        print('i')
                        
                    elif min_value < price_open - 0.618*tp or profit >= 1000*pos_volume*100:
                        #SL se lleva a 38,2%
                        print("Se alcanza el 61,8% de tp para ticket: " + str(ticket))
                        new_sl = 0.382*(price_open - current)
                        new_tp = tp
                        modify_sltp(position, symbol, price_open - new_sl, price_open - new_tp)
                        positions_df.loc[ticket_index, 'new_sl'].value = price_open + new_sl
                        positions_df.loc[ticket_index, 'new_tp'].value = price_open - new_tp
                        print('j')
                        
                    elif profit >= 600*pos_volume*100:
                        #SL se lleva a 38,2%
                        print("Se alcanzan 600 de profit para ticket: " + str(ticket))
                        new_sl = 0.1*(price_open - current)
                        new_tp = tp
                        modify_sltp(position, symbol, price_open - new_sl, price_open - new_tp)
                        positions_df.loc[ticket_index, 'new_sl'].value = price_open + new_sl
                        positions_df.loc[ticket_index, 'new_tp'].value = price_open - new_tp
                        print('k')
                        
                    elif profit >= 400*pos_volume*100:
                        #SL se lleva a 38,2%
                        print("Se alcanzan 400 de profit para ticket: " + str(ticket))
                        new_sl = 0.5*sl
                        new_tp = tp
                        modify_sltp(position, symbol, price_open + new_sl, price_open - new_tp)
                        positions_df.loc[ticket_index, 'new_sl'].value = price_open + new_sl
                        positions_df.loc[ticket_index, 'new_tp'].value = price_open - new_tp
                        print('l')
                        
                    else:
                        #Se asegura al menos un sl  = 0
                        print("Se alcanza el 50% de tp para ticket: " + str(ticket))
                        new_sl = 0.1*(price_open - current)
                        new_tp = tp
                        modify_sltp(position, symbol, price_open - new_sl, price_open - new_tp)
                        positions_df.loc[ticket_index, 'new_sl'].value = price_open + new_sl
                        positions_df.loc[ticket_index, 'new_tp'].value = price_open - new_tp
                        print('m')
                        
            #Estrategia de salida
            if new_max == 1:
                max_value = positions_df.loc[ticket_index, 'max'].values[0]
                sl = positions_df.loc[ticket_index, 'new_sl'].values[0] - price_open
                tp = price_open - positions_df.loc[ticket_index, 'new_tp'].values[0]
                aux = 0.5*tp + price_open
                
                if max_value > price_open + 0.618*sl:
                    
                    #Se disminuye el SL a 76.4%
                    print("Se alcanza el -61,8% de sl para ticket: " + str(ticket))
                    new_sl = 1*sl
                    new_tp = tp
                    modify_sltp(position, symbol, price_open + new_sl, price_open - new_tp)
                    positions_df.loc[ticket_index, 'new_sl'].value = price_open + new_sl
                    positions_df.loc[ticket_index, 'new_tp'].value = price_open - new_tp
                    print('n')
                    
    new_max = 0
    new_min = 0
    if(positions_cant > 1 and margin > 0 and total_profit/margin >= 0.15):
        print(margin,total_profit)
        for position in positions:
            result = mt5.Close(symbol,ticket=position.ticket)
        print('Cierre de posiciones: ',total_profit, f' {total_profit/margin}')
        positions_cant = 0
        
    if (datetime.now().minute == 0 or datetime.now().minute == 15 or datetime.now().minute == 30 or datetime.now().minute == 45) and datetime.now().hour <= 13 and flag_out == 0:
        
        print(datetime.now())
        time.sleep(2)
        #M1 Modelo Datos Macro Up#########################################################################################################
        model_name = 'm1_Up'
        merged_macro_up = dpcl.data_macro(start_date, end_date, timeframe)

        #Etiqueta Para cada ventana móvil de tamaño n
        merged_macro_up = dpcl.etiquetas_up(merged_macro_up, n, fut, umbral) 

        #Generar predicciones con Modelo 1

        #Ventana movil datos nuevos M1
        X_windows_pred = []
        features = merged_macro_up.columns[1:-1]

        for i in tqdm(range(n-1, len(merged_macro_up))):
            
            window = merged_macro_up.iloc[i-n+1:i+1].copy() # Crear ventana móvil
            window.loc[window.index[-26:], 'chikou_span USDIDX'] = 0

            # Aplicar scaler a la ventana móvil
            window_scaled = scaler.fit_transform(window[features].values)
            X_windows_pred.append(window_scaled)

        # Convertir la lista a un array numpy
        X_windows_pred = np.array(X_windows_pred)

        model1_up = load_model('m1_up.h5')

        # Realizar predicciones con el modelo cargado
        predictions = model1_up.predict(X_windows_pred)

        # Convertir el array numpy a un DataFrame de pandas
        predictions_df = pd.DataFrame(predictions, columns=['ProbC0 ' + model_name, 'ProbC1 ' + model_name])

        # Crear un DataFrame con 200 filas de ceros y las mismas columnas que predictions_df
        zeros_df = pd.DataFrame(0, index=np.arange(n-1), columns=predictions_df.columns)

        # Concatenar zeros_df y predictions_df
        predictions_df = pd.concat([zeros_df, predictions_df], ignore_index=True)

        # Añadir las predicciones al DataFrame original (X_original)
        merged_macro_up = pd.concat([merged_macro_up, predictions_df], axis=1)

        #M1 Modelo Datos Macro Down#########################################################################################################
        model_name = 'm1_Down'
        merged_macro_down = dpcl.data_macro(start_date, end_date, timeframe)

        #Etiqueta Para cada ventana móvil de tamaño n
        merged_macro_down = dpcl.etiquetas_down(merged_macro_down, n, fut, umbral) 
         
        #Ventana movil datos nuevos M1 Down
        X_windows_pred = []
        features = merged_macro_down.columns[1:-1]

        for i in tqdm(range(n-1, len(merged_macro_down))):
            
            window = merged_macro_down.iloc[i-n+1:i+1].copy() # Crear ventana móvil
            window.loc[window.index[-26:], 'chikou_span USDIDX'] = 0

            # Aplicar scaler a la ventana móvil
            window_scaled = scaler.fit_transform(window[features].values)
            X_windows_pred.append(window_scaled)

        # Convertir la lista a un array numpy
        X_windows_pred = np.array(X_windows_pred)

        model1_down = load_model('m1_down.h5')

        # Realizar predicciones con el modelo cargado
        predictions = model1_down.predict(X_windows_pred)

        # Convertir el array numpy a un DataFrame de pandas
        predictions_df = pd.DataFrame(predictions, columns=['ProbC0 ' + model_name, 'ProbC1 ' + model_name])

        # Crear un DataFrame con 200 filas de ceros y las mismas columnas que predictions_df
        zeros_df = pd.DataFrame(0, index=np.arange(n-1), columns=predictions_df.columns)

        # Concatenar zeros_df y predictions_df
        predictions_df = pd.concat([zeros_df, predictions_df], ignore_index=True)

        # Añadir las predicciones al DataFrame original (X_original)
        merged_macro_down = pd.concat([merged_macro_down, predictions_df], axis=1)

        #M2 Modelo Datos Tech ###########################################################################################################
        model_name = 'm2_Up'
        merged_tech_aux = dpcl.data_tech(start_date-delta, end_date, symbol, timeframe)
        merged_tech_up = merged_tech_aux[merged_tech_aux['time'] >= start_date].copy().reset_index(drop = True)

        #Etiqueta Para cada ventana móvil de tamaño n
        merged_tech_up = dpcl.etiquetas_up(merged_tech_up, n, fut, umbral) 

        X_windows_pred = []
        features = merged_tech_up.columns[1:-1]

        for i in tqdm(range(n-1, len(merged_tech_up))):
            
            window = merged_tech_up.iloc[i-n+1:i+1].copy() # Crear ventana móvil
            window.loc[window.index[-26:], 'chikou_span USDCLP'] = 0

            # Aplicar scaler a la ventana móvil
            window_scaled = scaler.fit_transform(window[features].values)
            X_windows_pred.append(window_scaled)

        # Convertir la lista a un array numpy
        X_windows_pred = np.array(X_windows_pred)

        model2_up = load_model('m2_up.h5')

        # Realizar predicciones con el modelo cargado
        predictions = model2_up.predict(X_windows_pred)

        # Convertir el array numpy a un DataFrame de pandas
        predictions_df = pd.DataFrame(predictions, columns=['ProbC0 ' + model_name, 'ProbC1 ' + model_name])

        # Crear un DataFrame con 200 filas de ceros y las mismas columnas que predictions_df
        zeros_df = pd.DataFrame(0, index=np.arange(n-1), columns=predictions_df.columns)

        # Concatenar zeros_df y predictions_df
        predictions_df = pd.concat([zeros_df, predictions_df], ignore_index=True)

        # Añadir las predicciones al DataFrame original (X_original)
        merged_tech_up = pd.concat([merged_tech_up, predictions_df], axis=1)

        #M2 Down Modelo Datos Tech ###########################################################################################################
        model_name = 'm2_Down'
        merged_tech_aux = dpcl.data_tech(start_date-delta, end_date, symbol, timeframe)
        merged_tech_down = merged_tech_aux[merged_tech_aux['time'] >= start_date].copy().reset_index(drop = True)

        #Etiqueta Para cada ventana móvil de tamaño n
        merged_tech_down = dpcl.etiquetas_down(merged_tech_down, n, fut, umbral) 

        X_windows_pred = []
        features = merged_tech_down.columns[1:-1]

        for i in tqdm(range(n-1, len(merged_tech_down))):
            
            window = merged_tech_down.iloc[i-n+1:i+1].copy() # Crear ventana móvil
            window.loc[window.index[-26:], 'chikou_span USDCLP'] = 0

            # Aplicar scaler a la ventana móvil
            window_scaled = scaler.fit_transform(window[features].values)
            X_windows_pred.append(window_scaled)

        # Convertir la lista a un array numpy
        X_windows_pred = np.array(X_windows_pred)

        model2_down = load_model('m2_down.h5')

        # Realizar predicciones con el modelo cargado
        predictions = model2_down.predict(X_windows_pred)

        # Convertir el array numpy a un DataFrame de pandas
        predictions_df = pd.DataFrame(predictions, columns=['ProbC0 ' + model_name, 'ProbC1 ' + model_name])

        # Crear un DataFrame con 200 filas de ceros y las mismas columnas que predictions_df
        zeros_df = pd.DataFrame(0, index=np.arange(n-1), columns=predictions_df.columns)

        # Concatenar zeros_df y predictions_df
        predictions_df = pd.concat([zeros_df, predictions_df], ignore_index=True)

        # Añadir las predicciones al DataFrame original (X_original)
        merged_tech_down = pd.concat([merged_tech_down, predictions_df], axis=1)

        #Modelo Final ##############################################################################################################
        merged_final_aux = dpcl.data_tech(start_date-delta, end_date, symbol, timeframe)
        merged_final = merged_final_aux[merged_final_aux['time'] >= start_date].copy().reset_index(drop = True)
        merged_macro_up = merged_macro_up[['time','ProbC1 m1_Up']].copy()
        merged_tech_up = merged_tech_up[['time','ProbC1 m2_Up']].copy()
        merged_macro_down = merged_macro_down[['time','ProbC1 m1_Down']].copy()
        merged_tech_down = merged_tech_down[['time','ProbC1 m2_Down']].copy()
        merged_final = pd.merge(merged_final, merged_macro_up, on = 'time', how='inner')
        merged_final = pd.merge(merged_final, merged_tech_up, on = 'time', how='inner')
        merged_final = pd.merge(merged_final, merged_macro_down, on = 'time', how='inner')
        merged_final = pd.merge(merged_final, merged_tech_down, on = 'time', how='inner')

        #Etiqueta Para cada ventana móvil de tamaño n
        merged_final = dpcl.etiquetas_up(merged_final, n, fut, umbral)

        X_windows_pred = []
        features = merged_final.columns[1:-1]

        for i in tqdm(range(n-1, len(merged_final))):
            
            window = merged_final.iloc[i-n+1:i+1].copy() # Crear ventana móvil
            window.loc[window.index[-26:], 'chikou_span USDCLP'] = 0

            # Aplicar scaler a la ventana móvil
            window_scaled = scaler.fit_transform(window[features].values)
            X_windows_pred.append(window_scaled)

        # Convertir la lista a un array numpy
        X_windows_pred = np.array(X_windows_pred)

        model3 = load_model('m3_final.h5')

        # Realizar predicciones con el modelo cargado
        predictions = model3.predict(X_windows_pred)
        model_name = 'm3'
        # Convertir el array numpy a un DataFrame de pandas
        predictions_df = pd.DataFrame(predictions, columns=['ProbC0 ' + model_name, 'ProbC1 ' + model_name])

        # Crear un DataFrame con 200 filas de ceros y las mismas columnas que predictions_df
        zeros_df = pd.DataFrame(0, index=np.arange(n-1), columns=predictions_df.columns)

        # Concatenar zeros_df y predictions_df
        predictions_df = pd.concat([zeros_df, predictions_df], ignore_index=True)
        
        print(predictions_df.tail(5))
        
        #Etiqueta Para cada ventana móvil de tamaño n
        merged_final = dpcl.etiquetas_down(merged_final, n, fut, umbral)

        X_windows_pred = []
        features = merged_final.columns[1:-1]

        for i in tqdm(range(n-1, len(merged_final))):
            
            window = merged_final.iloc[i-n+1:i+1].copy() # Crear ventana móvil
            window.loc[window.index[-26:], 'chikou_span USDCLP'] = 0

            # Aplicar scaler a la ventana móvil
            window_scaled = scaler.fit_transform(window[features].values)
            X_windows_pred.append(window_scaled)

        # Convertir la lista a un array numpy
        X_windows_pred = np.array(X_windows_pred)

        model4 = load_model('m3_final_down.h5')

        # Realizar predicciones con el modelo cargado
        predictions = model4.predict(X_windows_pred)
        model_name = 'm3_down'
        # Convertir el array numpy a un DataFrame de pandas
        predictions_df_down = pd.DataFrame(predictions, columns=['ProbC0 ' + model_name, 'ProbC1 ' + model_name])

        # Crear un DataFrame con 200 filas de ceros y las mismas columnas que predictions_df
        zeros_df = pd.DataFrame(0, index=np.arange(n-1), columns=predictions_df_down.columns)

        # Concatenar zeros_df y predictions_df
        predictions_df_down = pd.concat([zeros_df, predictions_df_down], ignore_index=True)
        
        print(predictions_df_down.tail(5))
        
        
        tp = (merged_final['BBUp USDCLP'].iloc[-1] - merged_final['USDCLP'].iloc[-1])*0.7
        sl = (merged_final['BBMid USDCLP'].iloc[-1] - merged_final['BBLow USDCLP'].iloc[-1])*0.7
        new_range = volume/min_vol
        print(tp, sl)
        if (tp > 1 and 
            predictions_df['ProbC1 m3'].iloc[-1] > 0.5 and
            predictions_df['ProbC1 m3'].iloc[-1] > predictions_df_down['ProbC1 m3_down'].iloc[-1] and
            predictions_df['ProbC1 m3'].iloc[-1] - predictions_df_down['ProbC1 m3_down'].iloc[-1] > 0.1 and
            ((positions_cant > 0 and (margin_level >= min_margin or margin == 0)) or positions_cant == 0)):
            # Establece la información de la orden de compra
            price = mt5.symbol_info_tick(symbol).ask  # Precio de compra actual     
            # Configura la estructura de la orden
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "type": mt5.ORDER_TYPE_BUY,
                "volume": min_vol,
                "price": price,
                "tp": price + tp,
                "sl": price - sl,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
                "deviation": 5,  # Desviación permitida en el precio
                "magic": 123456,  # Número mágico para identificar la orden
                "comment": "Up {:.2%}".format(predictions_df['ProbC1 m3'].iloc[-1]) ,
                }
    
            for i in range(0,int(new_range)):
                # Envía la orden de compra
                result = mt5.order_send(request)
                print(f"Se ejecuta compra por {price}")
            
        tp = (merged_final['USDCLP'].iloc[-1] - merged_final['BBLow USDCLP'].iloc[-1])*0.7
        sl = (merged_final['BBUp USDCLP'].iloc[-1] - merged_final['BBMid USDCLP'].iloc[-1])*0.7
        print(tp, sl)
        if (tp > 1 and 
            predictions_df_down['ProbC1 m3_down'].iloc[-1] > 0.5 and
            predictions_df_down['ProbC1 m3_down'].iloc[-1] > predictions_df['ProbC1 m3'].iloc[-1] and
            predictions_df_down['ProbC1 m3_down'].iloc[-1] - predictions_df['ProbC1 m3'].iloc[-1] > 0.1 and
            ((positions_cant > 0 and (margin_level >= min_margin or margin == 0)) or positions_cant == 0)):
            # Establece la información de la orden de compra
            price = mt5.symbol_info_tick(symbol).bid  # Precio de compra actual     
            # Configura la estructura de la orden
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "type": mt5.ORDER_TYPE_SELL,
                "volume": min_vol,
                "price": price,
                "tp": price - tp,
                "sl": price + sl,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
                "deviation": 5,  # Desviación permitida en el precio
                "magic": 123456,  # Número mágico para identificar la orden
                "comment": "Down {:.2%}".format(predictions_df_down['ProbC1 m3_down'].iloc[-1]) ,
                }
    
            for i in range(0,int(new_range)):
                # Envía la orden de compra
                result = mt5.order_send(request)
                print(f"Se ejecuta venta por {price}")
        
        del model1_up, model1_down, model2_up, model2_down, model3, model4
        print('Se eliminan modelos')
        flag_out = 1
        print(datetime.now())
        
    if (datetime.now().minute == 1 or datetime.now().minute == 16 or datetime.now().minute == 31 or datetime.now().minute == 46):
        flag_out = 0
        


    


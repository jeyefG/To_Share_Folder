# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 10:06:59 2025

@author: jfcog

Genera los niveles de Taylor y VWAP para n_días. Luego filtra los días en que no 
hay tick_volume

Luego se identifican los eventos de toque en las zonas de taylor (4 valores), 
si se está en la zona de intersección, y si se tocan las líneas de VWAP

Se definen las etiquetas en base a si hubo un toque y luego un rebote que 
alcanzara un nivel jerárquico posterior
"""

import MetaTrader5 as mt5
import pandas as pd
import ta
from ta.volatility import AverageTrueRange
from datetime import datetime, timedelta
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración centralizada de símbolos
SYMBOL_CONFIGS = {
    "US500.spot.mg": {"premarket": (15, 30), "expected_gain": 240, "min_dist": 4},
    "USDCLP.mg": {"premarket": (15, 30), "expected_gain": 2500, "min_dist": 1.5},
    "XAUUSD.mg": {"premarket": (5, 0), "expected_gain": 7000, "min_dist": 3},
    "EURUSD.mg": {"premarket": (14, 0), "expected_gain": 2000, "min_dist": 0.001},
}



class MT5Connector:
    def __init__(self):
        if not mt5.initialize():
            raise RuntimeError(f"No se pudo conectar a MetaTrader 5: {mt5.last_error()}")

    def shutdown(self):
        mt5.shutdown()

    def obtener_d1(self, symbol, fecha_sesion, num_dias, n=10):
        """Obtiene `n` velas diarias previas a ``fecha_sesion``.

        Esta lógica replica la empleada en ``Taylor_zone_V5_Max_min_VWAP.py``:
        se piden las últimas ``n`` velas D1 y se descarta la vela del día en
        curso si está presente.
        """
        rates_d1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, num_dias, n)
        if rates_d1 is None:
            return None

        df = pd.DataFrame(rates_d1)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        df = df[df.index < fecha_sesion]
        if len(df) < 4:
            return None  # Esto previene el error más arriba
        return df

    def obtener_m5(self, symbol, fecha_sesion):
        rates = mt5.copy_rates_range(
            symbol,
            mt5.TIMEFRAME_M5,
            fecha_sesion - timedelta(hours=4),
            fecha_sesion + timedelta(days=1)
        )
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df[df.index.normalize() == fecha_sesion]

class TaylorVWAPCalculator:
    def __init__(self, vwap_win=14):
        self.vwap_win = vwap_win

    def calcular_zonas_taylor(self, df_d1):
        df_d1['PP'] = (df_d1['high'] + df_d1['low'] + df_d1['close']) / 3
        df_d1['Rally'] = df_d1['high'] - df_d1['low'].shift(1)
        df_d1['Rally_avg'] = df_d1['Rally'].rolling(3).mean()
        RR1 = df_d1.iloc[-1]['low'] + df_d1.iloc[-1]['Rally_avg']
        df_d1['BH'] = df_d1['high'] - df_d1['high'].shift(1)
        df_d1['BH_avg'] = df_d1['BH'].rolling(3).mean()
        RR3 = df_d1.iloc[-1]['high'] + df_d1.iloc[-1]['BH_avg']
        RR2 = df_d1.iloc[-1]['high']
        PP = df_d1.iloc[-1]['PP']
        RR4 = 2 * PP - df_d1.iloc[-1]['low']
        zona_alta = (RR1 + RR2 + RR3 + RR4) / 4

        df_d1['Decline'] = df_d1['high'].shift(1) - df_d1['low']
        df_d1['Decline_avg'] = df_d1['Decline'].rolling(3).mean()
        SS1 = df_d1.iloc[-1]['high'] - df_d1.iloc[-1]['Decline_avg']
        df_d1['BL'] = df_d1['low'].shift(1) - df_d1['low']
        df_d1['BL_avg'] = df_d1['BL'].rolling(3).mean()
        SS3 = df_d1.iloc[-1]['low'] - df_d1.iloc[-1]['BL_avg']
        SS2 = df_d1.iloc[-1]['low']
        SS4 = 2 * PP - df_d1.iloc[-1]['high']
        zona_baja = (SS1 + SS2 + SS3 + SS4) / 4

        return zona_baja, zona_alta, zona_alta - zona_baja

    def calcular_vwap(self, df):
        df['typical'] = df[['high', 'low', 'close']].mean(axis=1)
        df['HH'] = df['high'].rolling(self.vwap_win, min_periods=1).max()
        df['LL'] = df['low'].rolling(self.vwap_win, min_periods=1).min()
        df['HV'] = df['tick_volume'].rolling(self.vwap_win, min_periods=1).max()
        df['pivot_price'] = (df['HH'] + df['LL'] + df['close']) / 3.0
        df['pivot_vol'] = df['HV']
        df['cum_pv'] = df['pivot_price'] * df['pivot_vol']
        df['vwap'] = df['cum_pv'].cumsum() / df['pivot_vol'].cumsum()
        df['cum_p2v'] = (df['pivot_price'] ** 2 * df['pivot_vol'])
        df['sigma'] = ((df['cum_p2v'].cumsum() / df['pivot_vol'].cumsum()) - df['vwap'] ** 2).pow(0.5)
        df['vwap_hi'] = df['vwap'] + 2 * df['sigma']
        df['vwap_lo'] = df['vwap'] - 2 * df['sigma']
        return df
    
class FeatureEnricher:
    def __init__(self, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9, ema_periods=[5, 14, 50, 200]):
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.ema_periods = ema_periods

    def enriquecer(self, df):
        df = df.copy()

        # Triggers de toque en t-1
        zonas = ['buy_low', 'buy_high', 'sell_low', 'sell_high', 'vwap_lo', 'vwap_hi']
        for zona in zonas:
            if 'high' in zona:
                df[f'toque_{zona}'] = df['high'].shift(1) >= df[zona].shift(1)
            else:
                df[f'toque_{zona}'] = df['low'].shift(1) <= df[zona].shift(1)

        df['toque_vwap'] = df['low'].shift(1) <= df['vwap'].shift(1)
        df['en_interseccion'] = (
            (df['high'].shift(1) >= df['interseccion_low'].shift(1)) &
            (df['low'].shift(1) <= df['interseccion_high'].shift(1))
        )

        # Precios previos para cálculo de cruces fuera de entrenamiento
        df['vwap_prev'] = df['vwap'].shift(1)
        df['close_prev'] = df['close'].shift(1)

        # Distancias
        df['above_vwap'] = df['close'] > df['vwap']
        df['dist_to_vwap'] = abs(df['close'] - df['vwap'])
        df['dist_to_buy_low'] = df['close'] - df['buy_low']
        df['dist_to_sell_high'] = df['sell_high'] - df['close']
        df['dist_to_buy_high'] = df['close'] - df['buy_high']
        df['dist_to_sell_low'] = df['sell_low'] - df['close']

        # Indicadores técnicos
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=self.rsi_period).rsi()
        macd = ta.trend.MACD(df['close'], window_slow=self.macd_slow, window_fast=self.macd_fast, window_sign=self.macd_signal)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        for p in self.ema_periods:
            df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()

        df['vol_surge'] = df['tick_volume'] / df['tick_volume'].rolling(14).mean()
        df['momentum_3'] = df['close'].pct_change(3)
        df['momentum_5'] = df['close'].pct_change(5)
        df['hora_normalizada'] = df.index.hour + df.index.minute / 60
        
        # Nuevos indicadores experimentales
        
        # 1. Velocidad del precio (cambio por unidad de tiempo)
        df['velocity'] = df['close'].diff() / df['hora_normalizada'].diff()
        
        # 2. Pendiente del VWAP (slope VWAP en últimas 3 velas)
        #df['vwap_slope_3'] = df['vwap'].diff(3) / 3
        
        # 4. Tendencia reciente del VWAP (promedio móvil del slope)
        #df['vwap_slope_ema'] = df['vwap'].diff().ewm(span=5).mean()
        
        # 5. Señales tipo price action
        """
        rango = df['high'] - df['low']
        cuerpo = abs(df['close'] - df['open'])
        
        df['body_ratio'] = cuerpo / rango.replace(0, 1e-9)
        df['upper_wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / rango.replace(0, 1e-9)
        df['lower_wick_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / rango.replace(0, 1e-9)
        
        # Gaps y breakout
        df['gap_up'] = df['open'] - df['close'].shift(1)
        df['gap_down'] = df['close'].shift(1) - df['open']
        df['close_breaks_prev_high'] = (df['close'] > df['high'].shift(1)).astype(int)
        df['close_breaks_prev_low'] = (df['close'] < df['low'].shift(1)).astype(int)
        
        # Secuencias de altos y bajos crecientes/decrecientes
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_high'] = (df['high'] < df['high'].shift(1)).astype(int)
        df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        """
        
        # EMA del RSI
        df['rsi_ema'] = df['rsi'].ewm(span=5).mean()
        df['rsi_vs_ema'] = df['rsi'] - df['rsi_ema']
        df['rsi_slope'] = df['rsi'].diff()
        
        # ATR
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=(min(len(df),14)), fillna=False)
        df['atr_14'] = atr.average_true_range()
                
        # --- Vecindades respecto a niveles críticos ---
        
        ancho_zona = abs(df['zona_alta'].iloc[0] - df['zona_baja'].iloc[0])

        def _vecindad(close, nivel, sigma, ancho_zona=None, es_taylor=False, direction=None):
            if es_taylor:
                umbral = 0.17 * ancho_zona
            else:
                umbral = 0.2 * sigma
        
            base = (close - nivel).abs() <= umbral
        
            if direction == 'support':
                base |= close < (nivel - umbral)
            elif direction == 'resistance':
                base |= close > (nivel + umbral)
        
            return base.astype(int)

        niveles_vec = {
            'buy_low': ('support', True),         # extremo inferior: sí aplica dirección
            'buy_high': (None, True),             # zona intermedia: sin dirección
            'sell_low': (None, True),             # zona intermedia: sin dirección
            'sell_high': ('resistance', True),    # extremo superior: sí aplica dirección
            'vwap_lo': ('support', False),
            'vwap_hi': ('resistance', False),
            'ema_50': (None, False),
            'ema_200': (None, False),
        }

        for nivel, (direction, es_taylor) in niveles_vec.items():
            ancho = ancho_zona if es_taylor else None
            df[f'vecindad_{nivel}'] = _vecindad(
                df['close'], df[nivel], df['sigma'], ancho, es_taylor, direction
            )
            df[f'vecindad_persist_{nivel}'] = (
                df[f'vecindad_{nivel}'].rolling(5, min_periods=1).sum() >= 3
            ).astype(int)

        df['vecindad_acumulada'] = df[[f'vecindad_{n}' for n in niveles_vec]].sum(axis=1)
        
        
        return df

class LabelGenerator:
    def __init__(self, n_ahead=24, umbral_rebote=0.8):
        self.n_ahead = n_ahead
        self.umbral_rebote = umbral_rebote

    def generar_etiquetas_cruce_y_rebote(self, df, niveles, min_dist=0):
        etiquetas = pd.DataFrame(index=df.index)

        for nivel in niveles:
            rebote_label = f"etiqueta_rebote_vwap_{nivel}"
            escape_label = f"etiqueta_escape_tendencia_{nivel}"
            etiquetas[rebote_label] = 0
            etiquetas[escape_label] = 0

            vecs = df[df[f"vecindad_{nivel}"] == 1]

            for idx in vecs.index:
                if idx not in df.index:
                    continue
                actual_idx = df.index.get_loc(idx)
                if actual_idx + self.n_ahead >= len(df):
                    continue

                sub_df = df.iloc[actual_idx + 1 : actual_idx + 1 + self.n_ahead]
                close_ini = df.loc[idx, 'close']
                vwap_ini = df.loc[idx, 'vwap']
                distancia_vwap = abs(vwap_ini - close_ini)
                if distancia_vwap < min_dist:
                    continue

                # Tendencia de corto plazo
                if 'ema_50' in df.columns and 'ema_200' in df.columns:
                    tendencia_alcista = df.loc[idx, 'ema_50'] > df.loc[idx, 'ema_200']
                else:
                    tendencia_alcista = df.loc[idx, 'vwap_slope_ema'] > 0

                if (sub_df['close'] - sub_df['vwap']).abs().min() == 0 or (
                    (sub_df['close'] > sub_df['vwap']).any() and close_ini < vwap_ini
                ) or (
                    (sub_df['close'] < sub_df['vwap']).any() and close_ini > vwap_ini
                ):
                    etiquetas.at[idx, rebote_label] = 1
                else:
                    if close_ini > vwap_ini:
                        avance = sub_df['close'].max() - close_ini
                        movimiento_alcista = True
                    else:
                        avance = close_ini - sub_df['close'].min()
                        movimiento_alcista = False

                    if avance >= self.umbral_rebote * distancia_vwap and (
                        (movimiento_alcista and tendencia_alcista)
                        or (not movimiento_alcista and not tendencia_alcista)
                    ):
                        etiquetas.at[idx, escape_label] = 1

        return etiquetas

calculator = TaylorVWAPCalculator(vwap_win=14)

class SesionProcessor:
    def __init__(self, symbol, fecha_sesion, apertura_mq, df_d1, rates_m5, calculator, label_generator):
        self.symbol = symbol
        self.fecha_sesion = fecha_sesion
        self.apertura_mq = apertura_mq
        self.df_d1 = df_d1
        self.rates_m5 = rates_m5
        self.calculator = calculator
        self.label_generator = label_generator

    def procesar(self):
        df = self.rates_m5.copy()
        if df.empty:
            return None

        zona_baja, zona_alta, _ = self.calculator.calcular_zonas_taylor(self.df_d1)
        ancho_zona = zona_alta - zona_baja
        df = self.calculator.calcular_vwap(df)
        df_premarket = df[df.index < self.apertura_mq]
        if df_premarket.empty:
            return None

        precio_min = df_premarket['low'].min()
        precio_max = df_premarket['high'].max()
        buy_low = precio_min
        buy_high = buy_low + ancho_zona
        sell_high = precio_max
        sell_low = sell_high - ancho_zona
        interseccion_low = max(buy_low, sell_low)
        interseccion_high = min(buy_high, sell_high)
        

        df['symbol'] = self.symbol
        df['fecha'] = self.fecha_sesion.date()
        df['buy_low'] = buy_low
        df['buy_high'] = buy_high
        df['sell_low'] = sell_low
        df['sell_high'] = sell_high
        df['interseccion_low'] = interseccion_low
        df['interseccion_high'] = interseccion_high

        #zona_baja, zona_alta, _ = self.calculator.calcular_zonas_taylor(self.df_d1)
        df['zona_baja'] = zona_baja
        df['zona_alta'] = zona_alta
        
        df['en_premarket'] = df.index < self.apertura_mq

        return df
    
class DatasetBuilder:
    def __init__(self, connector, calculator, enricher, label_generator):
        self.connector = connector
        self.calculator = calculator
        self.enricher = enricher
        self.label_generator = label_generator

    def procesar_sesion(self, symbol, fecha_sesion, apertura_mq, num_dias, min_dist=0):
        df_d1 = self.connector.obtener_d1(symbol, fecha_sesion, num_dias)
        if df_d1 is None:
            print(f"[{symbol} - {fecha_sesion}] ❌ df_d1 no disponible")
            return None

        rates_m5 = self.connector.obtener_m5(symbol, fecha_sesion)
        if rates_m5 is None:
            #print(f"[{symbol} - {fecha_sesion}] ❌ M5 no disponible")
            return None

        procesador = SesionProcessor(
            symbol, fecha_sesion, apertura_mq,
            df_d1, rates_m5,
            self.calculator, self.label_generator
        )

        df_base = procesador.procesar()
        if df_base is None:
            return None

        df_enriquecido = self.enricher.enriquecer(df_base)
        
        niveles = ['buy_low', 'buy_high', 'sell_low', 'sell_high', 'vwap_lo', 'vwap_hi']
        etiquetas = self.label_generator.generar_etiquetas_cruce_y_rebote(df_enriquecido, niveles, min_dist)
        df_final = pd.concat([df_enriquecido, etiquetas], axis=1)
        
        return df_final

class ModelTrainer:
    def __init__(self, df, etiquetas_objetivo, excluir_columnas=[]):
        self.df = df.copy()
        self.etiquetas_objetivo = etiquetas_objetivo
        self.excluir_columnas = excluir_columnas
        self.modelos = {}
        self.reportes = {}
        # Almacenan resultados específicos por símbolo
        self.modelos_por_symbol = {}
        self.reportes_por_symbol = {}

    def _seleccionar_features(self, target):
        etiquetas = [col for col in self.df.columns if col.startswith('etiqueta_')]

        # Exclusiones base
        excluidas = ['symbol', 'fecha', 'time'] + etiquetas + self.excluir_columnas
    
        # Evitar leakage por flags (aunque se calculen en t-1)
        excluidas += [col for col in self.df.columns if col.startswith('toque_')]

        # Excluir vecindad puntual para evitar leakage (mantener persistencia y acumulada)
        for col in self.df.columns:
            if col.startswith('vecindad_') and not col.startswith('vecindad_persist') and col != 'vecindad_acumulada':
                excluidas.append(col)
    
        # También puedes excluir otros flags específicos si los tuvieras
        # excluidas += [col for col in self.df.columns if col.startswith('cruce_') or 'rebote_' in col]
    
        features = [col for col in self.df.columns if col not in excluidas]
        return features

    def _split_por_dias(self, df, features, target, proporcion=0.8):
        """Divide el DataFrame en conjuntos de entrenamiento y prueba por días."""
        dias = df['fecha'].drop_duplicates().sort_values()
        if len(dias) < 2:
            return df[features], df[features], df[target], df[target]

        split_idx = int(proporcion * len(dias))
        dias_entrenamiento = dias.iloc[:split_idx]

        mask_train = df['fecha'].isin(dias_entrenamiento)
        X_train = df.loc[mask_train, features]
        X_test = df.loc[~mask_train, features]
        y_train = df.loc[mask_train, target]
        y_test = df.loc[~mask_train, target]

        return X_train, X_test, y_train, y_test
    
    def simular_rentabilidad(self, etiqueta, umbral=0.7, ganancia_unitaria=2.5, perdida_unitaria=0.6):
        if etiqueta not in self.reportes:
            print(f"⚠️ No hay reporte para la etiqueta '{etiqueta}'. ¿Ejecutaste 'entrenar_todos()'?")
            return None

        reporte = self.reportes[etiqueta]
        y_test = reporte['y_test']
        y_prob = reporte['y_prob']

        # Simular entradas solo si prob > umbral
        decisiones = y_prob > umbral
        if decisiones.sum() == 0:
            print(f"⚠️ Ninguna señal superó el umbral de {umbral}.")
            return None

        aciertos = (y_test[decisiones] == 1).sum()
        errores = (y_test[decisiones] == 0).sum()

        ganancia_total = aciertos * ganancia_unitaria - errores * perdida_unitaria
        promedio_por_trade = ganancia_total / (aciertos + errores)

        print(f"\n💰 Simulación para '{etiqueta}' con umbral {umbral}")
        print(f"Operaciones simuladas: {aciertos + errores}")
        print(f"Aciertos: {aciertos}")
        print(f"Errores: {errores}")
        print(f"Ganancia total simulada: {ganancia_total:.2f}")
        print(f"Promedio por operación: {promedio_por_trade:.2f}")

        return {
            "etiqueta": etiqueta,
            "umbral": umbral,
            "aciertos": aciertos,
            "errores": errores,
            "ganancia_total": ganancia_total,
            "promedio_por_operacion": promedio_por_trade,
            "operaciones": aciertos + errores
        }

    def evaluar_todos_por_rentabilidad(self, umbrales=[0.65, 0.7, 0.75, 0.8], ganancia_unitaria=2.5, perdida_unitaria=0.6):
        resultados = []

        for etiqueta in self.etiquetas_objetivo:
            if etiqueta not in self.reportes:
                print(f"⚠️ No hay modelo entrenado para '{etiqueta}'")
                continue

            mejor = None
            for u in umbrales:
                resultado = self.simular_rentabilidad(etiqueta, umbral=u, ganancia_unitaria=ganancia_unitaria, perdida_unitaria=perdida_unitaria)
                if resultado is not None:
                    if mejor is None or resultado['ganancia_total'] > mejor['ganancia_total']:
                        mejor = resultado

            if mejor:
                resultados.append(mejor)

        # Ordenar por ganancia total
        df_resultados = pd.DataFrame(resultados).sort_values(by="ganancia_total", ascending=False)
        print("\n📊 Ranking por ganancia total:")
        print(df_resultados[['etiqueta', 'umbral', 'ganancia_total', 'operaciones', 'promedio_por_operacion']])

        return df_resultados
    
    def entrenar_todos(self, verbose=True, plot=True):
        for etiqueta in self.etiquetas_objetivo:
            if verbose:
                print(f"\n📈 Entrenando para: {etiqueta}")
            if etiqueta not in self.df.columns:
                print(f"⚠️ Etiqueta {etiqueta} no encontrada en DataFrame.")
                continue

            features = self._seleccionar_features(etiqueta)
            X_train, X_test, y_train, y_test = self._split_por_dias(
                self.df, features, etiqueta
            )

            model = LGBMClassifier(random_state=42, class_weight='balanced', verbose = -1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            if verbose:
                print(classification_report(y_test, y_pred))
        
            if plot:
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f"Matriz de Confusión: {etiqueta}")
                plt.xlabel("Predicción")
                plt.ylabel("Real")
                plt.show()

            self.modelos[etiqueta] = model
            self.reportes[etiqueta] = {
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
        }

    def entrenar_por_symbol(self, verbose=True, plot=True):
        """Entrena modelos individualmente para cada símbolo."""
        self.modelos_por_symbol = {}
        self.reportes_por_symbol = {}
    
        for symbol in self.df['symbol'].unique():
            df_sym = self.df[self.df['symbol'] == symbol]
            self.modelos_por_symbol[symbol] = {}
            self.reportes_por_symbol[symbol] = {}
    
            for etiqueta in self.etiquetas_objetivo:
                if verbose:
                    print(f"\n📈 Entrenando {symbol} para: {etiqueta}")
                if etiqueta not in df_sym.columns:
                    print(f"⚠️ Etiqueta {etiqueta} no encontrada en DataFrame.")
                    continue
    
                features = self._seleccionar_features(etiqueta)

                if len(df_sym['fecha'].unique()) < 2:
                    print(f"⚠️ No hay suficientes días para {symbol} - {etiqueta}")
                    continue

                X_train, X_test, y_train, y_test = self._split_por_dias(
                    df_sym, features, etiqueta
                )
    
                model = LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)
                model.fit(X_train, y_train)
                #y_pred = model.predict(X_test)
                #y_prob = model.predict_proba(X_test)[:, 1]
                y_prob = model.predict_proba(X_test)[:, 1]
                umbral = 0.8
                y_pred = (y_prob >= umbral).astype(int)
    
                if verbose:
                    print(classification_report(y_test, y_pred))
    
                if plot:
                    cm = confusion_matrix(y_test, y_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title(f"Matriz de Confusión: {symbol} - {etiqueta}")
                    plt.xlabel("Predicción")
                    plt.ylabel("Real")
                    plt.show()
    
                self.modelos_por_symbol[symbol][etiqueta] = model
                self.reportes_por_symbol[symbol][etiqueta] = {
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'y_prob': y_prob,
                }
    
    def obtener_modelo(self, etiqueta):
        return self.modelos.get(etiqueta, None)
    
    def obtener_reporte(self, etiqueta):
        return self.reportes.get(etiqueta, None)
    
    def obtener_importancias(self, etiqueta, symbol=None, top_n=10):
        """Devuelve las importancias de features para un modelo entrenado."""
        if symbol:
            model = self.modelos_por_symbol.get(symbol, {}).get(etiqueta)
        else:
            model = self.modelos.get(etiqueta)
    
        if model is None:
            print(f"⚠️ Modelo no encontrado para '{etiqueta}'" + (f" en {symbol}" if symbol else ""))
            return pd.DataFrame(columns=["feature", "importance"])
    
        features = self._seleccionar_features(etiqueta)
        df_imp = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        return df_imp.head(top_n)
    
    def resumen_resultados(self, symbol=None, top_n=5):
        """Genera un DataFrame resumido de métricas y top features."""
        if symbol:
            items = {symbol: self.reportes_por_symbol.get(symbol, {})}.items()
        else:
            items = self.reportes_por_symbol.items()
    
        resumen = []
        for sym, rep_dic in items:
            for etiqueta, datos in rep_dic.items():
                y_true = datos['y_test']
                y_pred = datos['y_pred']
                cm = confusion_matrix(y_true, y_pred)
                if cm.size == 4:
                    tn, fp, fn, tp = cm.ravel()
                else:
                    tn = fp = fn = tp = 0
                accuracy = (tp + tn) / cm.sum() if cm.sum() else 0
                precision = tp / (tp + fp) if (tp + fp) else 0
                recall = tp / (tp + fn) if (tp + fn) else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
                top_feats = self.obtener_importancias(etiqueta, symbol=sym, top_n=top_n)['feature'].tolist()
                resumen.append({
                    'symbol': sym,
                    'etiqueta': etiqueta,
                    'accuracy': round(accuracy, 3),
                    'precision': round(precision, 3),
                    'recall': round(recall, 3),
                    'f1': round(f1, 3),
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'tn': tn,
                    'top_features': ', '.join(top_feats)
                })

        return pd.DataFrame(resumen)
    
if __name__ == "__main__":
    from tqdm import tqdm

    # --- Configuraciones ---
    symbols = list(SYMBOL_CONFIGS.keys())
    vwap_win = 14
    n_dias = 350

    # --- Inicialización de clases ---
    connector = MT5Connector()
    calculator = TaylorVWAPCalculator(vwap_win=vwap_win)
    enricher = FeatureEnricher()
    label_generator = LabelGenerator()
    builder = DatasetBuilder(connector, calculator, enricher, label_generator)

    fecha_base = (datetime.now() + timedelta(hours=6)).replace(hour=0, minute=0, second=0, microsecond=0)
    df_total = []

    for symbol in symbols:
        print(f"Procesando: {symbol}")
        for i in tqdm(range(n_dias), desc=f"{symbol}"):
            fecha_sesion = fecha_base - timedelta(days=i)
            apertura_mq = fecha_sesion.replace(
                hour=SYMBOL_CONFIGS[symbol]["premarket"][0],
                minute=SYMBOL_CONFIGS[symbol]["premarket"][1]
            )

            df = builder.procesar_sesion(symbol, fecha_sesion, apertura_mq, i, SYMBOL_CONFIGS[symbol]["min_dist"])

            if df is not None:
                df_total.append(df)

    if df_total:
        df_concat = pd.concat(df_total)
        df_concat = df_concat[df_concat['tick_volume'] > 0]
        bool_cols = df_concat.select_dtypes(include='bool').columns
        df_concat[bool_cols] = df_concat[bool_cols].astype(int)
        df_concat.sort_values(by=['symbol', 'fecha', 'time'], ascending=[True, True, True], inplace=True)
        #df_concat.to_csv("v5.csv", index=True, sep=';', decimal=',')
        #print("✅ Dataset guardado como 'v5.csv'")
    else:
        print("⚠️ No se generaron datos válidos.")

def revisar_coincidencias_etiquetas_flags(df, verbose=True):
    etiquetas = [col for col in df.columns if col.startswith('etiqueta_')]
    resumen = []

    for etiqueta in etiquetas:
        if 'rebote_vwap_' in etiqueta:
            nivel = etiqueta.replace('etiqueta_rebote_vwap_', '')
        elif 'escape_tendencia_' in etiqueta:
            nivel = etiqueta.replace('etiqueta_escape_tendencia_', '')
        else:
            if verbose:
                print(f"❌ Tipo de etiqueta desconocido o no relevante: {etiqueta}")
            continue

        flag_toque = f'vecindad_{nivel}'
        
        if flag_toque not in df.columns:
            if verbose:
                print(f"⚠️ Flag no encontrado para {etiqueta}: {flag_toque}")
            continue

        df_etiqueta = df[df[etiqueta] == 1]
        total = len(df_etiqueta)
        verdaderas = df_etiqueta[flag_toque].sum()
        porcentaje = verdaderas / total if total > 0 else 0

        resumen.append({
            'etiqueta': etiqueta,
            'flag': flag_toque,
            'total_etiquetas': total,
            'coincidencias': verdaderas,
            'porcentaje': porcentaje
        })

        if verbose:
            print(f"🔍 {etiqueta} → Flag: {flag_toque} | {verdaderas}/{total} coincidencias ({porcentaje:.2%})")

    if resumen:
        return pd.DataFrame(resumen).sort_values(by='porcentaje', ascending=False)
    else:
        print("⚠️ No se generaron coincidencias válidas.")
        return pd.DataFrame(columns=['etiqueta', 'flag', 'total_etiquetas', 'coincidencias', 'porcentaje'])


def plot_vecindad(df, niveles):
    """Grafica precios y marca las velas con vecindad en los niveles dados."""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label='close', color='black')
    plt.plot(df.index, df['vwap'], label='vwap', color='blue')
    if 'vwap_hi' in df:
        plt.plot(df.index, df['vwap_hi'], '--', label='vwap_hi', color='orange')
    if 'vwap_lo' in df:
        plt.plot(df.index, df['vwap_lo'], '--', label='vwap_lo', color='orange')
    if 'ema_50' in df:
        plt.plot(df.index, df['ema_50'], label='ema_50', color='green')
    if 'ema_200' in df:
        plt.plot(df.index, df['ema_200'], label='ema_200', color='red')

    base_levels = {'vwap_hi', 'vwap_lo', 'ema_50', 'ema_200'}

    for nivel in niveles:
        if nivel in df.columns:
            if nivel not in base_levels:
                plt.plot(df.index, df[nivel], linestyle=':', label=nivel)
            mask = df.get(f'vecindad_{nivel}', pd.Series(False, index=df.index)).astype(bool)
            plt.scatter(df.index[mask], df['close'][mask], s=20, label=f'vec_{nivel}')

    plt.legend()
    plt.tight_layout()
    plt.show()

#resumen_coincidencias = revisar_coincidencias_etiquetas_flags(df_concat)


etiquetas = [col for col in df_concat.columns if col.startswith('etiqueta_')]
trainer = ModelTrainer(df_concat, etiquetas)

# Entrenamiento global sin imprimir detalles
#trainer.entrenar_todos(verbose=False, plot=False)
#trainer.evaluar_todos_por_rentabilidad()

# Entrenamiento por símbolo y resumen compacto
trainer.entrenar_por_symbol(verbose=False, plot=False)
df_resumen = trainer.resumen_resultados(top_n=5)

# Ordenar por tp, f1 y precision de mayor a menor
df_resumen = df_resumen.sort_values(by=['tp', 'f1', 'precision'], ascending=False).reset_index(drop=True)

# Asignar expected gain por fila desde la configuración
df_resumen['expected_gain'] = df_resumen['symbol'].map(
    {k: v['expected_gain'] for k, v in SYMBOL_CONFIGS.items()}
)

# Calcular ganancias y pérdidas esperadas
df_resumen['ganancia_tp'] = df_resumen['tp'] * df_resumen['expected_gain']
df_resumen['perdida_fp'] = -df_resumen['fp'] * df_resumen['expected_gain']
df_resumen['resultado_estimado'] = df_resumen['ganancia_tp'] + df_resumen['perdida_fp']
#print("\n=== Resumen por símbolo ===")
#print(df_resumen.to_string(index=False))
df_resumen.to_csv("resultados_nf.csv", index=False, sep=';', decimal=',')


# --- Visualización para inspección de vecindades activadas ---
for symbol in df_concat['symbol'].unique():
    # Seleccionamos un símbolo de interés (puedes cambiarlo manualmente si quieres otro)
    symbol_viz = symbol  # Cambia por cualquier símbolo disponible en tu dataset
    df_viz = df_concat[df_concat['symbol'] == symbol_viz].copy()
    
    # Definimos los niveles que nos interesa visualizar
    niveles_vecindad = ['buy_low', 'buy_high', 'sell_low', 'sell_high',
                        'vwap_lo', 'vwap_hi', 'ema_50', 'ema_200']
    
    # Mostrar últimas 300 velas para mayor claridad
    #print(f"\n🔍 Visualización de vecindades para: {symbol_viz}")
    fecha_base_dt = datetime.combine(fecha_base, datetime.min.time())
    df_viz['fecha'] = pd.to_datetime(df_viz['fecha'], errors='coerce')
    df_viz = df_viz[df_viz['fecha'] >= fecha_base_dt]
    plot_vecindad(df_viz, niveles_vecindad)

# --- Predicciones para la última vela ---
resultados_prediccion = []

for etiqueta in etiquetas:
    features = trainer._seleccionar_features(etiqueta)

    for symbol in symbols:
        df_sym = df_concat[df_concat['symbol'] == symbol]
        if etiqueta not in df_sym.columns or len(df_sym) < 50:
            continue

        X = df_sym[features]
        y = df_sym[etiqueta]

        model = LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)
        model.fit(X[:-1], y[:-1])  # Entrenar sin la última fila

        X_pred = df_sym[features].iloc[[-1]]  # Última fila
        prob = model.predict_proba(X_pred)[:, 1][0]

        resultados_prediccion.append({
            "symbol": symbol,
            "etiqueta": etiqueta,
            "probabilidad": round(prob, 4),
            "distancia": abs(df_sym['vwap'].iloc[-1] - df_sym['close'].iloc[-1])
        })

# Mostrar predicciones ordenadas por mayor probabilidad
df_preds = pd.DataFrame(resultados_prediccion).sort_values(by="probabilidad", ascending=False)
df_preds = df_preds[df_preds['probabilidad'] > 0.1]
    
print("\n📊 Predicciones para la última vela:")
print(df_preds.to_string(index=False))
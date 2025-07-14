# -*- coding: utf-8 -*-
"""
Created on Tue May 20 16:55:29 2025

@author: jfcog
"""

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import MetaTrader5 as mt5
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta

matplotlib.use('Agg')

# ───────────────────── Configuración general ────────────────────────────
#symbols = ["US500.spot.mg", "USDCLP.mg", "USDIDX.mg", "XAUUSD.mg", "COPPER.mg", "VIX.mg", "BTCUSD.mg"]
symbols = ["US500.spot.mg", "USDCLP.mg", "USDIDX.mg", "XAUUSD.mg", "EURUSD.mg","COPPER.mg", "VIX.mg", "BTCUSD.mg", "GBPUSD.mg"]
offsets  = [0] * len(symbols)
timeframe_visual = mt5.TIMEFRAME_M5
n_candles_visual = 500
# ─────────────────────────────────────────────────────────────────────────

# ──────────────── Inicializar MetaTrader 5 ──────────────────────────────
# --- Inicializar MT5 ---
if not mt5.initialize():
    print("Error al inicializar MT5:", mt5.last_error())
    quit()

# Hora servidor simulada (servidor UTC+2 en tu script original)
ahora       = datetime.utcnow() + timedelta(hours=2)
inicio_hoy  = ahora.replace(hour=0, minute=0, second=0, microsecond=0)
apertura_mq = inicio_hoy.replace(hour=15, minute=30)   # 09:30 NY + 2 h

# --- Crear ventana tkinter con scroll ---
root = tk.Tk()
root.title("Gráficos con Envelopes y Scroll")

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=1)

canvas = tk.Canvas(main_frame)
scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# --- Proceso para cada símbolo ---
for symbol, offset in zip(symbols, offsets):
    print(f"\nProcesando: {symbol}")

    # --- Vela diaria ---
    rates_d1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 10)
    if rates_d1 is None or len(rates_d1) < 4:
        print(f"No se pudo obtener suficientes velas diarias para {symbol}")
        continue

    df_d1 = pd.DataFrame(rates_d1)
    df_d1['time'] = pd.to_datetime(df_d1['time'], unit='s')
    df_d1.set_index('time', inplace=True)
    df_d1 = df_d1[df_d1.index < inicio_hoy]

    # --- PP y Envelopes ---
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
    close_val = df_d1.iloc[-1]['close']

    ancho_zona = zona_alta - zona_baja
    
    # --- Datos visuales ---
    rates_visual = mt5.copy_rates_from_pos(symbol, timeframe_visual, 0, n_candles_visual)
    if rates_visual is None or len(rates_visual) == 0:
        print(f"No se pudo obtener datos visuales para {symbol}")
        continue

    df = pd.DataFrame(rates_visual)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df = df[(df.index >= inicio_hoy) & (df.index <= ahora)]

    if df.empty:
        dummy_index = pd.date_range(start=inicio_hoy, periods=5, freq='min')
        df = pd.DataFrame({
            'open': [close_val]*5,
            'high': [close_val]*5,
            'low': [close_val]*5,
            'close': [close_val]*5,
            'tick_volume': [1]*5
        }, index=dummy_index)
        
    #VWAP
    win = 14
    # 2. columna típica de cada vela (H+L+C)/3
    df['typical'] = df[['high', 'low', 'close']].mean(axis=1)

    # 3. para cada barra: highest-high, lowest-low y highest-volume
    df['HH'] = df['high'].rolling(win, min_periods=1).max()
    df['LL'] = df['low'] .rolling(win, min_periods=1).min()
    df['HV'] = df['tick_volume'].rolling(win, min_periods=1).max()

    # 4. precio “pivote” = (HH + LL + C) / 3  (tu paso 2)
    df['pivot_price'] = (df['HH'] + df['LL'] + df['close']) / 3.0

    # 5. volumen “ancla” = HV de la ventana (se repite en toda la fila)
    #    Si prefieres el volumen real de cada vela, cambia por df['vol_tv']
    df['pivot_vol'] = df['HV']

    # 8. VWAP acumulado desde la ancla
    df['cum_pv'] = (df['pivot_price'] * df['pivot_vol'])
    df['vwap']   = (df['cum_pv'].cumsum() /df['pivot_vol'].cumsum())

    # 9. # acumulado de (precio^2 * volumen)
    df['cum_p2v'] = (df['pivot_price']**2 * df['pivot_vol'])

    # σ² = Σp²v / Σv  –  VWAP²   (por grupo anchor)
    cum_p2v   = df['cum_p2v'].cumsum()
    cum_vol   = df['pivot_vol'].cumsum()
    df['sigma'] = ((cum_p2v / cum_vol) - df['vwap']**2).pow(0.5)

    df['vwap_hi'] = df['vwap'] + 2*df['sigma']
    df['vwap_lo'] = df['vwap'] - 2*df['sigma']

    # --- Filtrar velas antes de la apertura del mercado (09:30 horario servidor) ---
    df_apertura = df[(df.index >= inicio_hoy) & (df.index < apertura_mq)]
    
    if df_apertura.empty:
        print(f"Advertencia: No hay velas antes de la apertura para {symbol}")
        precio_min = df['low'].min()
        precio_max = df['high'].max()
    else:
        precio_min = df_apertura['low'].min()
        precio_max = df_apertura['high'].max()
        

    buy_low = precio_min + offset
    buy_high = buy_low + ancho_zona + offset
    sell_high = precio_max + offset
    sell_low = sell_high - ancho_zona + offset

    interseccion_low = max(buy_low, sell_low)
    interseccion_high = min(buy_high, sell_high)

    print(f"buy_high: {buy_high:.5f} | buy_low: {buy_low:.5f} | sell_high: {sell_high:.5f} | sell_low: {sell_low:.5f}")
    # ───── Dibujar gráfico con mplfinance ─────
    add_plots = []
    if 'vwap' in df.columns:
        add_plots += [mpf.make_addplot(df['vwap']   , color='cyan',    width=1)]
        add_plots += [mpf.make_addplot(df['vwap_hi'], color='lime',    linestyle='--', width=0.8)]
        add_plots += [mpf.make_addplot(df['vwap_lo'], color='magenta', linestyle='--', width=0.8)]
    
    fig, axlist = mpf.plot(
        df,
        type='candle',
        style='charles',
        title=symbol,
        ylabel='Precio',
        volume=False,
        mav=(5, 20),
        addplot=add_plots,          # ← aquí va la VWAP y bandas
        returnfig=True
    )

    plt.close(fig)
    ax = axlist[0]

    # Líneas fijas Taylor shift
    ax.axhline(buy_low, color='red', linestyle='--')
    ax.axhline(buy_high, color='green', linestyle='--')
    ax.axhline(sell_high, color='green', linestyle='--')
    ax.axhline(sell_low, color='red', linestyle='--')
    ax.axhline(close_val, color='blue', linestyle='--')

    # Intersección achurada
    if interseccion_low < interseccion_high:
        ax.axhspan(interseccion_low, interseccion_high, facecolor='gray', alpha=0.3)
        
    #ax.legend(loc="center left", fontsize=8)

    canvas_fig = FigureCanvasTkAgg(fig, master=scrollable_frame)
    canvas_fig.draw()
    canvas_fig.get_tk_widget().pack()

matplotlib.use('QtAgg')
mt5.shutdown()
root.mainloop()

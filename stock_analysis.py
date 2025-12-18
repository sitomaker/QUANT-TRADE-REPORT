import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from datetime import datetime, timedelta

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Quant Dashboard Pro", layout="wide", page_icon="📈")

# Estilo CSS Dark Mode
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
</style>
""", unsafe_allow_html=True)

# --- BACKEND ---
@st.cache_data
def get_data(ticker, benchmark='SPY'):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*50)
    
    try:
        # CORRECCIÓN WARNING 1: 'auto_adjust=False' para evitar avisos de yfinance
        data = yf.download([ticker, benchmark], start=start_date, end=end_date, progress=False, auto_adjust=False)
        
        # Limpieza robusta
        if 'Adj Close' in data.columns:
            df = data['Adj Close']
        elif 'Close' in data.columns:
            df = data['Close']
        else:
            return None, None
            
        df = df.dropna()
        # Verificamos que existan ambas columnas
        if ticker not in df.columns or benchmark not in df.columns:
            return None, None
            
        return df[ticker], df[benchmark]
    except Exception as e:
        return None, None

def calculate_metrics(stock_series, bench_series):
    # Retornos logarítmicos
    log_ret_s = np.log(stock_series / stock_series.shift(1)).dropna()
    log_ret_b = np.log(bench_series / bench_series.shift(1)).dropna()
    
    # Alineación de fechas
    idx = log_ret_s.index.intersection(log_ret_b.index)
    log_ret_s = log_ret_s.loc[idx]
    log_ret_b = log_ret_b.loc[idx]
    
    # Cálculos
    mean_ret = log_ret_s.mean() * 252
    volatility = log_ret_s.std() * np.sqrt(252)
    sharpe = mean_ret / volatility if volatility > 0 else 0
    
    covariance = np.cov(log_ret_s, log_ret_b)
    beta = covariance[0, 1] / covariance[1, 1]
    alpha = mean_ret - (beta * (log_ret_b.mean() * 252))
    
    var_95 = np.percentile(log_ret_s, 5)
    cvar_95 = log_ret_s[log_ret_s <= var_95].mean()
    kurtosis = stats.kurtosis(log_ret_s)
    
    cum_ret = (1 + log_ret_s).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()
    
    return {
        "Price": stock_series.iloc[-1],
        "Ret_Year": mean_ret,
        "Vol": volatility,
        "Sharpe": sharpe,
        "Beta": beta,
        "Alpha": alpha,
        "VaR": var_95,
        "CVaR": cvar_95,
        "Max_DD": max_dd,
        "Kurt": kurtosis
    }, log_ret_s, drawdown

def run_monte_carlo(last_price, log_returns, days=252, simulations=1000):
    mu = log_returns.mean()
    sigma = log_returns.std()
    
    paths = np.zeros((days, simulations))
    paths[0] = last_price
    
    for t in range(1, days):
        z = np.random.standard_normal(simulations)
        paths[t] = paths[t-1] * np.exp((mu - 0.5 * sigma**2) + sigma * z)
        
    return paths

# --- FRONTEND ---
st.sidebar.title("🎛️ Panel Quant")
ticker = st.sidebar.text_input("Ticker", value="NVDA").upper()
benchmark = st.sidebar.text_input("Benchmark", value="SPY").upper()

if ticker and benchmark:
    stock_data, bench_data = get_data(ticker, benchmark)
    
    if stock_data is not None:
        metrics, log_ret, dd_series = calculate_metrics(stock_data, bench_data)
        
        st.title(f"📊 Dashboard: {ticker}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precio", f"${metrics['Price']:.2f}")
        c2.metric("CAGR", f"{metrics['Ret_Year']:.2%}")
        c3.metric("Sharpe", f"{metrics['Sharpe']:.2f}")
        c4.metric("Max DD", f"{metrics['Max_DD']:.2%}")

        tab1, tab2, tab3 = st.tabs(["Análisis Técnico", "Métricas Riesgo", "Monte Carlo"])

        with tab1:
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=stock_data.index, y=stock_data, name=ticker, line=dict(color='#00F0FF')))
            fig_p.add_trace(go.Scatter(x=bench_data.index, y=bench_data * (stock_data.iloc[0]/bench_data.iloc[0]), 
                                     name="Benchmark (Norm)", line=dict(color='gray', dash='dash')))
            fig_p.update_layout(template="plotly_dark", title="Precio vs Benchmark", height=500)
            st.plotly_chart(fig_p, use_container_width=True) # CORRECCIÓN WARNING 2

            fig_d = go.Figure()
            fig_d.add_trace(go.Scatter(x=dd_series.index, y=dd_series, fill='tozeroy', line=dict(color='red')))
            fig_d.update_layout(template="plotly_dark", title="Drawdown", height=300)
            st.plotly_chart(fig_d, use_container_width=True)

        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                fig_h = px.histogram(log_ret, nbins=50, title="Distribución de Retornos", color_discrete_sequence=['purple'])
                fig_h.update_layout(template="plotly_dark")
                st.plotly_chart(fig_h, use_container_width=True)
            with c2:
                st.write("### KPIs Institucionales")
                st.markdown(f"""
                - **Alpha:** `{metrics['Alpha']:.2%}`
                - **Beta:** `{metrics['Beta']:.2f}`
                - **Kurtosis:** `{metrics['Kurt']:.2f}` (>3 Riesgo alto)
                - **VaR (95%):** `{metrics['VaR']:.2%}`
                """)

        with tab3:
            dias = st.slider("Días Proyección", 30, 365, 252)
            if st.button("Simular"):
                paths = run_monte_carlo(metrics['Price'], log_ret, days=dias)
                fig_m = go.Figure()
                fig_m.add_trace(go.Scatter(y=paths.mean(axis=1), name='Media', line=dict(color='white', width=2)))
                fig_m.add_trace(go.Scatter(y=np.percentile(paths, 90, axis=1), name='Optimista', line=dict(color='lime', dash='dot')))
                fig_m.add_trace(go.Scatter(y=np.percentile(paths, 10, axis=1), name='Pesimista', line=dict(color='red', dash='dot')))
                # Muestra visual de fondo
                for i in range(30): 
                    fig_m.add_trace(go.Scatter(y=paths[:, i], line=dict(color='cyan', width=0.5), opacity=0.3, showlegend=False))
                
                fig_m.update_layout(template="plotly_dark", title=f"Proyección Monte Carlo ({dias} días)", height=500)
                st.plotly_chart(fig_m, use_container_width=True)

    else:
        st.error("Error cargando datos. Verifica el ticker.")
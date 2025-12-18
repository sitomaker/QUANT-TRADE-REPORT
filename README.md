# QUANT-TRADE-REPORT
This code give's a quantum trade report of the ticket you want.
# 📈 QuantRisk: Dashboard de Análisis Cuantitativo y Simulación Monte Carlo

Este proyecto es una herramienta de análisis financiero de nivel institucional desarrollada en Python. Permite a traders y analistas ir más allá de los gráficos de precios básicos, evaluando métricas de riesgo estadístico y proyecciones probabilísticas.

## 🚀 Funcionalidades Principales

* **Motor de Datos:** Descarga automática de datos ajustados (Yahoo Finance API) y Benchmark (S&P 500).
* **Métricas de Riesgo (Risk Engine):** Cálculo automático de VaR (95%), CVaR, Kurtosis (Riesgo de cola) y Max Drawdown.
* **Performance:** Ratios de Sharpe, Alpha y Beta comparativos.
* **Simulación Estocástica:** Motor de Monte Carlo basado en Movimiento Browniano Geométrico (GBM) para proyectar 1,000 escenarios futuros.
* **Interfaz:** Dashboard web interactivo construido con Streamlit y Plotly.

## 🛠️ Tecnologías

* **Python 3.10+**
* **Streamlit** (Frontend)
* **Plotly** (Visualización GPU-accelerated)
* **Scipy & Numpy** (Cálculo estadístico)

## 💻 Cómo ejecutarlo en tu PC

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/TU_USUARIO/QuantRisk.git](https://github.com/TU_USUARIO/QuantRisk.git)
    cd QuantRisk
    ```

2.  **Instalar dependencias:**
    ```bash
    pip install -r MLLibraries.txt
    ```

3.  **Ejecutar la aplicación:**
    ```bash
    streamlit run stock_analysis.py
    ```

---
Desarrollado por Andrés Míguez Rodríguez

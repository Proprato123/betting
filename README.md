# ⚽ Soccer Quant System

> Sistema cuantitativo escalable para detección de **value bets** en ligas menores de fútbol.

---

## 📋 Tabla de Contenidos

1. [Descripción del Problema](#descripción-del-problema)
2. [Diseño del Sistema](#diseño-del-sistema)
3. [Modelo Matemático](#modelo-matemático)
4. [Estrategia de Validación](#estrategia-de-validación)
5. [Gestión de Riesgo](#gestión-de-riesgo)
6. [Instalación y Uso](#instalación-y-uso)
7. [Estructura del Proyecto](#estructura-del-proyecto)
8. [Limitaciones](#limitaciones)
9. [Roadmap](#roadmap)

---

## 🎯 Descripción del Problema

Las casas de apuestas establecen odds que implican probabilidades para los resultados de partidos de fútbol. Estas probabilidades incluyen un **overround** (margen de la casa) y están influenciadas por el volumen de apuestas del público.

En **ligas menores** (2ª divisiones, ligas escandinavas, ligas sudamericanas menores), los mercados tienden a ser menos eficientes debido a:

- **Menor liquidez** → las odds se ajustan menos al "precio justo"
- **Menor cobertura mediática** → menos información disponible para los apostadores
- **Menos análisis cuantitativo** → oportunidades de edge no explotadas

El sistema busca explotar estas ineficiencias mediante modelos probabilísticos calibrados que estimen la probabilidad real de cada resultado con mayor precisión que las odds del mercado.

---

## 🏗️ Diseño del Sistema

### Arquitectura Modular (POO)

```
soccer_quant_system/
│
├── config/settings.py          # Configuración central
│
├── src/
│   ├── data/
│   │   ├── data_loader.py      # Carga multi-fuente
│   │   ├── data_cleaner.py     # Limpieza y validación
│   │   └── feature_engineer.py # Generación de features
│   │
│   ├── models/
│   │   ├── base_model.py       # Interfaz abstracta
│   │   ├── logistic_model.py   # Regresión logística (baseline)
│   │   ├── xgboost_model.py    # XGBoost + LightGBM + Poisson
│   │   └── calibration.py      # Calibración probabilística
│   │
│   ├── betting/
│   │   ├── odds_converter.py   # Conversión de odds
│   │   ├── value_detector.py   # Detección de edge
│   │   └── staking.py          # Gestión de stakes
│   │
│   ├── backtesting/
│   │   ├── backtester.py       # Walk-forward validation
│   │   └── performance_metrics.py  # Métricas de rendimiento
│   │
│   └── pipeline/
│       └── daily_pipeline.py   # Orquestación diaria
│
├── main.py                     # Punto de entrada CLI
├── dashboard.py                # Dashboard Streamlit
└── requirements.txt
```

### Principios de Diseño

- **Patrón Strategy**: Modelos intercambiables vía `BaseModel`
- **Sin data leakage**: Todas las features son estrictamente backward-looking
- **Separación train/val/test**: Temporal, nunca aleatoria
- **Versionado de modelos**: Cada modelo se guarda con timestamp

---

## 📐 Modelo Matemático

### Features

| Feature | Descripción | Tipo |
|---------|-------------|------|
| `home_goals_avg` | Media de goles anotados (rolling N) | Obligatoria |
| `away_conceded_avg` | Media de goles concedidos (rolling N) | Obligatoria |
| `home_points_last_n` | Puntos acumulados últimos N | Obligatoria |
| `home_goal_diff_last_n` | Diferencia de goles últimos N | Obligatoria |
| `home_elo` / `away_elo` | Elo rating dinámico | Obligatoria |
| `home_rest_days` | Días desde último partido | Obligatoria |
| `home_momentum` | Forma exponencial (α=0.85) | Avanzada |
| `poisson_lambda_home` | λ del modelo Poisson implícito | Avanzada |
| `h2h_home_win_rate` | Historial directo | Avanzada |

### Elo Rating

```
E_home = 1 / (1 + 10^((R_away - R_home - HFA) / 400))
R_new = R_old + K × (S - E)
```

Donde HFA = Home Field Advantage = 100 puntos Elo.

### Modelo Poisson

```
λ_home = attack_strength_home × defense_weakness_away × league_avg
λ_away = attack_strength_away × defense_weakness_home × league_avg

P(Home wins) = Σ Σ P(X=i) × P(Y=j) para i > j
P(Draw)      = Σ P(X=i) × P(Y=i)
P(Away wins) = Σ Σ P(X=i) × P(Y=j) para i < j
```

### Detección de Edge

```
p_market = (1/odds_i) / Σ(1/odds_j)     # Eliminar overround
EV = p_model × odds - 1                   # Expected Value
Apostar si EV > 3%
```

### Calibración

- **Platt Scaling**: Regresión logística sobre probabilidades
- **Isotonic Regression**: Calibración no-paramétrica
- **Temperature Scaling**: T óptimo por minimización de NLL

```
ECE = Σ (|B_m|/n) × |accuracy(B_m) - confidence(B_m)|
```

### Staking (Kelly Criterion)

```
f* = (p × b - q) / b
f_fractional = 0.25 × f*
stake = bankroll × f_fractional
```

---

## 🔬 Estrategia de Validación

### Walk-Forward Validation

```
|-------- Train (200) --------|--- Test (50) ---|
                    |-------- Train (200) --------|--- Test (50) ---|
                                       |-------- Train (200) --------|--- Test (50) ---|
```

- **NO** se usa random split
- El modelo se re-entrena en cada fold
- La calibración se entrena en el último 20% del train
- Métricas se calculan solo en test

### Métricas de Optimización

Se optimiza exclusivamente por:
1. **Log Loss** (cross-entropy)
2. **Brier Score** (error cuadrático de probabilidades)
3. **ECE** (Expected Calibration Error)

**Nunca se optimiza por ROI directamente** (evita overfitting a ruido).

### Monte Carlo

10,000 simulaciones resampleando las apuestas del backtest con reemplazo para obtener distribución de ROI, drawdown y bankroll final.

---

## 💰 Gestión de Riesgo

| Control | Valor |
|---------|-------|
| Kelly Fraction | 0.25 (conservador) |
| Max bet % bankroll | 5% |
| Min EV threshold | 3% |
| Max Drawdown | 30% (stop automático) |
| Min Odds | 1.30 |
| Max Odds | 8.00 |
| Max Overround | 15% |

---

## 🚀 Instalación y Uso

### Requisitos Previos

- Python 3.9+
- pip

### Instalación

```bash
cd soccer_quant_system
pip install -r requirements.txt
```

### Variables de Entorno (Opcional)

```bash
export API_FOOTBALL_KEY="tu_api_key"
export ODDS_API_KEY="tu_api_key"
```

### Uso

```bash
# Pipeline completo
python main.py

# Solo actualizar datos
python main.py --data-only

# Backtesting con XGBoost (default)
python main.py --backtest

# Backtesting con LightGBM  
python main.py --backtest --model lightgbm

# Backtesting con regresión logística
python main.py --backtest --model logistic

# Dashboard
python main.py --dashboard
# o directamente:
streamlit run dashboard.py
```

---

## ⚠️ Limitaciones

1. **Calidad de datos**: Football-data.co.uk puede tener cobertura incompleta para ligas muy menores
2. **Odds de cierre**: No siempre disponibles para todas las casas de apuestas
3. **API rate limits**: Las APIs gratuitas tienen límites de consulta
4. **Liquidez**: En ligas muy menores, las odds reales pueden diferir de las publicadas
5. **Latencia**: Las odds cambian rápidamente; el sistema no es de trading de alta frecuencia
6. **xG**: No incluido por defecto (requiere fuente de datos adicional)
7. **Costos de transacción**: No se modelan comisiones de exchange

---

## 🗺️ Roadmap

### Fase 1 (Actual) ✅
- [x] Arquitectura modular POO
- [x] Data pipeline multi-fuente
- [x] Feature engineering completo
- [x] Modelos: Logistic, XGBoost, LightGBM, Poisson
- [x] Calibración probabilística
- [x] Detección de value bets
- [x] Backtesting walk-forward
- [x] Monte Carlo simulation
- [x] Dashboard Streamlit

### Fase 2 (Próxima)
- [ ] Integración con exchanges (Betfair, Pinnacle)
- [ ] xG como feature (Understat, FBref)
- [ ] Red neuronal simple (MLP)
- [ ] Ensemble de modelos
- [ ] Alertas por email/Telegram

### Fase 3 (Futuro)
- [ ] Despliegue en cloud (AWS/GCP)
- [ ] Base de datos PostgreSQL
- [ ] API REST para integraciones
- [ ] Modelo de goles por minutos
- [ ] Mercados adicionales (Over/Under, Asian Handicap)
- [ ] Optimización bayesiana de hiperparámetros

---

## 📄 Licencia

Uso personal. Sistema para investigación cuantitativa.

---

*Desarrollado con 🧠 y ⚽ — Soccer Quant System v1.0*

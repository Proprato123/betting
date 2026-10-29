"""
Dashboard Streamlit — Visualización del sistema cuantitativo.

Ejecutar con: streamlit run dashboard.py
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import RESULTS_DIR, PROCESSED_DATA_DIR, MODELS_DIR

# ─────────────────────── CONFIG ──────────────────────────────
st.set_page_config(
    page_title="⚽ Soccer Quant System",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────── ESTILOS ─────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1DB954;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        text-align: center;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #333;
        text-align: center;
    }
    .positive { color: #1DB954; }
    .negative { color: #e74c3c; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────── FUNCIONES ───────────────────────────
@st.cache_data
def load_backtest_results():
    """Carga el backtest más reciente."""
    backtest_files = sorted(RESULTS_DIR.glob("backtest_2*.json"), reverse=True)
    if not backtest_files:
        return None
    with open(backtest_files[0], "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_bet_log():
    """Carga el log de apuestas más reciente."""
    bet_files = sorted(RESULTS_DIR.glob("backtest_bets_*.csv"), reverse=True)
    if not bet_files:
        return None
    return pd.read_csv(bet_files[0], parse_dates=["date"])


@st.cache_data
def load_picks():
    """Carga los picks más recientes."""
    pick_files = sorted(RESULTS_DIR.glob("picks_*.csv"), reverse=True)
    if not pick_files:
        return None
    return pd.read_csv(pick_files[0])


@st.cache_data
def load_featured_data():
    """Carga datos con features."""
    path = PROCESSED_DATA_DIR / "featured_data.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"])


# ─────────────────────── SIDEBAR ─────────────────────────────
st.sidebar.image("https://img.icons8.com/emoji/96/soccer-ball.png", width=60)
st.sidebar.title("Soccer Quant System")

page = st.sidebar.radio(
    "Navegación",
    ["📊 Dashboard", "🎯 Value Bets", "📈 Backtest", "🏆 Ligas", "⚙️ Configuración"],
)


# ─────────────────────── PÁGINAS ─────────────────────────────
if page == "📊 Dashboard":
    st.markdown('<h1 class="main-header">⚽ Soccer Quant System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sistema cuantitativo para detección de value bets en ligas menores</p>', unsafe_allow_html=True)
    st.divider()

    # Cargar datos
    results = load_backtest_results()
    bet_log = load_bet_log()

    if results:
        # KPIs principales
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            roi = results.get("roi", 0)
            st.metric("ROI", f"{roi:.2%}", delta=f"{roi:.2%}")
        with col2:
            st.metric("Total Apuestas", results.get("total_bets", 0))
        with col3:
            st.metric("Win Rate", f"{results.get('win_rate', 0):.1%}")
        with col4:
            dd = results.get("max_drawdown", 0)
            st.metric("Max Drawdown", f"{dd:.1%}", delta=f"-{dd:.1%}", delta_color="inverse")
        with col5:
            st.metric("Sharpe", f"{results.get('sharpe', 0):.3f}")

        st.divider()

        # Equity Curve
        if bet_log is not None and not bet_log.empty:
            col_left, col_right = st.columns([2, 1])

            with col_left:
                st.subheader("📈 Equity Curve")
                equity = [1000]
                for _, row in bet_log.iterrows():
                    equity.append(row["bankroll"])

                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(
                    y=equity, mode='lines',
                    name='Bankroll',
                    line=dict(color='#1DB954', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(29, 185, 84, 0.1)',
                ))
                fig_equity.add_hline(y=1000, line_dash="dash", line_color="gray", annotation_text="Bankroll Inicial")
                fig_equity.update_layout(
                    template="plotly_dark",
                    height=400,
                    margin=dict(l=20, r=20, t=20, b=20),
                    yaxis_title="Bankroll ($)",
                    xaxis_title="Apuestas",
                )
                st.plotly_chart(fig_equity, use_container_width=True)

            with col_right:
                st.subheader("📊 Distribución de Profit")
                fig_hist = px.histogram(
                    bet_log, x="profit", nbins=30,
                    color_discrete_sequence=["#1DB954"],
                    template="plotly_dark",
                )
                fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
                fig_hist.update_layout(
                    height=400,
                    margin=dict(l=20, r=20, t=20, b=20),
                    xaxis_title="Profit/Loss ($)",
                    yaxis_title="Frecuencia",
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # Monte Carlo
            if "monte_carlo" in results:
                st.subheader("🎲 Simulación Monte Carlo")
                mc = results["monte_carlo"]

                mc_col1, mc_col2, mc_col3, mc_col4 = st.columns(4)
                with mc_col1:
                    st.metric("ROI Medio", f"{mc['roi']['mean']:.2%}")
                with mc_col2:
                    st.metric("P(ROI > 0)", f"{mc['roi']['prob_positive']:.1%}")
                with mc_col3:
                    st.metric("ROI P5", f"{mc['roi']['p5']:.2%}")
                with mc_col4:
                    st.metric("DD Medio", f"{mc['max_drawdown']['mean']:.1%}")

            # Resultados por fold
            if "fold_details" in results:
                st.subheader("📋 Resultados por Fold")
                folds_df = pd.DataFrame(results["fold_details"])
                if not folds_df.empty:
                    fig_folds = px.bar(
                        folds_df, x="fold", y="roi",
                        color="roi",
                        color_continuous_scale=["#e74c3c", "#f39c12", "#1DB954"],
                        template="plotly_dark",
                        labels={"roi": "ROI", "fold": "Fold"},
                    )
                    fig_folds.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=20, b=20),
                    )
                    st.plotly_chart(fig_folds, use_container_width=True)
    else:
        st.info("⚠️ No hay resultados de backtest. Ejecuta: `python main.py --backtest`")


elif page == "🎯 Value Bets":
    st.header("🎯 Value Bets Detectadas")

    picks = load_picks()
    if picks is not None and not picks.empty:
        # Filtros
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            if "league" in picks.columns:
                leagues = ["Todas"] + sorted(picks["league"].unique().tolist())
                sel_league = st.selectbox("Liga", leagues)
        with col_f2:
            min_ev_filter = st.slider("EV mínimo", 0.0, 0.5, 0.03, 0.01)

        # Filtrar
        filtered = picks[picks["ev"] >= min_ev_filter]
        if "league" in picks.columns and sel_league != "Todas":
            filtered = filtered[filtered["league"] == sel_league]

        # Métricas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Value Bets", len(filtered))
        with col2:
            st.metric("EV Promedio", f"{filtered['ev'].mean():.2%}")
        with col3:
            st.metric("Odds Promedio", f"{filtered['odds'].mean():.2f}")

        # Tabla
        st.dataframe(
            filtered.style.format({
                "model_prob": "{:.1%}",
                "market_prob": "{:.1%}",
                "ev": "{:.2%}",
                "edge": "{:.2%}",
                "odds": "{:.2f}",
                "amount": "${:.2f}",
            }),
            height=500,
            use_container_width=True,
        )
    else:
        st.info("⚠️ No hay picks. Ejecuta: `python main.py`")


elif page == "📈 Backtest":
    st.header("📈 Análisis de Backtest")

    bet_log = load_bet_log()
    if bet_log is not None and not bet_log.empty:
        # Análisis por tipo de apuesta
        st.subheader("Rendimiento por Tipo de Apuesta")
        type_stats = bet_log.groupby("bet_type").agg(
            n_bets=("profit", "count"),
            wins=("won", "sum"),
            total_profit=("profit", "sum"),
            total_staked=("stake", "sum"),
            avg_odds=("odds", "mean"),
            avg_ev=("ev", "mean"),
        ).reset_index()
        type_stats["roi"] = type_stats["total_profit"] / type_stats["total_staked"]
        type_stats["win_rate"] = type_stats["wins"] / type_stats["n_bets"]

        st.dataframe(
            type_stats.style.format({
                "roi": "{:.2%}",
                "win_rate": "{:.1%}",
                "avg_odds": "{:.2f}",
                "avg_ev": "{:.2%}",
                "total_profit": "${:.2f}",
            }),
            use_container_width=True,
        )

        # Análisis por liga
        if "league" in bet_log.columns:
            st.subheader("Rendimiento por Liga")
            league_stats = bet_log.groupby("league").agg(
                n_bets=("profit", "count"),
                total_profit=("profit", "sum"),
                total_staked=("stake", "sum"),
            ).reset_index()
            league_stats["roi"] = league_stats["total_profit"] / league_stats["total_staked"]
            league_stats = league_stats.sort_values("roi", ascending=False)

            fig_league = px.bar(
                league_stats, x="league", y="roi",
                color="roi",
                color_continuous_scale=["#e74c3c", "#f39c12", "#1DB954"],
                template="plotly_dark",
                labels={"roi": "ROI", "league": "Liga"},
            )
            fig_league.update_layout(height=400)
            st.plotly_chart(fig_league, use_container_width=True)

        # Profit acumulado en el tiempo
        if "date" in bet_log.columns:
            st.subheader("Profit Acumulado")
            bet_log_sorted = bet_log.sort_values("date")
            bet_log_sorted["cum_profit"] = bet_log_sorted["profit"].cumsum()

            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=bet_log_sorted["date"],
                y=bet_log_sorted["cum_profit"],
                mode='lines',
                name='Profit Acumulado',
                line=dict(color='#1DB954', width=2),
                fill='tozeroy',
                fillcolor='rgba(29, 185, 84, 0.1)',
            ))
            fig_cum.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_cum.update_layout(
                template="plotly_dark",
                height=400,
                yaxis_title="Profit ($)",
                xaxis_title="Fecha",
            )
            st.plotly_chart(fig_cum, use_container_width=True)

        # Distribución de EV
        st.subheader("Distribución de EV en Apuestas")
        col1, col2 = st.columns(2)
        with col1:
            fig_ev = px.histogram(
                bet_log, x="ev", nbins=30,
                color="won", color_discrete_map={True: "#1DB954", False: "#e74c3c"},
                template="plotly_dark",
                labels={"ev": "Expected Value", "won": "Ganó"},
            )
            fig_ev.update_layout(height=350, barmode="overlay")
            fig_ev.update_traces(opacity=0.7)
            st.plotly_chart(fig_ev, use_container_width=True)

        with col2:
            fig_odds = px.scatter(
                bet_log, x="odds", y="ev",
                color="won", color_discrete_map={True: "#1DB954", False: "#e74c3c"},
                template="plotly_dark",
                labels={"odds": "Odds", "ev": "EV", "won": "Ganó"},
            )
            fig_odds.update_layout(height=350)
            st.plotly_chart(fig_odds, use_container_width=True)
    else:
        st.info("⚠️ No hay datos de backtest. Ejecuta: `python main.py --backtest`")


elif page == "🏆 Ligas":
    st.header("🏆 Datos por Liga")

    data = load_featured_data()
    if data is not None and not data.empty:
        leagues = data["league"].unique()

        sel_league = st.selectbox("Seleccionar Liga", sorted(leagues))
        league_data = data[data["league"] == sel_league]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Partidos", len(league_data))
        with col2:
            st.metric("Equipos", league_data["home_team"].nunique())
        with col3:
            home_win = (league_data["result"] == "H").mean()
            st.metric("% Victoria Local", f"{home_win:.1%}")
        with col4:
            avg_goals = (league_data["home_goals"] + league_data["away_goals"]).mean()
            st.metric("Goles/Partido", f"{avg_goals:.2f}")

        # Distribución de resultados
        st.subheader("Distribución de Resultados")
        result_counts = league_data["result"].value_counts()
        fig_pie = px.pie(
            values=result_counts.values,
            names=result_counts.index.map({"H": "Local", "D": "Empate", "A": "Visitante"}),
            color_discrete_sequence=["#1DB954", "#f39c12", "#e74c3c"],
            template="plotly_dark",
        )
        fig_pie.update_layout(height=350)
        st.plotly_chart(fig_pie, use_container_width=True)

        # Top equipos por Elo
        if "home_elo" in league_data.columns:
            st.subheader("Top Equipos (Elo Rating)")
            latest = league_data.sort_values("date").groupby("home_team").last()
            top_elo = latest.nlargest(10, "home_elo")[["home_elo"]].reset_index()
            top_elo.columns = ["Equipo", "Elo"]

            fig_elo = px.bar(
                top_elo, x="Equipo", y="Elo",
                color="Elo",
                color_continuous_scale=["#16213e", "#1DB954"],
                template="plotly_dark",
            )
            fig_elo.update_layout(height=400)
            st.plotly_chart(fig_elo, use_container_width=True)
    else:
        st.info("⚠️ No hay datos. Ejecuta: `python main.py --data-only`")


elif page == "⚙️ Configuración":
    st.header("⚙️ Configuración del Sistema")

    from config.settings import (
        TARGET_LEAGUES, MIN_EV_THRESHOLD, KELLY_FRACTION,
        INITIAL_BANKROLL, ROLLING_WINDOW, ELO_K_FACTOR,
        WALK_FORWARD_TRAIN_WINDOW, WALK_FORWARD_TEST_WINDOW,
    )

    st.subheader("Ligas Configuradas")
    leagues_df = pd.DataFrame(TARGET_LEAGUES)
    st.dataframe(leagues_df, use_container_width=True)

    st.subheader("Parámetros")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Apuestas**")
        st.write(f"- Min EV: {MIN_EV_THRESHOLD:.0%}")
        st.write(f"- Kelly Fraction: {KELLY_FRACTION}")
        st.write(f"- Bankroll: ${INITIAL_BANKROLL:.0f}")
    with col2:
        st.markdown("**Features**")
        st.write(f"- Rolling Window: {ROLLING_WINDOW}")
        st.write(f"- Elo K Factor: {ELO_K_FACTOR}")
    with col3:
        st.markdown("**Backtesting**")
        st.write(f"- Train Window: {WALK_FORWARD_TRAIN_WINDOW}")
        st.write(f"- Test Window: {WALK_FORWARD_TEST_WINDOW}")

    # Modelos guardados
    st.subheader("Modelos Guardados")
    model_files = sorted(MODELS_DIR.glob("*.joblib"), reverse=True)
    if model_files:
        for mf in model_files[:10]:
            st.text(f"  📦 {mf.name} ({mf.stat().st_size / 1024:.1f} KB)")
    else:
        st.info("No hay modelos guardados")

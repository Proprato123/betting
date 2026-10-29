"""
Soccer Quant System — Punto de entrada principal.

Uso:
    python main.py                    # Pipeline completo
    python main.py --backtest         # Solo backtesting
    python main.py --data-only        # Solo actualizar datos
    python main.py --predict          # Solo predicciones (requiere modelo)
    python main.py --dashboard        # Lanzar dashboard Streamlit
    python main.py --model xgboost    # Seleccionar modelo
"""
import argparse
import logging
import sys
from pathlib import Path

# Añadir el directorio raíz al path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import LOG_LEVEL, LOG_FORMAT, LOGS_DIR
from src.pipeline.daily_pipeline import DailyPipeline
from src.models.logistic_model import LogisticModel
from src.models.xgboost_model import XGBoostModel, LightGBMModel, PoissonGoalsModel


def setup_logging():
    """Configura logging con archivo y consola."""
    log_file = LOGS_DIR / "system.log"

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def get_model_class(model_name: str):
    """Retorna la clase del modelo según su nombre."""
    models = {
        "logistic": LogisticModel,
        "xgboost": XGBoostModel,
        "lightgbm": LightGBMModel,
        "poisson": PoissonGoalsModel,
    }
    if model_name not in models:
        raise ValueError(
            f"Modelo '{model_name}' no reconocido. "
            f"Opciones: {list(models.keys())}"
        )
    return models[model_name]


def main():
    parser = argparse.ArgumentParser(
        description="⚽ Soccer Quant System — Detección de Value Bets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py                         Pipeline completo
  python main.py --backtest              Backtesting walk-forward
  python main.py --model lightgbm        Usar LightGBM
  python main.py --backtest --model logistic   Backtest con regresión logística
  python main.py --dashboard             Lanzar dashboard
        """,
    )

    parser.add_argument(
        "--backtest", action="store_true",
        help="Ejecutar backtesting walk-forward"
    )
    parser.add_argument(
        "--data-only", action="store_true",
        help="Solo actualizar y limpiar datos"
    )
    parser.add_argument(
        "--predict", action="store_true",
        help="Solo generar predicciones (requiere modelo entrenado)"
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Lanzar dashboard Streamlit"
    )
    parser.add_argument(
        "--model", type=str, default="xgboost",
        choices=["logistic", "xgboost", "lightgbm", "poisson"],
        help="Modelo a utilizar (default: xgboost)"
    )
    parser.add_argument(
        "--train-window", type=int, default=200,
        help="Ventana de entrenamiento para backtest (default: 200)"
    )
    parser.add_argument(
        "--test-window", type=int, default=50,
        help="Ventana de test para backtest (default: 50)"
    )
    parser.add_argument(
        "--force-retrain", action="store_true",
        help="Forzar re-entrenamiento del modelo"
    )

    args = parser.parse_args()

    # Setup
    setup_logging()
    logger = logging.getLogger("main")

    logger.info("=" * 60)
    logger.info("⚽ SOCCER QUANT SYSTEM")
    logger.info("=" * 60)
    logger.info(f"Modelo seleccionado: {args.model}")

    # Dashboard
    if args.dashboard:
        import subprocess
        dashboard_path = ROOT_DIR / "dashboard.py"
        if dashboard_path.exists():
            logger.info("Lanzando dashboard Streamlit...")
            subprocess.run(
                [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
                cwd=str(ROOT_DIR),
            )
        else:
            logger.error("dashboard.py no encontrado")
        return

    # Inicializar pipeline
    model_class = get_model_class(args.model)
    pipeline = DailyPipeline(model_class=model_class)

    # Solo datos
    if args.data_only:
        logger.info("Modo: Solo actualización de datos")
        result = pipeline.update_data()
        result_features = pipeline.update_features()
        logger.info(f"Datos: {result}")
        logger.info(f"Features: {result_features}")
        return

    # Backtest
    if args.backtest:
        logger.info("Modo: Backtesting walk-forward")
        pipeline.update_data()
        pipeline.update_features()
        results = pipeline.run_backtest(
            train_window=args.train_window,
            test_window=args.test_window,
        )

        # Imprimir resumen
        print("\n" + "=" * 60)
        print("📊 RESULTADOS DEL BACKTEST")
        print("=" * 60)

        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return

        print(f"Total apuestas:    {results.get('total_bets', 0)}")
        print(f"Ganadas:           {results.get('wins', 0)}")
        print(f"Perdidas:          {results.get('losses', 0)}")
        print(f"Win Rate:          {results.get('win_rate', 0):.1%}")
        print(f"ROI:               {results.get('roi', 0):.2%}")
        print(f"Yield:             {results.get('yield_pct', 0):.2f}%")
        print(f"Max Drawdown:      {results.get('max_drawdown', 0):.1%}")
        print(f"Sharpe Ratio:      {results.get('sharpe', 0):.3f}")
        print(f"Profit Factor:     {results.get('profit_factor', 0):.3f}")
        print(f"Final Bankroll:    ${results.get('final_bankroll', 0):.2f}")

        if "monte_carlo" in results:
            mc = results["monte_carlo"]
            print(f"\n📈 Monte Carlo ({mc['n_simulations']} simulaciones):")
            print(f"  ROI medio:       {mc['roi']['mean']:.2%}")
            print(f"  ROI P5-P95:      [{mc['roi']['p5']:.2%}, {mc['roi']['p95']:.2%}]")
            print(f"  P(ROI > 0):      {mc['roi']['prob_positive']:.1%}")
            print(f"  DD medio:        {mc['max_drawdown']['mean']:.1%}")

        if "clv" in results:
            print(f"CLV:               {results['clv']:.2%}")

        print("=" * 60)
        return

    # Pipeline completo
    logger.info("Modo: Pipeline completo")
    results = pipeline.run(full_pipeline=True)

    # Mostrar picks si hay
    if results.get("steps", {}).get("value", {}).get("total_bets", 0) > 0:
        print("\n" + "=" * 60)
        print("🎯 VALUE BETS DETECTADAS")
        print("=" * 60)
        value_info = results["steps"]["value"]
        print(f"Total:       {value_info['total_bets']}")
        print(f"EV promedio: {value_info['avg_ev']:.2%}")
        print(f"Edge prom.:  {value_info['avg_edge']:.2%}")
        print(f"Por tipo:    {value_info.get('by_type', {})}")
        print("=" * 60)
    else:
        print("\n⚠️ No se detectaron value bets en esta ejecución.")

    logger.info(f"Pipeline finalizado: {results['status']}")


if __name__ == "__main__":
    main()

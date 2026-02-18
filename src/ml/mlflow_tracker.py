"""
src/ml/mlflow_tracker.py
==========================
Intégration MLflow pour tracker les expériences ML :
  - Scoring produit
  - Modèle de churn
  - Prévision de ventes

Installation : pip install mlflow

Usage :
    from src.ml.mlflow_tracker import MLTracker
    with MLTracker("churn_model") as tracker:
        tracker.log_param("n_estimators", 100)
        tracker.log_metric("accuracy", 0.92)
        tracker.log_model(clf, "churn_rf")
"""

from __future__ import annotations
import os
from contextlib import contextmanager
from datetime import datetime


MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")


class MLTracker:
    """Wrapper MLflow avec fallback gracieux si mlflow n'est pas installé."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self._run = None
        self._mlflow = None

    def __enter__(self):
        try:
            import mlflow
            self._mlflow = mlflow
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(self.experiment_name)
            self._run = mlflow.start_run(run_name=f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M')}")
            print(f"[mlflow] Experiment '{self.experiment_name}' — Run ID : {self._run.info.run_id}")
        except ImportError:
            print("[mlflow] MLflow non installé — tracking désactivé. `pip install mlflow`")
        return self

    def __exit__(self, *args):
        if self._mlflow and self._run:
            self._mlflow.end_run()

    def log_param(self, key: str, value) -> None:
        if self._mlflow:
            self._mlflow.log_param(key, value)

    def log_params(self, params: dict) -> None:
        if self._mlflow:
            self._mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        if self._mlflow:
            self._mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if self._mlflow:
            self._mlflow.log_metrics(metrics, step=step)

    def log_model(self, model, artifact_path: str = "model") -> None:
        """Log un modèle sklearn."""
        if self._mlflow:
            try:
                self._mlflow.sklearn.log_model(model, artifact_path)
                print(f"[mlflow] Modèle enregistré : {artifact_path}")
            except Exception as e:
                print(f"[mlflow] Erreur log_model : {e}")

    def log_figure(self, fig, filename: str) -> None:
        """Log une figure matplotlib ou plotly."""
        if self._mlflow:
            try:
                import tempfile, os
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = os.path.join(tmpdir, filename)
                    fig.write_image(path)
                    self._mlflow.log_artifact(path)
            except Exception as e:
                print(f"[mlflow] Erreur log_figure : {e}")

    def log_dataframe(self, df, filename: str) -> None:
        """Log un DataFrame comme artifact CSV."""
        if self._mlflow:
            try:
                import tempfile, os
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = os.path.join(tmpdir, filename)
                    df.to_csv(path, index=False)
                    self._mlflow.log_artifact(path)
            except Exception as e:
                print(f"[mlflow] Erreur log_dataframe : {e}")


def list_experiments() -> list[dict]:
    """Retourne la liste des expériences MLflow."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        experiments = mlflow.search_experiments()
        return [{"name": e.name, "id": e.experiment_id, "lifecycle": e.lifecycle_stage}
                for e in experiments]
    except ImportError:
        return []
    except Exception as e:
        print(f"[mlflow] Erreur list_experiments : {e}")
        return []

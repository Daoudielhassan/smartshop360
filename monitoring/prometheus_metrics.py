"""
monitoring/prometheus_metrics.py
==================================
Métriques Prometheus exposées via un endpoint HTTP (port 8000).
Importer et lancer dans app.py si Prometheus est disponible.

Usage:
    from monitoring.prometheus_metrics import start_metrics_server, METRICS
    start_metrics_server()      # démarre en thread background
    METRICS.etl_runs.inc()      # incrémente un compteur
"""

from __future__ import annotations

import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Tentative d'import prometheus_client (optionnel)
# ─────────────────────────────────────────────────────────────

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        start_http_server, REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # Stub silencieux
    class _Noop:
        def inc(self, *a, **kw):   pass
        def dec(self, *a, **kw):   pass
        def set(self, *a, **kw):   pass
        def observe(self, *a, **kw): pass
        def labels(self, *a, **kw): return self
        def time(self):
            import contextlib
            return contextlib.nullcontext()

    class Counter(_Noop):
        def __init__(self, *a, **kw): pass
    class Gauge(_Noop):
        def __init__(self, *a, **kw): pass
    class Histogram(_Noop):
        def __init__(self, *a, **kw): pass
    class Summary(_Noop):
        def __init__(self, *a, **kw): pass

    def start_http_server(*a, **kw): pass


# ─────────────────────────────────────────────────────────────
#  Définition des métriques
# ─────────────────────────────────────────────────────────────

@dataclass
class AppMetrics:
    # ETL
    etl_runs: Counter = field(default_factory=lambda: Counter(
        "smartshop_etl_runs_total",
        "Nombre total de runs ETL",
        ["status"],        # success / failure / skipped
    ))
    etl_rows_loaded: Gauge = field(default_factory=lambda: Gauge(
        "smartshop_etl_rows_loaded",
        "Nombre de lignes chargées lors du dernier ETL",
        ["table"],
    ))
    etl_duration_seconds: Histogram = field(default_factory=lambda: Histogram(
        "smartshop_etl_duration_seconds",
        "Durée du pipeline ETL (secondes)",
        buckets=[5, 10, 30, 60, 120, 300],
    ))
    data_quality_checks: Counter = field(default_factory=lambda: Counter(
        "smartshop_dq_checks_total",
        "Résultats des vérifications de qualité",
        ["result"],        # pass / fail_error / fail_warning
    ))

    # Application
    streamlit_sessions: Gauge = field(default_factory=lambda: Gauge(
        "smartshop_active_sessions",
        "Sessions Streamlit actives",
    ))
    query_duration_seconds: Histogram = field(default_factory=lambda: Histogram(
        "smartshop_query_duration_seconds",
        "Durée des requêtes SQL (secondes)",
        ["query_type"],
        buckets=[0.01, 0.05, 0.1, 0.5, 1, 5],
    ))
    agent_requests: Counter = field(default_factory=lambda: Counter(
        "smartshop_agent_requests_total",
        "Requêtes adressées à l'agent IA",
        ["status"],        # success / error
    ))
    alerts_sent: Counter = field(default_factory=lambda: Counter(
        "smartshop_alerts_sent_total",
        "Alertes envoyées (email/slack)",
        ["channel"],       # email / slack
    ))

    # ML
    model_predictions: Counter = field(default_factory=lambda: Counter(
        "smartshop_model_predictions_total",
        "Prédictions ML effectuées",
        ["model"],         # churn / forecast / scoring
    ))
    model_training_duration: Histogram = field(default_factory=lambda: Histogram(
        "smartshop_model_training_seconds",
        "Durée d'entraînement des modèles",
        ["model"],
        buckets=[0.1, 0.5, 1, 5, 10, 30, 60],
    ))


# Singleton partagé
METRICS = AppMetrics()


# ─────────────────────────────────────────────────────────────
#  Démarrage du serveur de métriques
# ─────────────────────────────────────────────────────────────

_SERVER_STARTED = False


def start_metrics_server(port: int = 8000) -> bool:
    """
    Lance le serveur Prometheus sur *port* dans un thread daemon.
    Retourne True si démarré avec succès, False sinon.
    """
    global _SERVER_STARTED
    if _SERVER_STARTED:
        return True
    if not PROMETHEUS_AVAILABLE:
        logger.info("prometheus_client non installé — métriques désactivées.")
        return False

    def _run():
        start_http_server(port)
        logger.info("Serveur Prometheus démarré sur http://localhost:%d/metrics", port)

    t = threading.Thread(target=_run, daemon=True, name="prometheus-server")
    t.start()
    _SERVER_STARTED = True
    return True


# ─────────────────────────────────────────────────────────────
#  Décorateur pratique pour mesurer la durée d'une fonction
# ─────────────────────────────────────────────────────────────

def timed(metric: Histogram, **label_kwargs):
    """
    Décorateur : mesure la durée d'exécution et l'enregistre dans *metric*.

    @timed(METRICS.query_duration_seconds, query_type="dashboard")
    def my_query():
        ...
    """
    import functools

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                if label_kwargs:
                    metric.labels(**label_kwargs).observe(elapsed)
                else:
                    metric.observe(elapsed)
        return wrapper
    return decorator

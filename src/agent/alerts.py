"""
src/agent/alerts.py
=====================
Alertes proactives — envoie des notifications email et/ou Slack
quand un produit passe en statut CRITIQUE.

Configuration (.env) :
    ALERT_EMAIL_FROM=alerts@smartshop.com
    ALERT_EMAIL_TO=manager@smartshop.com
    ALERT_SMTP_HOST=smtp.gmail.com
    ALERT_SMTP_PORT=587
    ALERT_SMTP_USER=alerts@smartshop.com
    ALERT_SMTP_PASS=yourpassword
    ALERT_SLACK_WEBHOOK=https://hooks.slack.com/services/XXX/YYY/ZZZ

Usage :
    from src.agent.alerts import check_and_alert
    check_and_alert()   # vérifie les produits critiques et envoie les alertes
"""

from __future__ import annotations
import os
import smtplib
import json
import urllib.request
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from sqlalchemy import text

from src.db_config import get_engine


# ─────────────────────────────────────────────────────────────
#  Récupération des produits CRITIQUES
# ─────────────────────────────────────────────────────────────

def get_critical_products() -> list[dict]:
    """Retourne les produits en statut CRITIQUE depuis v_alerts."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT "ProductName" AS name,
                   "Category"    AS category,
                   ROUND("Notemoyenne"::NUMERIC, 2) AS note,
                   "NbAvis"      AS nb_avis,
                   "AvisNegatifs" AS avis_neg,
                   "CA"          AS ca
            FROM v_alerts
            WHERE "Statut" = 'CRITIQUE'
            ORDER BY "Notemoyenne" ASC
        """)).fetchall()
    return [dict(r._mapping) for r in rows]


# ─────────────────────────────────────────────────────────────
#  Formatage des messages
# ─────────────────────────────────────────────────────────────

def _format_alert_text(products: list[dict]) -> str:
    lines = [
        f"🚨 SmartShop 360 — Alerte CRITIQUE ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
        f"{len(products)} produit(s) en statut CRITIQUE :\n",
    ]
    for p in products:
        lines.append(
            f"  • {p['name']} [{p['category']}]\n"
            f"    Note : {p['note']}/5 | {p['avis_neg']} avis négatifs / {p['nb_avis']} total\n"
            f"    CA : {p['ca']:,.0f} EUR"
        )
    lines.append("\n→ Action recommandée : vérifier la qualité produit et contacter le fournisseur.")
    return "\n".join(lines)


def _format_slack_payload(products: list[dict]) -> dict:
    text_body = _format_alert_text(products)
    return {
        "text": "🚨 *Alerte SmartShop 360 — Produits CRITIQUES*",
        "blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": f"```{text_body}```"}},
        ]
    }


# ─────────────────────────────────────────────────────────────
#  Envoi Email
# ─────────────────────────────────────────────────────────────

def send_email_alert(products: list[dict]) -> bool:
    """Envoie un email d'alerte via SMTP. Retourne True si succès."""
    smtp_host = os.getenv("ALERT_SMTP_HOST", "")
    smtp_port = int(os.getenv("ALERT_SMTP_PORT", "587"))
    smtp_user = os.getenv("ALERT_SMTP_USER", "")
    smtp_pass = os.getenv("ALERT_SMTP_PASS", "")
    from_addr = os.getenv("ALERT_EMAIL_FROM", smtp_user)
    to_addr   = os.getenv("ALERT_EMAIL_TO", "")

    if not all([smtp_host, smtp_user, smtp_pass, to_addr]):
        print("[alerts] Email non configuré — variables ALERT_SMTP_* manquantes")
        return False

    body = _format_alert_text(products)
    msg  = MIMEMultipart("alternative")
    msg["Subject"] = f"🚨 SmartShop 360 — {len(products)} produit(s) CRITIQUE(s)"
    msg["From"]    = from_addr
    msg["To"]      = to_addr
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(from_addr, to_addr, msg.as_string())
        print(f"[alerts] Email envoyé à {to_addr}")
        return True
    except Exception as e:
        print(f"[alerts] Erreur email : {e}")
        return False


# ─────────────────────────────────────────────────────────────
#  Envoi Slack
# ─────────────────────────────────────────────────────────────

def send_slack_alert(products: list[dict]) -> bool:
    """Envoie une notification Slack via Incoming Webhook. Retourne True si succès."""
    webhook = os.getenv("ALERT_SLACK_WEBHOOK", "")
    if not webhook:
        print("[alerts] Slack non configuré — variable ALERT_SLACK_WEBHOOK manquante")
        return False

    payload = json.dumps(_format_slack_payload(products)).encode("utf-8")
    req = urllib.request.Request(
        webhook,
        data    = payload,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            ok = resp.status == 200
        if ok:
            print("[alerts] Notification Slack envoyée")
        return ok
    except Exception as e:
        print(f"[alerts] Erreur Slack : {e}")
        return False


# ─────────────────────────────────────────────────────────────
#  Orchestrateur
# ─────────────────────────────────────────────────────────────

def check_and_alert(dry_run: bool = False) -> list[dict]:
    """
    Vérifie les produits critiques et envoie les alertes configurées.

    Parameters
    ----------
    dry_run : si True, affiche les alertes sans les envoyer

    Returns
    -------
    Liste des produits critiques détectés
    """
    products = get_critical_products()

    if not products:
        print("[alerts] Aucun produit CRITIQUE — tout va bien ✅")
        return []

    print(f"[alerts] {len(products)} produit(s) CRITIQUE(s) détecté(s)")
    print(_format_alert_text(products))

    if dry_run:
        print("[alerts] Mode dry_run — alertes non envoyées")
        return products

    send_email_alert(products)
    send_slack_alert(products)
    return products


if __name__ == "__main__":
    check_and_alert(dry_run=False)

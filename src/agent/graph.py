"""
src/agent/graph.py
===================
Chaîne LLM (Text-to-SQL) — compatible multi-provider.

Providers supportés (par ordre de priorité) :
  1. Groq      GROQ_API_KEY      → llama-3.3-70b-versatile
  2. Mistral   MISTRAL_API_KEY   → mistral-large-latest
  3. OpenAI    OPENAI_API_KEY    → gpt-4o-mini
  4. Anthropic ANTHROPIC_API_KEY → claude-3-5-haiku-20241022
  5. Fallback  —                 → règles SQL basées sur mots-clés

Point d'entrée public :
    run_agent(question, api_key, conversation_history) → dict
    get_active_provider(api_key) → str
"""

import os
import json
import re
import sys
import requests
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.agent.tools import execute_sql, python_analysis, SYSTEM_PROMPT

# Mots-clés déclenchant une analyse Python post-SQL
_ANALYSIS_KEYWORDS = {
    "segmentation": ["segment",      "rentable",    "satisfait",    "quadrant"],
    "clustering":   ["cluster",       "groupe",      "automatique",  "k-means", "natural"],
    "correlation":  ["corrélation",   "corrélé",     "lien entre",   "impact de"],
    "trend":        ["tendance",       "evolution",   "évolution",    "mois par mois", "mensuel"],
    "rfm":          ["rfm",            "récence",     "fréquence",    "monétaire",   "profil client"],
}


# ────────────────────────────────────────────────────────────
#  Détection du provider
# ────────────────────────────────────────────────────────────

def _detect_provider(api_key: str | None = None) -> tuple:
    """Auto-détecte le provider LLM disponible. Retourne (name, key)."""
    if api_key:
        if api_key.startswith("gsk_"):                              return ("groq",      api_key)
        if api_key.startswith("sk-ant-"):                           return ("anthropic", api_key)
        if api_key.startswith("sk-"):                               return ("openai",    api_key)
        if 28 <= len(api_key) <= 64 and not api_key.startswith("sk"):
            return ("mistral", api_key)

    for env, name in [
        ("GROQ_API_KEY",      "groq"),
        ("MISTRAL_API_KEY",   "mistral"),
        ("OPENAI_API_KEY",    "openai"),
        ("ANTHROPIC_API_KEY", "anthropic"),
    ]:
        val = os.environ.get(env)
        if val:
            return (name, val)

    return ("fallback", "")


def get_active_provider(api_key: str | None = None) -> str:
    """Retourne une chaîne lisible décrivant le provider actif."""
    labels = {
        "groq":      " Groq (Llama 3.3-70B)",
        "mistral":   " Mistral (Large)",
        "openai":    " OpenAI (GPT-4o-mini)",
        "anthropic": " Anthropic (Claude 3.5 Haiku)",
        "fallback":  " Mode Hors-ligne (règles SQL)",
    }
    provider, _ = _detect_provider(api_key)
    return labels.get(provider, " Inconnu")


# ────────────────────────────────────────────────────────────
#  Appels individuels aux providers
# ────────────────────────────────────────────────────────────

def _call_groq(messages: list, system: str, key: str, max_tokens: int = 1024) -> str:
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model":      "llama-3.3-70b-versatile",
            "max_tokens": max_tokens,
            "messages":   [{"role": "system", "content": system}] + messages,
        },
        timeout=30,
    )
    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"]
    raise RuntimeError(f"Groq {r.status_code}: {r.text[:200]}")


def _call_mistral(messages: list, system: str, key: str, max_tokens: int = 1024) -> str:
    r = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model":      "mistral-large-latest",
            "max_tokens": max_tokens,
            "messages":   [{"role": "system", "content": system}] + messages,
        },
        timeout=30,
    )
    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"]
    raise RuntimeError(f"Mistral {r.status_code}: {r.text[:200]}")


def _call_openai(messages: list, system: str, key: str, max_tokens: int = 1024) -> str:
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model":      "gpt-4o-mini",
            "max_tokens": max_tokens,
            "messages":   [{"role": "system", "content": system}] + messages,
        },
        timeout=30,
    )
    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"]
    raise RuntimeError(f"OpenAI {r.status_code}: {r.text[:200]}")


def _call_anthropic(messages: list, system: str, key: str, max_tokens: int = 1024) -> str:
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key":         key,
            "anthropic-version": "2023-06-01",
            "Content-Type":      "application/json",
        },
        json={
            "model":      "claude-3-5-haiku-20241022",
            "max_tokens": max_tokens,
            "system":     system,
            "messages":   messages,
        },
        timeout=30,
    )
    if r.status_code == 200:
        return r.json()["content"][0]["text"]
    raise RuntimeError(f"Anthropic {r.status_code}: {r.text[:200]}")


def call_llm(messages: list, api_key: str | None = None, max_tokens: int = 1024) -> str:
    """Route vers le bon provider et gère le fallback."""
    provider, key = _detect_provider(api_key)
    callers = {
        "groq":      _call_groq,
        "mistral":   _call_mistral,
        "openai":    _call_openai,
        "anthropic": _call_anthropic,
    }
    if provider in callers:
        try:
            return callers[provider](messages, SYSTEM_PROMPT, key, max_tokens)
        except Exception as e:
            print(f"[graph] Provider {provider} échoué : {e} — basculement fallback")

    return _fallback_sql(messages[-1]["content"] if messages else "")


# ────────────────────────────────────────────────────────────
#  Fallback : règles SQL par mots-clés
# ────────────────────────────────────────────────────────────

def _fallback_sql(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["critique", "alerte", "mauvais", "négatif"]):
        sql = 'SELECT "ProductName", "Notemoyenne", "AvisNegatifs", "Statut" FROM v_alerts WHERE "Statut" IN (\'CRITIQUE\', \'A_SURVEILLER\') ORDER BY "Notemoyenne" ASC LIMIT 10'
    elif any(w in q for w in ["produit", "vente", "best", "top", "meilleur"]):
        sql = 'SELECT "ProductName", "Category", "CA", "QuantiteVendue", "Notemoyenne" FROM v_product_kpi ORDER BY "CA" DESC LIMIT 10'
    elif any(w in q for w in ["segment", "rentable", "satisfait", "quadrant", "cluster"]):
        sql = (
            'SELECT c."ClientID", c."Nom", c."Pays", '
            'ROUND(c."CA_Total"::numeric, 2) AS "CA_Total", '
            'c."NbCommandes", '
            'ROUND(c."PanierMoyen"::numeric, 2) AS "PanierMoyen", '
            'ROUND(COALESCE(AVG(rf."Rating"), 0)::numeric, 2) AS "NoteMoyenne" '
            'FROM v_customer_kpi c '
            'JOIN sales_facts sf ON sf."CustomerID" = c."ClientID" '
            'JOIN product_mapping pm ON pm."ERP_StockCode" = sf."StockCode" '
            'JOIN review_facts rf ON rf."ProductID" = pm."Review_ProductCode" '
            'GROUP BY c."ClientID", c."Nom", c."Pays", c."CA_Total", c."NbCommandes", c."PanierMoyen" '
            'ORDER BY c."CA_Total" DESC LIMIT 200'
        )
    elif any(w in q for w in ["client", "acheteur", "fidèle"]):
        sql = 'SELECT "Nom", "Pays", "NbCommandes", "CA_Total" FROM v_customer_kpi ORDER BY "CA_Total" DESC LIMIT 10'
    elif any(w in q for w in ["avis", "note", "sentiment", "satisfaction"]):
        sql = 'SELECT "ProductName", "Notemoyenne", "NbAvis", "AvisPositifs", "AvisNegatifs" FROM v_product_kpi ORDER BY "Notemoyenne" DESC LIMIT 10'
    elif any(w in q for w in ["qualité", "mdm", "couverture", "mapping"]):
        sql = 'SELECT * FROM v_data_quality'
    elif any(w in q for w in ["ca", "chiffre", "revenu", "revenue"]):
        sql = 'SELECT "Category", SUM("CA") AS "CA_Categorie" FROM v_product_kpi GROUP BY "Category" ORDER BY "CA_Categorie" DESC'
    else:
        sql = 'SELECT "ProductName", "CA", "Notemoyenne" FROM v_product_kpi ORDER BY "CA" DESC LIMIT 10'

    return json.dumps({
        "sql":             sql,
        "reasoning":       "Mode hors-ligne — règle SQL par mots-clés.",
        "answer_template": "Voici les résultats basés sur votre question.",
    })


# ────────────────────────────────────────────────────────────
#  Formatage de la réponse naturelle
# ────────────────────────────────────────────────────────────

def format_natural_response(question: str, sql: str, data: list,
                             api_key: str | None = None) -> str:
    if not data:
        return "Aucun résultat trouvé pour cette requête."

    preview = str(data[:5])
    messages = [{
        "role": "user",
        "content": (
            f"Question : {question}\n"
            f"SQL exécuté : {sql}\n"
            f"Données (extrait) : {preview}\n"
            f"Nombre total de lignes : {len(data)}\n\n"
            "Formule une réponse claire et synthétique en français pour un manager."
        ),
    }]

    provider, key = _detect_provider(api_key)
    callers = {
        "groq":      _call_groq,
        "mistral":   _call_mistral,
        "openai":    _call_openai,
        "anthropic": _call_anthropic,
    }
    if provider in callers:
        try:
            return callers[provider](messages, "Tu es un assistant data analytique concis.", key, 512)
        except Exception:
            pass

    # Fallback textuel
    lines = [f"**{k}** : {v}" for k, v in data[0].items()]
    return (
        f"J'ai trouvé **{len(data)} résultat(s)**."
        f"\n\nPremier résultat :\n" + "\n".join(lines)
    )


# ────────────────────────────────────────────────────────────
#  Point d'entrée principal : run_agent
# ────────────────────────────────────────────────────────────

def run_agent(question: str, api_key: str | None = None,
              conversation_history: list | None = None) -> dict:
    """
    Orchestre le pipeline Text-to-SQL :
      1. LLM génère le SQL (ou fallback)
      2. SQL exécuté sur PostgreSQL
      3. LLM formule la réponse naturelle

    Returns
    -------
    dict avec clés : question, sql, reasoning, data, columns,
                     answer, row_count, success
    """
    history = conversation_history or []
    messages = history + [{"role": "user", "content": question}]

    # Pré-détection : certaines questions complexes nécessitent un SQL prédéfini
    # (jointures multi-tables que le LLM simplifie souvent trop)
    q_lower = question.lower()
    _forced_sql: str | None = None
    for kw_list in [
        ["segment", "rentable", "satisfait", "quadrant", "cluster"],
        ["rfm", "récence", "fréquence", "monétaire", "profil client"],
    ]:
        if any(kw in q_lower for kw in kw_list):
            try:
                _forced_sql = json.loads(_fallback_sql(question)).get("sql", "")
            except Exception:
                pass
            break

    # Étape 1 : génération SQL
    llm_raw = call_llm(messages, api_key)

    # Extraction JSON (le LLM peut inclure texto, code fences, etc.)
    sql, reasoning, answer_template = "", "", ""
    try:
        # 1) Enlever les balises de code Markdown (```json, ```sql, ``` etc.)
        cleaned = re.sub(r"```\w*\s*", "", llm_raw).replace("```", "")
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            # 2) Normaliser les newlines littéraux dans les strings JSON
            #    (le LLM met souvent un SELECT multiligne non échappé)
            json_block = match.group()
            # Extraire "sql": "..." de façon ciblée avant d'essayer json.loads
            sql_field = re.search(
                r'"sql"\s*:\s*"(.*?)",\s*"(?:reasoning|answer)',
                json_block, re.DOTALL
            )
            if sql_field:
                sql = sql_field.group(1).replace('\\"', '"').replace("\\n", "\n").strip()
            # Tenter quand même le json.loads sur le reste (reasoning, template)
            try:
                norm = json_block.replace("\\n", "\n")
                # Remplacer les guillemets typographiques qui brisent json.loads
                norm = re.sub(r'"([^"]*?)"(?=\s*:)', r'"\1"', norm)
                parsed          = json.loads(norm)
                sql             = sql or parsed.get("sql", "").strip()
                reasoning       = parsed.get("reasoning", "")
                answer_template = parsed.get("answer_template", "")
            except json.JSONDecodeError:
                pass  # sql already extracted above if sql_field matched
    except (json.JSONDecodeError, AttributeError):
        pass

    # 3) Ultime recours : extraire un SELECT...LIMIT depuis le texte brut
    if not sql:
        raw_match = re.search(
            r"(SELECT\b.+?(?:LIMIT\s+\d+\s*;?|;\s*$))",
            llm_raw, re.DOTALL | re.IGNORECASE
        )
        if raw_match:
            sql = raw_match.group(1).replace('\\"', '"').strip()

    # Sécurité : si le SQL est toujours vide ou semble être un template vide,
    # utiliser le fallback par mots-clés
    _is_placeholder = (
        not sql
        or sql.strip() in ("SELECT ...", "SELECT…", "...", "")
        or not re.search(r"\bFROM\b", sql, re.IGNORECASE)
    )
    if _is_placeholder:
        try:
            fallback_parsed = json.loads(_fallback_sql(question))
            sql             = fallback_parsed.get("sql", "")
            reasoning       = reasoning or fallback_parsed.get("reasoning", "Règle fallback")
        except Exception:
            pass

    # Override pour les requêtes complexes pré-définies (segmentation, RFM…)
    if _forced_sql:
        sql      = _forced_sql
        reasoning = reasoning or "SQL pré-défini (segmentation multi-tables)"

    # Étape 2 : exécution SQL
    sql_result = execute_sql(sql) if sql else {"success": False, "error": "SQL vide — aucune règle applicable", "data": [], "columns": [], "row_count": 0}

    # Étape 3 : réponse naturelle
    if sql_result["success"] and sql_result["data"]:
        answer = format_natural_response(question, sql, sql_result["data"], api_key)
    elif sql_result["success"]:
        answer = "La requête n'a retourné aucun résultat."
    else:
        answer = f"Erreur SQL : {sql_result.get('error', 'inconnue')}"

    # Étape 4 : Analyse Python automatique (si la question le justifie)
    analysis: dict | None = None
    if sql_result["success"] and sql_result["data"]:
        q_lower = question.lower()
        for atype, keywords in _ANALYSIS_KEYWORDS.items():
            if any(kw in q_lower for kw in keywords):
                analysis = python_analysis(sql_result["data"], atype)
                break

    return {
        "question":  question,
        "sql":       sql,
        "reasoning": reasoning,
        "data":      sql_result.get("data", []),
        "columns":   sql_result.get("columns", []),
        "answer":    answer,
        "row_count": sql_result.get("row_count", 0),
        "success":   sql_result.get("success", False),
        "analysis":  analysis,
    }

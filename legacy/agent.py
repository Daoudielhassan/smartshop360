"""
SmartShop 360 - Agent IA (Mini-Orchestrateur)
===============================================
Architecture :
- L'agent reçoit une question en langage naturel
- Il génère une requête SQL grâce au LLM (multi-provider)
- Il exécute la requête sur la base SQLite
- Il formule une réponse en langage naturel

Pattern : Text-to-SQL (RAG structuré)

Providers LLM supportés (par ordre de priorité) :
  1. Groq     — GROQ_API_KEY      (llama-3.3-70b-versatile)
  2. Mistral  — MISTRAL_API_KEY   (mistral-large-latest)
  3. OpenAI   — OPENAI_API_KEY    (gpt-4o-mini)
  4. Anthropic— ANTHROPIC_API_KEY (claude-sonnet-4-20250514)
  5. Fallback — règles SQL intégrées (sans API)
"""

import sqlite3
import os
import json
import re
import requests

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "smartshop360.db")

# ─────────────────────────────────────────────
# SCHÉMA DE LA BASE (injecté dans le prompt)
# ─────────────────────────────────────────────
DB_SCHEMA = """
Base de données SQLite - SmartShop 360

TABLES PRINCIPALES :
--------------------
CUSTOMERS(ClientID, Nom, Pays)
PRODUCTS(ProductID, ProductName, Category)
INVOICES(FactureID, ClientID, Date, MontantTotal)
INVOICE_LINES(LigneID, FactureID, ProduitID, Quantite, PrixUnitaire, Revenue, Margin)
REVIEWS(ReviewID, ReviewText, Sentiment, Note, ReviewDate, ProduitID)
PRODUCT_MAPPING(MappingID, ERP_ProductCode, ERP_ProductName, Review_ProductCode, Review_ProductName, Category, GoldenRecordName)

VUES ANALYTIQUES (préférer pour les KPIs) :
--------------------------------------------
V_PRODUCT_KPI(ProductID, ProductName, Category, CA, Marge, QuantiteVendue, Notemoyenne, NbAvis, AvisPositifs, AvisNegatifs, AvisNeutres)
V_CUSTOMER_KPI(ClientID, Nom, Pays, NbCommandes, CA_Total, PanierMoyen)
V_ALERTS(ProductID, ProductName, Category, CA, Notemoyenne, NbAvis, AvisNegatifs, QuantiteVendue, Statut)
  → Statut peut être : 'CRITIQUE', 'A_SURVEILLER', 'OK'
V_DATA_QUALITY(Nb_Produits_ERP, Nb_Produits_Avis, Nb_Mappings, Nb_Avis_Total, Nb_Avis_Lies, Nb_Factures, Nb_Clients, Taux_Couverture_MDM)

NOTES :
- Sentiment dans REVIEWS : 'positive', 'negative', 'neutral'
- Note dans REVIEWS : de 1.0 à 5.0
- CA = Chiffre d'Affaires (Revenue total)
- Utiliser SQLite (pas de ILIKE, utiliser LIKE en majuscules ou LOWER())
- Les données transactions viennent du CSV réel Online Retail II (Kaggle)
"""

SYSTEM_PROMPT = f"""Tu es un Data Analyst expert pour SmartShop 360, un e-commerçant B2C spécialisé en Décoration & Cadeaux.

Tu as accès à une base de données SQLite avec le schéma suivant :
{DB_SCHEMA}

Ta mission est de :
1. Comprendre la question métier de l'utilisateur
2. Générer une requête SQL valide et optimisée
3. Analyser les résultats et formuler une réponse claire en français

Format de réponse OBLIGATOIRE (JSON strict) :
{{
  "sql": "SELECT ...",
  "reasoning": "Explication courte de l'approche",
  "answer_template": "Template de réponse à compléter avec les données"
}}

Règles SQL :
- Utilise uniquement du SQL compatible SQLite
- Préfère les vues analytiques (V_PRODUCT_KPI, V_CUSTOMER_KPI, V_ALERTS) aux tables brutes
- Limite les résultats à 20 lignes max avec LIMIT
- Arrondis les montants avec ROUND(x, 2)
- Ne génère qu'une seule requête SQL
"""

# ─────────────────────────────────────────────
# TOOL 1 : SQL EXECUTOR
# ─────────────────────────────────────────────
def execute_sql(query: str) -> dict:
    """Exécute une requête SQL et retourne les résultats."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description] if cur.description else []
        data = [dict(zip(columns, row)) for row in rows]
        conn.close()
        return {"success": True, "data": data, "columns": columns, "row_count": len(data)}
    except Exception as e:
        return {"success": False, "error": str(e), "data": [], "columns": []}

# ─────────────────────────────────────────────
# TOOL 2 : PYTHON ANALYTIQUE (calculs avancés)
# ─────────────────────────────────────────────
def python_analysis(data: list, analysis_type: str = "summary") -> dict:
    """Calculs statistiques complémentaires sur les données."""
    if not data:
        return {"result": "Aucune donnée à analyser"}
    
    import statistics
    
    if analysis_type == "summary":
        numeric_cols = {}
        for row in data:
            for k, v in row.items():
                if isinstance(v, (int, float)):
                    numeric_cols.setdefault(k, []).append(v)
        
        summary = {}
        for col, values in numeric_cols.items():
            summary[col] = {
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "moyenne": round(statistics.mean(values), 2),
                "mediane": round(statistics.median(values), 2),
            }
        return {"result": summary}
    
    return {"result": "Analyse non reconnue"}

# ─────────────────────────────────────────────
# AGENT PRINCIPAL
# ─────────────────────────────────────────────

def _detect_provider(api_key: str | None = None) -> tuple[str, str]:
    """
    Détecte automatiquement le provider LLM disponible.
    Priorité : Groq > Mistral > OpenAI > Anthropic > Fallback
    Retourne (provider_name, api_key)
    """
    # Clé explicitement passée → détecter son type par préfixe (priorité sur longueur)
    if api_key:
        if api_key.startswith("gsk_"):      return ("groq", api_key)
        if api_key.startswith("sk-ant-"):   return ("anthropic", api_key)
        if api_key.startswith("sk-"):       return ("openai", api_key)
        # Mistral : clé sans préfixe standard, entre 32 et 64 chars hexadécimaux
        if 28 <= len(api_key) <= 64 and not api_key.startswith("sk"):
            return ("mistral", api_key)

    # Variables d'environnement
    if os.environ.get("GROQ_API_KEY"):
        return ("groq", os.environ["GROQ_API_KEY"])
    if os.environ.get("MISTRAL_API_KEY"):
        return ("mistral", os.environ["MISTRAL_API_KEY"])
    if os.environ.get("OPENAI_API_KEY"):
        return ("openai", os.environ["OPENAI_API_KEY"])
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ("anthropic", os.environ["ANTHROPIC_API_KEY"])

    return ("fallback", "")


def _call_groq(messages: list, system: str, key: str, max_tokens: int = 1024) -> str:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}] + messages,
    }
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                      headers=headers, json=payload, timeout=30)
    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"]
    raise RuntimeError(f"Groq error {r.status_code}: {r.text[:200]}")


def _call_mistral(messages: list, system: str, key: str, max_tokens: int = 1024) -> str:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": "mistral-large-latest",
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}] + messages,
    }
    r = requests.post("https://api.mistral.ai/v1/chat/completions",
                      headers=headers, json=payload, timeout=30)
    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"]
    raise RuntimeError(f"Mistral error {r.status_code}: {r.text[:200]}")


def _call_openai(messages: list, system: str, key: str, max_tokens: int = 1024) -> str:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}] + messages,
    }
    r = requests.post("https://api.openai.com/v1/chat/completions",
                      headers=headers, json=payload, timeout=30)
    if r.status_code == 200:
        return r.json()["choices"][0]["message"]["content"]
    raise RuntimeError(f"OpenAI error {r.status_code}: {r.text[:200]}")


def _call_anthropic(messages: list, system: str, key: str, max_tokens: int = 1024) -> str:
    headers = {
        "Content-Type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    r = requests.post("https://api.anthropic.com/v1/messages",
                      headers=headers, json=payload, timeout=30)
    if r.status_code == 200:
        return r.json()["content"][0]["text"]
    raise RuntimeError(f"Anthropic error {r.status_code}: {r.text[:200]}")


def call_llm(messages: list, api_key: str| None = None, max_tokens: int = 1024) -> str:
    """
    Appelle le meilleur LLM disponible.
    Détection automatique du provider via les clés API (env ou paramètre).
    """
    provider, key = _detect_provider(api_key)

    try:
        if provider == "groq":
            return _call_groq(messages, SYSTEM_PROMPT, key, max_tokens)
        elif provider == "mistral":
            return _call_mistral(messages, SYSTEM_PROMPT, key, max_tokens)
        elif provider == "openai":
            return _call_openai(messages, SYSTEM_PROMPT, key, max_tokens)
        elif provider == "anthropic":
            return _call_anthropic(messages, SYSTEM_PROMPT, key, max_tokens)
    except Exception as e:
        print(f"⚠️ [{provider}] Erreur API : {e} — basculement sur le fallback SQL")

    # Fallback sans LLM
    return generate_sql_fallback(messages[-1]["content"])


def get_active_provider(api_key: str| None = None) -> str:
    """Retourne le nom du provider actif (pour affichage dans l'UI)."""
    provider, key = _detect_provider(api_key)
    labels = {
        "groq":      "🟢 Groq (Llama 3.3-70B)",
        "mistral":   "🟠 Mistral (Large)",
        "openai":    "🔵 OpenAI (GPT-4o-mini)",
        "anthropic": "🟣 Anthropic (Claude Sonnet 4)",
        "fallback":  "⚫ Mode hors-ligne (règles SQL)",
    }
    return labels.get(provider, provider)

def generate_sql_fallback(question: str) -> str:
    """Génération SQL basée sur des règles simples (sans API LLM)."""
    q = question.lower()
    
    if any(w in q for w in ["alerte", "surveiller", "critique", "mauvais avis"]):
        sql = "SELECT ProductName, CA, Notemoyenne, AvisNegatifs, Statut FROM V_ALERTS WHERE Statut != 'OK' ORDER BY Notemoyenne ASC LIMIT 10"
        reasoning = "Recherche des produits avec des alertes qualité"
    elif any(w in q for w in ["top", "meilleur", "best", "vente", "ca"]):
        sql = "SELECT ProductName, Category, CA, QuantiteVendue, Notemoyenne FROM V_PRODUCT_KPI ORDER BY CA DESC LIMIT 10"
        reasoning = "Classement des produits par chiffre d'affaires"
    elif any(w in q for w in ["client", "segment", "fidèle", "rentable"]):
        sql = "SELECT Nom, Pays, NbCommandes, CA_Total, PanierMoyen FROM V_CUSTOMER_KPI ORDER BY CA_Total DESC LIMIT 10"
        reasoning = "Analyse des meilleurs clients"
    elif any(w in q for w in ["catégorie", "categorie"]):
        sql = "SELECT Category, ROUND(SUM(CA),2) as CA_Total, ROUND(AVG(Notemoyenne),2) as Note_Moy, SUM(QuantiteVendue) as Qte FROM V_PRODUCT_KPI GROUP BY Category ORDER BY CA_Total DESC"
        reasoning = "Performance par catégorie"
    elif any(w in q for w in ["sentiment", "avis", "satisfaction"]):
        sql = "SELECT ProductName, Notemoyenne, NbAvis, AvisPositifs, AvisNegatifs FROM V_PRODUCT_KPI WHERE NbAvis > 5 ORDER BY Notemoyenne DESC LIMIT 15"
        reasoning = "Analyse du sentiment client par produit"
    elif any(w in q for w in ["pays", "country", "géographie"]):
        sql = "SELECT Pays, COUNT(*) as NbClients, ROUND(SUM(CA_Total),2) as CA FROM V_CUSTOMER_KPI GROUP BY Pays ORDER BY CA DESC"
        reasoning = "Analyse géographique"
    else:
        sql = "SELECT ProductName, CA, Marge, QuantiteVendue, Notemoyenne FROM V_PRODUCT_KPI ORDER BY CA DESC LIMIT 10"
        reasoning = "Vue générale des KPIs produits"
    
    return json.dumps({
        "sql": sql,
        "reasoning": reasoning,
        "answer_template": "Voici les résultats de votre analyse."
    })

def parse_agent_response(response_text: str) -> dict:
    """Parse la réponse JSON de l'agent."""
    # Cherche le JSON dans la réponse
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Fallback
    return {
        "sql": "SELECT ProductName, CA, Notemoyenne FROM V_PRODUCT_KPI ORDER BY CA DESC LIMIT 10",
        "reasoning": "Requête par défaut",
        "answer_template": "Voici les données disponibles."
    }

def format_natural_response(question: str, sql: str, data: list, reasoning: str, api_key: str | None = None) -> str:
    """Génère la réponse finale en langage naturel."""
    if not data:
        return "Aucune donnée ne correspond à votre question."

    data_str = json.dumps(data[:10], ensure_ascii=False, indent=2)
    provider, key = _detect_provider(api_key)

    if key and provider != "fallback":
        user_msg = f"""Question de l'utilisateur : {question}

Résultats SQL (top 10) :
{data_str}

Génère une réponse claire et utile en français pour un responsable marketing ou qualité.
Sois concis (3-5 phrases max), mets en avant les insights clés et recommande des actions si pertinent."""
        system_msg = "Tu es un analyste data senior pour SmartShop 360. Tes réponses sont concises, orientées action et en français."
        messages = [{"role": "user", "content": user_msg}]

        try:
            if provider == "groq":
                return _call_groq(messages, system_msg, key, max_tokens=512)
            elif provider == "mistral":
                return _call_mistral(messages, system_msg, key, max_tokens=512)
            elif provider == "openai":
                return _call_openai(messages, system_msg, key, max_tokens=512)
            elif provider == "anthropic":
                return _call_anthropic(messages, system_msg, key, max_tokens=512)
        except Exception:
            pass

    # Fallback : réponse structurée simple
    lines = [f"📊 **Analyse : {question}**\n"]
    lines.append(f"*Approche : {reasoning}*\n")
    lines.append(f"**{len(data)} résultat(s) trouvé(s) :**\n")
    for i, row in enumerate(data[:5], 1):
        parts = [f"{k}: **{v}**" for k, v in row.items()]
        lines.append(f"{i}. {' | '.join(parts)}")
    if len(data) > 5:
        lines.append(f"\n*...et {len(data)-5} autres résultats.*")
    return "\n".join(lines)

def run_agent(question: str, api_key: str | None = None, conversation_history: list | None = None) -> dict:
    """
    Boucle d'orchestration principale.
    
    Retourne :
    {
        "question": str,
        "sql": str,
        "reasoning": str,
        "data": list,
        "answer": str,
        "row_count": int
    }
    """
    if conversation_history is None:
        conversation_history = []
    
    # Étape 1 : L'agent génère le SQL
    messages = conversation_history + [{"role": "user", "content": question}]
    llm_response = call_llm(messages, api_key)
    parsed = parse_agent_response(llm_response)
    
    sql_query = parsed.get("sql", "")
    reasoning = parsed.get("reasoning", "")
    
    # Étape 2 : Exécution SQL (Tool)
    sql_result = execute_sql(sql_query)
    
    # Étape 3 : Réponse en langage naturel
    if sql_result["success"]:
        answer = format_natural_response(question, sql_query, sql_result["data"], reasoning, api_key)
    else:
        answer = f"❌ Erreur lors de l'exécution de la requête : {sql_result.get('error', 'Erreur inconnue')}"
    
    return {
        "question": question,
        "sql": sql_query,
        "reasoning": reasoning,
        "data": sql_result.get("data", []),
        "columns": sql_result.get("columns", []),
        "answer": answer,
        "row_count": sql_result.get("row_count", 0),
        "success": sql_result["success"]
    }

if __name__ == "__main__":
    print("🤖 Test de l'agent SmartShop 360\n")
    questions = [
        "Quels produits vendus à plus de 50 unités ont une note inférieure à 3 ?",
        "Quels sont nos 5 meilleurs produits en chiffre d'affaires ?",
        "Quels segments de clients sont les plus rentables ?",
    ]
    for q in questions:
        print(f"\n❓ {q}")
        result = run_agent(q)
        print(f"🔍 SQL: {result['sql']}")
        print(f"💡 {result['answer'][:300]}")
        print("-" * 60)

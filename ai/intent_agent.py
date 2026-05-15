"""
Intent Detection Agent — Healthcare AI Platform
Analyse les requêtes en langage naturel et détecte l'intention SQL.

Architecture :
  OllamaClient   → client HTTP bas niveau avec retry et timeout
  RuleBasedFallback → fallback si Ollama est indisponible
  IntentAgent    → orchestrateur principal (Ollama + fallback)

Output standard :
{
  "intent": "READ_ONLY" | "READ_WRITE",
  "action": "SELECT" | "INSERT" | "UPDATE" | "DELETE",
  "tables": ["Patient", ...],
  "attributes": ["first_name", ...],
  "filters": [{"column": "...", "operator": "...", "value": "..."}],
  "joins": [{"from_table": "...", "to_table": "...", "on": "..."}],
  "confidence": 0.92,
  "reasoning": "...",
  "source": "llm" | "fallback"
}
"""

import re
import json
import time
import logging
from typing import Optional

import requests

from ai.prompts import (
    SYSTEM_INTENT_ANALYZER,
    SYSTEM_TABLE_PREDICTOR,
    SYSTEM_ATTRIBUTE_EXTRACTOR,
    SYSTEM_ACTION_CLASSIFIER,
    build_analysis_prompt,
    build_table_prompt,
    build_attribute_prompt,
    build_action_prompt,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Constantes
# ─────────────────────────────────────────────────────────────────────────────

VALID_TABLES = {
    "Patient", "Medical_records", "Consultation",
    "Medical_staff", "Service", "Users", "AI_Query_Logs",
}

VALID_INTENTS  = {"READ_ONLY", "READ_WRITE"}
VALID_ACTIONS  = {"SELECT", "INSERT", "UPDATE", "DELETE"}

# Schéma de validation du résultat final
_RESULT_DEFAULTS = {
    "intent":     "READ_ONLY",
    "action":     "SELECT",
    "tables":     [],
    "attributes": [],
    "filters":    [],
    "joins":      [],
    "confidence": 0.0,
    "reasoning":  "",
    "source":     "llm",
}


# ─────────────────────────────────────────────────────────────────────────────
#  OllamaClient — client HTTP avec retry et timeout
# ─────────────────────────────────────────────────────────────────────────────

class OllamaClient:
    """
    Client HTTP pour l'API Ollama.
    Gère les retries, timeouts et parsing JSON robuste.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout:  int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Args:
            base_url:    URL de base d'Ollama (ex: http://localhost:11434).
            timeout:     Timeout HTTP en secondes.
            max_retries: Nombre de tentatives en cas d'échec.
            retry_delay: Délai entre les tentatives (secondes).
        """
        self.base_url    = base_url.rstrip("/")
        self.timeout     = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._session    = requests.Session()

    def generate(
        self,
        model:  str,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
    ) -> str:
        """
        Envoie une requête de génération à Ollama avec retry automatique.

        Args:
            model:       Nom du modèle Ollama (ex: "llama3").
            prompt:      Message utilisateur.
            system:      Prompt système (instructions).
            temperature: Température de génération (0.1 = déterministe).

        Returns:
            Texte généré par le modèle.

        Raises:
            OllamaUnavailableError: Si Ollama est inaccessible après tous les retries.
            OllamaTimeoutError:     Si le timeout est dépassé.
        """
        payload = {
            "model":  model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 1024,
                "stop": ["```", "---"],
            },
        }

        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "Ollama generate — model=%s attempt=%d/%d",
                    model, attempt, self.max_retries,
                )
                response = self._session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json().get("response", "").strip()
                logger.debug("Ollama réponse reçue — longueur=%d chars", len(result))
                return result

            except requests.exceptions.ConnectionError as exc:
                last_error = exc
                logger.warning(
                    "Ollama connexion échouée — attempt=%d/%d url=%s",
                    attempt, self.max_retries, self.base_url,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)  # backoff exponentiel

            except requests.exceptions.Timeout as exc:
                last_error = exc
                logger.warning(
                    "Ollama timeout (%ds) — attempt=%d/%d",
                    self.timeout, attempt, self.max_retries,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)

            except requests.exceptions.HTTPError as exc:
                logger.error("Ollama HTTP error — %s", str(exc))
                raise OllamaError(f"Erreur HTTP Ollama : {exc}") from exc

            except Exception as exc:
                logger.error("Erreur inattendue Ollama — %s", str(exc), exc_info=True)
                raise OllamaError(f"Erreur inattendue : {exc}") from exc

        raise OllamaUnavailableError(
            f"Ollama inaccessible après {self.max_retries} tentatives. "
            f"Dernière erreur : {last_error}"
        )

    def is_available(self) -> bool:
        """
        Vérifie si Ollama est disponible (health check rapide).

        Returns:
            True si Ollama répond, False sinon.
        """
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def extract_json(text: str) -> Optional[dict]:
        """
        Extrait un objet JSON depuis une réponse Ollama potentiellement bruitée.
        Gère les cas où le modèle ajoute du texte avant/après le JSON,
        ou entoure le JSON de balises markdown.

        Args:
            text: Texte brut retourné par Ollama.

        Returns:
            Dictionnaire Python ou None si aucun JSON valide trouvé.
        """
        if not text:
            return None

        # Stratégie 1 : parsing direct
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Stratégie 2 : extraire depuis un bloc ```json ... ```
        md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if md_match:
            try:
                return json.loads(md_match.group(1))
            except json.JSONDecodeError:
                pass

        # Stratégie 3 : trouver le premier { ... } complet
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        # Stratégie 4 : nettoyer les caractères parasites et réessayer
        cleaned = re.sub(r"[^\x20-\x7E\u00C0-\u024F\u0400-\u04FF\n\r\t]", "", text)
        brace_match2 = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if brace_match2:
            try:
                return json.loads(brace_match2.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning("Impossible d'extraire un JSON valide depuis : %s", text[:200])
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Exceptions personnalisées
# ─────────────────────────────────────────────────────────────────────────────

class OllamaError(Exception):
    """Erreur générique Ollama."""


class OllamaUnavailableError(OllamaError):
    """Ollama est inaccessible (connexion refusée ou timeout)."""


class OllamaTimeoutError(OllamaError):
    """Timeout dépassé lors de l'appel Ollama."""


# ─────────────────────────────────────────────────────────────────────────────
#  RuleBasedFallback — détection par règles si Ollama est indisponible
# ─────────────────────────────────────────────────────────────────────────────

class RuleBasedFallback:
    """
    Système de détection d'intention basé sur des règles lexicales.
    Utilisé quand Ollama est indisponible.
    Moins précis que le LLM mais toujours fonctionnel.
    """

    # Mots-clés par action
    _INSERT_KEYWORDS = {
        "ajouter", "créer", "creer", "nouveau", "nouvelle", "enregistrer",
        "inscrire", "admettre", "saisir", "insérer", "inserer", "ajoute",
        "crée", "cree", "nouvel", "nouvelle",
    }
    _UPDATE_KEYWORDS = {
        "modifier", "mettre à jour", "mettre a jour", "changer", "corriger",
        "actualiser", "éditer", "editer", "mise à jour", "mise a jour",
        "modifie", "change", "corrige", "update",
    }
    _DELETE_KEYWORDS = {
        "supprimer", "effacer", "retirer", "enlever", "annuler", "supprimer",
        "supprime", "efface", "retire", "enleve", "delete",
    }

    # Mots-clés par table
    _TABLE_KEYWORDS = {
        "Patient":         {"patient", "malade", "personne", "individu", "patients"},
        "Medical_records": {"dossier", "antécédent", "antecedent", "allergie", "allergies",
                            "groupe sanguin", "chronique", "chroniques", "médical", "medical",
                            "historique", "pathologie"},
        "Consultation":    {"consultation", "consultations", "diagnostic", "traitement",
                            "rapport", "visite", "rendez-vous", "rdv"},
        "Medical_staff":   {"médecin", "medecin", "docteur", "infirmier", "infirmière",
                            "personnel", "staff", "technicien", "administrateur", "dr"},
        "Service":         {"service", "département", "departement", "cardiologie",
                            "neurologie", "pédiatrie", "pediatrie", "urgences"},
    }

    # Mots-clés par attribut
    _ATTRIBUTE_KEYWORDS = {
        "first_name":        {"prénom", "prenom", "firstname"},
        "last_name":         {"nom", "lastname"},
        "birthdate":         {"naissance", "né", "nee", "né le", "date de naissance"},
        "age":               {"âge", "age", "ans"},
        "gender":            {"genre", "sexe", "homme", "femme", "masculin", "féminin"},
        "email":             {"email", "mail", "courriel"},
        "phone":             {"téléphone", "telephone", "tel", "phone"},
        "allergies":         {"allergie", "allergies", "allergique"},
        "chronic_diseases":  {"chronique", "chroniques", "diabète", "diabete", "hypertension",
                              "asthme", "cancer", "pathologie"},
        "blood_group":       {"groupe sanguin", "sang", "blood"},
        "medical_history":   {"historique", "antécédent", "antecedent"},
        "diagnosis":         {"diagnostic", "diagnostics"},
        "treatment":         {"traitement", "traitements", "prescription", "médicament"},
        "medical_report":    {"rapport", "compte-rendu", "compte rendu"},
        "name_staff":        {"nom du médecin", "nom du docteur", "nom du staff"},
        "speciality":        {"spécialité", "specialite", "spécialiste"},
        "service_name":      {"nom du service", "service"},
    }

    def analyze(self, prompt: str) -> dict:
        """
        Analyse une requête par règles lexicales.

        Args:
            prompt: Requête en langage naturel.

        Returns:
            Dictionnaire d'intention (même format que IntentAgent).
        """
        prompt_lower = prompt.lower()
        words        = set(re.findall(r"\b\w+\b", prompt_lower))

        # ── Détection action ──────────────────────────────────────────────
        action = "SELECT"
        intent = "READ_ONLY"

        if words & self._DELETE_KEYWORDS or any(k in prompt_lower for k in self._DELETE_KEYWORDS):
            action = "DELETE"
            intent = "READ_WRITE"
        elif words & self._UPDATE_KEYWORDS or any(k in prompt_lower for k in self._UPDATE_KEYWORDS):
            action = "UPDATE"
            intent = "READ_WRITE"
        elif words & self._INSERT_KEYWORDS or any(k in prompt_lower for k in self._INSERT_KEYWORDS):
            action = "INSERT"
            intent = "READ_WRITE"

        # ── Détection tables ──────────────────────────────────────────────
        tables = []
        for table, keywords in self._TABLE_KEYWORDS.items():
            if words & keywords or any(k in prompt_lower for k in keywords):
                tables.append(table)

        # Si aucune table détectée, Patient par défaut
        if not tables:
            tables = ["Patient"]

        # Ajouter Medical_records si Patient est présent et contexte médical
        medical_context = {"allergie", "chronique", "dossier", "groupe sanguin", "historique"}
        if "Patient" in tables and (words & medical_context or any(k in prompt_lower for k in medical_context)):
            if "Medical_records" not in tables:
                tables.append("Medical_records")

        # ── Détection attributs ───────────────────────────────────────────
        attributes = []
        for attr, keywords in self._ATTRIBUTE_KEYWORDS.items():
            if words & keywords or any(k in prompt_lower for k in keywords):
                attributes.append(attr)

        # ── Détection filtres simples ─────────────────────────────────────
        filters = []

        # Filtre genre
        if "homme" in prompt_lower or "masculin" in prompt_lower:
            filters.append({"column": "gender", "operator": "=", "value": "Male"})
        elif "femme" in prompt_lower or "féminin" in prompt_lower or "feminine" in prompt_lower:
            filters.append({"column": "gender", "operator": "=", "value": "Female"})

        # Filtre maladies chroniques
        diseases = ["diabète", "diabete", "hypertension", "asthme", "cancer",
                    "insuffisance rénale", "bpco", "arthrite"]
        for disease in diseases:
            if disease in prompt_lower:
                filters.append({
                    "column":   "chronic_diseases",
                    "operator": "LIKE",
                    "value":    f"%{disease}%",
                })

        # Filtre groupe sanguin
        blood_groups = ["a+", "a-", "b+", "b-", "ab+", "ab-", "o+", "o-"]
        for bg in blood_groups:
            if bg in prompt_lower:
                filters.append({
                    "column":   "blood_group",
                    "operator": "=",
                    "value":    bg.upper(),
                })

        # ── Calcul confiance ──────────────────────────────────────────────
        # La confiance est plus basse pour le fallback (max 0.65)
        confidence = 0.40
        if tables:
            confidence += 0.10
        if attributes:
            confidence += 0.10
        if filters:
            confidence += 0.05
        confidence = min(confidence, 0.65)

        return {
            "intent":     intent,
            "action":     action,
            "tables":     tables,
            "attributes": attributes,
            "filters":    filters,
            "joins":      [],
            "confidence": round(confidence, 2),
            "reasoning":  f"Analyse par règles lexicales (Ollama indisponible). "
                          f"Action détectée : {action}. Tables : {', '.join(tables)}.",
            "source":     "fallback",
        }


# ─────────────────────────────────────────────────────────────────────────────
#  IntentAgent — orchestrateur principal
# ─────────────────────────────────────────────────────────────────────────────

class IntentAgent:
    """
    Agent de détection d'intention pour les requêtes médicales.

    Stratégie :
      1. Tenter l'analyse via Ollama (LLM) — résultat précis
      2. Si Ollama indisponible → fallback par règles lexicales
      3. Valider et normaliser le résultat dans les deux cas

    Usage:
        agent = IntentAgent(base_url="http://localhost:11434", model="llama3")
        result = agent.analyze("Montre-moi les patients diabétiques")
    """

    def __init__(
        self,
        base_url:    str = "http://localhost:11434",
        model:       str = "llama3",
        timeout:     int = 30,
        max_retries: int = 3,
    ):
        """
        Args:
            base_url:    URL Ollama.
            model:       Modèle à utiliser (llama3, qwen2.5-coder, etc.).
            timeout:     Timeout HTTP en secondes.
            max_retries: Nombre de retries en cas d'échec.
        """
        self.model    = model
        self.client   = OllamaClient(
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.fallback = RuleBasedFallback()
        self._ollama_available: Optional[bool] = None  # Cache du statut

    # ── Méthode principale ────────────────────────────────────────────────────

    def analyze(self, prompt: str) -> dict:
        """
        Analyse une requête en langage naturel et retourne l'intention détectée.

        Args:
            prompt: Requête médicale en langage naturel (français ou anglais).

        Returns:
            Dictionnaire d'intention avec intent, action, tables, attributes,
            filters, joins, confidence, reasoning, source.
        """
        if not prompt or not prompt.strip():
            return {**_RESULT_DEFAULTS, "reasoning": "Requête vide", "confidence": 0.0}

        prompt = prompt.strip()
        start_time = time.perf_counter()

        # ── Tentative LLM ─────────────────────────────────────────────────
        try:
            result = self._analyze_with_llm(prompt)
            result["source"] = "llm"
            elapsed = round((time.perf_counter() - start_time) * 1000, 1)
            logger.info(
                "Intent détecté via LLM — action=%s tables=%s confidence=%.2f latency=%dms",
                result.get("action"), result.get("tables"), result.get("confidence", 0), elapsed,
            )
            return result

        except OllamaUnavailableError:
            logger.warning("Ollama indisponible — basculement sur le fallback par règles")
            result = self.fallback.analyze(prompt)
            elapsed = round((time.perf_counter() - start_time) * 1000, 1)
            logger.info(
                "Intent détecté via fallback — action=%s tables=%s confidence=%.2f latency=%dms",
                result.get("action"), result.get("tables"), result.get("confidence", 0), elapsed,
            )
            return result

        except Exception as exc:
            logger.error("Erreur inattendue IntentAgent — %s", str(exc), exc_info=True)
            # Fallback de sécurité
            result = self.fallback.analyze(prompt)
            result["reasoning"] += f" [Erreur LLM : {str(exc)[:100]}]"
            return result

    # ── Analyse via LLM ───────────────────────────────────────────────────────

    def _analyze_with_llm(self, prompt: str) -> dict:
        """
        Analyse via Ollama en une seule passe (prompt principal).
        Si le parsing JSON échoue, tente une analyse multi-étapes.

        Args:
            prompt: Requête utilisateur.

        Returns:
            Dictionnaire d'intention validé.

        Raises:
            OllamaUnavailableError: Si Ollama est inaccessible.
        """
        # Passe 1 : analyse complète en une seule requête
        raw_response = self.client.generate(
            model=self.model,
            prompt=build_analysis_prompt(prompt),
            system=SYSTEM_INTENT_ANALYZER,
            temperature=0.05,  # Très déterministe pour le JSON
        )

        parsed = OllamaClient.extract_json(raw_response)

        if parsed and self._is_valid_result(parsed):
            return self._normalize(parsed)

        # Passe 2 : analyse multi-étapes si la passe 1 échoue
        logger.debug("Passe 1 échouée — tentative analyse multi-étapes")
        return self._analyze_multistep(prompt)

    def _analyze_multistep(self, prompt: str) -> dict:
        """
        Analyse en plusieurs étapes spécialisées.
        Utilisée quand la passe unique retourne un JSON invalide.

        Étapes :
          1. Classifier l'action (SELECT/INSERT/UPDATE/DELETE)
          2. Détecter les tables
          3. Extraire les attributs

        Args:
            prompt: Requête utilisateur.

        Returns:
            Dictionnaire d'intention assemblé depuis les 3 étapes.
        """
        result = dict(_RESULT_DEFAULTS)

        # Étape 1 : classification action
        try:
            raw = self.client.generate(
                model=self.model,
                prompt=build_action_prompt(prompt),
                system=SYSTEM_ACTION_CLASSIFIER,
                temperature=0.05,
            )
            action_data = OllamaClient.extract_json(raw) or {}
            result["intent"]     = action_data.get("intent", "READ_ONLY")
            result["action"]     = action_data.get("action", "SELECT")
            result["confidence"] = action_data.get("confidence", 0.5)
        except OllamaUnavailableError:
            raise
        except Exception as exc:
            logger.warning("Étape 1 (action) échouée : %s", str(exc))

        # Étape 2 : détection tables
        try:
            raw = self.client.generate(
                model=self.model,
                prompt=build_table_prompt(prompt),
                system=SYSTEM_TABLE_PREDICTOR,
                temperature=0.05,
            )
            table_data = OllamaClient.extract_json(raw) or {}
            tables = [t for t in table_data.get("tables", []) if t in VALID_TABLES]
            result["tables"] = tables or ["Patient"]
        except OllamaUnavailableError:
            raise
        except Exception as exc:
            logger.warning("Étape 2 (tables) échouée : %s", str(exc))
            result["tables"] = ["Patient"]

        # Étape 3 : extraction attributs
        try:
            raw = self.client.generate(
                model=self.model,
                prompt=build_attribute_prompt(prompt, result["tables"]),
                system=SYSTEM_ATTRIBUTE_EXTRACTOR,
                temperature=0.05,
            )
            attr_data = OllamaClient.extract_json(raw) or {}
            result["attributes"] = attr_data.get("attributes", [])
        except OllamaUnavailableError:
            raise
        except Exception as exc:
            logger.warning("Étape 3 (attributs) échouée : %s", str(exc))

        result["reasoning"] = f"Analyse multi-étapes — {result['action']} sur {', '.join(result['tables'])}"
        return self._normalize(result)

    # ── Validation et normalisation ───────────────────────────────────────────

    @staticmethod
    def _is_valid_result(data: dict) -> bool:
        """
        Vérifie qu'un résultat LLM contient les champs minimaux valides.

        Args:
            data: Dictionnaire à valider.

        Returns:
            True si le résultat est utilisable.
        """
        return (
            isinstance(data, dict)
            and data.get("intent")  in VALID_INTENTS
            and data.get("action")  in VALID_ACTIONS
            and isinstance(data.get("tables"), list)
        )

    @staticmethod
    def _normalize(data: dict) -> dict:
        """
        Normalise et valide un résultat d'analyse.
        Corrige les valeurs invalides et applique les defaults.

        Args:
            data: Résultat brut à normaliser.

        Returns:
            Résultat normalisé et garanti valide.
        """
        result = dict(_RESULT_DEFAULTS)
        result.update(data)

        # Normaliser intent
        if result["intent"] not in VALID_INTENTS:
            result["intent"] = "READ_ONLY"

        # Normaliser action
        if result["action"] not in VALID_ACTIONS:
            result["action"] = "SELECT"

        # Cohérence intent/action
        if result["action"] in ("INSERT", "UPDATE", "DELETE"):
            result["intent"] = "READ_WRITE"
        elif result["action"] == "SELECT":
            result["intent"] = "READ_ONLY"

        # Filtrer les tables invalides
        result["tables"] = [
            t for t in (result.get("tables") or [])
            if t in VALID_TABLES
        ]
        if not result["tables"]:
            result["tables"] = ["Patient"]

        # Normaliser confidence
        try:
            conf = float(result.get("confidence", 0.5))
            result["confidence"] = round(max(0.0, min(1.0, conf)), 2)
        except (TypeError, ValueError):
            result["confidence"] = 0.5

        # Garantir les listes
        for key in ("attributes", "filters", "joins"):
            if not isinstance(result.get(key), list):
                result[key] = []

        # Garantir reasoning string
        if not isinstance(result.get("reasoning"), str):
            result["reasoning"] = ""

        return result

    # ── Utilitaires ───────────────────────────────────────────────────────────

    def is_ollama_available(self) -> bool:
        """Vérifie si Ollama est disponible."""
        return self.client.is_available()

    def get_status(self) -> dict:
        """
        Retourne le statut de l'agent.

        Returns:
            Dictionnaire avec model, ollama_available, fallback_ready.
        """
        available = self.client.is_available()
        return {
            "model":            self.model,
            "ollama_url":       self.client.base_url,
            "ollama_available": available,
            "fallback_ready":   True,
            "mode":             "llm" if available else "fallback",
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Factory — création depuis la config Flask
# ─────────────────────────────────────────────────────────────────────────────

def create_intent_agent(app=None) -> IntentAgent:
    """
    Crée un IntentAgent configuré depuis la config Flask ou les variables d'env.

    Args:
        app: Instance Flask optionnelle (pour lire la config).

    Returns:
        IntentAgent configuré.
    """
    import os

    if app is not None:
        base_url = app.config.get("OLLAMA_BASE_URL", "http://localhost:11434")
        model    = app.config.get("LLAMA3_MODEL",    "llama3")
        timeout  = app.config.get("OLLAMA_TIMEOUT",  30)
    else:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model    = os.getenv("LLAMA3_MODEL",    "llama3")
        timeout  = int(os.getenv("OLLAMA_TIMEOUT", "30"))

    return IntentAgent(
        base_url=base_url,
        model=model,
        timeout=min(timeout, 30),  # Cap à 30s pour l'intent detection
        max_retries=3,
    )

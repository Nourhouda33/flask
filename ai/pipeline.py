"""
AI Pipeline — Healthcare AI Platform
Orchestrateur principal du pipeline Text2SQL.

Deux sous-pipelines :

READ_ONLY :
  prompt → IntentAgent → TableMatcher → SchemaBuilder
         → SQLGenerator → SQLValidator → SQL final

READ_WRITE :
  prompt → IntentAgent → MissingAttributesDetector
         → [si manquants requis: retourner form_schema]
         → TableMatcher → SchemaBuilder
         → SQLGenerator → SQLValidator → SQL final

Logging : chaque requête est enregistrée dans AI_Query_Logs.
"""

import time
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  PipelineResult — résultat structuré du pipeline
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PipelineStep:
    """Représente une étape du pipeline avec son statut et sa durée."""

    name: str
    status: str  # "success" | "skipped" | "error" | "fallback"
    duration_ms: float = 0.0
    details: str = ""


@dataclass
class PipelineResult:
    """
    Résultat complet du pipeline Text2SQL.

    Champs :
      sql           → Requête SQL finale (vide si form_required=True)
      intent        → Intention détectée (READ_ONLY / READ_WRITE)
      action        → Action SQL (SELECT / INSERT / UPDATE / DELETE)
      tables        → Tables identifiées
      attributes    → Attributs extraits
      filters       → Filtres détectés
      missing_attrs → Attributs manquants (READ-WRITE uniquement)
      form_schema   → Schéma Angular si attributs manquants
      form_required → True si un formulaire doit être affiché
      confidence    → Score de confiance global (0-1)
      valid_sql     → True si le SQL est valide
      sql_errors    → Erreurs de validation SQL
      sql_warnings  → Avertissements SQL
      steps         → Étapes du pipeline avec durées
      latency_ms    → Latence totale en millisecondes
      source        → Source de l'intent ("llm" | "fallback")
      error         → Message d'erreur si le pipeline a échoué
    """

    sql: str = ""
    intent: str = "READ_ONLY"
    action: str = "SELECT"
    tables: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    filters: List[Dict] = field(default_factory=list)
    missing_attrs: List[Dict] = field(default_factory=list)
    form_schema: Optional[Dict] = None
    form_required: bool = False
    confidence: float = 0.0
    valid_sql: bool = False
    sql_errors: List[str] = field(default_factory=list)
    sql_warnings: List[str] = field(default_factory=list)
    steps: List[PipelineStep] = field(default_factory=list)
    latency_ms: float = 0.0
    source: str = "llm"
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        """Sérialise le résultat en dictionnaire JSON-safe."""
        d = asdict(self)
        # Convertir les PipelineStep en dicts
        d["steps"] = [asdict(s) for s in self.steps]
        return d

    def is_success(self) -> bool:
        """Retourne True si le pipeline a produit un résultat utilisable."""
        return self.error is None and (self.valid_sql or self.form_required)


# ─────────────────────────────────────────────────────────────────────────────
#  AIPipeline — orchestrateur principal
# ─────────────────────────────────────────────────────────────────────────────


class AIPipeline:
    """
    Orchestrateur du pipeline Text2SQL Healthcare.

    Usage:
        pipeline = AIPipeline.from_config(app)
        result   = pipeline.process("Liste les patients diabétiques", user_role="doctor")
    """

    def __init__(
        self,
        intent_agent,
        table_matcher,
        sql_generator,
        sql_validator,
        missing_detector,
        schema_builder,
        ollama_client,
        confidence_threshold: float = 0.75,
        log_to_db: bool = True,
    ):
        """
        Args:
            intent_agent:         Instance IntentAgent.
            table_matcher:        Instance TableMatcher.
            sql_generator:        Instance SQLGenerator.
            sql_validator:        Instance SQLValidator.
            missing_detector:     Instance MissingAttributesDetector.
            schema_builder:       Instance SchemaBuilder.
            ollama_client:        Instance OllamaClient.
            confidence_threshold: Seuil de confiance minimum.
            log_to_db:            Enregistrer les requêtes dans AI_Query_Logs.
        """
        self.intent_agent = intent_agent
        self.table_matcher = table_matcher
        self.sql_generator = sql_generator
        self.sql_validator = sql_validator
        self.missing_detector = missing_detector
        self.schema_builder = schema_builder
        self.client = ollama_client
        self.confidence_threshold = confidence_threshold
        self.log_to_db = log_to_db

    # ── Méthode principale ────────────────────────────────────────────────────

    def process(
        self,
        prompt: str,
        user_role: str = "staff",
        user_id: Optional[int] = None,
    ) -> PipelineResult:
        """
        Traite une requête en langage naturel et retourne le SQL généré.

        Args:
            prompt:    Requête médicale en langage naturel.
            user_role: Rôle de l'utilisateur (admin/doctor/staff/patient).
            user_id:   ID de l'utilisateur (pour le logging).

        Returns:
            PipelineResult avec le SQL final ou le schéma de formulaire.
        """
        pipeline_start = time.perf_counter()
        result = PipelineResult()

        if not prompt or not prompt.strip():
            result.error = "Requête vide"
            return result

        prompt = prompt.strip()

        try:
            # ── Étape 1 : Détection d'intention ───────────────────────────
            intent_info = self._step_intent(prompt, result)

            result.intent = intent_info.get("intent", "READ_ONLY")
            result.action = intent_info.get("action", "SELECT")
            result.attributes = intent_info.get("attributes", [])
            result.filters = intent_info.get("filters", [])
            result.confidence = intent_info.get("confidence", 0.0)
            result.source = intent_info.get("source", "llm")

            # ── Étape 2 : Vérification des droits ─────────────────────────
            access_error = self._check_access(result.action, user_role)
            if access_error:
                result.error = access_error
                result.steps.append(
                    PipelineStep(
                        name="access_check", status="error", details=access_error
                    )
                )
                return result

            result.steps.append(PipelineStep(name="access_check", status="success"))

            # ── Étape 3 : Détection attributs manquants (READ-WRITE) ───────
            if result.intent == "READ_WRITE":
                missing = self._step_missing_attrs(intent_info, result)
                result.missing_attrs = missing

                # Si des champs OBLIGATOIRES manquent → retourner le formulaire
                required_missing = [m for m in missing if m["required"]]
                if required_missing:
                    result.form_schema = self.missing_detector.generate_form_schema(
                        missing
                    )
                    result.form_required = True
                    result.steps.append(
                        PipelineStep(
                            name="missing_attrs",
                            status="success",
                            details=f"{len(required_missing)} champ(s) obligatoire(s) manquant(s)",
                        )
                    )
                    result.latency_ms = round(
                        (time.perf_counter() - pipeline_start) * 1000, 1
                    )
                    self._log_to_db(prompt, result, user_id)
                    return result

            # ── Étape 4 : Table Matching ───────────────────────────────────
            tables = self._step_table_matching(prompt, intent_info, result)
            result.tables = tables

            # ── Étape 5 : Contexte schéma ─────────────────────────────────
            schema_context = self._step_schema_context(tables, result)

            # ── Étape 6 : Génération SQL ───────────────────────────────────
            raw_sql = self._step_sql_generation(
                prompt, schema_context, intent_info, result
            )

            # ── Étape 7 : Validation SQL ───────────────────────────────────
            validation = self._step_sql_validation(raw_sql, schema_context, result)

            result.sql = validation["fixed_sql"]
            result.valid_sql = validation["valid"]
            result.sql_errors = validation["errors"]
            result.sql_warnings = validation["warnings"]

        except Exception as exc:
            logger.error("Erreur pipeline : %s", str(exc), exc_info=True)
            result.error = f"Erreur interne du pipeline : {str(exc)[:200]}"
            result.steps.append(
                PipelineStep(
                    name="pipeline_error", status="error", details=str(exc)[:200]
                )
            )

        finally:
            result.latency_ms = round((time.perf_counter() - pipeline_start) * 1000, 1)
            self._log_to_db(prompt, result, user_id)

        return result

    # ── Étapes du pipeline ────────────────────────────────────────────────────

    def _step_intent(self, prompt: str, result: PipelineResult) -> Dict:
        """Étape 1 : Détection d'intention via IntentAgent."""
        start = time.perf_counter()
        try:
            intent_info = self.intent_agent.analyze(prompt)
            duration = round((time.perf_counter() - start) * 1000, 1)
            result.steps.append(
                PipelineStep(
                    name="intent_detection",
                    status=(
                        "success" if intent_info.get("source") == "llm" else "fallback"
                    ),
                    duration_ms=duration,
                    details=(
                        f"action={intent_info.get('action')} "
                        f"confidence={intent_info.get('confidence', 0):.2f} "
                        f"source={intent_info.get('source')}"
                    ),
                )
            )
            return intent_info
        except Exception as exc:
            duration = round((time.perf_counter() - start) * 1000, 1)
            result.steps.append(
                PipelineStep(
                    name="intent_detection",
                    status="error",
                    duration_ms=duration,
                    details=str(exc),
                )
            )
            raise

    def _step_missing_attrs(
        self, intent_info: Dict, result: PipelineResult
    ) -> List[Dict]:
        """Étape 3 : Détection des attributs manquants."""
        start = time.perf_counter()
        try:
            missing = self.missing_detector.detect(intent_info)
            duration = round((time.perf_counter() - start) * 1000, 1)
            result.steps.append(
                PipelineStep(
                    name="missing_attrs_detection",
                    status="success",
                    duration_ms=duration,
                    details=f"{len(missing)} attribut(s) manquant(s)",
                )
            )
            return missing
        except Exception as exc:
            duration = round((time.perf_counter() - start) * 1000, 1)
            result.steps.append(
                PipelineStep(
                    name="missing_attrs_detection",
                    status="error",
                    duration_ms=duration,
                    details=str(exc),
                )
            )
            return []

    def _step_table_matching(
        self,
        prompt: str,
        intent_info: Dict,
        result: PipelineResult,
    ) -> List[str]:
        """Étape 4 : Table Matching hybride FAISS + LLM."""
        start = time.perf_counter()
        try:
            tables = self.table_matcher.match_tables(prompt, intent_info)
            duration = round((time.perf_counter() - start) * 1000, 1)
            result.steps.append(
                PipelineStep(
                    name="table_matching",
                    status="success",
                    duration_ms=duration,
                    details=f"tables={tables}",
                )
            )
            return tables
        except Exception as exc:
            duration = round((time.perf_counter() - start) * 1000, 1)
            # Fallback : utiliser les tables de l'IntentAgent
            fallback_tables = intent_info.get("tables", ["Patient"])
            result.steps.append(
                PipelineStep(
                    name="table_matching",
                    status="fallback",
                    duration_ms=duration,
                    details=f"Fallback tables={fallback_tables} — {str(exc)[:100]}",
                )
            )
            return fallback_tables

    def _step_schema_context(self, tables: List[str], result: PipelineResult) -> str:
        """Étape 5 : Construction du contexte schéma."""
        start = time.perf_counter()
        try:
            context = self.table_matcher.get_schema_context(tables)
            duration = round((time.perf_counter() - start) * 1000, 1)
            result.steps.append(
                PipelineStep(
                    name="schema_context",
                    status="success",
                    duration_ms=duration,
                    details=f"{len(context)} chars",
                )
            )
            return context
        except Exception as exc:
            duration = round((time.perf_counter() - start) * 1000, 1)
            fallback = self.schema_builder.build_schema_context(tables)
            result.steps.append(
                PipelineStep(
                    name="schema_context",
                    status="fallback",
                    duration_ms=duration,
                    details=str(exc)[:100],
                )
            )
            return fallback

    def _step_sql_generation(
        self,
        prompt: str,
        schema_context: str,
        intent_info: Dict,
        result: PipelineResult,
    ) -> str:
        """Étape 6 : Génération SQL via Qwen."""
        start = time.perf_counter()
        try:
            sql = self.sql_generator.generate(prompt, schema_context, intent_info)
            duration = round((time.perf_counter() - start) * 1000, 1)
            result.steps.append(
                PipelineStep(
                    name="sql_generation",
                    status="success",
                    duration_ms=duration,
                    details=f"{len(sql)} chars",
                )
            )
            return sql
        except Exception as exc:
            duration = round((time.perf_counter() - start) * 1000, 1)
            result.steps.append(
                PipelineStep(
                    name="sql_generation",
                    status="error",
                    duration_ms=duration,
                    details=str(exc)[:200],
                )
            )
            raise

    def _step_sql_validation(
        self,
        sql: str,
        schema_context: str,
        result: PipelineResult,
    ) -> Dict:
        """Étape 7 : Validation et correction SQL."""
        start = time.perf_counter()
        try:
            validation = self.sql_validator.validate_and_fix(sql, schema_context)
            duration = round((time.perf_counter() - start) * 1000, 1)
            status = "success" if validation["valid"] else "error"
            result.steps.append(
                PipelineStep(
                    name="sql_validation",
                    status=status,
                    duration_ms=duration,
                    details=(
                        f"valid={validation['valid']} "
                        f"fixes={len(validation.get('fixes', []))} "
                        f"errors={len(validation.get('errors', []))}"
                    ),
                )
            )
            return validation
        except Exception as exc:
            duration = round((time.perf_counter() - start) * 1000, 1)
            result.steps.append(
                PipelineStep(
                    name="sql_validation",
                    status="error",
                    duration_ms=duration,
                    details=str(exc)[:200],
                )
            )
            return {
                "valid": False,
                "fixed_sql": sql,
                "errors": [str(exc)],
                "warnings": [],
                "fixes": [],
            }

    # ── Contrôle d'accès ──────────────────────────────────────────────────────

    @staticmethod
    def _check_access(action: str, user_role: str) -> Optional[str]:
        """
        Vérifie que l'utilisateur a les droits pour l'action SQL demandée.
        Délègue au module RBAC centralisé (auth/rbac.py).

        Args:
            action:    Action SQL (SELECT/INSERT/UPDATE/DELETE/DDL).
            user_role: Rôle de l'utilisateur.

        Returns:
            Message d'erreur si accès refusé, None si autorisé.
        """
        from auth.rbac import check_sql_access

        return check_sql_access(user_role, action)

    # ── Logging DB ────────────────────────────────────────────────────────────

    def _log_to_db(
        self,
        prompt: str,
        result: PipelineResult,
        user_id: Optional[int],
    ) -> None:
        """
        Enregistre la requête dans AI_Query_Logs.
        Silencieux en cas d'erreur (ne doit pas bloquer le pipeline).
        """
        if not self.log_to_db:
            return

        try:
            from database.db import db
            from models.ai_query_log import AIQueryLog

            log = AIQueryLog(
                user_id=user_id,
                prompt=prompt,
                detected_intent=result.intent,
                detected_tables=result.tables,
                generated_sql=result.sql or None,
                execution_result=None,
                exact_match=None,
                confidence_score=result.confidence,
                latency_ms=int(result.latency_ms),
            )
            db.session.add(log)
            db.session.commit()
            logger.debug(
                "Requête loggée — id=%s latency=%dms", log.id_log, result.latency_ms
            )

        except Exception as exc:
            logger.warning("Impossible de logger la requête : %s", str(exc))

    # ── Statut ────────────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Retourne le statut de tous les composants du pipeline."""
        return {
            "intent_agent": self.intent_agent.get_status(),
            "table_matcher": self.table_matcher.get_status(),
            "sql_generator": {"model": self.sql_generator.model},
            "sql_validator": {"sqlcoder_model": self.sql_validator.sqlcoder_model},
            "confidence_threshold": self.confidence_threshold,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Factory — création depuis la config Flask
# ─────────────────────────────────────────────────────────────────────────────


def create_pipeline(app=None) -> AIPipeline:
    """
    Crée un AIPipeline complet depuis la config Flask ou les variables d'env.

    Args:
        app: Instance Flask optionnelle.

    Returns:
        AIPipeline configuré et prêt à l'emploi.
    """
    import os

    if app is not None:
        base_url = app.config.get("OLLAMA_BASE_URL", "http://localhost:11434")
        timeout = app.config.get("OLLAMA_TIMEOUT", 120)
        llama_model = app.config.get("LLAMA3_MODEL", "llama3")
        qwen_model = app.config.get("QWEN_CODER_MODEL", "qwen2.5-coder:7b-instruct")
        sqlcoder_model = app.config.get("SQLCODER_MODEL", "sqlcoder")
        index_path = app.config.get(
            "FAISS_INDEX_PATH", "embeddings/faiss_store/healthcare.index"
        )
        meta_path = app.config.get(
            "FAISS_METADATA_PATH", "embeddings/faiss_store/metadata.json"
        )
        conf_threshold = app.config.get("CONFIDENCE_THRESHOLD", 0.75)
    else:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        timeout = int(os.getenv("OLLAMA_TIMEOUT", "120"))
        llama_model = os.getenv("LLAMA3_MODEL", "llama3")
        qwen_model = os.getenv("QWEN_CODER_MODEL", "qwen2.5-coder:7b-instruct")
        sqlcoder_model = os.getenv("SQLCODER_MODEL", "sqlcoder")
        index_path = os.getenv(
            "FAISS_INDEX_PATH", "embeddings/faiss_store/healthcare.index"
        )
        meta_path = os.getenv(
            "FAISS_METADATA_PATH", "embeddings/faiss_store/metadata.json"
        )
        conf_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))

    # Créer le client Ollama partagé
    from ai.intent_agent import OllamaClient, IntentAgent

    ollama_client = OllamaClient(base_url=base_url, timeout=timeout, max_retries=3)

    # Créer les composants
    intent_agent = IntentAgent(base_url=base_url, model=llama_model, timeout=30)

    from ai.table_matcher import create_table_matcher

    table_matcher = create_table_matcher(
        index_path=index_path,
        metadata_path=meta_path,
    )

    from ai.sql_generator import SQLGenerator

    sql_generator = SQLGenerator(ollama_client=ollama_client, model=qwen_model)

    from ai.sql_validator import SQLValidator

    sql_validator = SQLValidator(
        ollama_client=ollama_client,
        sqlcoder_model=sqlcoder_model,
    )

    from ai.missing_attributes import MissingAttributesDetector

    missing_detector = MissingAttributesDetector()

    from database.schema_builder import schema_builder

    return AIPipeline(
        intent_agent=intent_agent,
        table_matcher=table_matcher,
        sql_generator=sql_generator,
        sql_validator=sql_validator,
        missing_detector=missing_detector,
        schema_builder=schema_builder,
        ollama_client=ollama_client,
        confidence_threshold=conf_threshold,
        log_to_db=True,
    )

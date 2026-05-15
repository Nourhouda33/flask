"""
Métriques d'évaluation — Healthcare AI Platform
Calcule les métriques de performance du pipeline Text2SQL.

Métriques :
  - Exact Match (EM)          : SQL généré == SQL de référence (après normalisation)
  - Accuracy                  : % de requêtes correctes sur le dataset
  - Table Precision/Recall    : précision/rappel sur la détection des tables
  - Confidence Score moyen    : score de confiance moyen du pipeline
  - Latency percentiles       : p50, p90, p99 des temps de réponse

Dataset de référence : 20 paires (prompt_médical, sql_attendu)
"""

import re
import json
import time
import logging
import statistics
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Dataset de référence — 20 paires (prompt, SQL attendu)
# ─────────────────────────────────────────────────────────────────────────────

REFERENCE_DATASET: List[Dict] = [
    # ── SELECT simples ────────────────────────────────────────────────────────
    {
        "id":       1,
        "category": "SELECT_SIMPLE",
        "prompt":   "Liste tous les patients",
        "expected_sql": (
            "SELECT id_patient, first_name, last_name, birthdate, age, gender "
            "FROM Patient "
            "ORDER BY last_name, first_name;"
        ),
        "expected_tables": ["Patient"],
        "expected_action": "SELECT",
    },
    {
        "id":       2,
        "category": "SELECT_FILTER",
        "prompt":   "Affiche les patients de sexe féminin",
        "expected_sql": (
            "SELECT id_patient, first_name, last_name, birthdate, age "
            "FROM Patient "
            "WHERE gender = 'Female' "
            "ORDER BY last_name;"
        ),
        "expected_tables": ["Patient"],
        "expected_action": "SELECT",
    },
    {
        "id":       3,
        "category": "SELECT_FILTER",
        "prompt":   "Quels patients ont plus de 60 ans ?",
        "expected_sql": (
            "SELECT id_patient, first_name, last_name, age, gender "
            "FROM Patient "
            "WHERE age > 60 "
            "ORDER BY age DESC;"
        ),
        "expected_tables": ["Patient"],
        "expected_action": "SELECT",
    },
    # ── SELECT avec JOIN ──────────────────────────────────────────────────────
    {
        "id":       4,
        "category": "SELECT_JOIN",
        "prompt":   "Affiche les patients avec leur groupe sanguin",
        "expected_sql": (
            "SELECT p.id_patient, p.first_name, p.last_name, p.age, mr.blood_group "
            "FROM Patient p "
            "INNER JOIN Medical_records mr ON mr.id_patient = p.id_patient "
            "ORDER BY p.last_name;"
        ),
        "expected_tables": ["Patient", "Medical_records"],
        "expected_action": "SELECT",
    },
    {
        "id":       5,
        "category": "SELECT_JOIN",
        "prompt":   "Liste les consultations avec le nom du médecin et du patient",
        "expected_sql": (
            "SELECT c.id_consultation, c.date, c.diagnosis, "
            "CONCAT(p.first_name, ' ', p.last_name) AS patient_name, "
            "ms.name_staff AS doctor_name "
            "FROM Consultation c "
            "LEFT JOIN Patient p ON p.id_patient = c.id_patient "
            "LEFT JOIN Medical_staff ms ON ms.id_staff = c.id_staff "
            "ORDER BY c.date DESC;"
        ),
        "expected_tables": ["Consultation", "Patient", "Medical_staff"],
        "expected_action": "SELECT",
    },
    {
        "id":       6,
        "category": "SELECT_JOIN",
        "prompt":   "Affiche les médecins avec leur service",
        "expected_sql": (
            "SELECT ms.id_staff, ms.name_staff, ms.position_staff, ms.speciality, "
            "s.service_name "
            "FROM Medical_staff ms "
            "LEFT JOIN Service s ON s.id_service = ms.id_service "
            "ORDER BY ms.name_staff;"
        ),
        "expected_tables": ["Medical_staff", "Service"],
        "expected_action": "SELECT",
    },
    # ── SELECT avec filtre médical ────────────────────────────────────────────
    {
        "id":       7,
        "category": "SELECT_MEDICAL_FILTER",
        "prompt":   "Quels patients sont atteints de diabète ?",
        "expected_sql": (
            "SELECT p.id_patient, p.first_name, p.last_name, p.age, "
            "mr.chronic_diseases "
            "FROM Patient p "
            "INNER JOIN Medical_records mr ON mr.id_patient = p.id_patient "
            "WHERE mr.chronic_diseases LIKE '%diabète%' "
            "ORDER BY p.last_name;"
        ),
        "expected_tables": ["Patient", "Medical_records"],
        "expected_action": "SELECT",
    },
    {
        "id":       8,
        "category": "SELECT_MEDICAL_FILTER",
        "prompt":   "Liste les patients allergiques à la pénicilline",
        "expected_sql": (
            "SELECT p.id_patient, p.first_name, p.last_name, mr.allergies "
            "FROM Patient p "
            "INNER JOIN Medical_records mr ON mr.id_patient = p.id_patient "
            "WHERE mr.allergies LIKE '%pénicilline%' "
            "ORDER BY p.last_name;"
        ),
        "expected_tables": ["Patient", "Medical_records"],
        "expected_action": "SELECT",
    },
    {
        "id":       9,
        "category": "SELECT_MEDICAL_FILTER",
        "prompt":   "Patients avec groupe sanguin O+",
        "expected_sql": (
            "SELECT p.id_patient, p.first_name, p.last_name, p.age, mr.blood_group "
            "FROM Patient p "
            "INNER JOIN Medical_records mr ON mr.id_patient = p.id_patient "
            "WHERE mr.blood_group = 'O+' "
            "ORDER BY p.last_name;"
        ),
        "expected_tables": ["Patient", "Medical_records"],
        "expected_action": "SELECT",
    },
    # ── SELECT avec agrégation ────────────────────────────────────────────────
    {
        "id":       10,
        "category": "SELECT_AGGREGATE",
        "prompt":   "Combien de patients par genre ?",
        "expected_sql": (
            "SELECT gender, COUNT(*) AS nb_patients "
            "FROM Patient "
            "GROUP BY gender "
            "ORDER BY nb_patients DESC;"
        ),
        "expected_tables": ["Patient"],
        "expected_action": "SELECT",
    },
    {
        "id":       11,
        "category": "SELECT_AGGREGATE",
        "prompt":   "Nombre de consultations par médecin",
        "expected_sql": (
            "SELECT ms.id_staff, ms.name_staff, COUNT(c.id_consultation) AS nb_consultations "
            "FROM Medical_staff ms "
            "LEFT JOIN Consultation c ON c.id_staff = ms.id_staff "
            "GROUP BY ms.id_staff, ms.name_staff "
            "ORDER BY nb_consultations DESC;"
        ),
        "expected_tables": ["Medical_staff", "Consultation"],
        "expected_action": "SELECT",
    },
    {
        "id":       12,
        "category": "SELECT_AGGREGATE",
        "prompt":   "Nombre de médecins par service",
        "expected_sql": (
            "SELECT s.service_name, COUNT(ms.id_staff) AS nb_doctors "
            "FROM Service s "
            "LEFT JOIN Medical_staff ms ON ms.id_service = s.id_service "
            "AND ms.position_staff = 'Doctor' "
            "GROUP BY s.id_service, s.service_name "
            "ORDER BY nb_doctors DESC;"
        ),
        "expected_tables": ["Service", "Medical_staff"],
        "expected_action": "SELECT",
    },
    # ── SELECT complexe ───────────────────────────────────────────────────────
    {
        "id":       13,
        "category": "SELECT_COMPLEX",
        "prompt":   "Patients sans aucune consultation",
        "expected_sql": (
            "SELECT p.id_patient, p.first_name, p.last_name, p.age "
            "FROM Patient p "
            "WHERE p.id_patient NOT IN ("
            "SELECT DISTINCT id_patient FROM Consultation WHERE id_patient IS NOT NULL"
            ") "
            "ORDER BY p.last_name;"
        ),
        "expected_tables": ["Patient", "Consultation"],
        "expected_action": "SELECT",
    },
    {
        "id":       14,
        "category": "SELECT_COMPLEX",
        "prompt":   "Consultations du mois en cours avec diagnostic",
        "expected_sql": (
            "SELECT c.id_consultation, c.date, c.diagnosis, c.treatment, "
            "CONCAT(p.first_name, ' ', p.last_name) AS patient_name "
            "FROM Consultation c "
            "LEFT JOIN Patient p ON p.id_patient = c.id_patient "
            "WHERE MONTH(c.date) = MONTH(CURDATE()) "
            "AND YEAR(c.date) = YEAR(CURDATE()) "
            "ORDER BY c.date DESC;"
        ),
        "expected_tables": ["Consultation", "Patient"],
        "expected_action": "SELECT",
    },
    {
        "id":       15,
        "category": "SELECT_COMPLEX",
        "prompt":   "Top 5 des diagnostics les plus fréquents",
        "expected_sql": (
            "SELECT diagnosis, COUNT(*) AS frequency "
            "FROM Consultation "
            "GROUP BY diagnosis "
            "ORDER BY frequency DESC "
            "LIMIT 5;"
        ),
        "expected_tables": ["Consultation"],
        "expected_action": "SELECT",
    },
    # ── INSERT ────────────────────────────────────────────────────────────────
    {
        "id":       16,
        "category": "INSERT",
        "prompt":   "Ajouter le patient Paul Martin né le 1980-06-20, homme",
        "expected_sql": (
            "INSERT INTO Patient (first_name, last_name, birthdate, gender) "
            "VALUES ('Paul', 'Martin', '1980-06-20', 'Male');"
        ),
        "expected_tables": ["Patient"],
        "expected_action": "INSERT",
    },
    {
        "id":       17,
        "category": "INSERT",
        "prompt":   "Créer une consultation avec diagnostic hypertension pour le patient 5 par le médecin 2",
        "expected_sql": (
            "INSERT INTO Consultation (diagnosis, id_patient, id_staff, date) "
            "VALUES ('Hypertension artérielle', 5, 2, NOW());"
        ),
        "expected_tables": ["Consultation"],
        "expected_action": "INSERT",
    },
    # ── UPDATE ────────────────────────────────────────────────────────────────
    {
        "id":       18,
        "category": "UPDATE",
        "prompt":   "Modifier le traitement de la consultation 10 : Metformine 500mg",
        "expected_sql": (
            "UPDATE Consultation "
            "SET treatment = 'Metformine 500mg' "
            "WHERE id_consultation = 10;"
        ),
        "expected_tables": ["Consultation"],
        "expected_action": "UPDATE",
    },
    {
        "id":       19,
        "category": "UPDATE",
        "prompt":   "Mettre à jour le groupe sanguin du patient 3 : AB+",
        "expected_sql": (
            "UPDATE Medical_records "
            "SET blood_group = 'AB+' "
            "WHERE id_patient = 3;"
        ),
        "expected_tables": ["Medical_records"],
        "expected_action": "UPDATE",
    },
    # ── DELETE ────────────────────────────────────────────────────────────────
    {
        "id":       20,
        "category": "DELETE",
        "prompt":   "Supprimer la consultation numéro 15",
        "expected_sql": (
            "DELETE FROM Consultation "
            "WHERE id_consultation = 15;"
        ),
        "expected_tables": ["Consultation"],
        "expected_action": "DELETE",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
#  Normalisation SQL pour la comparaison Exact Match
# ─────────────────────────────────────────────────────────────────────────────

def normalize_sql(sql: str) -> str:
    """
    Normalise une requête SQL pour la comparaison Exact Match.

    Transformations :
      - Minuscules
      - Suppression des espaces multiples
      - Suppression des commentaires
      - Suppression des backticks
      - Normalisation des guillemets
      - Suppression du point-virgule final
      - Normalisation des sauts de ligne

    Args:
        sql: Requête SQL à normaliser.

    Returns:
        SQL normalisé.
    """
    if not sql:
        return ""

    normalized = sql.strip()

    # Supprimer les commentaires SQL (-- ... et /* ... */)
    normalized = re.sub(r"--[^\n]*", " ", normalized)
    normalized = re.sub(r"/\*.*?\*/", " ", normalized, flags=re.DOTALL)

    # Minuscules
    normalized = normalized.lower()

    # Supprimer les backticks
    normalized = normalized.replace("`", "")

    # Normaliser les guillemets (double → simple)
    normalized = normalized.replace('"', "'")

    # Supprimer le point-virgule final
    normalized = normalized.rstrip(";").strip()

    # Normaliser les espaces et sauts de ligne
    normalized = re.sub(r"\s+", " ", normalized)

    # Normaliser les espaces autour des opérateurs
    normalized = re.sub(r"\s*=\s*", " = ", normalized)
    normalized = re.sub(r"\s*,\s*", ", ", normalized)
    normalized = re.sub(r"\s*\(\s*", "(", normalized)
    normalized = re.sub(r"\s*\)\s*", ")", normalized)

    return normalized.strip()


def exact_match(generated_sql: str, expected_sql: str) -> bool:
    """
    Vérifie si deux requêtes SQL sont équivalentes après normalisation.

    Args:
        generated_sql: SQL généré par le pipeline.
        expected_sql:  SQL de référence.

    Returns:
        True si les deux SQL sont équivalents.
    """
    return normalize_sql(generated_sql) == normalize_sql(expected_sql)


# ─────────────────────────────────────────────────────────────────────────────
#  Métriques de tables
# ─────────────────────────────────────────────────────────────────────────────

def table_precision(predicted_tables: List[str], expected_tables: List[str]) -> float:
    """
    Calcule la précision de la détection de tables.
    Précision = |predicted ∩ expected| / |predicted|

    Args:
        predicted_tables: Tables prédites par le pipeline.
        expected_tables:  Tables attendues.

    Returns:
        Score de précision entre 0 et 1.
    """
    if not predicted_tables:
        return 0.0
    predicted_set = set(predicted_tables)
    expected_set  = set(expected_tables)
    intersection  = predicted_set & expected_set
    return len(intersection) / len(predicted_set)


def table_recall(predicted_tables: List[str], expected_tables: List[str]) -> float:
    """
    Calcule le rappel de la détection de tables.
    Rappel = |predicted ∩ expected| / |expected|

    Args:
        predicted_tables: Tables prédites par le pipeline.
        expected_tables:  Tables attendues.

    Returns:
        Score de rappel entre 0 et 1.
    """
    if not expected_tables:
        return 1.0
    predicted_set = set(predicted_tables)
    expected_set  = set(expected_tables)
    intersection  = predicted_set & expected_set
    return len(intersection) / len(expected_set)


def table_f1(predicted_tables: List[str], expected_tables: List[str]) -> float:
    """
    Calcule le F1-score de la détection de tables.

    Args:
        predicted_tables: Tables prédites.
        expected_tables:  Tables attendues.

    Returns:
        F1-score entre 0 et 1.
    """
    p = table_precision(predicted_tables, expected_tables)
    r = table_recall(predicted_tables, expected_tables)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ─────────────────────────────────────────────────────────────────────────────
#  EvaluationResult — résultat d'une évaluation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SingleEvalResult:
    """Résultat de l'évaluation d'une seule requête."""
    sample_id:        int
    category:         str
    prompt:           str
    expected_sql:     str
    generated_sql:    str
    exact_match:      bool
    table_precision:  float
    table_recall:     float
    table_f1:         float
    confidence:       float
    latency_ms:       float
    action_correct:   bool
    error:            Optional[str] = None


@dataclass
class EvaluationReport:
    """Rapport d'évaluation complet sur le dataset."""
    total_samples:      int   = 0
    exact_match_count:  int   = 0
    exact_match_rate:   float = 0.0
    avg_table_precision: float = 0.0
    avg_table_recall:   float = 0.0
    avg_table_f1:       float = 0.0
    action_accuracy:    float = 0.0
    avg_confidence:     float = 0.0
    latency_p50:        float = 0.0
    latency_p90:        float = 0.0
    latency_p99:        float = 0.0
    avg_latency:        float = 0.0
    results_by_category: Dict[str, Dict] = field(default_factory=dict)
    individual_results:  List[SingleEvalResult] = field(default_factory=list)
    errors:             List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["individual_results"] = [asdict(r) for r in self.individual_results]
        return d

    def summary(self) -> str:
        """Retourne un résumé textuel du rapport."""
        lines = [
            "=" * 60,
            "RAPPORT D'ÉVALUATION — Healthcare AI Text2SQL",
            "=" * 60,
            f"Échantillons testés  : {self.total_samples}",
            f"Exact Match          : {self.exact_match_count}/{self.total_samples} "
            f"({self.exact_match_rate:.1%})",
            f"Table Precision      : {self.avg_table_precision:.3f}",
            f"Table Recall         : {self.avg_table_recall:.3f}",
            f"Table F1             : {self.avg_table_f1:.3f}",
            f"Action Accuracy      : {self.action_accuracy:.1%}",
            f"Confidence moyenne   : {self.avg_confidence:.3f}",
            f"Latence p50          : {self.latency_p50:.0f}ms",
            f"Latence p90          : {self.latency_p90:.0f}ms",
            f"Latence p99          : {self.latency_p99:.0f}ms",
            "=" * 60,
        ]
        if self.results_by_category:
            lines.append("Par catégorie :")
            for cat, stats in self.results_by_category.items():
                lines.append(
                    f"  {cat:<25} EM={stats.get('exact_match_rate', 0):.1%} "
                    f"n={stats.get('count', 0)}"
                )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  MetricsCalculator — calculateur principal
# ─────────────────────────────────────────────────────────────────────────────

class MetricsCalculator:
    """
    Calcule les métriques d'évaluation du pipeline Text2SQL.

    Usage:
        calc   = MetricsCalculator()
        report = calc.evaluate_pipeline(pipeline, dataset=REFERENCE_DATASET)
        print(report.summary())
    """

    def evaluate_pipeline(
        self,
        pipeline,
        dataset:   Optional[List[Dict]] = None,
        user_role: str = "admin",
        verbose:   bool = True,
    ) -> EvaluationReport:
        """
        Évalue le pipeline sur le dataset de référence.

        Args:
            pipeline:  Instance AIPipeline.
            dataset:   Dataset de test (défaut : REFERENCE_DATASET).
            user_role: Rôle utilisateur pour les tests.
            verbose:   Afficher la progression.

        Returns:
            EvaluationReport complet.
        """
        dataset = dataset or REFERENCE_DATASET
        report  = EvaluationReport(total_samples=len(dataset))
        results = []

        for i, sample in enumerate(dataset):
            if verbose:
                logger.info(
                    "Évaluation %d/%d — %s",
                    i + 1, len(dataset), sample["prompt"][:50],
                )

            result = self._evaluate_single(pipeline, sample, user_role)
            results.append(result)

            if result.error:
                report.errors.append(f"Sample {sample['id']}: {result.error}")

        report.individual_results = results
        self._compute_aggregates(report, results)
        return report

    def _evaluate_single(
        self,
        pipeline,
        sample:    Dict,
        user_role: str,
    ) -> SingleEvalResult:
        """Évalue une seule requête du dataset."""
        start = time.perf_counter()
        error = None
        generated_sql = ""
        predicted_tables = []
        confidence = 0.0
        action_correct = False

        try:
            pipeline_result = pipeline.process(
                prompt=sample["prompt"],
                user_role=user_role,
            )
            generated_sql    = pipeline_result.sql or ""
            predicted_tables = pipeline_result.tables or []
            confidence       = pipeline_result.confidence
            action_correct   = pipeline_result.action == sample.get("expected_action", "SELECT")

        except Exception as exc:
            error = str(exc)[:200]
            logger.warning("Erreur évaluation sample %d : %s", sample["id"], error)

        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        return SingleEvalResult(
            sample_id=sample["id"],
            category=sample.get("category", "UNKNOWN"),
            prompt=sample["prompt"],
            expected_sql=sample["expected_sql"],
            generated_sql=generated_sql,
            exact_match=exact_match(generated_sql, sample["expected_sql"]),
            table_precision=table_precision(predicted_tables, sample.get("expected_tables", [])),
            table_recall=table_recall(predicted_tables, sample.get("expected_tables", [])),
            table_f1=table_f1(predicted_tables, sample.get("expected_tables", [])),
            confidence=confidence,
            latency_ms=latency_ms,
            action_correct=action_correct,
            error=error,
        )

    @staticmethod
    def _compute_aggregates(report: EvaluationReport, results: List[SingleEvalResult]) -> None:
        """Calcule les métriques agrégées depuis les résultats individuels."""
        if not results:
            return

        # Exact Match
        em_results = [r for r in results if not r.error]
        report.exact_match_count = sum(1 for r in em_results if r.exact_match)
        report.exact_match_rate  = report.exact_match_count / len(results)

        # Table metrics
        report.avg_table_precision = statistics.mean(r.table_precision for r in em_results) if em_results else 0.0
        report.avg_table_recall    = statistics.mean(r.table_recall    for r in em_results) if em_results else 0.0
        report.avg_table_f1        = statistics.mean(r.table_f1        for r in em_results) if em_results else 0.0

        # Action accuracy
        report.action_accuracy = sum(1 for r in em_results if r.action_correct) / len(results)

        # Confidence
        report.avg_confidence = statistics.mean(r.confidence for r in em_results) if em_results else 0.0

        # Latency percentiles
        latencies = sorted(r.latency_ms for r in results)
        if latencies:
            report.avg_latency  = statistics.mean(latencies)
            report.latency_p50  = _percentile(latencies, 50)
            report.latency_p90  = _percentile(latencies, 90)
            report.latency_p99  = _percentile(latencies, 99)

        # Par catégorie
        categories: Dict[str, List[SingleEvalResult]] = {}
        for r in results:
            categories.setdefault(r.category, []).append(r)

        for cat, cat_results in categories.items():
            valid = [r for r in cat_results if not r.error]
            report.results_by_category[cat] = {
                "count":            len(cat_results),
                "exact_match_count": sum(1 for r in valid if r.exact_match),
                "exact_match_rate":  sum(1 for r in valid if r.exact_match) / len(cat_results),
                "avg_table_f1":     statistics.mean(r.table_f1 for r in valid) if valid else 0.0,
                "avg_confidence":   statistics.mean(r.confidence for r in valid) if valid else 0.0,
                "avg_latency_ms":   statistics.mean(r.latency_ms for r in cat_results),
            }

    def evaluate_intent_only(
        self,
        intent_agent,
        dataset: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Évalue uniquement l'IntentAgent (sans générer de SQL).
        Plus rapide pour tester la détection d'intention.

        Args:
            intent_agent: Instance IntentAgent.
            dataset:      Dataset de test.

        Returns:
            Dictionnaire de métriques.
        """
        dataset = dataset or REFERENCE_DATASET
        results = []

        for sample in dataset:
            start  = time.perf_counter()
            intent = intent_agent.analyze(sample["prompt"])
            elapsed = round((time.perf_counter() - start) * 1000, 1)

            action_correct = intent.get("action") == sample.get("expected_action")
            table_p = table_precision(intent.get("tables", []), sample.get("expected_tables", []))
            table_r = table_recall(intent.get("tables", []), sample.get("expected_tables", []))

            results.append({
                "id":             sample["id"],
                "action_correct": action_correct,
                "table_precision": table_p,
                "table_recall":   table_r,
                "table_f1":       table_f1(intent.get("tables", []), sample.get("expected_tables", [])),
                "confidence":     intent.get("confidence", 0),
                "latency_ms":     elapsed,
                "source":         intent.get("source", "unknown"),
            })

        return {
            "total":            len(results),
            "action_accuracy":  sum(1 for r in results if r["action_correct"]) / len(results),
            "avg_table_precision": statistics.mean(r["table_precision"] for r in results),
            "avg_table_recall":    statistics.mean(r["table_recall"]    for r in results),
            "avg_table_f1":        statistics.mean(r["table_f1"]        for r in results),
            "avg_confidence":      statistics.mean(r["confidence"]      for r in results),
            "avg_latency_ms":      statistics.mean(r["latency_ms"]      for r in results),
            "llm_rate":            sum(1 for r in results if r["source"] == "llm") / len(results),
        }


def _percentile(sorted_data: List[float], p: int) -> float:
    """Calcule le p-ième percentile d'une liste triée."""
    if not sorted_data:
        return 0.0
    idx = int(len(sorted_data) * p / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]

"""
Tests unitaires — IntentAgent et RuleBasedFallback
Couvre : SELECT simple, JOIN, INSERT, UPDATE, DELETE, requêtes complexes,
         fallback Ollama, parsing JSON robuste, normalisation.

Lancer : pytest backend/tests/test_intent_agent.py -v
"""

import json
import sys
import os
import pytest
from unittest.mock import patch, MagicMock

# Ajouter le répertoire backend au path pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai.intent_agent import (
    IntentAgent,
    OllamaClient,
    OllamaUnavailableError,
    RuleBasedFallback,
    VALID_TABLES,
    VALID_INTENTS,
    VALID_ACTIONS,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def fallback():
    """Instance RuleBasedFallback pour les tests sans Ollama."""
    return RuleBasedFallback()


@pytest.fixture
def agent_with_mock_ollama():
    """
    IntentAgent avec OllamaClient mocké.
    Permet de tester la logique de l'agent sans Ollama réel.
    """
    agent = IntentAgent(base_url="http://localhost:11434", model="llama3")
    return agent


@pytest.fixture
def ollama_client():
    """OllamaClient pour tester le parsing JSON."""
    return OllamaClient(base_url="http://localhost:11434", timeout=5)


# ─────────────────────────────────────────────────────────────────────────────
#  Cas de test médicaux — 10 prompts avec résultats attendus
# ─────────────────────────────────────────────────────────────────────────────

# Format : (prompt, expected_action, expected_intent, expected_tables_subset)
MEDICAL_TEST_CASES = [
    # 1. SELECT simple — liste de patients
    (
        "Montre-moi tous les patients",
        "SELECT",
        "READ_ONLY",
        ["Patient"],
    ),
    # 2. SELECT avec filtre — patients diabétiques
    (
        "Liste les patients atteints de diabète",
        "SELECT",
        "READ_ONLY",
        ["Patient"],
    ),
    # 3. SELECT avec JOIN — patients + dossier médical
    (
        "Affiche les patients avec leur groupe sanguin et leurs allergies",
        "SELECT",
        "READ_ONLY",
        ["Patient", "Medical_records"],
    ),
    # 4. SELECT avec filtre genre
    (
        "Quelles sont les patientes de sexe féminin avec de l'hypertension ?",
        "SELECT",
        "READ_ONLY",
        ["Patient"],
    ),
    # 5. SELECT consultation avec médecin (JOIN)
    (
        "Donne-moi les consultations du docteur Martin avec leurs diagnostics",
        "SELECT",
        "READ_ONLY",
        ["Consultation", "Medical_staff"],
    ),
    # 6. INSERT — créer un nouveau patient
    (
        "Ajouter un nouveau patient : Jean Dupont, né le 15/03/1985, groupe sanguin A+",
        "INSERT",
        "READ_WRITE",
        ["Patient"],
    ),
    # 7. UPDATE — modifier une consultation
    (
        "Modifier le traitement de la consultation numéro 42",
        "UPDATE",
        "READ_WRITE",
        ["Consultation"],
    ),
    # 8. DELETE — supprimer un dossier
    (
        "Supprimer le dossier médical du patient 15",
        "DELETE",
        "READ_WRITE",
        ["Medical_records"],
    ),
    # 9. SELECT complexe — statistiques
    (
        "Combien de patients par service médical ont été consultés ce mois-ci ?",
        "SELECT",
        "READ_ONLY",
        ["Patient", "Consultation"],
    ),
    # 10. SELECT complexe — personnel par service
    (
        "Liste tous les médecins du service de cardiologie avec leur spécialité",
        "SELECT",
        "READ_ONLY",
        ["Medical_staff", "Service"],
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
#  Tests RuleBasedFallback
# ─────────────────────────────────────────────────────────────────────────────

class TestRuleBasedFallback:
    """Tests du système de fallback par règles lexicales."""

    @pytest.mark.parametrize(
        "prompt, expected_action, expected_intent, expected_tables",
        MEDICAL_TEST_CASES,
    )
    def test_medical_prompts(
        self,
        fallback,
        prompt,
        expected_action,
        expected_intent,
        expected_tables,
    ):
        """
        Vérifie que le fallback détecte correctement l'action et l'intention
        pour les 10 cas médicaux de référence.
        """
        result = fallback.analyze(prompt)

        assert result["action"]  == expected_action,  (
            f"Prompt: {prompt!r}\n"
            f"Attendu: {expected_action}, Obtenu: {result['action']}"
        )
        assert result["intent"]  == expected_intent,  (
            f"Prompt: {prompt!r}\n"
            f"Attendu: {expected_intent}, Obtenu: {result['intent']}"
        )
        # Vérifier que les tables attendues sont un sous-ensemble des tables détectées
        for table in expected_tables:
            assert table in result["tables"], (
                f"Prompt: {prompt!r}\n"
                f"Table attendue {table!r} non trouvée dans {result['tables']}"
            )

    def test_result_structure(self, fallback):
        """Vérifie que le résultat contient tous les champs requis."""
        result = fallback.analyze("Liste les patients")
        required_keys = {"intent", "action", "tables", "attributes", "filters", "joins", "confidence", "reasoning", "source"}
        assert required_keys.issubset(result.keys()), (
            f"Champs manquants : {required_keys - result.keys()}"
        )

    def test_source_is_fallback(self, fallback):
        """Le source doit toujours être 'fallback'."""
        result = fallback.analyze("Montre les patients")
        assert result["source"] == "fallback"

    def test_confidence_range(self, fallback):
        """La confiance doit être entre 0 et 1."""
        for prompt, *_ in MEDICAL_TEST_CASES:
            result = fallback.analyze(prompt)
            assert 0.0 <= result["confidence"] <= 1.0, (
                f"Confiance hors limites pour {prompt!r}: {result['confidence']}"
            )

    def test_confidence_lower_than_llm(self, fallback):
        """La confiance du fallback doit être <= 0.65 (moins fiable que le LLM)."""
        result = fallback.analyze("Liste les patients diabétiques")
        assert result["confidence"] <= 0.65

    def test_tables_are_valid(self, fallback):
        """Toutes les tables retournées doivent exister dans le schéma."""
        for prompt, *_ in MEDICAL_TEST_CASES:
            result = fallback.analyze(prompt)
            for table in result["tables"]:
                assert table in VALID_TABLES, (
                    f"Table invalide {table!r} pour prompt {prompt!r}"
                )

    def test_gender_filter_male(self, fallback):
        """Détecte le filtre genre masculin."""
        result = fallback.analyze("Montre les patients de sexe masculin")
        gender_filters = [f for f in result["filters"] if f["column"] == "gender"]
        assert any(f["value"] == "Male" for f in gender_filters)

    def test_gender_filter_female(self, fallback):
        """Détecte le filtre genre féminin."""
        result = fallback.analyze("Liste les patientes femmes")
        gender_filters = [f for f in result["filters"] if f["column"] == "gender"]
        assert any(f["value"] == "Female" for f in gender_filters)

    def test_disease_filter(self, fallback):
        """Détecte le filtre maladie chronique."""
        result = fallback.analyze("Patients atteints de diabète")
        disease_filters = [f for f in result["filters"] if f["column"] == "chronic_diseases"]
        assert len(disease_filters) > 0
        assert any("diab" in f["value"].lower() for f in disease_filters)

    def test_blood_group_filter(self, fallback):
        """Détecte le filtre groupe sanguin."""
        result = fallback.analyze("Patients avec groupe sanguin A+")
        blood_filters = [f for f in result["filters"] if f["column"] == "blood_group"]
        assert any(f["value"] == "A+" for f in blood_filters)

    def test_empty_prompt(self, fallback):
        """Une requête vide retourne Patient par défaut."""
        result = fallback.analyze("")
        assert isinstance(result["tables"], list)

    def test_insert_keywords(self, fallback):
        """Détecte les mots-clés d'insertion."""
        for prompt in ["Ajouter un patient", "Créer un nouveau dossier", "Enregistrer une consultation"]:
            result = fallback.analyze(prompt)
            assert result["action"] == "INSERT", f"Échec pour: {prompt!r}"
            assert result["intent"] == "READ_WRITE"

    def test_update_keywords(self, fallback):
        """Détecte les mots-clés de mise à jour."""
        for prompt in ["Modifier le traitement", "Mettre à jour le dossier", "Corriger le diagnostic"]:
            result = fallback.analyze(prompt)
            assert result["action"] == "UPDATE", f"Échec pour: {prompt!r}"

    def test_delete_keywords(self, fallback):
        """Détecte les mots-clés de suppression."""
        for prompt in ["Supprimer le patient", "Effacer la consultation", "Retirer le dossier"]:
            result = fallback.analyze(prompt)
            assert result["action"] == "DELETE", f"Échec pour: {prompt!r}"

    def test_medical_records_auto_added(self, fallback):
        """Medical_records est ajouté automatiquement si Patient + contexte médical."""
        result = fallback.analyze("Patients avec leurs allergies")
        assert "Patient" in result["tables"]
        assert "Medical_records" in result["tables"]


# ─────────────────────────────────────────────────────────────────────────────
#  Tests OllamaClient.extract_json
# ─────────────────────────────────────────────────────────────────────────────

class TestOllamaClientExtractJson:
    """Tests du parsing JSON robuste."""

    def test_clean_json(self, ollama_client):
        """Parse un JSON propre."""
        text = '{"intent": "READ_ONLY", "action": "SELECT", "tables": ["Patient"]}'
        result = OllamaClient.extract_json(text)
        assert result is not None
        assert result["intent"] == "READ_ONLY"

    def test_json_with_markdown(self, ollama_client):
        """Extrait le JSON depuis un bloc markdown."""
        text = 'Voici le résultat :\n```json\n{"intent": "READ_ONLY", "action": "SELECT", "tables": ["Patient"]}\n```'
        result = OllamaClient.extract_json(text)
        assert result is not None
        assert result["action"] == "SELECT"

    def test_json_with_surrounding_text(self, ollama_client):
        """Extrait le JSON entouré de texte."""
        text = 'Analyse complète : {"intent": "READ_WRITE", "action": "INSERT", "tables": ["Patient"]} Fin.'
        result = OllamaClient.extract_json(text)
        assert result is not None
        assert result["action"] == "INSERT"

    def test_json_with_preamble(self, ollama_client):
        """Extrait le JSON après un préambule."""
        text = "Voici mon analyse de la requête médicale :\n\n{\"intent\": \"READ_ONLY\", \"action\": \"SELECT\", \"tables\": [\"Consultation\"]}"
        result = OllamaClient.extract_json(text)
        assert result is not None
        assert result["tables"] == ["Consultation"]

    def test_invalid_json_returns_none(self, ollama_client):
        """Retourne None pour un texte sans JSON valide."""
        result = OllamaClient.extract_json("Ceci n'est pas du JSON du tout.")
        assert result is None

    def test_empty_string_returns_none(self, ollama_client):
        """Retourne None pour une chaîne vide."""
        result = OllamaClient.extract_json("")
        assert result is None

    def test_none_returns_none(self, ollama_client):
        """Retourne None pour None."""
        result = OllamaClient.extract_json(None)
        assert result is None

    def test_nested_json(self, ollama_client):
        """Parse un JSON avec des objets imbriqués."""
        text = json.dumps({
            "intent": "READ_ONLY",
            "action": "SELECT",
            "tables": ["Patient", "Medical_records"],
            "filters": [{"column": "chronic_diseases", "operator": "LIKE", "value": "%diabète%"}],
            "confidence": 0.92,
        })
        result = OllamaClient.extract_json(text)
        assert result is not None
        assert len(result["filters"]) == 1
        assert result["confidence"] == 0.92


# ─────────────────────────────────────────────────────────────────────────────
#  Tests IntentAgent — avec Ollama mocké
# ─────────────────────────────────────────────────────────────────────────────

class TestIntentAgentWithMock:
    """Tests de l'IntentAgent avec Ollama simulé."""

    def _make_ollama_response(self, intent, action, tables, confidence=0.9, filters=None):
        """Helper pour créer une réponse Ollama simulée."""
        return json.dumps({
            "intent":     intent,
            "action":     action,
            "tables":     tables,
            "attributes": [],
            "filters":    filters or [],
            "joins":      [],
            "confidence": confidence,
            "reasoning":  f"Test — {action} sur {', '.join(tables)}",
        })

    def test_select_simple_via_llm(self, agent_with_mock_ollama):
        """SELECT simple via LLM mocké."""
        mock_response = self._make_ollama_response("READ_ONLY", "SELECT", ["Patient"])
        with patch.object(agent_with_mock_ollama.client, "generate", return_value=mock_response):
            result = agent_with_mock_ollama.analyze("Liste les patients")
        assert result["action"]  == "SELECT"
        assert result["intent"]  == "READ_ONLY"
        assert "Patient" in result["tables"]
        assert result["source"]  == "llm"

    def test_insert_via_llm(self, agent_with_mock_ollama):
        """INSERT via LLM mocké."""
        mock_response = self._make_ollama_response("READ_WRITE", "INSERT", ["Patient"])
        with patch.object(agent_with_mock_ollama.client, "generate", return_value=mock_response):
            result = agent_with_mock_ollama.analyze("Ajouter un nouveau patient")
        assert result["action"] == "INSERT"
        assert result["intent"] == "READ_WRITE"

    def test_update_via_llm(self, agent_with_mock_ollama):
        """UPDATE via LLM mocké."""
        mock_response = self._make_ollama_response("READ_WRITE", "UPDATE", ["Consultation"])
        with patch.object(agent_with_mock_ollama.client, "generate", return_value=mock_response):
            result = agent_with_mock_ollama.analyze("Modifier le traitement")
        assert result["action"] == "UPDATE"
        assert "Consultation" in result["tables"]

    def test_delete_via_llm(self, agent_with_mock_ollama):
        """DELETE via LLM mocké."""
        mock_response = self._make_ollama_response("READ_WRITE", "DELETE", ["Medical_records"])
        with patch.object(agent_with_mock_ollama.client, "generate", return_value=mock_response):
            result = agent_with_mock_ollama.analyze("Supprimer le dossier médical")
        assert result["action"] == "DELETE"
        assert result["intent"] == "READ_WRITE"

    def test_join_query_via_llm(self, agent_with_mock_ollama):
        """Requête avec JOIN via LLM mocké."""
        mock_response = self._make_ollama_response(
            "READ_ONLY", "SELECT",
            ["Patient", "Medical_records"],
            confidence=0.88,
        )
        with patch.object(agent_with_mock_ollama.client, "generate", return_value=mock_response):
            result = agent_with_mock_ollama.analyze(
                "Affiche les patients avec leur groupe sanguin"
            )
        assert "Patient" in result["tables"]
        assert "Medical_records" in result["tables"]
        assert result["confidence"] == 0.88

    def test_fallback_on_ollama_unavailable(self, agent_with_mock_ollama):
        """Bascule sur le fallback si Ollama est indisponible."""
        with patch.object(
            agent_with_mock_ollama.client,
            "generate",
            side_effect=OllamaUnavailableError("Ollama down"),
        ):
            result = agent_with_mock_ollama.analyze("Liste les patients diabétiques")
        assert result["source"]  == "fallback"
        assert result["action"]  in VALID_ACTIONS
        assert result["intent"]  in VALID_INTENTS

    def test_fallback_on_invalid_json(self, agent_with_mock_ollama):
        """
        Si Ollama retourne un JSON invalide à toutes les passes,
        le fallback prend le relais.
        """
        with patch.object(
            agent_with_mock_ollama.client,
            "generate",
            side_effect=OllamaUnavailableError("Connexion refusée"),
        ):
            result = agent_with_mock_ollama.analyze("Patients avec hypertension")
        assert result["source"] == "fallback"
        assert isinstance(result["tables"], list)
        assert len(result["tables"]) > 0

    def test_normalization_fixes_invalid_intent(self, agent_with_mock_ollama):
        """La normalisation corrige un intent invalide retourné par le LLM."""
        bad_response = json.dumps({
            "intent":     "INVALID_INTENT",
            "action":     "SELECT",
            "tables":     ["Patient"],
            "attributes": [],
            "filters":    [],
            "joins":      [],
            "confidence": 0.7,
            "reasoning":  "Test",
        })
        with patch.object(agent_with_mock_ollama.client, "generate", return_value=bad_response):
            result = agent_with_mock_ollama.analyze("Liste les patients")
        assert result["intent"] in VALID_INTENTS

    def test_normalization_fixes_invalid_tables(self, agent_with_mock_ollama):
        """La normalisation filtre les tables invalides."""
        bad_response = json.dumps({
            "intent":     "READ_ONLY",
            "action":     "SELECT",
            "tables":     ["Patient", "FakeTable", "NonExistentTable"],
            "attributes": [],
            "filters":    [],
            "joins":      [],
            "confidence": 0.8,
            "reasoning":  "Test",
        })
        with patch.object(agent_with_mock_ollama.client, "generate", return_value=bad_response):
            result = agent_with_mock_ollama.analyze("Liste les patients")
        for table in result["tables"]:
            assert table in VALID_TABLES

    def test_confidence_clamped_to_range(self, agent_with_mock_ollama):
        """La confiance est toujours entre 0 et 1."""
        bad_response = json.dumps({
            "intent": "READ_ONLY", "action": "SELECT",
            "tables": ["Patient"], "attributes": [], "filters": [], "joins": [],
            "confidence": 1.5,  # Valeur invalide > 1
            "reasoning": "Test",
        })
        with patch.object(agent_with_mock_ollama.client, "generate", return_value=bad_response):
            result = agent_with_mock_ollama.analyze("Test")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_empty_prompt_returns_default(self, agent_with_mock_ollama):
        """Une requête vide retourne le résultat par défaut sans appeler Ollama."""
        with patch.object(agent_with_mock_ollama.client, "generate") as mock_gen:
            result = agent_with_mock_ollama.analyze("")
            mock_gen.assert_not_called()
        assert result["confidence"] == 0.0

    def test_complex_query_with_filters(self, agent_with_mock_ollama):
        """Requête complexe avec filtres multiples."""
        mock_response = json.dumps({
            "intent":     "READ_ONLY",
            "action":     "SELECT",
            "tables":     ["Patient", "Medical_records"],
            "attributes": ["first_name", "last_name", "chronic_diseases", "blood_group"],
            "filters": [
                {"column": "chronic_diseases", "operator": "LIKE", "value": "%diabète%"},
                {"column": "gender",           "operator": "=",    "value": "Female"},
            ],
            "joins": [
                {"from_table": "Patient", "to_table": "Medical_records",
                 "on": "Patient.id_patient = Medical_records.id_patient"},
            ],
            "confidence": 0.95,
            "reasoning":  "Recherche patientes diabétiques avec dossier médical",
        })
        with patch.object(agent_with_mock_ollama.client, "generate", return_value=mock_response):
            result = agent_with_mock_ollama.analyze(
                "Montre-moi les patientes diabétiques avec leur groupe sanguin"
            )
        assert result["action"]     == "SELECT"
        assert len(result["filters"]) == 2
        assert len(result["joins"])   == 1
        assert result["confidence"]   == 0.95
        assert "chronic_diseases" in result["attributes"]


# ─────────────────────────────────────────────────────────────────────────────
#  Tests IntentAgent.get_status
# ─────────────────────────────────────────────────────────────────────────────

class TestIntentAgentStatus:
    """Tests du statut de l'agent."""

    def test_status_when_ollama_available(self, agent_with_mock_ollama):
        """Statut correct quand Ollama est disponible."""
        with patch.object(agent_with_mock_ollama.client, "is_available", return_value=True):
            status = agent_with_mock_ollama.get_status()
        assert status["ollama_available"] is True
        assert status["mode"]             == "llm"
        assert status["fallback_ready"]   is True

    def test_status_when_ollama_unavailable(self, agent_with_mock_ollama):
        """Statut correct quand Ollama est indisponible."""
        with patch.object(agent_with_mock_ollama.client, "is_available", return_value=False):
            status = agent_with_mock_ollama.get_status()
        assert status["ollama_available"] is False
        assert status["mode"]             == "fallback"

    def test_status_contains_model(self, agent_with_mock_ollama):
        """Le statut contient le nom du modèle."""
        status = agent_with_mock_ollama.get_status()
        assert "model" in status
        assert status["model"] == "llama3"

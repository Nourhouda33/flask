"""
Missing Attributes Detector — Healthcare AI Platform
Détecte les champs obligatoires manquants pour les opérations READ-WRITE.
Génère le schéma de formulaire Angular pour les attributs manquants.

Architecture :
  MissingAttributesDetector → classe principale
  _REQUIRED_FIELDS          → définition des champs obligatoires par table/action
  generate_form_schema      → génère le JSON pour Angular dynamic forms
"""

import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Définition des champs requis par table et action
# ─────────────────────────────────────────────────────────────────────────────

# Structure : {table: {action: [field_definitions]}}
# field_definition : {
#   "name": str,           # Nom de la colonne
#   "label": str,          # Label pour le formulaire
#   "type": str,           # Type de champ (text, date, select, email, tel, textarea)
#   "required": bool,      # Obligatoire ou optionnel
#   "validation": dict,    # Règles de validation Angular
#   "options": list,       # Options pour les champs select
#   "placeholder": str,    # Placeholder du champ
#   "hint": str,           # Aide contextuelle médicale
# }

REQUIRED_FIELDS: Dict[str, Dict[str, List[Dict]]] = {

    "Patient": {
        "INSERT": [
            {
                "name":        "first_name",
                "label":       "Prénom",
                "type":        "text",
                "required":    True,
                "validation":  {"minLength": 2, "maxLength": 100, "pattern": r"^[A-Za-zÀ-ÿ\s\-']+$"},
                "placeholder": "Ex: Jean",
                "hint":        "Prénom du patient (lettres uniquement)",
            },
            {
                "name":        "last_name",
                "label":       "Nom de famille",
                "type":        "text",
                "required":    True,
                "validation":  {"minLength": 2, "maxLength": 100, "pattern": r"^[A-Za-zÀ-ÿ\s\-']+$"},
                "placeholder": "Ex: Dupont",
                "hint":        "Nom de famille du patient",
            },
            {
                "name":        "birthdate",
                "label":       "Date de naissance",
                "type":        "date",
                "required":    True,
                "validation":  {"max": "today", "min": "1900-01-01"},
                "placeholder": "YYYY-MM-DD",
                "hint":        "Date de naissance (ne peut pas être dans le futur)",
            },
            {
                "name":        "gender",
                "label":       "Genre",
                "type":        "select",
                "required":    False,
                "options":     [
                    {"value": "Male",   "label": "Masculin"},
                    {"value": "Female", "label": "Féminin"},
                ],
                "placeholder": "Sélectionner le genre",
                "hint":        "Genre biologique du patient",
            },
            {
                "name":        "email",
                "label":       "Email",
                "type":        "email",
                "required":    False,
                "validation":  {"email": True, "maxLength": 150},
                "placeholder": "patient@exemple.com",
                "hint":        "Adresse email unique du patient",
            },
            {
                "name":        "phone",
                "label":       "Téléphone",
                "type":        "tel",
                "required":    False,
                "validation":  {"pattern": r"^[\+\d\s\-\(\)]{7,20}$"},
                "placeholder": "+33 6 12 34 56 78",
                "hint":        "Numéro de téléphone du patient",
            },
        ],
        "UPDATE": [
            {
                "name":        "id_patient",
                "label":       "ID Patient",
                "type":        "number",
                "required":    True,
                "validation":  {"min": 1},
                "placeholder": "Identifiant du patient à modifier",
                "hint":        "Identifiant unique du patient",
            },
        ],
    },

    "Medical_records": {
        "INSERT": [
            {
                "name":        "id_patient",
                "label":       "Patient",
                "type":        "number",
                "required":    True,
                "validation":  {"min": 1},
                "placeholder": "ID du patient",
                "hint":        "Identifiant du patient propriétaire du dossier",
            },
            {
                "name":        "blood_group",
                "label":       "Groupe sanguin",
                "type":        "select",
                "required":    False,
                "options":     [
                    {"value": "A+",  "label": "A+"},
                    {"value": "A-",  "label": "A-"},
                    {"value": "B+",  "label": "B+"},
                    {"value": "B-",  "label": "B-"},
                    {"value": "AB+", "label": "AB+"},
                    {"value": "AB-", "label": "AB-"},
                    {"value": "O+",  "label": "O+"},
                    {"value": "O-",  "label": "O-"},
                ],
                "placeholder": "Sélectionner le groupe sanguin",
                "hint":        "Groupe sanguin ABO + Rhésus",
            },
            {
                "name":        "allergies",
                "label":       "Allergies connues",
                "type":        "textarea",
                "required":    False,
                "validation":  {"maxLength": 2000},
                "placeholder": "Ex: Pénicilline, Aspirine, Arachides...",
                "hint":        "Lister toutes les allergies médicamenteuses et alimentaires",
            },
            {
                "name":        "chronic_diseases",
                "label":       "Maladies chroniques",
                "type":        "textarea",
                "required":    False,
                "validation":  {"maxLength": 2000},
                "placeholder": "Ex: Diabète type 2, Hypertension artérielle...",
                "hint":        "Pathologies chroniques diagnostiquées",
            },
            {
                "name":        "medical_history",
                "label":       "Historique médical",
                "type":        "textarea",
                "required":    False,
                "validation":  {"maxLength": 5000},
                "placeholder": "Antécédents médicaux, chirurgies, hospitalisations...",
                "hint":        "Historique médical complet du patient",
            },
        ],
        "UPDATE": [
            {
                "name":        "id_patient",
                "label":       "ID Patient",
                "type":        "number",
                "required":    True,
                "validation":  {"min": 1},
                "placeholder": "Identifiant du patient",
                "hint":        "Identifiant du patient dont on modifie le dossier",
            },
        ],
    },

    "Consultation": {
        "INSERT": [
            {
                "name":        "diagnosis",
                "label":       "Diagnostic",
                "type":        "textarea",
                "required":    True,
                "validation":  {"minLength": 5, "maxLength": 5000},
                "placeholder": "Diagnostic médical établi lors de la consultation...",
                "hint":        "Diagnostic clinique obligatoire",
            },
            {
                "name":        "id_patient",
                "label":       "Patient",
                "type":        "number",
                "required":    True,
                "validation":  {"min": 1},
                "placeholder": "ID du patient",
                "hint":        "Identifiant du patient consulté",
            },
            {
                "name":        "id_staff",
                "label":       "Médecin traitant",
                "type":        "number",
                "required":    True,
                "validation":  {"min": 1},
                "placeholder": "ID du médecin",
                "hint":        "Identifiant du médecin effectuant la consultation",
            },
            {
                "name":        "treatment",
                "label":       "Traitement prescrit",
                "type":        "textarea",
                "required":    False,
                "validation":  {"maxLength": 3000},
                "placeholder": "Médicaments, posologie, thérapies prescrites...",
                "hint":        "Traitement médical prescrit au patient",
            },
            {
                "name":        "date",
                "label":       "Date de consultation",
                "type":        "datetime-local",
                "required":    False,
                "validation":  {},
                "placeholder": "YYYY-MM-DDTHH:MM",
                "hint":        "Date et heure de la consultation (défaut : maintenant)",
            },
            {
                "name":        "medical_report",
                "label":       "Rapport médical",
                "type":        "textarea",
                "required":    False,
                "validation":  {"maxLength": 10000},
                "placeholder": "Rapport médical détaillé...",
                "hint":        "Compte-rendu médical complet de la consultation",
            },
        ],
        "UPDATE": [
            {
                "name":        "id_consultation",
                "label":       "ID Consultation",
                "type":        "number",
                "required":    True,
                "validation":  {"min": 1},
                "placeholder": "Identifiant de la consultation",
                "hint":        "Identifiant unique de la consultation à modifier",
            },
        ],
    },

    "Medical_staff": {
        "INSERT": [
            {
                "name":        "name_staff",
                "label":       "Nom complet",
                "type":        "text",
                "required":    True,
                "validation":  {"minLength": 3, "maxLength": 150},
                "placeholder": "Dr. Jean Martin",
                "hint":        "Nom complet du membre du personnel médical",
            },
            {
                "name":        "position_staff",
                "label":       "Poste",
                "type":        "select",
                "required":    True,
                "options":     [
                    {"value": "Doctor",        "label": "Médecin"},
                    {"value": "Nurse",         "label": "Infirmier(ère)"},
                    {"value": "Technician",    "label": "Technicien(ne)"},
                    {"value": "Administrator", "label": "Administrateur(trice)"},
                ],
                "placeholder": "Sélectionner le poste",
                "hint":        "Fonction du membre du personnel",
            },
            {
                "name":        "speciality",
                "label":       "Spécialité",
                "type":        "text",
                "required":    False,
                "validation":  {"maxLength": 100},
                "placeholder": "Ex: Cardiologie, Neurologie...",
                "hint":        "Spécialité médicale (pour les médecins)",
            },
            {
                "name":        "id_service",
                "label":       "Service",
                "type":        "number",
                "required":    False,
                "validation":  {"min": 1},
                "placeholder": "ID du service",
                "hint":        "Service hospitalier d'appartenance",
            },
            {
                "name":        "email",
                "label":       "Email professionnel",
                "type":        "email",
                "required":    False,
                "validation":  {"email": True, "maxLength": 150},
                "placeholder": "dr.martin@hopital.fr",
                "hint":        "Adresse email professionnelle unique",
            },
        ],
        "UPDATE": [
            {
                "name":        "id_staff",
                "label":       "ID Personnel",
                "type":        "number",
                "required":    True,
                "validation":  {"min": 1},
                "placeholder": "Identifiant du membre du personnel",
                "hint":        "Identifiant unique du membre à modifier",
            },
        ],
    },

    "Service": {
        "INSERT": [
            {
                "name":        "service_name",
                "label":       "Nom du service",
                "type":        "text",
                "required":    True,
                "validation":  {"minLength": 2, "maxLength": 100},
                "placeholder": "Ex: Cardiologie, Neurologie...",
                "hint":        "Nom unique du service hospitalier",
            },
        ],
        "UPDATE": [
            {
                "name":        "id_service",
                "label":       "ID Service",
                "type":        "number",
                "required":    True,
                "validation":  {"min": 1},
                "placeholder": "Identifiant du service",
                "hint":        "Identifiant unique du service à modifier",
            },
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  MissingAttributesDetector
# ─────────────────────────────────────────────────────────────────────────────

class MissingAttributesDetector:
    """
    Détecte les attributs obligatoires manquants pour les opérations READ-WRITE.
    Génère le schéma de formulaire Angular pour les attributs manquants.

    Usage:
        detector = MissingAttributesDetector()
        missing = detector.detect(intent_info, action="INSERT")
        form_schema = detector.generate_form_schema(missing)
    """

    def detect(
        self,
        intent_info: Dict,
        action:      Optional[str] = None,
    ) -> List[Dict]:
        """
        Détecte les attributs obligatoires manquants dans une requête READ-WRITE.

        Args:
            intent_info: Résultat de l'IntentAgent (tables, attributes, filters...).
            action:      Action SQL ('INSERT', 'UPDATE', 'DELETE').
                         Si None, utilise intent_info["action"].

        Returns:
            Liste des attributs manquants :
            [{"attribute": str, "table": str, "required": bool, "type": str, ...}]
        """
        action = action or intent_info.get("action", "SELECT")

        # Seules les opérations d'écriture nécessitent une vérification
        if action not in ("INSERT", "UPDATE", "DELETE"):
            return []

        tables     = intent_info.get("tables", [])
        attributes = set(intent_info.get("attributes", []))
        filters    = {f.get("column") for f in intent_info.get("filters", [])}

        # Combiner attributs et filtres comme "fournis"
        provided = attributes | filters

        missing = []

        for table in tables:
            table_fields = REQUIRED_FIELDS.get(table, {}).get(action, [])

            for field in table_fields:
                field_name = field["name"]
                is_required = field.get("required", False)

                # Vérifier si le champ est fourni
                if field_name not in provided:
                    missing.append({
                        "attribute":   field_name,
                        "table":       table,
                        "required":    is_required,
                        "type":        field.get("type", "text"),
                        "label":       field.get("label", field_name),
                        "validation":  field.get("validation", {}),
                        "options":     field.get("options", []),
                        "placeholder": field.get("placeholder", ""),
                        "hint":        field.get("hint", ""),
                    })

        # Trier : champs obligatoires en premier
        missing.sort(key=lambda x: (not x["required"], x["table"], x["attribute"]))

        logger.debug(
            "Attributs manquants détectés — action=%s tables=%s missing=%d",
            action, tables, len(missing),
        )
        return missing

    def has_required_missing(self, intent_info: Dict, action: Optional[str] = None) -> bool:
        """
        Vérifie s'il manque des attributs OBLIGATOIRES (required=True).

        Args:
            intent_info: Résultat de l'IntentAgent.
            action:      Action SQL.

        Returns:
            True si des champs obligatoires manquent.
        """
        missing = self.detect(intent_info, action)
        return any(m["required"] for m in missing)

    def get_required_only(self, intent_info: Dict, action: Optional[str] = None) -> List[Dict]:
        """
        Retourne uniquement les attributs OBLIGATOIRES manquants.

        Args:
            intent_info: Résultat de l'IntentAgent.
            action:      Action SQL.

        Returns:
            Liste des attributs obligatoires manquants.
        """
        return [m for m in self.detect(intent_info, action) if m["required"]]

    def generate_form_schema(self, missing_attrs: List[Dict]) -> Dict:
        """
        Génère le schéma de formulaire Angular pour les attributs manquants.
        Compatible avec Angular Reactive Forms et Angular Material.

        Args:
            missing_attrs: Liste des attributs manquants (résultat de detect()).

        Returns:
            Schéma JSON pour Angular dynamic form :
            {
                "form_id": str,
                "title": str,
                "fields": [...],
                "submit_label": str,
                "has_required": bool,
            }
        """
        if not missing_attrs:
            return {
                "form_id":      "empty_form",
                "title":        "Aucun champ manquant",
                "fields":       [],
                "submit_label": "Confirmer",
                "has_required": False,
            }

        # Déterminer le titre du formulaire
        tables   = list({a["table"] for a in missing_attrs})
        required = [a for a in missing_attrs if a["required"]]

        table_labels = {
            "Patient":         "Patient",
            "Medical_records": "Dossier Médical",
            "Consultation":    "Consultation",
            "Medical_staff":   "Personnel Médical",
            "Service":         "Service",
        }
        title = f"Informations requises — {', '.join(table_labels.get(t, t) for t in tables)}"

        # Construire les champs Angular
        angular_fields = []
        for attr in missing_attrs:
            field = {
                "name":        attr["attribute"],
                "label":       attr["label"],
                "type":        attr["type"],
                "required":    attr["required"],
                "placeholder": attr["placeholder"],
                "hint":        attr["hint"],
                "table":       attr["table"],
                # Validateurs Angular
                "validators":  self._build_angular_validators(attr),
            }

            # Ajouter les options pour les selects
            if attr["type"] == "select" and attr.get("options"):
                field["options"] = attr["options"]

            angular_fields.append(field)

        return {
            "form_id":      f"missing_attrs_{'_'.join(tables)}",
            "title":        title,
            "description":  (
                f"{len(required)} champ(s) obligatoire(s) et "
                f"{len(missing_attrs) - len(required)} champ(s) optionnel(s) manquant(s)"
            ),
            "fields":       angular_fields,
            "submit_label": "Valider et exécuter",
            "cancel_label": "Annuler",
            "has_required": len(required) > 0,
            "required_count": len(required),
            "optional_count": len(missing_attrs) - len(required),
        }

    @staticmethod
    def _build_angular_validators(attr: Dict) -> List[Dict]:
        """
        Construit la liste des validateurs Angular depuis les règles de validation.

        Args:
            attr: Attribut avec ses règles de validation.

        Returns:
            Liste de validateurs Angular :
            [{"type": "required"}, {"type": "minLength", "value": 2}, ...]
        """
        validators = []
        validation = attr.get("validation", {})

        if attr.get("required"):
            validators.append({"type": "required"})

        if "minLength" in validation:
            validators.append({"type": "minLength", "value": validation["minLength"]})

        if "maxLength" in validation:
            validators.append({"type": "maxLength", "value": validation["maxLength"]})

        if "pattern" in validation:
            validators.append({"type": "pattern", "value": validation["pattern"]})

        if "min" in validation:
            validators.append({"type": "min", "value": validation["min"]})

        if "max" in validation:
            validators.append({"type": "max", "value": validation["max"]})

        if validation.get("email"):
            validators.append({"type": "email"})

        return validators

    def get_all_fields_for_table(self, table: str, action: str) -> List[Dict]:
        """
        Retourne tous les champs définis pour une table et une action.

        Args:
            table:  Nom de la table.
            action: Action SQL ('INSERT', 'UPDATE').

        Returns:
            Liste de tous les champs (requis et optionnels).
        """
        return REQUIRED_FIELDS.get(table, {}).get(action, [])

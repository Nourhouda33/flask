"""
Service métier — Patient
Sépare la logique métier des routes Flask.
Toutes les opérations complexes sur les patients passent par ce module.
"""

import logging
from datetime import date
from typing import Optional

from sqlalchemy import or_, func

from database.db import db, paginate_query, pagination_meta
from models.patient        import Patient
from models.medical_record import MedicalRecord
from models.consultation   import Consultation

logger = logging.getLogger(__name__)

# Valeurs acceptées pour le filtre gender
VALID_GENDERS = {"Male", "Female"}

# Valeurs acceptées pour blood_group
VALID_BLOOD_GROUPS = {"A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"}


# ─────────────────────────────────────────────────────────────────────────────
#  search_patients
# ─────────────────────────────────────────────────────────────────────────────

def search_patients(
    search:   Optional[str] = None,
    gender:   Optional[str] = None,
    age_min:  Optional[int] = None,
    age_max:  Optional[int] = None,
    page:     int = 1,
    per_page: int = 20,
) -> dict:
    """
    Recherche paginée de patients avec filtres multiples.

    Args:
        search:   Recherche textuelle sur prénom, nom ou email.
        gender:   Filtre par genre ('Male' ou 'Female').
        age_min:  Âge minimum (utilise la colonne virtuelle MySQL).
        age_max:  Âge maximum.
        page:     Numéro de page (1-indexé).
        per_page: Éléments par page (max 100).

    Returns:
        Dictionnaire {"items": [...], "meta": {...}}.
    """
    query = Patient.query

    # ── Filtre recherche textuelle ─────────────────────────────────────────
    if search:
        term = f"%{search.strip()}%"
        query = query.filter(
            or_(
                Patient.first_name.ilike(term),
                Patient.last_name.ilike(term),
                Patient.email.ilike(term),
                func.concat(Patient.first_name, " ", Patient.last_name).ilike(term),
            )
        )

    # ── Filtre genre ───────────────────────────────────────────────────────
    if gender and gender in VALID_GENDERS:
        query = query.filter(Patient.gender == gender)

    # ── Filtre âge (colonne virtuelle MySQL) ───────────────────────────────
    # age = TIMESTAMPDIFF(YEAR, birthdate, CURDATE()) — lecture seule
    if age_min is not None:
        query = query.filter(Patient.age >= age_min)
    if age_max is not None:
        query = query.filter(Patient.age <= age_max)

    # ── Tri par défaut : plus récents en premier ───────────────────────────
    query = query.order_by(Patient.created_at.desc())

    pagination = paginate_query(query, page=page, per_page=per_page)

    return {
        "items": [p.to_dict() for p in pagination.items],
        "meta":  pagination_meta(pagination),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  get_patient_full_profile
# ─────────────────────────────────────────────────────────────────────────────

def get_patient_full_profile(patient_id: int) -> Optional[dict]:
    """
    Retourne le profil complet d'un patient :
    données démographiques + dossier médical + consultations récentes.

    Args:
        patient_id: Identifiant du patient.

    Returns:
        Dictionnaire complet ou None si introuvable.
    """
    patient = Patient.query.get(patient_id)
    if not patient:
        return None

    # Récupérer les 10 dernières consultations (triées par date desc)
    recent_consultations = (
        Consultation.query
        .filter_by(id_patient=patient_id)
        .order_by(Consultation.date.desc())
        .limit(10)
        .all()
    )

    profile = patient.to_dict(include_record=True)
    profile["recent_consultations"] = [
        c.to_dict(include_staff=True) for c in recent_consultations
    ]
    profile["consultation_count"] = (
        Consultation.query.filter_by(id_patient=patient_id).count()
    )

    return profile


# ─────────────────────────────────────────────────────────────────────────────
#  create_patient_with_record
# ─────────────────────────────────────────────────────────────────────────────

def create_patient_with_record(data: dict) -> tuple[Patient, MedicalRecord]:
    """
    Crée un patient ET son dossier médical dans une seule transaction.
    Si la création du dossier échoue, le patient est aussi annulé (rollback).

    Args:
        data: Dictionnaire contenant les champs patient et medical_record optionnels.
              Champs patient  : first_name, last_name, birthdate, gender, email, phone
              Champs record   : allergies, chronic_diseases, blood_group, medical_history

    Returns:
        Tuple (Patient, MedicalRecord) créés.

    Raises:
        ValueError: Si des champs obligatoires sont manquants ou invalides.
        Exception:  Propagée après rollback en cas d'erreur DB.
    """
    # ── Validation champs obligatoires ────────────────────────────────────
    required = ["first_name", "last_name", "birthdate"]
    missing  = [f for f in required if not data.get(f)]
    if missing:
        raise ValueError(f"Champs obligatoires manquants : {', '.join(missing)}")

    # ── Parsing de la date de naissance ───────────────────────────────────
    birthdate_raw = data["birthdate"]
    if isinstance(birthdate_raw, str):
        try:
            birthdate = date.fromisoformat(birthdate_raw)
        except ValueError:
            raise ValueError("Format de date invalide. Attendu : YYYY-MM-DD")
    elif isinstance(birthdate_raw, date):
        birthdate = birthdate_raw
    else:
        raise ValueError("birthdate doit être une chaîne ISO ou un objet date")

    if birthdate >= date.today():
        raise ValueError("La date de naissance doit être dans le passé")

    # ── Validation genre ───────────────────────────────────────────────────
    gender = data.get("gender")
    if gender and gender not in VALID_GENDERS:
        raise ValueError(f"Genre invalide. Valeurs acceptées : {', '.join(VALID_GENDERS)}")

    # ── Vérification unicité email ─────────────────────────────────────────
    email = data.get("email", "").strip().lower() or None
    if email and Patient.query.filter_by(email=email).first():
        raise ValueError(f"L'email {email!r} est déjà utilisé")

    try:
        # ── Création du patient ────────────────────────────────────────────
        patient = Patient(
            first_name=data["first_name"].strip(),
            last_name=data["last_name"].strip(),
            birthdate=birthdate,
            gender=gender,
            email=email,
            phone=data.get("phone", "").strip() or None,
        )
        db.session.add(patient)
        db.session.flush()  # Obtenir l'id_patient sans commit

        # ── Création automatique du dossier médical ────────────────────────
        blood_group = data.get("blood_group")
        if blood_group and blood_group not in VALID_BLOOD_GROUPS:
            raise ValueError(f"Groupe sanguin invalide. Valeurs : {', '.join(VALID_BLOOD_GROUPS)}")

        record = MedicalRecord(
            id_patient=patient.id_patient,
            allergies=data.get("allergies"),
            chronic_diseases=data.get("chronic_diseases"),
            blood_group=blood_group,
            medical_history=data.get("medical_history"),
        )
        db.session.add(record)
        db.session.commit()

        logger.info(
            "Patient créé — id=%s name=%s %s",
            patient.id_patient, patient.first_name, patient.last_name,
        )
        return patient, record

    except Exception:
        db.session.rollback()
        raise


# ─────────────────────────────────────────────────────────────────────────────
#  update_patient
# ─────────────────────────────────────────────────────────────────────────────

def update_patient(patient: Patient, data: dict) -> Patient:
    """
    Met à jour les champs modifiables d'un patient.
    Ignore les champs non fournis (PATCH-style).

    Args:
        patient: Instance Patient à modifier.
        data:    Dictionnaire des champs à mettre à jour.

    Returns:
        Patient mis à jour.

    Raises:
        ValueError: Si les données sont invalides.
    """
    # Champs modifiables (age est virtuel — exclu)
    updatable = ["first_name", "last_name", "birthdate", "gender", "email", "phone"]

    if "birthdate" in data and data["birthdate"]:
        raw = data["birthdate"]
        if isinstance(raw, str):
            try:
                data["birthdate"] = date.fromisoformat(raw)
            except ValueError:
                raise ValueError("Format de date invalide. Attendu : YYYY-MM-DD")
        if data["birthdate"] >= date.today():
            raise ValueError("La date de naissance doit être dans le passé")

    if "gender" in data and data["gender"] and data["gender"] not in VALID_GENDERS:
        raise ValueError(f"Genre invalide. Valeurs acceptées : {', '.join(VALID_GENDERS)}")

    if "email" in data and data["email"]:
        new_email = data["email"].strip().lower()
        existing  = Patient.query.filter_by(email=new_email).first()
        if existing and existing.id_patient != patient.id_patient:
            raise ValueError(f"L'email {new_email!r} est déjà utilisé")
        data["email"] = new_email

    for field in updatable:
        if field in data and data[field] is not None:
            setattr(patient, field, data[field])

    try:
        db.session.commit()
        logger.info("Patient mis à jour — id=%s", patient.id_patient)
        return patient
    except Exception:
        db.session.rollback()
        raise


# ─────────────────────────────────────────────────────────────────────────────
#  get_patient_consultations
# ─────────────────────────────────────────────────────────────────────────────

def get_patient_consultations(
    patient_id: int,
    page:       int = 1,
    per_page:   int = 20,
) -> Optional[dict]:
    """
    Retourne l'historique paginé des consultations d'un patient.

    Args:
        patient_id: Identifiant du patient.
        page:       Numéro de page.
        per_page:   Éléments par page.

    Returns:
        Dictionnaire {"items": [...], "meta": {...}} ou None si patient introuvable.
    """
    patient = Patient.query.get(patient_id)
    if not patient:
        return None

    query = (
        Consultation.query
        .filter_by(id_patient=patient_id)
        .order_by(Consultation.date.desc())
    )

    pagination = paginate_query(query, page=page, per_page=per_page)

    return {
        "patient": {"id_patient": patient.id_patient, "full_name": f"{patient.first_name} {patient.last_name}"},
        "items":   [c.to_dict(include_staff=True) for c in pagination.items],
        "meta":    pagination_meta(pagination),
    }

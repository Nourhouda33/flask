"""
Blueprint patients — CRUD complet + sous-ressources.

Endpoints :
  GET    /api/patients                        → liste paginée avec filtres
  GET    /api/patients/<id>                   → profil complet
  POST   /api/patients                        → créer patient + dossier médical
  PUT    /api/patients/<id>                   → modifier patient
  DELETE /api/patients/<id>                   → suppression (admin uniquement)
  GET    /api/patients/<id>/consultations     → historique consultations
  GET    /api/patients/<id>/medical-record    → dossier médical
  PUT    /api/patients/<id>/medical-record    → modifier dossier médical
"""

import logging
from flask import Blueprint, request, g

from database.db import db
from models.patient import Patient
from models.medical_record import MedicalRecord
from auth.decorators import token_required, role_required
from utils.response_helper import success_response, error_response, paginated_response
from services.patient_service import (
    search_patients,
    get_patient_full_profile,
    create_patient_with_record,
    update_patient,
    get_patient_consultations,
)

logger = logging.getLogger(__name__)
patients_bp = Blueprint("patients", __name__)

VALID_BLOOD_GROUPS = {"A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"}


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/patients
# ─────────────────────────────────────────────────────────────────────────────


@patients_bp.route("", methods=["GET"])
@token_required
def list_patients():
    """
    Liste paginée des patients avec filtres optionnels.
    RBAC : patient → voit uniquement son propre profil.
    """
    user = g.current_user

    # ── Patient : redirige vers son propre profil ──────────────────────────
    if user.role == "patient":
        if not user.id_patient:
            return error_response("Aucun profil patient associé à ce compte", 404)
        profile = get_patient_full_profile(user.id_patient)
        if not profile:
            return error_response("Profil patient introuvable", 404)
        return success_response(
            data={"items": [profile], "total": 1},
            message="Votre profil patient",
        )

    try:
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 20))
    except ValueError:
        return error_response("page et per_page doivent être des entiers", 400)

    try:
        age_min = (
            int(request.args.get("age_min")) if request.args.get("age_min") else None
        )
        age_max = (
            int(request.args.get("age_max")) if request.args.get("age_max") else None
        )
    except ValueError:
        return error_response("age_min et age_max doivent être des entiers", 400)

    result = search_patients(
        search=request.args.get("search"),
        gender=request.args.get("gender"),
        age_min=age_min,
        age_max=age_max,
        page=page,
        per_page=per_page,
    )

    return paginated_response(
        items=result["items"],
        pagination_meta=result["meta"],
        message=f"{result['meta']['total']} patient(s) trouvé(s)",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/patients/<id>
# ─────────────────────────────────────────────────────────────────────────────


@patients_bp.route("/<int:patient_id>", methods=["GET"])
@token_required
def get_patient(patient_id: int):
    """
    Retourne le profil complet d'un patient.
    RBAC : patient → peut voir uniquement son propre profil.
    """
    user = g.current_user

    # Patient : accès uniquement à son propre profil
    if user.role == "patient" and user.id_patient != patient_id:
        return error_response(
            "Accès refusé : vous ne pouvez consulter que votre propre profil", 403
        )

    profile = get_patient_full_profile(patient_id)
    if not profile:
        return error_response(f"Patient {patient_id} introuvable", 404)

    return success_response(data=profile, message="Profil patient récupéré avec succès")


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/patients
# ─────────────────────────────────────────────────────────────────────────────


@patients_bp.route("", methods=["POST"])
@token_required
@role_required("admin", "doctor", "staff")
def create_patient():
    """
    Crée un nouveau patient avec son dossier médical automatiquement.

    Body JSON:
        first_name       (str, requis)  : Prénom.
        last_name        (str, requis)  : Nom.
        birthdate        (str, requis)  : Date de naissance ISO (YYYY-MM-DD).
        gender           (str)          : 'Male' ou 'Female'.
        email            (str)          : Email unique.
        phone            (str)          : Téléphone.
        allergies        (str)          : Allergies connues.
        chronic_diseases (str)          : Maladies chroniques.
        blood_group      (str)          : Groupe sanguin.
        medical_history  (str)          : Historique médical.

    Returns:
        201 : { success, data: {patient, medical_record}, message }
        400 : Validation échouée.
        409 : Email déjà utilisé.
    """
    body = request.get_json(silent=True) or {}

    try:
        patient, record = create_patient_with_record(body)
    except ValueError as exc:
        status = 409 if "déjà utilisé" in str(exc) else 400
        return error_response(str(exc), status)
    except Exception as exc:
        logger.error("Erreur création patient : %s", str(exc), exc_info=True)
        return error_response("Erreur interne lors de la création du patient", 500)

    logger.info(
        "Patient créé — id=%s par user_id=%s",
        patient.id_patient,
        g.current_user.id_user,
    )

    return success_response(
        data={
            "patient": patient.to_dict(),
            "medical_record": record.to_dict(),
        },
        message="Patient créé avec succès",
        status_code=201,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PUT /api/patients/<id>
# ─────────────────────────────────────────────────────────────────────────────


@patients_bp.route("/<int:patient_id>", methods=["PUT"])
@token_required
@role_required("admin", "doctor", "staff")
def update_patient_route(patient_id: int):
    """
    Modifie les données démographiques d'un patient.
    Seuls les champs fournis sont mis à jour (PATCH-style).

    Body JSON (tous optionnels) :
        first_name, last_name, birthdate, gender, email, phone

    Returns:
        200 : { success, data: {patient}, message }
        400 : Validation échouée.
        404 : Patient introuvable.
        409 : Email déjà utilisé.
    """
    patient = Patient.query.get(patient_id)
    if not patient:
        return error_response(f"Patient {patient_id} introuvable", 404)

    body = request.get_json(silent=True) or {}
    if not body:
        return error_response("Corps de requête vide", 400)

    try:
        patient = update_patient(patient, body)
    except ValueError as exc:
        status = 409 if "déjà utilisé" in str(exc) else 400
        return error_response(str(exc), status)
    except Exception as exc:
        logger.error(
            "Erreur mise à jour patient %s : %s", patient_id, str(exc), exc_info=True
        )
        return error_response("Erreur interne lors de la mise à jour", 500)

    return success_response(
        data={"patient": patient.to_dict()},
        message="Patient mis à jour avec succès",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  DELETE /api/patients/<id>
# ─────────────────────────────────────────────────────────────────────────────


@patients_bp.route("/<int:patient_id>", methods=["DELETE"])
@token_required
@role_required("admin")
def delete_patient(patient_id: int):
    """
    Supprime un patient et son dossier médical (cascade DB).
    Réservé aux administrateurs.

    Returns:
        200 : { success, message }
        404 : Patient introuvable.
    """
    patient = Patient.query.get(patient_id)
    if not patient:
        return error_response(f"Patient {patient_id} introuvable", 404)

    patient_name = f"{patient.first_name} {patient.last_name}"

    try:
        db.session.delete(patient)
        db.session.commit()
        logger.warning(
            "Patient supprimé — id=%s name=%s par user_id=%s",
            patient_id,
            patient_name,
            g.current_user.id_user,
        )
    except Exception as exc:
        db.session.rollback()
        logger.error(
            "Erreur suppression patient %s : %s", patient_id, str(exc), exc_info=True
        )
        return error_response("Erreur interne lors de la suppression", 500)

    return success_response(
        message=f"Patient '{patient_name}' supprimé avec succès",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/patients/<id>/consultations
# ─────────────────────────────────────────────────────────────────────────────


@patients_bp.route("/<int:patient_id>/consultations", methods=["GET"])
@token_required
def get_patient_consultations_route(patient_id: int):
    """
    Retourne l'historique paginé des consultations d'un patient.
    RBAC : patient → uniquement ses propres consultations.
    """
    user = g.current_user

    # Patient : accès uniquement à ses propres consultations
    if user.role == "patient" and user.id_patient != patient_id:
        return error_response(
            "Accès refusé : vous ne pouvez consulter que vos propres consultations", 403
        )

    try:
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 20))
    except ValueError:
        return error_response("page et per_page doivent être des entiers", 400)

    result = get_patient_consultations(patient_id, page=page, per_page=per_page)
    if result is None:
        return error_response(f"Patient {patient_id} introuvable", 404)

    return paginated_response(
        items=result["items"],
        pagination_meta=result["meta"],
        message=f"{result['meta']['total']} consultation(s) trouvée(s)",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/patients/<id>/medical-record
# ─────────────────────────────────────────────────────────────────────────────


@patients_bp.route("/<int:patient_id>/medical-record", methods=["GET"])
@token_required
def get_medical_record(patient_id: int):
    """
    Retourne le dossier médical d'un patient.
    RBAC : patient → uniquement son propre dossier.
    """
    user = g.current_user

    # Patient : accès uniquement à son propre dossier
    if user.role == "patient" and user.id_patient != patient_id:
        return error_response(
            "Accès refusé : vous ne pouvez consulter que votre propre dossier médical",
            403,
        )

    patient = Patient.query.get(patient_id)
    if not patient:
        return error_response(f"Patient {patient_id} introuvable", 404)

    if not patient.medical_record:
        return error_response(
            f"Aucun dossier médical pour le patient {patient_id}", 404
        )

    return success_response(
        data={"medical_record": patient.medical_record.to_dict()},
        message="Dossier médical récupéré avec succès",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PUT /api/patients/<id>/medical-record
# ─────────────────────────────────────────────────────────────────────────────


@patients_bp.route("/<int:patient_id>/medical-record", methods=["PUT"])
@token_required
@role_required("admin", "doctor")
def update_medical_record(patient_id: int):
    """
    Modifie le dossier médical d'un patient.
    Crée le dossier s'il n'existe pas encore.

    Body JSON (tous optionnels) :
        allergies, chronic_diseases, blood_group, medical_history

    Returns:
        200 : { success, data: {medical_record}, message }
        400 : Validation échouée.
        404 : Patient introuvable.
    """
    patient = Patient.query.get(patient_id)
    if not patient:
        return error_response(f"Patient {patient_id} introuvable", 404)

    body = request.get_json(silent=True) or {}
    if not body:
        return error_response("Corps de requête vide", 400)

    # Valider blood_group si fourni
    blood_group = body.get("blood_group")
    if blood_group and blood_group not in VALID_BLOOD_GROUPS:
        return error_response(
            f"Groupe sanguin invalide. Valeurs acceptées : {', '.join(sorted(VALID_BLOOD_GROUPS))}",
            400,
        )

    # Créer le dossier s'il n'existe pas
    record = patient.medical_record
    if not record:
        record = MedicalRecord(id_patient=patient_id)
        db.session.add(record)

    # Mettre à jour les champs fournis
    updatable = ["allergies", "chronic_diseases", "blood_group", "medical_history"]
    for field in updatable:
        if field in body:
            setattr(record, field, body[field])

    try:
        db.session.commit()
        logger.info(
            "Dossier médical mis à jour — patient_id=%s par user_id=%s",
            patient_id,
            g.current_user.id_user,
        )
    except Exception as exc:
        db.session.rollback()
        logger.error("Erreur mise à jour dossier médical : %s", str(exc), exc_info=True)
        return error_response("Erreur interne lors de la mise à jour du dossier", 500)

    return success_response(
        data={"medical_record": record.to_dict()},
        message="Dossier médical mis à jour avec succès",
    )

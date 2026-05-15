"""
Blueprint consultations — CRUD complet + génération de rapport IA.

Endpoints :
  GET    /api/consultations                        → liste paginée avec filtres
  GET    /api/consultations/<id>                   → détail complet
  POST   /api/consultations                        → créer (doctor/staff)
  PUT    /api/consultations/<id>                   → modifier (doctor)
  DELETE /api/consultations/<id>                   → supprimer (admin)
  POST   /api/consultations/<id>/generate-report   → rapport IA via Ollama
"""

import logging
from datetime import datetime

from flask import Blueprint, request, g

from database.db import db, paginate_query, pagination_meta
from models.consultation import Consultation
from models.patient import Patient
from models.staff import Staff
from auth.decorators import token_required, role_required
from utils.response_helper import success_response, error_response, paginated_response

logger = logging.getLogger(__name__)
consultations_bp = Blueprint("consultations", __name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers de validation
# ─────────────────────────────────────────────────────────────────────────────


def _validate_consultation_body(
    body: dict, require_diagnosis: bool = True
) -> list[str]:
    """
    Valide le corps d'une requête de consultation.

    Returns:
        Liste des messages d'erreur (vide si valide).
    """
    errors = []

    if require_diagnosis and not body.get("diagnosis", "").strip():
        errors.append("Le champ 'diagnosis' est obligatoire")

    if body.get("id_patient"):
        if not Patient.query.get(body["id_patient"]):
            errors.append(f"Patient {body['id_patient']} introuvable")

    if body.get("id_staff"):
        if not Staff.query.get(body["id_staff"]):
            errors.append(f"Membre du personnel {body['id_staff']} introuvable")

    if body.get("date"):
        try:
            datetime.fromisoformat(body["date"])
        except ValueError:
            errors.append(
                "Format de date invalide. Attendu : ISO 8601 (YYYY-MM-DDTHH:MM:SS)"
            )

    return errors


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/consultations
# ─────────────────────────────────────────────────────────────────────────────


@consultations_bp.route("", methods=["GET"])
@token_required
def list_consultations():
    """
    Liste paginée des consultations avec filtres optionnels.
    RBAC : patient → uniquement ses propres consultations (filtré par id_patient).
    """
    user = g.current_user
    query = Consultation.query

    # Patient : filtre automatique sur ses propres consultations
    if user.role == "patient":
        if not user.id_patient:
            return success_response(data=[], message="Aucune consultation")
        query = query.filter_by(id_patient=user.id_patient)
    else:
        # Filtres manuels pour les autres rôles
        if request.args.get("patient_id"):
            try:
                query = query.filter_by(id_patient=int(request.args["patient_id"]))
            except ValueError:
                return error_response("patient_id doit être un entier", 400)

        if request.args.get("staff_id"):
            try:
                query = query.filter_by(id_staff=int(request.args["staff_id"]))
            except ValueError:
                return error_response("staff_id doit être un entier", 400)

    try:
        page = int(request.args.get("page", 1))
        per_page = int(request.args.get("per_page", 20))
    except ValueError:
        return error_response("page et per_page doivent être des entiers", 400)

    if request.args.get("date_from"):
        try:
            query = query.filter(
                Consultation.date >= datetime.fromisoformat(request.args["date_from"])
            )
        except ValueError:
            return error_response(
                "date_from invalide. Format attendu : YYYY-MM-DD", 400
            )

    if request.args.get("date_to"):
        try:
            query = query.filter(
                Consultation.date <= datetime.fromisoformat(request.args["date_to"])
            )
        except ValueError:
            return error_response("date_to invalide. Format attendu : YYYY-MM-DD", 400)

    query = query.order_by(Consultation.date.desc())
    pagination = paginate_query(query, page=page, per_page=per_page)
    items = [
        c.to_dict(include_patient=True, include_staff=True) for c in pagination.items
    ]

    return paginated_response(
        items=items,
        pagination_meta=pagination_meta(pagination),
        message=f"{pagination.total} consultation(s) trouvée(s)",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/consultations/<id>
# ─────────────────────────────────────────────────────────────────────────────


@consultations_bp.route("/<int:consultation_id>", methods=["GET"])
@token_required
def get_consultation(consultation_id: int):
    """
    Retourne le détail complet d'une consultation (patient + staff inclus).

    Returns:
        200 : { success, data: {consultation}, message }
        404 : Consultation introuvable.
    """
    consultation = Consultation.query.get(consultation_id)
    if not consultation:
        return error_response(f"Consultation {consultation_id} introuvable", 404)

    return success_response(
        data={
            "consultation": consultation.to_dict(
                include_patient=True, include_staff=True
            )
        },
        message="Consultation récupérée avec succès",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/consultations
# ─────────────────────────────────────────────────────────────────────────────


@consultations_bp.route("", methods=["POST"])
@token_required
@role_required("admin", "doctor", "staff")
def create_consultation():
    """
    Crée une nouvelle consultation médicale.

    Body JSON:
        diagnosis      (str, requis) : Diagnostic.
        treatment      (str)         : Traitement prescrit.
        medical_report (str)         : Rapport médical.
        date           (str)         : Date ISO (défaut : maintenant).
        id_staff       (int)         : ID du médecin traitant.
        id_patient     (int)         : ID du patient.

    Returns:
        201 : { success, data: {consultation}, message }
        400 : Validation échouée.
    """
    body = request.get_json(silent=True) or {}

    errors = _validate_consultation_body(body, require_diagnosis=True)
    if errors:
        return error_response("; ".join(errors), 400)

    # Résoudre la date
    date_val = datetime.utcnow()
    if body.get("date"):
        try:
            date_val = datetime.fromisoformat(body["date"])
        except ValueError:
            return error_response("Format de date invalide", 400)

    consultation = Consultation(
        diagnosis=body["diagnosis"].strip(),
        treatment=body.get("treatment", "").strip() or None,
        medical_report=body.get("medical_report", "").strip() or None,
        date=date_val,
        id_staff=body.get("id_staff"),
        id_patient=body.get("id_patient"),
    )

    try:
        db.session.add(consultation)
        db.session.commit()
        logger.info(
            "Consultation créée — id=%s patient=%s staff=%s par user_id=%s",
            consultation.id_consultation,
            consultation.id_patient,
            consultation.id_staff,
            g.current_user.id_user,
        )
    except Exception as exc:
        db.session.rollback()
        logger.error("Erreur création consultation : %s", str(exc), exc_info=True)
        return error_response("Erreur interne lors de la création", 500)

    return success_response(
        data={
            "consultation": consultation.to_dict(
                include_patient=True, include_staff=True
            )
        },
        message="Consultation créée avec succès",
        status_code=201,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PUT /api/consultations/<id>
# ─────────────────────────────────────────────────────────────────────────────


@consultations_bp.route("/<int:consultation_id>", methods=["PUT"])
@token_required
@role_required("admin", "doctor")
def update_consultation(consultation_id: int):
    """
    Modifie une consultation existante.
    Seuls les champs fournis sont mis à jour.

    Body JSON (tous optionnels) :
        diagnosis, treatment, medical_report, date, id_staff, id_patient

    Returns:
        200 : { success, data: {consultation}, message }
        400 : Validation échouée.
        404 : Consultation introuvable.
    """
    consultation = Consultation.query.get(consultation_id)
    if not consultation:
        return error_response(f"Consultation {consultation_id} introuvable", 404)

    body = request.get_json(silent=True) or {}
    if not body:
        return error_response("Corps de requête vide", 400)

    errors = _validate_consultation_body(body, require_diagnosis=False)
    if errors:
        return error_response("; ".join(errors), 400)

    # Mettre à jour les champs fournis
    updatable = ["diagnosis", "treatment", "medical_report", "id_staff", "id_patient"]
    for field in updatable:
        if field in body:
            val = body[field]
            if isinstance(val, str):
                val = val.strip() or None
            setattr(consultation, field, val)

    if "date" in body and body["date"]:
        try:
            consultation.date = datetime.fromisoformat(body["date"])
        except ValueError:
            return error_response("Format de date invalide", 400)

    try:
        db.session.commit()
        logger.info(
            "Consultation mise à jour — id=%s par user_id=%s",
            consultation_id,
            g.current_user.id_user,
        )
    except Exception as exc:
        db.session.rollback()
        logger.error(
            "Erreur mise à jour consultation %s : %s",
            consultation_id,
            str(exc),
            exc_info=True,
        )
        return error_response("Erreur interne lors de la mise à jour", 500)

    return success_response(
        data={
            "consultation": consultation.to_dict(
                include_patient=True, include_staff=True
            )
        },
        message="Consultation mise à jour avec succès",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  DELETE /api/consultations/<id>
# ─────────────────────────────────────────────────────────────────────────────


@consultations_bp.route("/<int:consultation_id>", methods=["DELETE"])
@token_required
@role_required("admin")
def delete_consultation(consultation_id: int):
    """
    Supprime définitivement une consultation.
    Réservé aux administrateurs.

    Returns:
        200 : { success, message }
        404 : Consultation introuvable.
    """
    consultation = Consultation.query.get(consultation_id)
    if not consultation:
        return error_response(f"Consultation {consultation_id} introuvable", 404)

    try:
        db.session.delete(consultation)
        db.session.commit()
        logger.warning(
            "Consultation supprimée — id=%s par user_id=%s",
            consultation_id,
            g.current_user.id_user,
        )
    except Exception as exc:
        db.session.rollback()
        logger.error(
            "Erreur suppression consultation %s : %s",
            consultation_id,
            str(exc),
            exc_info=True,
        )
        return error_response("Erreur interne lors de la suppression", 500)

    return success_response(
        message=f"Consultation {consultation_id} supprimée avec succès"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/consultations/<id>/generate-report
# ─────────────────────────────────────────────────────────────────────────────


@consultations_bp.route("/<int:consultation_id>/generate-report", methods=["POST"])
@token_required
@role_required("admin", "doctor")
def generate_report(consultation_id: int):
    """
    Génère un rapport médical structuré via Ollama (LLaMA3).
    Le rapport est sauvegardé dans medical_report de la consultation.

    Body JSON (optionnel) :
        model (str) : Modèle Ollama à utiliser (défaut : llama3).

    Returns:
        200 : { success, data: {consultation, report}, message }
        404 : Consultation introuvable.
        503 : Ollama indisponible.
    """
    from flask import current_app
    import requests as http_requests

    consultation = Consultation.query.get(consultation_id)
    if not consultation:
        return error_response(f"Consultation {consultation_id} introuvable", 404)

    body = request.get_json(silent=True) or {}
    model = body.get("model") or current_app.config.get("LLAMA3_MODEL", "llama3")

    # ── Construction du prompt médical ────────────────────────────────────
    patient_info = ""
    if consultation.patient:
        p = consultation.patient
        patient_info = (
            f"Patient : {p.first_name} {p.last_name}, "
            f"âge {p.age} ans, genre {p.gender or 'non renseigné'}"
        )

    staff_info = ""
    if consultation.staff:
        s = consultation.staff
        staff_info = f"Médecin : Dr {s.name_staff} ({s.speciality or s.position_staff})"

    prompt = f"""Tu es un médecin expert. Génère un rapport médical structuré et professionnel en français.

Informations de la consultation :
- {patient_info}
- {staff_info}
- Date : {consultation.date.strftime('%d/%m/%Y %H:%M') if consultation.date else 'Non renseignée'}
- Diagnostic : {consultation.diagnosis}
- Traitement : {consultation.treatment or 'Non renseigné'}

Génère un rapport médical complet avec les sections suivantes :
1. RÉSUMÉ CLINIQUE
2. DIAGNOSTIC DÉTAILLÉ
3. PLAN DE TRAITEMENT
4. RECOMMANDATIONS ET SUIVI
5. CONCLUSION

Le rapport doit être professionnel, précis et adapté au contexte médical."""

    # ── Appel Ollama ───────────────────────────────────────────────────────
    ollama_url = current_app.config.get("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_timeout = current_app.config.get("OLLAMA_TIMEOUT", 120)

    try:
        response = http_requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=ollama_timeout,
        )
        response.raise_for_status()
        report_text = response.json().get("response", "").strip()

    except http_requests.exceptions.ConnectionError:
        logger.error("Ollama indisponible — url=%s", ollama_url)
        return error_response(
            "Service IA indisponible. Vérifiez qu'Ollama est démarré.",
            503,
        )
    except http_requests.exceptions.Timeout:
        logger.error("Timeout Ollama après %ss", ollama_timeout)
        return error_response(
            f"Délai d'attente dépassé ({ollama_timeout}s). Réessayez.",
            503,
        )
    except Exception as exc:
        logger.error("Erreur Ollama : %s", str(exc), exc_info=True)
        return error_response("Erreur lors de la génération du rapport IA", 500)

    # ── Sauvegarde du rapport ──────────────────────────────────────────────
    consultation.medical_report = report_text
    try:
        db.session.commit()
        logger.info(
            "Rapport IA généré — consultation_id=%s model=%s par user_id=%s",
            consultation_id,
            model,
            g.current_user.id_user,
        )
    except Exception as exc:
        db.session.rollback()
        logger.error("Erreur sauvegarde rapport : %s", str(exc), exc_info=True)
        return error_response("Rapport généré mais non sauvegardé", 500)

    return success_response(
        data={
            "consultation": consultation.to_dict(
                include_patient=True, include_staff=True
            ),
            "report": report_text,
            "model_used": model,
        },
        message="Rapport médical généré avec succès",
    )

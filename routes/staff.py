"""
Blueprint staff — CRUD complet Medical_staff + sous-ressources.

Endpoints :
  GET    /api/staff                          → liste paginée avec filtres
  GET    /api/staff/<id>                     → détail complet
  POST   /api/staff                          → créer (admin)
  PUT    /api/staff/<id>                     → modifier (admin)
  DELETE /api/staff/<id>                     → supprimer (admin)
  GET    /api/staff/by-service/<id_service>  → personnel d'un service
  GET    /api/staff/<id>/consultations       → consultations d'un médecin
"""

import logging
from flask import Blueprint, request, g

from database.db     import db, paginate_query, pagination_meta
from models.staff    import Staff
from models.service  import Service
from models.consultation import Consultation
from auth.decorators import token_required, role_required
from utils.response_helper import success_response, error_response, paginated_response

logger   = logging.getLogger(__name__)
staff_bp = Blueprint("staff", __name__)

VALID_POSITIONS = {"Doctor", "Nurse", "Technician", "Administrator"}


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/staff
# ─────────────────────────────────────────────────────────────────────────────

@staff_bp.route("", methods=["GET"])
@token_required
def list_staff():
    """
    Liste paginée du personnel médical avec filtres optionnels.

    Query params:
        page       (int) : Numéro de page (défaut 1).
        per_page   (int) : Éléments par page (défaut 20).
        search     (str) : Recherche sur nom ou email.
        position   (str) : Filtre par position (Doctor/Nurse/Technician/Administrator).
        service_id (int) : Filtre par service.

    Returns:
        200 : { success, data: [...], meta: {pagination}, message }
    """
    try:
        page     = int(request.args.get("page",     1))
        per_page = int(request.args.get("per_page", 20))
    except ValueError:
        return error_response("page et per_page doivent être des entiers", 400)

    query = Staff.query

    # ── Filtre recherche ───────────────────────────────────────────────────
    if request.args.get("search"):
        term  = f"%{request.args['search'].strip()}%"
        query = query.filter(
            Staff.name_staff.ilike(term) | Staff.email.ilike(term)
        )

    # ── Filtre position ────────────────────────────────────────────────────
    position = request.args.get("position")
    if position:
        if position not in VALID_POSITIONS:
            return error_response(
                f"Position invalide. Valeurs : {', '.join(VALID_POSITIONS)}", 400
            )
        query = query.filter_by(position_staff=position)

    # ── Filtre service ─────────────────────────────────────────────────────
    if request.args.get("service_id"):
        try:
            query = query.filter_by(id_service=int(request.args["service_id"]))
        except ValueError:
            return error_response("service_id doit être un entier", 400)

    query      = query.order_by(Staff.name_staff.asc())
    pagination = paginate_query(query, page=page, per_page=per_page)

    items = [s.to_dict(include_service=True) for s in pagination.items]

    return paginated_response(
        items=items,
        pagination_meta=pagination_meta(pagination),
        message=f"{pagination.total} membre(s) du personnel trouvé(s)",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/staff/by-service/<id_service>
#  IMPORTANT : cette route doit être déclarée AVANT /<id> pour éviter
#  que Flask interprète "by-service" comme un entier.
# ─────────────────────────────────────────────────────────────────────────────

@staff_bp.route("/by-service/<int:service_id>", methods=["GET"])
@token_required
def staff_by_service(service_id: int):
    """
    Retourne tout le personnel rattaché à un service hospitalier.

    Query params:
        page     (int) : Numéro de page (défaut 1).
        per_page (int) : Éléments par page (défaut 50).

    Returns:
        200 : { success, data: {service, items}, meta: {pagination}, message }
        404 : Service introuvable.
    """
    service = Service.query.get(service_id)
    if not service:
        return error_response(f"Service {service_id} introuvable", 404)

    try:
        page     = int(request.args.get("page",     1))
        per_page = int(request.args.get("per_page", 50))
    except ValueError:
        return error_response("page et per_page doivent être des entiers", 400)

    query      = Staff.query.filter_by(id_service=service_id).order_by(Staff.name_staff.asc())
    pagination = paginate_query(query, page=page, per_page=per_page)

    return success_response(
        data={
            "service": service.to_dict(),
            "items":   [s.to_dict() for s in pagination.items],
        },
        meta=pagination_meta(pagination),
        message=f"{pagination.total} membre(s) dans le service '{service.service_name}'",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/staff/<id>
# ─────────────────────────────────────────────────────────────────────────────

@staff_bp.route("/<int:staff_id>", methods=["GET"])
@token_required
def get_staff(staff_id: int):
    """
    Retourne le détail complet d'un membre du personnel (avec service).

    Returns:
        200 : { success, data: {staff}, message }
        404 : Membre introuvable.
    """
    staff = Staff.query.get(staff_id)
    if not staff:
        return error_response(f"Membre du personnel {staff_id} introuvable", 404)

    return success_response(
        data={"staff": staff.to_dict(include_service=True)},
        message="Membre du personnel récupéré avec succès",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/staff
# ─────────────────────────────────────────────────────────────────────────────

@staff_bp.route("", methods=["POST"])
@token_required
@role_required("admin")
def create_staff():
    """
    Crée un nouveau membre du personnel médical.

    Body JSON:
        name_staff     (str, requis) : Nom complet.
        position_staff (str, requis) : Doctor/Nurse/Technician/Administrator.
        speciality     (str)         : Spécialité médicale.
        id_service     (int)         : Service d'appartenance.
        email          (str)         : Email unique.
        phone          (str)         : Téléphone.

    Returns:
        201 : { success, data: {staff}, message }
        400 : Validation échouée.
        409 : Email déjà utilisé.
    """
    body = request.get_json(silent=True) or {}

    # ── Validation ────────────────────────────────────────────────────────
    errors = []
    if not body.get("name_staff", "").strip():
        errors.append("name_staff est obligatoire")

    position = body.get("position_staff", "Doctor")
    if position not in VALID_POSITIONS:
        errors.append(f"position_staff invalide. Valeurs : {', '.join(VALID_POSITIONS)}")

    if errors:
        return error_response("; ".join(errors), 400)

    # Vérifier unicité email
    email = body.get("email", "").strip().lower() or None
    if email and Staff.query.filter_by(email=email).first():
        return error_response(f"L'email {email!r} est déjà utilisé", 409)

    # Vérifier que le service existe
    id_service = body.get("id_service")
    if id_service and not Service.query.get(id_service):
        return error_response(f"Service {id_service} introuvable", 400)

    staff = Staff(
        name_staff=body["name_staff"].strip(),
        position_staff=position,
        speciality=body.get("speciality", "").strip() or None,
        id_service=id_service,
        email=email,
        phone=body.get("phone", "").strip() or None,
    )

    try:
        db.session.add(staff)
        db.session.commit()
        logger.info(
            "Staff créé — id=%s name=%s par user_id=%s",
            staff.id_staff, staff.name_staff, g.current_user.id_user,
        )
    except Exception as exc:
        db.session.rollback()
        logger.error("Erreur création staff : %s", str(exc), exc_info=True)
        return error_response("Erreur interne lors de la création", 500)

    return success_response(
        data={"staff": staff.to_dict(include_service=True)},
        message="Membre du personnel créé avec succès",
        status_code=201,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PUT /api/staff/<id>
# ─────────────────────────────────────────────────────────────────────────────

@staff_bp.route("/<int:staff_id>", methods=["PUT"])
@token_required
@role_required("admin")
def update_staff(staff_id: int):
    """
    Modifie un membre du personnel médical.

    Body JSON (tous optionnels) :
        name_staff, position_staff, speciality, id_service, email, phone

    Returns:
        200 : { success, data: {staff}, message }
        400 : Validation échouée.
        404 : Membre introuvable.
        409 : Email déjà utilisé.
    """
    staff = Staff.query.get(staff_id)
    if not staff:
        return error_response(f"Membre du personnel {staff_id} introuvable", 404)

    body = request.get_json(silent=True) or {}
    if not body:
        return error_response("Corps de requête vide", 400)

    # Valider position si fournie
    if "position_staff" in body and body["position_staff"] not in VALID_POSITIONS:
        return error_response(
            f"position_staff invalide. Valeurs : {', '.join(VALID_POSITIONS)}", 400
        )

    # Vérifier unicité email
    if "email" in body and body["email"]:
        new_email = body["email"].strip().lower()
        existing  = Staff.query.filter_by(email=new_email).first()
        if existing and existing.id_staff != staff_id:
            return error_response(f"L'email {new_email!r} est déjà utilisé", 409)
        body["email"] = new_email

    # Vérifier service
    if "id_service" in body and body["id_service"]:
        if not Service.query.get(body["id_service"]):
            return error_response(f"Service {body['id_service']} introuvable", 400)

    updatable = ["name_staff", "position_staff", "speciality", "id_service", "email", "phone"]
    for field in updatable:
        if field in body:
            val = body[field]
            if isinstance(val, str):
                val = val.strip() or None
            setattr(staff, field, val)

    try:
        db.session.commit()
        logger.info("Staff mis à jour — id=%s par user_id=%s", staff_id, g.current_user.id_user)
    except Exception as exc:
        db.session.rollback()
        logger.error("Erreur mise à jour staff %s : %s", staff_id, str(exc), exc_info=True)
        return error_response("Erreur interne lors de la mise à jour", 500)

    return success_response(
        data={"staff": staff.to_dict(include_service=True)},
        message="Membre du personnel mis à jour avec succès",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  DELETE /api/staff/<id>
# ─────────────────────────────────────────────────────────────────────────────

@staff_bp.route("/<int:staff_id>", methods=["DELETE"])
@token_required
@role_required("admin")
def delete_staff(staff_id: int):
    """
    Supprime un membre du personnel médical.
    Les consultations liées auront id_staff = NULL (SET NULL en DB).

    Returns:
        200 : { success, message }
        404 : Membre introuvable.
    """
    staff = Staff.query.get(staff_id)
    if not staff:
        return error_response(f"Membre du personnel {staff_id} introuvable", 404)

    staff_name = staff.name_staff

    try:
        db.session.delete(staff)
        db.session.commit()
        logger.warning(
            "Staff supprimé — id=%s name=%s par user_id=%s",
            staff_id, staff_name, g.current_user.id_user,
        )
    except Exception as exc:
        db.session.rollback()
        logger.error("Erreur suppression staff %s : %s", staff_id, str(exc), exc_info=True)
        return error_response("Erreur interne lors de la suppression", 500)

    return success_response(message=f"Membre '{staff_name}' supprimé avec succès")


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/staff/<id>/consultations
# ─────────────────────────────────────────────────────────────────────────────

@staff_bp.route("/<int:staff_id>/consultations", methods=["GET"])
@token_required
def staff_consultations(staff_id: int):
    """
    Retourne l'historique paginé des consultations d'un médecin.

    Query params:
        page     (int) : Numéro de page (défaut 1).
        per_page (int) : Éléments par page (défaut 20).

    Returns:
        200 : { success, data: {staff, items}, meta: {pagination}, message }
        404 : Membre introuvable.
    """
    staff = Staff.query.get(staff_id)
    if not staff:
        return error_response(f"Membre du personnel {staff_id} introuvable", 404)

    try:
        page     = int(request.args.get("page",     1))
        per_page = int(request.args.get("per_page", 20))
    except ValueError:
        return error_response("page et per_page doivent être des entiers", 400)

    query = (
        Consultation.query
        .filter_by(id_staff=staff_id)
        .order_by(Consultation.date.desc())
    )
    pagination = paginate_query(query, page=page, per_page=per_page)

    return success_response(
        data={
            "staff": {
                "id_staff":   staff.id_staff,
                "name_staff": staff.name_staff,
                "speciality": staff.speciality,
            },
            "items": [c.to_dict(include_patient=True) for c in pagination.items],
        },
        meta=pagination_meta(pagination),
        message=f"{pagination.total} consultation(s) pour Dr {staff.name_staff}",
    )

"""
Blueprint services — CRUD complet Service + sous-ressources.

Endpoints :
  GET    /api/services              → liste de tous les services
  GET    /api/services/<id>         → détail d'un service
  POST   /api/services              → créer un service (admin)
  PUT    /api/services/<id>         → modifier un service (admin)
  DELETE /api/services/<id>         → supprimer un service (admin)
  GET    /api/services/<id>/staff   → personnel du service
"""

import logging
from flask import Blueprint, request, g

from database.db    import db, paginate_query, pagination_meta
from models.service import Service
from models.staff   import Staff
from auth.decorators import token_required, role_required
from utils.response_helper import success_response, error_response, paginated_response

logger      = logging.getLogger(__name__)
services_bp = Blueprint("services", __name__)


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/services
# ─────────────────────────────────────────────────────────────────────────────

@services_bp.route("", methods=["GET"])
@token_required
def list_services():
    """
    Retourne la liste complète des services hospitaliers.
    Inclut le nombre de membres du personnel par service.

    Query params:
        search (str) : Recherche sur le nom du service.

    Returns:
        200 : { success, data: [...], message }
    """
    query = Service.query

    if request.args.get("search"):
        term  = f"%{request.args['search'].strip()}%"
        query = query.filter(Service.service_name.ilike(term))

    services = query.order_by(Service.service_name.asc()).all()

    # Enrichir avec le nombre de membres
    items = []
    for svc in services:
        data = svc.to_dict()
        data["staff_count"] = Staff.query.filter_by(id_service=svc.id_service).count()
        items.append(data)

    return success_response(
        data=items,
        message=f"{len(items)} service(s) trouvé(s)",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/services/<id>
# ─────────────────────────────────────────────────────────────────────────────

@services_bp.route("/<int:service_id>", methods=["GET"])
@token_required
def get_service(service_id: int):
    """
    Retourne le détail d'un service avec le nombre de membres.

    Returns:
        200 : { success, data: {service}, message }
        404 : Service introuvable.
    """
    service = Service.query.get(service_id)
    if not service:
        return error_response(f"Service {service_id} introuvable", 404)

    data = service.to_dict()
    data["staff_count"] = Staff.query.filter_by(id_service=service_id).count()

    return success_response(
        data={"service": data},
        message="Service récupéré avec succès",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/services
# ─────────────────────────────────────────────────────────────────────────────

@services_bp.route("", methods=["POST"])
@token_required
@role_required("admin")
def create_service():
    """
    Crée un nouveau service hospitalier.

    Body JSON:
        service_name (str, requis) : Nom unique du service.

    Returns:
        201 : { success, data: {service}, message }
        400 : Validation échouée.
        409 : Nom de service déjà existant.
    """
    body = request.get_json(silent=True) or {}

    service_name = body.get("service_name", "").strip()
    if not service_name:
        return error_response("service_name est obligatoire", 400)

    if len(service_name) > 100:
        return error_response("service_name ne peut pas dépasser 100 caractères", 400)

    # Vérifier unicité
    if Service.query.filter_by(service_name=service_name).first():
        return error_response(f"Le service '{service_name}' existe déjà", 409)

    service = Service(service_name=service_name)

    try:
        db.session.add(service)
        db.session.commit()
        logger.info(
            "Service créé — id=%s name=%s par user_id=%s",
            service.id_service, service_name, g.current_user.id_user,
        )
    except Exception as exc:
        db.session.rollback()
        logger.error("Erreur création service : %s", str(exc), exc_info=True)
        return error_response("Erreur interne lors de la création", 500)

    return success_response(
        data={"service": service.to_dict()},
        message="Service créé avec succès",
        status_code=201,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PUT /api/services/<id>
# ─────────────────────────────────────────────────────────────────────────────

@services_bp.route("/<int:service_id>", methods=["PUT"])
@token_required
@role_required("admin")
def update_service(service_id: int):
    """
    Modifie le nom d'un service hospitalier.

    Body JSON:
        service_name (str, requis) : Nouveau nom du service.

    Returns:
        200 : { success, data: {service}, message }
        400 : Validation échouée.
        404 : Service introuvable.
        409 : Nom déjà utilisé.
    """
    service = Service.query.get(service_id)
    if not service:
        return error_response(f"Service {service_id} introuvable", 404)

    body = request.get_json(silent=True) or {}
    service_name = body.get("service_name", "").strip()

    if not service_name:
        return error_response("service_name est obligatoire", 400)

    if len(service_name) > 100:
        return error_response("service_name ne peut pas dépasser 100 caractères", 400)

    # Vérifier unicité (exclure le service courant)
    existing = Service.query.filter_by(service_name=service_name).first()
    if existing and existing.id_service != service_id:
        return error_response(f"Le service '{service_name}' existe déjà", 409)

    service.service_name = service_name

    try:
        db.session.commit()
        logger.info("Service mis à jour — id=%s par user_id=%s", service_id, g.current_user.id_user)
    except Exception as exc:
        db.session.rollback()
        logger.error("Erreur mise à jour service %s : %s", service_id, str(exc), exc_info=True)
        return error_response("Erreur interne lors de la mise à jour", 500)

    return success_response(
        data={"service": service.to_dict()},
        message="Service mis à jour avec succès",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  DELETE /api/services/<id>
# ─────────────────────────────────────────────────────────────────────────────

@services_bp.route("/<int:service_id>", methods=["DELETE"])
@token_required
@role_required("admin")
def delete_service(service_id: int):
    """
    Supprime un service hospitalier.
    Le personnel rattaché aura id_service = NULL (SET NULL en DB).

    Returns:
        200 : { success, message }
        404 : Service introuvable.
    """
    service = Service.query.get(service_id)
    if not service:
        return error_response(f"Service {service_id} introuvable", 404)

    service_name = service.service_name
    staff_count  = Staff.query.filter_by(id_service=service_id).count()

    try:
        db.session.delete(service)
        db.session.commit()
        logger.warning(
            "Service supprimé — id=%s name=%s (affectait %s membres) par user_id=%s",
            service_id, service_name, staff_count, g.current_user.id_user,
        )
    except Exception as exc:
        db.session.rollback()
        logger.error("Erreur suppression service %s : %s", service_id, str(exc), exc_info=True)
        return error_response("Erreur interne lors de la suppression", 500)

    return success_response(
        message=f"Service '{service_name}' supprimé. {staff_count} membre(s) désaffecté(s).",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/services/<id>/staff
# ─────────────────────────────────────────────────────────────────────────────

@services_bp.route("/<int:service_id>/staff", methods=["GET"])
@token_required
def service_staff(service_id: int):
    """
    Retourne le personnel médical rattaché à un service, paginé.

    Query params:
        page     (int) : Numéro de page (défaut 1).
        per_page (int) : Éléments par page (défaut 50).
        position (str) : Filtre par position.

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

    query = Staff.query.filter_by(id_service=service_id)

    # Filtre optionnel par position
    position = request.args.get("position")
    if position:
        query = query.filter_by(position_staff=position)

    query      = query.order_by(Staff.position_staff.asc(), Staff.name_staff.asc())
    pagination = paginate_query(query, page=page, per_page=per_page)

    return success_response(
        data={
            "service": service.to_dict(),
            "items":   [s.to_dict() for s in pagination.items],
        },
        meta=pagination_meta(pagination),
        message=f"{pagination.total} membre(s) dans '{service.service_name}'",
    )

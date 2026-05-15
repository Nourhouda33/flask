"""
Helpers pour les réponses JSON uniformes.
Format standard : { success, data, message, error, meta }
"""

from flask import jsonify
from typing import Any, Optional


def success_response(
    data: Any = None,
    message: str = "Succès",
    status_code: int = 200,
    meta: Optional[dict] = None,
):
    """
    Construit une réponse JSON de succès uniforme.

    Args:
        data:        Données à retourner (dict, list, etc.).
        message:     Message descriptif.
        status_code: Code HTTP (défaut 200).
        meta:        Métadonnées optionnelles (pagination, etc.).

    Returns:
        Tuple (Response Flask, status_code).
    """
    body: dict = {
        "success": True,
        "message": message,
        "data":    data,
    }
    if meta is not None:
        body["meta"] = meta

    return jsonify(body), status_code


def error_response(
    message: str = "Une erreur est survenue",
    status_code: int = 400,
    error: Optional[str] = None,
    data: Any = None,
):
    """
    Construit une réponse JSON d'erreur uniforme.

    Args:
        message:     Message d'erreur lisible.
        status_code: Code HTTP (défaut 400).
        error:       Code d'erreur technique optionnel.
        data:        Données supplémentaires (ex: erreurs de validation).

    Returns:
        Tuple (Response Flask, status_code).
    """
    body: dict = {
        "success": False,
        "message": message,
        "data":    data,
    }
    if error is not None:
        body["error"] = error

    return jsonify(body), status_code


def paginated_response(
    items: list,
    pagination_meta: dict,
    message: str = "Données récupérées avec succès",
):
    """
    Construit une réponse JSON paginée uniforme.

    Args:
        items:           Liste des éléments sérialisés.
        pagination_meta: Métadonnées de pagination (page, total, etc.).
        message:         Message descriptif.

    Returns:
        Tuple (Response Flask, 200).
    """
    return success_response(
        data=items,
        message=message,
        meta=pagination_meta,
    )

"""
Point d'entrée principal de l'application Flask — Healthcare AI Platform.
Utilise le pattern Application Factory pour la testabilité et la flexibilité.
"""

import os
import logging
from flask import Flask, jsonify
from flask_cors import CORS
from flask_migrate import Migrate

from config import get_config
from database.db import db
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


def create_app(config_override=None) -> Flask:
    """
    Factory function — crée et configure l'application Flask.

    Args:
        config_override: Classe de configuration optionnelle (utile pour les tests).

    Returns:
        Instance Flask configurée et prête à l'emploi.
    """
    app = Flask(__name__)

    # ── Configuration ─────────────────────────────────────────────────────────
    cfg = config_override or get_config()
    app.config.from_object(cfg)

    # ── Logging ───────────────────────────────────────────────────────────────
    setup_logger(app)

    # ── Extensions ────────────────────────────────────────────────────────────
    db.init_app(app)
    Migrate(app, db)

    CORS(
        app,
        origins=app.config.get("CORS_ORIGINS", ["http://localhost:4200"]),
        supports_credentials=True,
        allow_headers=["Content-Type", "Authorization"],
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )

    # ── Import des modèles (nécessaire pour Flask-Migrate) ────────────────────
    with app.app_context():
        import models  # noqa: F401 — enregistre tous les modèles auprès de SQLAlchemy

    # ── Blueprints ────────────────────────────────────────────────────────────
    _register_blueprints(app)

    # ── Gestionnaires d'erreurs ────────────────────────────────────────────────
    _register_error_handlers(app)

    # ── Health check ──────────────────────────────────────────────────────────
    @app.route("/api/health", methods=["GET"])
    def health_check():
        """Endpoint de vérification de l'état du service."""
        from database.db import check_db_connection
        db_status = check_db_connection()
        status    = "ok" if db_status["connected"] else "degraded"
        code      = 200  if db_status["connected"] else 503

        return jsonify({
            "success": True,
            "data": {
                "status":   status,
                "service":  "Healthcare AI Platform",
                "version":  "1.0.0",
                "database": db_status,
            },
            "message": f"Service {status}",
        }), code

    app.logger.info(
        "Healthcare AI Platform démarré — env=%s",
        os.getenv("FLASK_ENV", "development"),
    )
    return app


def _register_blueprints(app: Flask) -> None:
    """
    Enregistre tous les blueprints de l'application.
    Les imports sont locaux pour éviter les imports circulaires.
    """
    from routes.auth          import auth_bp
    from routes.patients      import patients_bp
    from routes.consultations import consultations_bp
    from routes.staff         import staff_bp
    from routes.services      import services_bp
    from routes.ai            import ai_bp
    from routes.metrics       import metrics_bp
    from routes.export        import export_bp

    app.register_blueprint(auth_bp,          url_prefix="/api/auth")
    app.register_blueprint(patients_bp,      url_prefix="/api/patients")
    app.register_blueprint(consultations_bp, url_prefix="/api/consultations")
    app.register_blueprint(staff_bp,         url_prefix="/api/staff")
    app.register_blueprint(services_bp,      url_prefix="/api/services")
    app.register_blueprint(ai_bp,            url_prefix="/api")
    app.register_blueprint(metrics_bp,       url_prefix="/api/metrics")
    app.register_blueprint(export_bp,        url_prefix="/api/export")

    app.logger.debug("Blueprints enregistrés")


def _register_error_handlers(app: Flask) -> None:
    """
    Enregistre les gestionnaires d'erreurs HTTP globaux.
    Toutes les erreurs retournent le format JSON uniforme { success, message, error }.
    """

    @app.errorhandler(400)
    def bad_request(e):
        logger.warning("400 Bad Request — %s", str(e))
        return jsonify({
            "success": False,
            "message": "Requête invalide",
            "error":   str(e),
            "data":    None,
        }), 400

    @app.errorhandler(401)
    def unauthorized(e):
        return jsonify({
            "success": False,
            "message": "Authentification requise",
            "error":   "UNAUTHORIZED",
            "data":    None,
        }), 401

    @app.errorhandler(403)
    def forbidden(e):
        return jsonify({
            "success": False,
            "message": "Accès refusé — permissions insuffisantes",
            "error":   "FORBIDDEN",
            "data":    None,
        }), 403

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({
            "success": False,
            "message": "Ressource introuvable",
            "error":   str(e),
            "data":    None,
        }), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({
            "success": False,
            "message": "Méthode HTTP non autorisée",
            "error":   str(e),
            "data":    None,
        }), 405

    @app.errorhandler(409)
    def conflict(e):
        return jsonify({
            "success": False,
            "message": "Conflit de données",
            "error":   str(e),
            "data":    None,
        }), 409

    @app.errorhandler(422)
    def unprocessable(e):
        return jsonify({
            "success": False,
            "message": "Données non traitables — vérifiez les champs envoyés",
            "error":   str(e),
            "data":    None,
        }), 422

    @app.errorhandler(429)
    def too_many_requests(e):
        return jsonify({
            "success": False,
            "message": "Trop de requêtes — réessayez plus tard",
            "error":   "RATE_LIMIT_EXCEEDED",
            "data":    None,
        }), 429

    @app.errorhandler(500)
    def internal_error(e):
        app.logger.error("500 Internal Server Error — %s", str(e), exc_info=True)
        return jsonify({
            "success": False,
            "message": "Erreur interne du serveur",
            "error":   "INTERNAL_SERVER_ERROR",
            "data":    None,
        }), 500

    @app.errorhandler(503)
    def service_unavailable(e):
        return jsonify({
            "success": False,
            "message": "Service temporairement indisponible",
            "error":   "SERVICE_UNAVAILABLE",
            "data":    None,
        }), 503


# ── Lancement direct ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    application = create_app()
    application.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=application.config.get("DEBUG", False),
    )

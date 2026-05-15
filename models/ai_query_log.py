"""
Modèle SQLAlchemy — AI_Query_Logs
Correspond exactement à la table `AI_Query_Logs` du schéma MySQL.
Historique de toutes les requêtes IA (Text2SQL) avec métriques.
"""

from datetime import datetime
from database.db import db


class AIQueryLog(db.Model):
    """
    Log d'une requête IA : prompt utilisateur, SQL généré, résultat et métriques.
    """

    __tablename__ = "AI_Query_Logs"

    # ── Colonnes ──────────────────────────────────────────────────────────────
    id_log           = db.Column("id_log",           db.Integer,  primary_key=True, autoincrement=True)
    user_id          = db.Column(
        "user_id",
        db.Integer,
        db.ForeignKey("Users.id_user", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
    )
    prompt           = db.Column("prompt",           db.Text,     nullable=False)
    detected_intent  = db.Column("detected_intent",  db.String(50), nullable=True)
    detected_tables  = db.Column("detected_tables",  db.JSON,     nullable=True)
    generated_sql    = db.Column("generated_sql",    db.Text,     nullable=True)
    execution_result = db.Column("execution_result", db.Text,     nullable=True)
    exact_match      = db.Column("exact_match",      db.Boolean,  nullable=True)
    confidence_score = db.Column("confidence_score", db.Float,    nullable=True)
    latency_ms       = db.Column("latency_ms",       db.Integer,  nullable=True)
    created_at       = db.Column("created_at",       db.DateTime, nullable=False, default=datetime.utcnow)

    # ── Relations ─────────────────────────────────────────────────────────────
    user = db.relationship(
        "User",
        backref=db.backref("ai_query_logs", lazy="dynamic"),
        foreign_keys=[user_id],
        lazy="select",
    )

    # ── Sérialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Sérialise le log IA en dictionnaire JSON-safe.

        Returns:
            Dictionnaire représentant le log de requête IA.
        """
        return {
            "id_log":           self.id_log,
            "user_id":          self.user_id,
            "prompt":           self.prompt,
            "detected_intent":  self.detected_intent,
            "detected_tables":  self.detected_tables,
            "generated_sql":    self.generated_sql,
            "execution_result": self.execution_result,
            "exact_match":      self.exact_match,
            "confidence_score": self.confidence_score,
            "latency_ms":       self.latency_ms,
            "created_at":       self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        return (
            f"<AIQueryLog id={self.id_log} user={self.user_id} "
            f"intent={self.detected_intent!r} score={self.confidence_score}>"
        )

"""
Modèle SQLAlchemy — Service
Correspond exactement à la table `Service` du schéma MySQL.
Représente les services hospitaliers (Cardiologie, Neurologie, etc.).
"""

from database.db import db


class Service(db.Model):
    """Service hospitalier auquel est rattaché le personnel médical."""

    __tablename__ = "Service"

    # ── Colonnes ──────────────────────────────────────────────────────────────
    id_service   = db.Column("id_service",   db.Integer,     primary_key=True, autoincrement=True)
    service_name = db.Column("service_name", db.String(100), nullable=False, unique=True)

    # ── Relations ─────────────────────────────────────────────────────────────
    # Personnel médical rattaché à ce service (1-to-many)
    staff_members = db.relationship(
        "Staff",
        backref=db.backref("service", lazy="select"),
        foreign_keys="Staff.id_service",
        lazy="dynamic",
    )

    # ── Sérialisation ─────────────────────────────────────────────────────────

    def to_dict(self, include_staff: bool = False) -> dict:
        """
        Sérialise le service en dictionnaire JSON-safe.

        Args:
            include_staff: Si True, inclut la liste du personnel du service.

        Returns:
            Dictionnaire représentant le service.
        """
        data = {
            "id_service":   self.id_service,
            "service_name": self.service_name,
        }

        if include_staff:
            data["staff_members"] = [s.to_dict() for s in self.staff_members]

        return data

    def __repr__(self) -> str:
        return f"<Service id={self.id_service} name={self.service_name!r}>"

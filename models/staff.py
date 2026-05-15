"""
Modèle SQLAlchemy — Medical_staff
Correspond exactement à la table `Medical_staff` du schéma MySQL.
Personnel médical : médecins, infirmiers, techniciens, administrateurs.
"""

from datetime import datetime
from database.db import db


class Staff(db.Model):
    """
    Membre du personnel médical rattaché à un service hospitalier.
    """

    __tablename__ = "Medical_staff"

    # ── Colonnes ──────────────────────────────────────────────────────────────
    id_staff       = db.Column("id_staff",       db.Integer,     primary_key=True, autoincrement=True)
    name_staff     = db.Column("name_staff",     db.String(150), nullable=False)
    position_staff = db.Column(
        "position_staff",
        db.Enum("Doctor", "Nurse", "Technician", "Administrator", name="staff_position_enum"),
        nullable=False,
        default="Doctor",
    )
    speciality  = db.Column("speciality",  db.String(100), nullable=True)
    id_service  = db.Column(
        "id_service",
        db.Integer,
        db.ForeignKey("Service.id_service", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
    )
    email      = db.Column("email",      db.String(150), nullable=True, unique=True)
    phone      = db.Column("phone",      db.String(20),  nullable=True)
    created_at = db.Column("created_at", db.DateTime,    nullable=False, default=datetime.utcnow)

    # ── Relations ─────────────────────────────────────────────────────────────
    # Consultations effectuées par ce membre du personnel (1-to-many)
    consultations = db.relationship(
        "Consultation",
        backref=db.backref("staff", lazy="select"),
        foreign_keys="Consultation.id_staff",
        lazy="dynamic",
    )

    # ── Sérialisation ─────────────────────────────────────────────────────────

    def to_dict(self, include_service: bool = False, include_consultations: bool = False) -> dict:
        """
        Sérialise le membre du personnel en dictionnaire JSON-safe.

        Args:
            include_service:       Si True, inclut les données du service.
            include_consultations: Si True, inclut la liste des consultations.

        Returns:
            Dictionnaire représentant le membre du personnel.
        """
        data = {
            "id_staff":       self.id_staff,
            "name_staff":     self.name_staff,
            "position_staff": self.position_staff,
            "speciality":     self.speciality,
            "id_service":     self.id_service,
            "email":          self.email,
            "phone":          self.phone,
            "created_at":     self.created_at.isoformat() if self.created_at else None,
        }

        if include_service:
            data["service"] = self.service.to_dict() if self.service else None

        if include_consultations:
            data["consultations"] = [c.to_dict() for c in self.consultations]

        return data

    def __repr__(self) -> str:
        return f"<Staff id={self.id_staff} name={self.name_staff!r} position={self.position_staff!r}>"

"""
Modèle SQLAlchemy — Patient
Correspond exactement à la table `Patient` du schéma MySQL.
La colonne `age` est VIRTUELLE côté MySQL (GENERATED ALWAYS) — non mappée en écriture.
"""

from datetime import datetime
from database.db import db


class Patient(db.Model):
    """
    Données démographiques d'un patient.
    La colonne `age` est calculée automatiquement par MySQL via TIMESTAMPDIFF.
    """

    __tablename__ = "Patient"

    # ── Colonnes ──────────────────────────────────────────────────────────────
    id_patient = db.Column("id_patient", db.Integer,     primary_key=True, autoincrement=True)
    first_name = db.Column("first_name", db.String(100), nullable=False)
    last_name  = db.Column("last_name",  db.String(100), nullable=False)
    birthdate  = db.Column("birthdate",  db.Date,        nullable=False)
    gender     = db.Column(
        "gender",
        db.Enum("Male", "Female", name="patient_gender_enum"),
        nullable=True,
    )
    # Colonne virtuelle MySQL — lecture seule, pas d'insertion/update
    age        = db.Column("age",        db.Integer,     nullable=True)
    email      = db.Column("email",      db.String(150), nullable=True, unique=True)
    phone      = db.Column("phone",      db.String(20),  nullable=True)
    created_at = db.Column("created_at", db.DateTime,    nullable=False, default=datetime.utcnow)

    # ── Relations ─────────────────────────────────────────────────────────────
    # Dossier médical (1-to-1, cascade delete)
    medical_record = db.relationship(
        "MedicalRecord",
        backref=db.backref("patient", lazy="select"),
        uselist=False,
        cascade="all, delete-orphan",
        lazy="select",
    )
    # Consultations (1-to-many)
    consultations = db.relationship(
        "Consultation",
        backref=db.backref("patient", lazy="select"),
        foreign_keys="Consultation.id_patient",
        lazy="dynamic",
    )

    # ── Sérialisation ─────────────────────────────────────────────────────────

    def to_dict(self, include_record: bool = False, include_consultations: bool = False) -> dict:
        """
        Sérialise le patient en dictionnaire JSON-safe.

        Args:
            include_record:        Si True, inclut le dossier médical.
            include_consultations: Si True, inclut la liste des consultations.

        Returns:
            Dictionnaire représentant le patient.
        """
        data = {
            "id_patient": self.id_patient,
            "first_name": self.first_name,
            "last_name":  self.last_name,
            "full_name":  f"{self.first_name} {self.last_name}",
            "birthdate":  self.birthdate.isoformat() if self.birthdate else None,
            "age":        self.age,
            "gender":     self.gender,
            "email":      self.email,
            "phone":      self.phone,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

        if include_record:
            data["medical_record"] = self.medical_record.to_dict() if self.medical_record else None

        if include_consultations:
            data["consultations"] = [c.to_dict() for c in self.consultations]

        return data

    def __repr__(self) -> str:
        return f"<Patient id={self.id_patient} name={self.first_name!r} {self.last_name!r}>"

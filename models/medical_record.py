"""
Modèle SQLAlchemy — Medical_records
Correspond exactement à la table `Medical_records` du schéma MySQL.
Relation 1-to-1 avec Patient (cascade delete).
"""

from datetime import datetime
from database.db import db


class MedicalRecord(db.Model):
    """
    Dossier médical complet d'un patient.
    Contient allergies, maladies chroniques, groupe sanguin et historique.
    """

    __tablename__ = "Medical_records"

    # ── Colonnes ──────────────────────────────────────────────────────────────
    id_record        = db.Column("id_record",        db.Integer, primary_key=True, autoincrement=True)
    id_patient       = db.Column(
        "id_patient",
        db.Integer,
        db.ForeignKey("Patient.id_patient", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
        unique=True,   # Relation 1-to-1 garantie par UNIQUE KEY
    )
    allergies        = db.Column("allergies",        db.Text,    nullable=True)
    chronic_diseases = db.Column("chronic_diseases", db.Text,    nullable=True)
    blood_group      = db.Column(
        "blood_group",
        db.Enum("A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", name="blood_group_enum"),
        nullable=True,
    )
    medical_history  = db.Column("medical_history",  db.Text,    nullable=True)
    last_updated     = db.Column(
        "last_updated",
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # ── Sérialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        Sérialise le dossier médical en dictionnaire JSON-safe.

        Returns:
            Dictionnaire représentant le dossier médical.
        """
        return {
            "id_record":        self.id_record,
            "id_patient":       self.id_patient,
            "allergies":        self.allergies,
            "chronic_diseases": self.chronic_diseases,
            "blood_group":      self.blood_group,
            "medical_history":  self.medical_history,
            "last_updated":     self.last_updated.isoformat() if self.last_updated else None,
        }

    def __repr__(self) -> str:
        return f"<MedicalRecord id={self.id_record} patient_id={self.id_patient}>"

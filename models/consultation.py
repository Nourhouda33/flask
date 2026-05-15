"""
Modèle SQLAlchemy — Consultation
Correspond exactement à la table `Consultation` du schéma MySQL.
Lie un patient à un membre du personnel médical pour une consultation.
"""

from datetime import datetime
from database.db import db


class Consultation(db.Model):
    """
    Consultation médicale : diagnostic, traitement et rapport.
    Liée à un Patient et à un Medical_staff (SET NULL si suppression).
    """

    __tablename__ = "Consultation"

    # ── Colonnes ──────────────────────────────────────────────────────────────
    id_consultation = db.Column("id_consultation", db.Integer,  primary_key=True, autoincrement=True)
    diagnosis       = db.Column("diagnosis",       db.Text,     nullable=False)
    treatment       = db.Column("treatment",       db.Text,     nullable=True)
    medical_report  = db.Column("medical_report",  db.Text,     nullable=True)
    date            = db.Column("date",            db.DateTime, nullable=False, default=datetime.utcnow)
    id_staff        = db.Column(
        "id_staff",
        db.Integer,
        db.ForeignKey("Medical_staff.id_staff", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
    )
    id_patient = db.Column(
        "id_patient",
        db.Integer,
        db.ForeignKey("Patient.id_patient", ondelete="SET NULL", onupdate="CASCADE"),
        nullable=True,
    )

    # ── Sérialisation ─────────────────────────────────────────────────────────

    def to_dict(self, include_patient: bool = False, include_staff: bool = False) -> dict:
        """
        Sérialise la consultation en dictionnaire JSON-safe.

        Args:
            include_patient: Si True, inclut les données du patient.
            include_staff:   Si True, inclut les données du médecin.

        Returns:
            Dictionnaire représentant la consultation.
        """
        data = {
            "id_consultation": self.id_consultation,
            "diagnosis":       self.diagnosis,
            "treatment":       self.treatment,
            "medical_report":  self.medical_report,
            "date":            self.date.isoformat() if self.date else None,
            "id_staff":        self.id_staff,
            "id_patient":      self.id_patient,
        }

        if include_patient and self.patient:
            data["patient"] = {
                "id_patient": self.patient.id_patient,
                "full_name":  f"{self.patient.first_name} {self.patient.last_name}",
                "age":        self.patient.age,
                "gender":     self.patient.gender,
            }

        if include_staff and self.staff:
            data["staff"] = {
                "id_staff":       self.staff.id_staff,
                "name_staff":     self.staff.name_staff,
                "position_staff": self.staff.position_staff,
                "speciality":     self.staff.speciality,
            }

        return data

    def __repr__(self) -> str:
        return (
            f"<Consultation id={self.id_consultation} "
            f"patient={self.id_patient} staff={self.id_staff} date={self.date}>"
        )

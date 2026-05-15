"""
Modèle SQLAlchemy — Users
Correspond exactement à la table `Users` du schéma MySQL.
Gère l'authentification RBAC (admin / doctor / staff / patient).
"""

import bcrypt
from datetime import datetime
from database.db import db


class User(db.Model):
    """
    Compte utilisateur avec rôle RBAC.
    Peut être lié à un Medical_staff ou à un Patient.
    """

    __tablename__ = "Users"

    # ── Colonnes ──────────────────────────────────────────────────────────────
    id_user       = db.Column("id_user",       db.Integer,      primary_key=True, autoincrement=True)
    username      = db.Column("username",      db.String(100),  nullable=False, unique=True)
    email         = db.Column("email",         db.String(150),  nullable=False, unique=True)
    password_hash = db.Column("password_hash", db.String(255),  nullable=False)
    role          = db.Column(
        "role",
        db.Enum("admin", "doctor", "staff", "patient", name="user_role_enum"),
        nullable=False,
        default="staff",
    )
    id_staff   = db.Column("id_staff",   db.Integer, db.ForeignKey("Medical_staff.id_staff",   ondelete="SET NULL", onupdate="CASCADE"), nullable=True)
    id_patient = db.Column("id_patient", db.Integer, db.ForeignKey("Patient.id_patient",       ondelete="SET NULL", onupdate="CASCADE"), nullable=True)
    is_active  = db.Column("is_active",  db.Boolean, nullable=False, default=True)
    created_at = db.Column("created_at", db.DateTime, nullable=False, default=datetime.utcnow)

    # ── Relations ─────────────────────────────────────────────────────────────
    # Un utilisateur peut être rattaché à un membre du personnel médical
    staff   = db.relationship("Staff",   backref=db.backref("user", uselist=False), foreign_keys=[id_staff],   lazy="select")
    # Un utilisateur peut être rattaché à un patient
    patient = db.relationship("Patient", backref=db.backref("user", uselist=False), foreign_keys=[id_patient], lazy="select")

    # ── Méthodes mot de passe ─────────────────────────────────────────────────

    def set_password(self, plain_password: str) -> None:
        """
        Hache le mot de passe en clair avec bcrypt et stocke le hash.

        Args:
            plain_password: Mot de passe en clair fourni par l'utilisateur.
        """
        salt = bcrypt.gensalt(rounds=12)
        self.password_hash = bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")

    def check_password(self, plain_password: str) -> bool:
        """
        Vérifie si le mot de passe en clair correspond au hash stocké.

        Args:
            plain_password: Mot de passe en clair à vérifier.

        Returns:
            True si le mot de passe est correct, False sinon.
        """
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            self.password_hash.encode("utf-8"),
        )

    # ── Sérialisation ─────────────────────────────────────────────────────────

    def to_dict(self, include_relations: bool = False) -> dict:
        """
        Sérialise l'utilisateur en dictionnaire JSON-safe.
        Le password_hash n'est jamais exposé.

        Args:
            include_relations: Si True, inclut les données staff/patient liées.

        Returns:
            Dictionnaire représentant l'utilisateur.
        """
        data = {
            "id_user":    self.id_user,
            "username":   self.username,
            "email":      self.email,
            "role":       self.role,
            "id_staff":   self.id_staff,
            "id_patient": self.id_patient,
            "is_active":  self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

        if include_relations:
            data["staff"]   = self.staff.to_dict()   if self.staff   else None
            data["patient"] = self.patient.to_dict() if self.patient else None

        return data

    def __repr__(self) -> str:
        return f"<User id={self.id_user} username={self.username!r} role={self.role!r}>"

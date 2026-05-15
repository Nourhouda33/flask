"""
Package models — exporte tous les modèles SQLAlchemy.
L'ordre d'import respecte les dépendances FK :
  Service → Staff → Patient → MedicalRecord → Consultation → User → AIQueryLog
"""

from models.service        import Service
from models.staff          import Staff
from models.patient        import Patient
from models.medical_record import MedicalRecord
from models.consultation   import Consultation
from models.user           import User
from models.ai_query_log   import AIQueryLog

__all__ = [
    "Service",
    "Staff",
    "Patient",
    "MedicalRecord",
    "Consultation",
    "User",
    "AIQueryLog",
]

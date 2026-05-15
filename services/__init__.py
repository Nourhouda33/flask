"""
Package services — logique métier séparée des routes Flask.
"""

from services.patient_service import (
    search_patients,
    get_patient_full_profile,
    create_patient_with_record,
    update_patient,
    get_patient_consultations,
)

__all__ = [
    "search_patients",
    "get_patient_full_profile",
    "create_patient_with_record",
    "update_patient",
    "get_patient_consultations",
]

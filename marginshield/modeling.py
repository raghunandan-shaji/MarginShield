"""Compatibility exports for the ring-native modelling workflow.

The former broad refund-abuse classifier has been retired. New code should import
from :mod:`marginshield.tournament` directly.
"""

from marginshield.tournament import classification_metrics, select_threshold

__all__ = ["classification_metrics", "select_threshold"]

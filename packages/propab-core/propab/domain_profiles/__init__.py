"""Domain profiles for artifact-gate verification (fixes.md Step 3)."""
from propab.domain_profiles.base import DomainProfile
from propab.domain_profiles.registry import get_profile, resolve_domain_profile

__all__ = ["DomainProfile", "get_profile", "resolve_domain_profile"]

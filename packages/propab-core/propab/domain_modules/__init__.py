"""
Domain modules — the single extension point for scientific domains.

Core Propab knows nothing about any specific domain. Everything domain-specific
(what to measure, how to verify, which features exist, what artifacts to suspect,
what "confirmed" means, how to run preflight) lives behind the ``DomainPlugin``
interface defined in :mod:`propab.domain_modules.base` and is looked up through
:mod:`propab.domain_modules.registry`.

Adding a new domain means adding one plugin here and registering it — no change
to any file under ``services/`` or to core campaign/verification logic.
"""
from propab.domain_modules.base import DomainPlugin, PreflightResult
from propab.domain_modules.registry import (
    all_plugins,
    get_domain_plugin,
    register_plugin,
    resolve_domain_plugin,
)

__all__ = [
    "DomainPlugin",
    "PreflightResult",
    "all_plugins",
    "get_domain_plugin",
    "register_plugin",
    "resolve_domain_plugin",
]

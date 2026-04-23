from __future__ import annotations

from fastapi import APIRouter

from propab.tools.registry import ToolRegistry

router = APIRouter(prefix="/tools", tags=["tools"])
registry = ToolRegistry()


@router.get("")
async def list_tools() -> dict:
    return {"tools": registry.get_all_specs()}


@router.get("/{domain}")
async def list_tools_by_domain(domain: str) -> dict:
    return {"domain": domain, "tools": registry.get_cluster(domain)}

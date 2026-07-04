from fastapi import APIRouter, Depends

from services.api.app.deps import get_app_meta

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    summary="Service health check",
    description="Returns `status: ok` plus build metadata. Used by Docker healthchecks and load balancers.",
    responses={200: {"description": "API process is up and connected to dependencies"}},
)
async def health(meta: dict = Depends(get_app_meta)) -> dict:
    return {"status": "ok", **meta}

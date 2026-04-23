from fastapi import APIRouter, Depends

from services.api.app.deps import get_app_meta

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(meta: dict = Depends(get_app_meta)) -> dict:
    return {"status": "ok", **meta}

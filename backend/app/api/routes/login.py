from fastapi import APIRouter
from pydantic import BaseModel

class Message(BaseModel):
    message: str


router = APIRouter(prefix="/utils", tags=["utils"])


@router.get("/health-check/")
async def health_check() -> bool:
    return True
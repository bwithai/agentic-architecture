from fastapi import APIRouter
from pydantic import BaseModel

class Message(BaseModel):
    message: str


router = APIRouter(prefix="/login", tags=["login"])


@router.post(
    "/",
    status_code=201,
)
def login() -> Message:
    return Message(message="Login successful")


@router.get("/health-check/")
async def health_check() -> bool:
    return True
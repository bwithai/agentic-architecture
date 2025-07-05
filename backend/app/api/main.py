from fastapi import APIRouter

from app.api.routes import login
from app.api.routes import qontak_webhook

api_router = APIRouter()
api_router.include_router(login.router)
api_router.include_router(qontak_webhook.router)
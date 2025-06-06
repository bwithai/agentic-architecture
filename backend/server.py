from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

from app.api.main import api_router
from app.core.config import config


def custom_generate_unique_id(route: APIRoute) -> str:
    return f"{route.tags[0]}-{route.name}"


app = FastAPI(
    title=config.PROJECT_NAME,
    openapi_url=f"{config.API_V1_STR}/openapi.json",
    generate_unique_id_function=custom_generate_unique_id,
)

if config.all_cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.all_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(api_router, prefix=config.API_V1_STR)
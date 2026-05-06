from __future__ import annotations

"""FastAPI application entrypoint."""

from fastapi import FastAPI

from src.api.routers.frontend_mock import router as frontend_mock_router
from src.api.routers.git import router as git_router
from src.api.routers.health import router as health_router
from src.api.routers.git import router as git_router
from src.api.routers.pipelines import router as pipelines_router
from src.api.routers.pipelines import ui_router as ui_pipelines_router
from src.api.service import PipelineService
from src.api.routers.settings import router as settings_router
from src.api.settings_service import SettingsService


_pipeline_service: PipelineService | None = None


def get_pipeline_service() -> PipelineService:
    """全局单例获取器，供 router Depends 使用。"""
    return _pipeline_service or app.state.pipeline_service


def create_app(engine=None, service: PipelineService | None = None) -> FastAPI:
    global _pipeline_service
    app = FastAPI(
        title="FlowState API",
        version="0.1.0",
        description="A minimal, standalone RESTful API scaffold for backend development.",
    )

    svc = service or PipelineService(engine=engine)
    app.state.pipeline_service = svc
    _pipeline_service = svc
    app.state.settings_service = SettingsService()

    @app.get("/", tags=["meta"])
    async def root() -> dict[str, str]:
        return {
            "name": "FlowState API",
            "status": "running",
        }

    app.include_router(health_router)
    app.include_router(ui_pipelines_router)
    app.include_router(agents_router)
    app.include_router(checkpoints_router)
    app.include_router(analytics_router)
    app.include_router(activities_router)
    app.include_router(pipelines_router)
    app.include_router(git_router)
    app.include_router(settings_router)
    app.include_router(git_router, prefix="/api/v1/pipelines", tags=["git"])
    return app


app = create_app()

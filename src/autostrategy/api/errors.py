"""API error handling."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from autostrategy.services.exceptions import (
    ArtifactNotFoundError,
    AutostrategyServiceError,
    BacktestServiceError,
    LLMConfigurationRequiredError,
    PaperRunServiceError,
    StrategyAlreadyExistsError,
    StrategyNotFoundError,
    ValidationServiceError,
)

_ERROR_STATUS = {
    StrategyNotFoundError: 404,
    StrategyAlreadyExistsError: 409,
    ValidationServiceError: 400,
    BacktestServiceError: 400,
    ArtifactNotFoundError: 404,
    LLMConfigurationRequiredError: 428,
    PaperRunServiceError: 400,
}


def register_error_handlers(app: FastAPI) -> None:
    """Register API exception handlers."""

    @app.exception_handler(AutostrategyServiceError)
    async def handle_service_error(
        request: Request, exc: AutostrategyServiceError
    ) -> JSONResponse:
        status_code = 500
        for error_type, mapped_status in _ERROR_STATUS.items():
            if isinstance(exc, error_type):
                status_code = mapped_status
                break
        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed.",
                    "details": {"errors": jsonable_encoder(exc.errors())},
                }
            },
        )

    @app.exception_handler(ValidationError)
    async def handle_validation_error(request: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed.",
                    "details": {"errors": jsonable_encoder(exc.errors())},
                }
            },
        )

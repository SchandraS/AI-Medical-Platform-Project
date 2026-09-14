"""FastAPI application entrypoint: app wiring, lifespan, middleware, and the
global exception handlers that guarantee no unhandled 500s reach the client
for any known failure mode (validation, model load, inference, DB integrity).
"""
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import get_settings
from app import db as db_module
from app.ml.errors import InferenceError, ModelLoadError, NoProductionModelError
from app.models_db import RequestLog
from app.routers import auth, health, jobs, logs, metrics, models, predict
from app.workers.batch import start_worker, stop_worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("manas.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_worker()
    logger.info("MANAS inference service started")
    yield
    await stop_worker()
    logger.info("MANAS inference service stopped")


app = FastAPI(
    title="MANAS Diabetes Risk Inference Service",
    description="Production-style inference service for diabetes risk prediction, "
    "with model versioning, promotion gates, and audited logging.",
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # Let it propagate to the catch-all handler below; still record it.
        latency_ms = (time.perf_counter() - start) * 1000
        _write_request_log(request_id, request, 500, latency_ms)
        raise
    latency_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    _write_request_log(request_id, request, response.status_code, latency_ms)
    return response


def _write_request_log(request_id: str, request: Request, status_code: int, latency_ms: float) -> None:
    if request.url.path in ("/health", "/ready"):
        return  # avoid polluting metrics with orchestrator polling noise
    db = db_module.new_session()
    try:
        db.add(
            RequestLog(
                request_id=request_id,
                endpoint=request.url.path,
                method=request.method,
                status_code=status_code,
                latency_ms=latency_ms,
            )
        )
        db.commit()
    except Exception:  # noqa: BLE001
        logger.exception("failed to write request log")
        db.rollback()
    finally:
        db.close()


# --- Exception handlers: every known failure mode maps to a clean response ---

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = [
        {"loc": [str(p) for p in e["loc"]], "msg": e["msg"], "type": e["type"]}
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "input validation failed",
            "error_code": "validation_error",
            "errors": errors,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(ModelLoadError)
async def model_load_error_handler(request: Request, exc: ModelLoadError):
    logger.error("model load error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": f"model artifact unavailable: {exc}",
            "error_code": "model_load_error",
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(NoProductionModelError)
async def no_production_model_handler(request: Request, exc: NoProductionModelError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "detail": str(exc),
            "error_code": "no_production_model",
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(InferenceError)
async def inference_error_handler(request: Request, exc: InferenceError):
    logger.error("inference error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "the model failed to produce a prediction for this input",
            "error_code": "inference_error",
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(IntegrityError)
async def integrity_error_handler(request: Request, exc: IntegrityError):
    logger.warning("integrity error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={
            "detail": "the request conflicts with existing data (e.g. duplicate or constraint violation)",
            "error_code": "integrity_error",
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_code": None,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(Exception)
async def catch_all_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", None)
    logger.exception("unhandled exception (request_id=%s): %s", request_id, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "an unexpected error occurred; this has been logged",
            "error_code": "internal_error",
            "request_id": request_id,
        },
    )


app.include_router(health.router)
app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(jobs.router)
app.include_router(models.router)
app.include_router(logs.router)
app.include_router(metrics.router)

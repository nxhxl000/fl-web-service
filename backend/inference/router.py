from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.inference.service import predict
from backend.projects.deps import get_project_or_404_public
from backend.projects.models import Project


router = APIRouter(prefix="/projects/{project_id}/inference", tags=["inference"])


MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/predict")
async def predict_endpoint(
    project: Project = Depends(get_project_or_404_public),
    image: UploadFile = File(...),
) -> dict:
    """Public — anyone can upload an image and get top-5 predictions."""
    target = project.inference_target
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="This project has no inference target.",
        )
    image_bytes = await image.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image too large (>{MAX_UPLOAD_BYTES // (1024 * 1024)} MB)",
        )
    try:
        results = predict(
            model_name=target.model_name,
            dataset=target.dataset,
            weights_path=target.weights_path,
            image_bytes=image_bytes,
            top_k=5,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Inference failed: {exc}")
    return {
        "model_name": target.model_name,
        "dataset": target.dataset,
        "predictions": results,
    }

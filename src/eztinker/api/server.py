"""FastAPI server for EZTinker."""

import os
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..core.state import state
from ..models.api import (
    BatchInput,
    CreateTrainingRunRequest,
    CreateTrainingRunResponse,
    EvaluationRequest,
    JobResponse,
    JobResult,
    OptimParams,
    SamplingParams,
)

app = FastAPI(
    title="EZTinker API",
    description="Minimal Tinker clone for distributed model training",
    version="0.1.0",
)


# CORS configuration - secure by default, allow override via env var
ALLOWED_ORIGINS = (
    os.environ.get("EZTINKER_ALLOWED_ORIGINS", "").split(",")
    if os.environ.get("EZTINKER_ALLOWED_ORIGINS")
    else []
)


app.add_middleware(
    CORSMiddleware,  # type: ignore
    allow_origins=ALLOWED_ORIGINS
    if ALLOWED_ORIGINS
    else ["http://localhost:8000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Simple job store
job_store = {}


def _create_job() -> str:
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    job_store[job_id] = {"status": "queued", "result": None, "error": None}
    return job_id


@app.post("/v1/runs", response_model=CreateTrainingRunResponse)
async def create_training_run(req: CreateTrainingRunRequest):
    """Create a new training run."""
    from ..models.api import LoRAConfig, LossFunctionConfig

    try:
        # Provide defaults if None
        lora_config = req.lora_config if req.lora_config is not None else LoRAConfig()
        loss_config = req.loss_config if req.loss_config is not None else LossFunctionConfig()

        run_id = state.create_run(
            base_model=req.base_model,
            lora_config=lora_config,
            loss_config=loss_config,
            run_id=req.run_id,
        )
        return CreateTrainingRunResponse(
            run_id=run_id, status="created", message="Training session initialized"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/v1/runs/{run_id}/forward_backward")
async def forward_backward(run_id: str, batch: BatchInput):
    """Add batch and perform forward-backward pass."""
    job_id = _create_job()
    try:
        run = state.get_run(run_id)
        run.add_batch(batch)
        result = run.forward_backward(accumulation_steps=1)
        job_store[job_id].update({"status": "completed", "result": result})
        return JobResponse(job_id=job_id, status="completed")
    except Exception as e:
        job_store[job_id].update({"status": "failed", "error": str(e)})
        return JobResponse(job_id=job_id, status="failed")


@app.post("/v1/runs/{run_id}/optim_step")
async def optim_step(run_id: str, optim_params: OptimParams):
    """Perform optimizer step."""
    job_id = _create_job()
    try:
        run = state.get_run(run_id)
        result = run.optim_step(optim_params)
        job_store[job_id].update({"status": "completed", "result": result})
        return JobResponse(job_id=job_id, status="completed")
    except Exception as e:
        job_store[job_id].update({"status": "failed", "error": str(e)})
        return JobResponse(job_id=job_id, status="failed")


@app.post("/v1/sample")
async def sample(params: SamplingParams):
    """Generate text from prompt."""
    job_id = _create_job()
    try:
        # Default model key
        model_key = "default_model"
        if model_key not in state.sampler.models:
            state.sampler.load_model("gpt2", model_key)  # TODO: make this configurable

        result = state.sampler.sample(
            model_key=model_key,
            prompt=params.prompt,
            max_new_tokens=params.max_new_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            do_sample=params.do_sample,
        )

        job_store[job_id].update({"status": "completed", "result": {"generated_text": result}})
        return JobResponse(job_id=job_id, status="completed")
    except Exception as e:
        job_store[job_id].update({"status": "failed", "error": str(e)})
        return JobResponse(job_id=job_id, status="failed")


@app.post("/v1/runs/{run_id}/save")
async def save_state(run_id: str, name: str):
    """Save checkpoint."""
    job_id = _create_job()
    try:
        import os

        checkpoint_dir = os.environ.get("CHECKPOINTS_DIR", "checkpoints")
        run = state.get_run(run_id)
        outputs = run.save_checkpoint(checkpoint_dir, name, sampler_optimized=False)

        job_store[job_id].update({"status": "completed", "result": outputs})
        return JobResponse(job_id=job_id, status="completed")
    except Exception as e:
        job_store[job_id].update({"status": "failed", "error": str(e)})
        return JobResponse(job_id=job_id, status="failed")


@app.get("/v1/jobs/{job_id}", response_model=JobResult)
async def get_job_result(job_id: str):
    """Get job result."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")

    entry = job_store[job_id]
    return JobResult(
        job_id=job_id,
        status=entry["status"],
        result=entry.get("result"),
        error=entry.get("error"),
    )


@app.get("/v1/runs")
async def list_runs():
    """List all active training runs."""
    return {"runs": state.list_runs()}


@app.delete("/v1/runs/{run_id}")
async def delete_run(run_id: str):
    """Delete a training run."""
    try:
        state.delete_run(run_id)
        return {"status": "deleted", "run_id": run_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/v1/runs/{run_id}/evaluate")
async def evaluate_responses(run_id: str, req: EvaluationRequest):
    """Evaluate multiple responses and return scores (loss values).

    This endpoint computes model loss for each prompt+response pair,
    which can be used for rejection sampling to select the best response.
    """
    job_id = _create_job()
    try:
        run = state.get_run(run_id)
        results = run.evaluate_responses(req.batches)
        job_store[job_id].update({"status": "completed", "result": results})
        return JobResponse(job_id=job_id, status="completed")
    except Exception as e:
        job_store[job_id].update({"status": "failed", "error": str(e)})
        return JobResponse(job_id=job_id, status="failed")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

"""
avatar_service.py
=================

FastAPI service that generates an avatar ZIP from an uploaded image, combining
outputs from LAM-20k and LAM-5k into a single archive. Drop this file into the
root of your LAM repository (same working directory the original gradio apps
are launched from) and run:

    python avatar_service.py
    # or
    uvicorn avatar_service:app --host 0.0.0.0 --port 8000

Endpoint
--------
POST /generate-avatar            multipart/form-data
    image: <file>                the input portrait

Response: application/zip. The archive contains one folder (named after the
uploaded image's base filename) with:

    animation.glb      copied from ./assets/sample_oac/animation.glb
    offset.ply         from LAM-20k
    offset_low.ply     from LAM-5k
    skin.glb           from LAM-20k via Blender
    skin_low.glb       from LAM-5k via Blender

Environment variables (all optional, defaults shown)
----------------------------------------------------
    LAM_20K_MODEL_NAME   ./model_zoo/lam_models/releases/lam/lam-20k/step_045500/
    LAM_20K_INFER        ./configs/inference/lam-20k-8gpu.yaml
    LAM_5K_MODEL_NAME    ./model_zoo/lam_models/releases/lam/lam-5k/step_045500/
    LAM_5K_INFER         ./configs/inference/lam-5k-8gpu.yaml
    BLENDER_PATH         blender
    SAMPLE_MOTION_DIR    (auto-detected under ./assets/sample_motion/export/*/flame_param)
    SERVICE_HOST         0.0.0.0
    SERVICE_PORT         8000

Notes
-----
* Both LAM models are held in GPU memory concurrently. On a single-GPU host
  that is the main memory cost; on a constrained GPU, load one, infer, free,
  load the other (the code is structured so this swap is a small change).
* Requests are GPU-serialized via an asyncio.Lock — one avatar at a time.
"""

import os
import io
import shutil
import zipfile
import asyncio
import tempfile
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import torch
from PIL import Image
from omegaconf import OmegaConf
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

# --- LAM imports (identical to the gradio apps) -----------------------------
from tools.flame_tracking_single_image import FlameTrackingSingleImage
from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image
from lam.models import ModelLAM
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LAM_20K_MODEL_NAME = os.environ.get(
    "LAM_20K_MODEL_NAME",
    "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/",
)
LAM_20K_INFER = os.environ.get(
    "LAM_20K_INFER",
    "./configs/inference/lam-20k-8gpu.yaml",
)
LAM_5K_MODEL_NAME = os.environ.get(
    "LAM_5K_MODEL_NAME",
    "./model_zoo/lam_models/releases/lam/lam-20k/step_045500/",
)
LAM_5K_INFER = os.environ.get(
    "LAM_5K_INFER",
    "./configs/inference/lam-5k-8gpu.yaml",
)
BLENDER_PATH = os.environ.get("BLENDER_PATH", "/")
SAMPLE_MOTION_DIR = os.environ.get("SAMPLE_MOTION_DIR", "")

ANIMATION_GLB_SRC = "./assets/sample_oac/animation.glb"
TEMPLATE_FBX_20K = "./assets/sample_oac/template_file.fbx"
TEMPLATE_FBX_5K = "./assets/sample_oac/template_file_5k.fbx"

DEVICE = "cuda"
DTYPE = torch.float32

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("avatar_service")


# ---------------------------------------------------------------------------
# Config + model construction
# (non-CLI replacement for parse_configs() / _build_model() in the gradio apps)
# ---------------------------------------------------------------------------

def build_cfg(model_name: str, infer_config_path: str, blender_path: str):
    """Build an OmegaConf cfg for a (model, inference-config) pair."""
    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.create()
    cli_cfg.model_name = model_name
    cfg.blender_path = blender_path

    cfg_train = OmegaConf.load(infer_config_path)
    cfg.source_size = cfg_train.dataset.source_image_res
    try:
        cfg.src_head_size = cfg_train.dataset.src_head_size
    except Exception:
        cfg.src_head_size = 112
    cfg.render_size = cfg_train.dataset.render_image.high

    _rel = os.path.join(
        cfg_train.experiment.parent,
        cfg_train.experiment.child,
        os.path.basename(cli_cfg.model_name).split("_")[-1],
    )
    cfg.save_tmp_dump = os.path.join("exps", "save_tmp", _rel)
    cfg.image_dump = os.path.join("exps", "images", _rel)
    cfg.video_dump = os.path.join("exps", "videos", _rel)

    cfg_infer = OmegaConf.load(infer_config_path)
    cfg.merge_with(cfg_infer)
    cfg.setdefault("mesh_dump", os.path.join("dumps", cli_cfg.model_name, "meshes"))

    cfg.motion_video_read_fps = 30
    cfg.merge_with(cli_cfg)
    cfg.setdefault("logger", "INFO")
    assert cfg.model_name, "model_name is required"
    return cfg


def load_lam_model(cfg):
    """Instantiate ModelLAM and load pretrained weights (same as _build_model)."""
    model = ModelLAM(**cfg.model)
    resume = os.path.join(cfg.model_name, "model.safetensors")
    logger.info("Loading LAM weights from %s", resume)
    if resume.endswith("safetensors"):
        ckpt = load_file(resume, device="cpu")
    else:
        ckpt = torch.load(resume, map_location="cpu")
    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict:
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                logger.warning(
                    "Shape mismatch for %s: ckpt %s vs model %s, ignored.",
                    k, v.shape, state_dict[k].shape,
                )
        else:
            logger.warning("Unexpected param %s: %s", k, v.shape)
    return model


def autodetect_sample_motion_dir() -> str:
    """Locate any exported motion's flame_param directory — needed to satisfy
    infer_single_view's signature even though we don't save the driven video."""
    if SAMPLE_MOTION_DIR:
        return SAMPLE_MOTION_DIR
    candidates = sorted(Path("./assets/sample_motion/export").glob("*/flame_param"))
    if not candidates:
        raise RuntimeError(
            "No sample motion with flame_param/ found under "
            "./assets/sample_motion/export. Set SAMPLE_MOTION_DIR env var."
        )
    return str(candidates[0])


# ---------------------------------------------------------------------------
# Service state
# ---------------------------------------------------------------------------

class AppState:
    flametracking: Optional[FlameTrackingSingleImage] = None
    cfg_20k = None
    cfg_5k = None
    lam_20k = None
    lam_5k = None
    motion_seqs_dir: str = ""
    gpu_lock: Optional[asyncio.Lock] = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading FLAME tracking models...")
    state.flametracking = FlameTrackingSingleImage(
        output_dir="output/tracking",
        alignment_model_path="./model_zoo/flame_tracking_models/68_keypoints_model.pkl",
        vgghead_model_path="./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd",
        human_matting_path="./model_zoo/flame_tracking_models/matting/stylematte_synth.pt",
        facebox_model_path="./model_zoo/flame_tracking_models/FaceBoxesV2.pth",
        detect_iris_landmarks=True,
    )

    logger.info("Building LAM-20k cfg + loading model...")
    state.cfg_20k = build_cfg(LAM_20K_MODEL_NAME, LAM_20K_INFER, BLENDER_PATH)
    state.lam_20k = load_lam_model(state.cfg_20k).to(DEVICE).eval()

    logger.info("Building LAM-5k cfg + loading model...")
    state.cfg_5k = build_cfg(LAM_5K_MODEL_NAME, LAM_5K_INFER, BLENDER_PATH)
    state.lam_5k = load_lam_model(state.cfg_5k).to(DEVICE).eval()

    state.motion_seqs_dir = autodetect_sample_motion_dir()
    logger.info("Using sample motion dir: %s", state.motion_seqs_dir)

    state.gpu_lock = asyncio.Lock()
    logger.info("Avatar service ready.")

    try:
        yield
    finally:
        state.lam_20k = None
        state.lam_5k = None
        state.flametracking = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


app = FastAPI(title="LAM Avatar Service", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Pipeline helpers (sync; called via run_in_executor)
# ---------------------------------------------------------------------------

def run_flame_tracking(image_raw_path: str) -> str:
    """preprocess → optimize → export. Returns flametracking output dir."""
    assert state.flametracking.preprocess(image_raw_path) == 0, "flametracking.preprocess failed"
    assert state.flametracking.optimize() == 0, "flametracking.optimize failed"
    code, out_dir = state.flametracking.export()
    assert code == 0, "flametracking.export failed"
    return out_dir


def run_lam_inference(lam, cfg, tracked_image_path: str, tracked_mask_path: str, tmp_dir: str):
    """Run a single LAM inference. Returns (canonical_gaussians, shape_param).

    Mirrors core_fn from the gradio app but skips everything after the
    forward pass (no per-frame RGB dump, no mp4 encoding, no audio).
    """
    aspect_standard = 1.0 / 1.0
    source_size = cfg.source_size
    render_size = cfg.render_size

    image, _, _, shape_param = preprocess_image(
        tracked_image_path,
        mask_path=tracked_mask_path,
        intr=None, pad_ratio=0, bg_color=1.0, max_tgt_size=None,
        aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
        render_tgt_size=source_size, multiply=14,
        need_mask=True, get_shape_param=True,
    )

    # The canonical Gaussians don't actually depend on the driving motion,
    # but infer_single_view requires motion tensors — we provide a minimal
    # one from the first available sample motion, same as the gradio app.
    src = tracked_image_path.split("/")[-3]
    driven = state.motion_seqs_dir.rstrip("/").split("/")[-2]
    motion_seq = prepare_motion_seqs(
        state.motion_seqs_dir, None,
        save_root=tmp_dir, fps=30,
        bg_color=1.0, aspect_standard=aspect_standard,
        enlarge_ratio=[1.0, 1.0],
        render_image_res=render_size, multiply=16,
        need_mask=cfg.get("motion_img_need_mask", False),
        vis_motion=False,
        shape_param=shape_param,
        test_sample=False, cross_id=False,
        src_driven=[src, driven],
    )
    motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)

    with torch.no_grad():
        res = lam.infer_single_view(
            image.unsqueeze(0).to(DEVICE, DTYPE), None, None,
            render_c2ws=motion_seq["render_c2ws"].to(DEVICE),
            render_intrs=motion_seq["render_intrs"].to(DEVICE),
            render_bg_colors=motion_seq["render_bg_colors"].to(DEVICE),
            flame_params={k: v.to(DEVICE) for k, v in motion_seq["flame_params"].items()},
        )
    return res["cano_gs_lst"][0], shape_param


def export_avatar_assets(lam, cano_gs, shape_param, out_dir: str,
                         template_fbx: str, generate_glb_fn,
                         offset_name: str, skin_name: str):
    """Save the offset ply + skin glb for a single variant into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    shaped_head_path = lam.renderer.flame_model.save_shaped_mesh(
        shape_param.unsqueeze(0).cuda(), fd=out_dir,
    )
    cano_gs.save_ply(os.path.join(out_dir, offset_name), rgb2sh=False, offset2xyz=True)
    generate_glb_fn(
        input_mesh=Path(shaped_head_path),
        template_fbx=Path(template_fbx),
        output_glb=Path(os.path.join(out_dir, skin_name)),
        blender_exec=Path(BLENDER_PATH),
    )
    # The shaped mesh is only a Blender intermediate; keep the folder clean.
    if os.path.exists(shaped_head_path):
        os.remove(shaped_head_path)


def generate_avatar_zip(image_bytes: bytes, original_name: str, work_root: str) -> str:
    """Full pipeline, runs synchronously. Returns path to the generated zip."""
    base_name = os.path.splitext(os.path.basename(original_name))[0] or "avatar"
    tmp_dir = os.path.join(work_root, "work")
    os.makedirs(tmp_dir, exist_ok=True)
 
    # Persist the uploaded image to disk as PNG (flametracking wants a path).
    raw_path = os.path.join(tmp_dir, "raw.png")
    with Image.open(io.BytesIO(image_bytes)).convert("RGB") as img:
        img.save(raw_path)
 
    # 1. FLAME tracking — runs ONCE, shared between the two LAM variants.
    output_dir = run_flame_tracking(raw_path)
    tracked_img = os.path.join(output_dir, "images/00000_00.png")
    tracked_mask = os.path.join(output_dir, "fg_masks/00000_00.png")
 
    avatar_dir = os.path.join(tmp_dir, base_name)
    os.makedirs(avatar_dir, exist_ok=True)
 
    # The 20k and 5k GLB generators are two different modules.
    from tools.generateARKITGLBWithBlender import generate_glb as generate_glb_20k
    from tools.generateARKITGLBWithBlender_5k import generate_glb as generate_glb_5k
 
    # 2. LAM-20k → offset.ply + skin.glb
    logger.info("[%s] LAM-20k inference...", base_name)
    cano_20k, shape_20k = run_lam_inference(
        state.lam_20k, state.cfg_20k, tracked_img, tracked_mask, tmp_dir,
    )
    export_avatar_assets(
        state.lam_20k, cano_20k, shape_20k,
        avatar_dir, TEMPLATE_FBX_20K, generate_glb_20k,
        offset_name="offset.ply", skin_name="skin.glb",
    )
    # Free the 20k forward-pass tensors before the next inference.
    del cano_20k
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
    # 3. LAM-5k → offset_low.ply + skin_low.glb
    logger.info("[%s] LAM-5k inference...", base_name)
    cano_5k, shape_5k = run_lam_inference(
        state.lam_5k, state.cfg_5k, tracked_img, tracked_mask, tmp_dir,
    )
    export_avatar_assets(
        state.lam_5k, cano_5k, shape_5k,
        avatar_dir, TEMPLATE_FBX_5K, generate_glb_5k,
        offset_name="offset_low.ply", skin_name="skin_low.glb",
    )
    del cano_5k
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
    # 4. animation.glb (identical asset for both variants)
    if not os.path.exists(ANIMATION_GLB_SRC):
        raise RuntimeError(f"Missing animation glb at {ANIMATION_GLB_SRC}")
    shutil.copy(ANIMATION_GLB_SRC, os.path.join(avatar_dir, "animation.glb"))
 
    # 5. Zip only the files we actually want. save_shaped_mesh / Blender may
    # write extra artifacts (vertex_order.json, lbs_weight_*.json, bone_tree.json,
    # etc.) into avatar_dir — we skip those by using an explicit allow-list
    # rather than walking the directory.
    expected_files = [
        "animation.glb",
        "offset.ply",
        "offset_low.ply",
        "skin.glb",
        "skin_low.glb",
    ]
    zip_path = os.path.join(work_root, base_name + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in expected_files:
            full = os.path.join(avatar_dir, name)
            if not os.path.exists(full):
                raise RuntimeError(f"expected output missing: {name}")
            # Files are always stored under a fixed "image/" folder inside the
            # archive, regardless of the uploaded filename.
            zf.write(full, os.path.join("image", name))
 
    logger.info("[%s] Done: %s", base_name, zip_path)
    return zip_path


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.post("/generate-avatar")
async def generate_avatar(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "file must be an image")
    data = await image.read()
    if not data:
        raise HTTPException(400, "empty upload")

    work_root = tempfile.mkdtemp(prefix="avatar_req_")

    try:
        # Serialize GPU work across concurrent requests.
        async with state.gpu_lock:
            loop = asyncio.get_event_loop()
            zip_path = await loop.run_in_executor(
                None,
                generate_avatar_zip,
                data,
                image.filename or "avatar.png",
                work_root,
            )
    except AssertionError as e:
        shutil.rmtree(work_root, ignore_errors=True)
        # Usually a face-detection / tracking failure — surface it as 422.
        raise HTTPException(422, f"avatar generation failed: {e}") from e
    except Exception as e:
        shutil.rmtree(work_root, ignore_errors=True)
        logger.exception("Unexpected failure while generating avatar")
        raise HTTPException(500, f"internal error: {e}") from e

    # Clean up the working tree AFTER the response finishes streaming.
    background_tasks.add_task(shutil.rmtree, work_root, ignore_errors=True)
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=os.path.basename(zip_path),
    )


@app.get("/health")
async def health():
    return {
        "ready": state.lam_20k is not None and state.lam_5k is not None,
        "motion_seqs_dir": state.motion_seqs_dir,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "avatar_service:app",
        host=os.environ.get("SERVICE_HOST", "0.0.0.0"),
        port=int(os.environ.get("SERVICE_PORT", "8000")),
        reload=False,
    )

"""Run any command from this repo on a Modal GPU.

    modal run scripts/modal_run.py --cmd "python bench.py"
    GPU=H100 modal run scripts/modal_run.py --cmd "python bench.py"
    GPU=H100:8 TIMEOUT_MIN=60 modal run scripts/modal_run.py --cmd "..."
    modal run scripts/modal_run.py --model deepseek-ai/DeepSeek-V2-Lite --cmd "..."

`--model` is a Hugging Face repo id; it is downloaded once into ~/huggingface/<name>
(e.g. ~/huggingface/DeepSeek-V2-Lite), which is the path the bench commands take.
nsys is in the image: `nsys profile ... python -m bench.run ...` works as is.

Each run is killed after TIMEOUT_MIN minutes (default 20) so a hung job can't
burn credits; raise it only for runs that really are long.

The repo is mounted at run time (not baked into the image), so local edits
take effect on the next run without rebuilding. Models live in the Modal
volume `hf-models`, mounted at ~/huggingface; the first run downloads them.

Anything the command writes under out/ is saved to the volume `nanovllm-out`.
Pull it back with:

    mkdir -p out && modal volume get nanovllm-out / out/ --force
"""
import os
import shutil
import subprocess

import modal

REPO = "/root/repo"
MODELS = "/root/huggingface"
OUT = "/root/out"
GPU = os.environ.get("GPU", "L4")
TIMEOUT_MIN = int(os.environ.get("TIMEOUT_MIN", "20"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("build-essential", "wget")  # triton compiles its launcher with gcc
    .run_commands(  # Nsight Systems CLI from NVIDIA's CUDA repo, for kernel-level timelines
        "wget -qO /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i /tmp/cuda-keyring.deb && apt-get update && apt-get install -y --no-install-recommends nsight-systems-2026.3.2",
        "ln -sf $(ls -d /opt/nvidia/nsight-systems/*/bin/nsys | head -1) /usr/local/bin/nsys",
    )
    .pip_install("torch==2.6.0", "transformers>=4.51,<5", "xxhash", "huggingface_hub", "pytest")
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
        "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
    )
    .add_local_dir(".", REPO, ignore=["reference", ".git", "**/__pycache__", "out", ".venv"])
)

app = modal.App("nanovllm2deepinfer", image=image)
models = modal.Volume.from_name("hf-models", create_if_missing=True)
outputs = modal.Volume.from_name("nanovllm-out", create_if_missing=True)


@app.function(gpu=GPU, volumes={MODELS: models, OUT: outputs}, timeout=TIMEOUT_MIN * 60)
def run(cmd: str, model: str):
    repo = model if "/" in model else f"Qwen/{model}"  # bare names are Qwen models, as before
    path = os.path.join(MODELS, repo.split("/")[-1])
    if not os.path.exists(os.path.join(path, "config.json")):
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_dir=path)
        models.commit()
    try:
        subprocess.run(cmd, shell=True, cwd=REPO, check=True)
    finally:  # keep partial outputs of a failed run too
        if os.path.isdir(os.path.join(REPO, "out")):
            shutil.copytree(os.path.join(REPO, "out"), OUT, dirs_exist_ok=True)
            outputs.commit()


@app.local_entrypoint()
def main(cmd: str = "python bench.py", model: str = "Qwen/Qwen3-0.6B"):
    print(f"GPU={GPU}  timeout={TIMEOUT_MIN}min  cmd={cmd!r}")
    run.remote(cmd, model)

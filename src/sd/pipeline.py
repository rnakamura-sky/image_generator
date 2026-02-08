# coding: utf-8
"""
text2imageの画像生成パイプライン
"""
import logging
import os

import torch
from diffusers import StableDiffusionPipeline

logger = logging.getLogger(__name__)

# diffusersがHTTP通信を行わないよう設定
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

## モデルについての変数設定
PROJECT_FOLDER_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", ".."))
STABLE_DIFFUSION_MODEL = os.getenv(
    "STABLE_DIFFUSION_MODEL", "stable-diffusion/sd15/base/v1-5-pruned-emaonly.safetensors")
MODEL_FILE_PATH = os.path.join(
    PROJECT_FOLDER_PATH,
    "rsc/models",
    STABLE_DIFFUSION_MODEL
)
NUM_INFERENCE_STEPS = int(os.getenv("NUM_INFERENCE_STEPS", "20"))
GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", "7.5"))
LORA_MODEL_FOLDER = os.path.join(
    PROJECT_FOLDER_PATH,
    "rsc/models",
    "stable-diffusion/sd15/lora/style"
)
LORA_MODEL_NAME = "Ghibli_v6.safetensors"

class StableDiffusionManager:
    """Stable Diffusionを管理するクラス
    """
    _pipe = None

    @classmethod
    def get_pipeline(cls):
        """
        pipeline取得
        """
        if cls._pipe is not None:
            return cls._pipe

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda"else torch.float32

        logger.info("Using device: %s", device)
        logger.info("Using model: %s", MODEL_FILE_PATH)
        logger.info("Using LoRA model: %s", os.path.join(LORA_MODEL_FOLDER, LORA_MODEL_NAME))

        # StableDiffusionのパイプラインを設定
        pipe = StableDiffusionPipeline.from_single_file(
            pretrained_model_link_or_path=MODEL_FILE_PATH,
            torch_dtype=dtype,
            use_safetensors=True,
            safety_checker=None,    # シンプルにするため、OFF
            local_files_only=True,
        )
        pipe = pipe.to(device)

        # LoRAを設定
        pipe.load_lora_weights(
            LORA_MODEL_FOLDER,
            weight_name=LORA_MODEL_NAME,
            adapter_name="style"
        )
        pipe.set_adapters(["style"], adapter_weights=[1.0])

        # GPU(デバイスがcuda)の場合はスライ寝具を設定
        if device == "cuda":
            pipe.enable_attention_slicing()

        cls._pipe = pipe
        return cls._pipe

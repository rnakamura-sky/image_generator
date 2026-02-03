# coding: utf-8
"""
画像生成プログラム
"""
import logging
import os
import time

from dotenv import load_dotenv

import torch
from diffusers import StableDiffusionPipeline

# 実行内容に合わせて変更する変数
PROMPT = """Ghibli style, A single turtle, fully visible head to tail, centered in frame.
Anime-style illustration with clean lineart and soft shading.
Biologically accurate turtle anatomy: four legs, shell, head, tail correctly formed.
Natural pose, realistic proportions.
Not chibi, not humanoid, not fantasy.
"""
NEGATIVE_PROMPT = """extra limbs, missing legs, deformed anatomy, cropped body, partial view,
humanoid features, cartoon exaggeration, chibi style, fantasy creature,
multiple turtles, blurry, low detail, bad proportions, unrealistic anatomy
"""
IMAGE_NAME = "test.png"

# diffusersがHTTP通信を行わないよう設定
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# .envファイルの読み込み
load_dotenv()

## プロジェクトについての変数う設定
PROJECT_FOLDER_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."))
OUTPUT_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "outputs")

## モデルについての変数設定
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
LORA_MODEL_NAME = "ghibli_style_offset.safetensors"

## 出力についての変数設定
IMAGE_FILE_PATH = os.path.join(
    OUTPUT_FOLDER_PATH,
    IMAGE_NAME,
)

## ログについての変数設定
LOG_FILE_PATH = os.path.join(
    OUTPUT_FOLDER_PATH,
    "image_generate.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    # デバイス設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda"else torch.float32
    logger.info("Using device: %s", device)
    logger.info ("Using model: %s", MODEL_FILE_PATH)
    logger.info("Loading model...")
    start = time.time()
    pipe = StableDiffusionPipeline.from_single_file(
        pretrained_model_link_or_path=MODEL_FILE_PATH,
        torch_dtype=dtype,
        use_safetensors=True,
        safety_checker=None,    # シンプルにするため、OFF
        local_files_only=True,
    )
    pipe = pipe.to(device)
    logger.info("Model loaded in %.2fs", time.time() - start)

    logger.info("Loading LoRA...")
    start = time.time()
    pipe.load_lora_weights(
        LORA_MODEL_FOLDER,
        weight_name=LORA_MODEL_NAME,
        adapter_name="style"
    )
    pipe.set_adapters(["style"], adapter_weights=[1.0])
    logger.info("Loaded in %.2fs", time.time() - start)

    logger.info("Start image generation")
    start = time.time()
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    ).images[0]
    logger.info("Generation finished in %.2fs", time.time() - start)

    logger.info("Saving image...")
    start = time.time()
    image.save(IMAGE_FILE_PATH)
    logger.info("Image saved in %.2fs", time.time() - start)

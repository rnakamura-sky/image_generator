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
PROMPT = "a cute anime-style tortoise, hight quality"
IMAGE_NAME = "test.png"

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
    "STABLE_DIFFUSION_MODEL", "stable-diffusion/v1-5-pruned-emaonly.safetensors")
MODEL_FILE_PATH = os.path.join(
    PROJECT_FOLDER_PATH,
    "rsc/models",
    STABLE_DIFFUSION_MODEL
)
NUM_INFERENCE_STEPS = int(os.getenv("NUM_INFERENCE_STEPS", "20"))
GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", "7.5"))

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
    )
    pipe = pipe.to(device)
    logger.info("Model loaded in %.2fs", time.time() - start)

    logger.info("Start image generation")
    start = time.time()
    image = pipe(
        PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    ).images[0]
    logger.info("Generation finished in %.2fs", time.time() - start)

    logger.info("Saving image...")
    start = time.time()
    image.save(IMAGE_FILE_PATH)
    logger.info("Image saved in %.2fs", time.time() - start)

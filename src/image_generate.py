# coding: utf-8
"""
画像生成プログラム
"""
import os

from dotenv import load_dotenv

import torch
from diffusers import StableDiffusionPipeline

# 実行内容に合わせて変更する変数
PROMPT = "a cute anime-style tortoise, hight quality"
IMAGE_NAME = "test.png"

# .envファイルの読み込み
load_dotenv()

STABLE_DIFFUSION_MODEL = os.getenv(
    "STABLE_DIFFUSION_MODEL", "stable-diffusion/v1-5-pruned-emaonly.safetensors")
NUM_INFERENCE_STEPS = int(os.getenv("NUM_INFERENCE_STEPS", "20"))
GUIDANCE_SCALE = float(os.getenv("GUIDANCE_SCALE", "7.5"))

PROJECT_FOLDER_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."))
MODEL_FILE_PATH = os.path.join(
    PROJECT_FOLDER_PATH,
    "rsc/models",
    STABLE_DIFFUSION_MODEL
)
IMAGE_FILE_PATH = os.path.join(
    PROJECT_FOLDER_PATH,
    "outputs",
    IMAGE_NAME,
)

if __name__ == '__main__':
    pipe = StableDiffusionPipeline.from_single_file(
        pretrained_model_link_or_path=MODEL_FILE_PATH,
        torch_dtype=torch.float16,
        use_safetensors=True,
        safety_checker=None,    # シンプルにするため、OFF
    )

    image = pipe(
        PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
    ).images[0]

    image.save(IMAGE_FILE_PATH)

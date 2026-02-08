# coding: utf-8
"""
画像生成用アプリケーション
"""
import random

import streamlit as st
import torch

from sd.pipeline import StableDiffusionManager

# アプリケーションページ初期設定
st.set_page_config(page_title="Stable Diffusion text2img", layout="wide")
st.title("  Stable Diffusion text2img")

# パイプラインの設定
pipe = StableDiffusionManager.get_pipeline()

# 状態管理
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "do_generate" not in st.session_state:
    st.session_state.do_generate = False
if "image" not in st.session_state:
    st.session_state.image = None
if "image_meta" not in st.session_state:
    st.session_state.image_meta = {
        "model": "",
        "LoRA": None,
        "Prompt": "",
        "Negative": "",
        "Step": -1,
        "CFG": 0.0,
        "Width": -1,
        "Height": -1,
        "Seed": -1,
    }

DEFAULT_PROMPT = """masterpiece, best quality, high detail,
a quiet mountain landscape at sunrise,
misty valley, soft light, natural colors,
cinematic composition"""

DEFAULT_NEGATIVE_PROMPT = """worst quality, low quality, blurry,
overexposed, underexposed,
jpeg artifacts, distorted"""

# --- UI ---
left, center, right = st.columns(
    [0.3, 0.4, 0.3],
)

with left:
    prompt = st.text_area(
        "Prompt",
        DEFAULT_PROMPT,
        disabled=st.session_state.is_generating
    )

    negative = st.text_area(
        "Negative Prompt",
        DEFAULT_NEGATIVE_PROMPT,
        disabled=st.session_state.is_generating
    )

    steps = st.slider(
        "Steps", 1, 50, 25,
        disabled=st.session_state.is_generating
    )
    cfg = st.slider(
        "CFG Scale", 3.0, 10.0, 7.5,
        disabled=st.session_state.is_generating
    )
    width = st.selectbox(
        "Width", [512, 640, 768], index=0,
        disabled=st.session_state.is_generating
    )
    height = st.selectbox(
        "Height", [512, 640, 768], index=0,
        disabled=st.session_state.is_generating
    )
    seed = st.number_input(
        "Seed (-1 = random)", value=-1,
        disabled=st.session_state.is_generating
    )

with center:
    generate = st.button("Generate", disabled=st.session_state.is_generating)
    if  generate:
        st.session_state.is_generating = True
        st.session_state.do_generate = True
        st.rerun()

    if st.session_state.do_generate:
        st.session_state.do_generate = False

        try:
            with st.spinner("Generating..."):
                if seed == -1:
                    seed = random.randint(0, 2**32 - 1)
                generator = torch.Generator(device=pipe.device).manual_seed(seed)

                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    width=width,
                    height=height,
                    generator=generator
                ).images[0]

            st.session_state.image = image
            st.session_state.image_meta = {
                "model": "v1-5-pruned-emaonly.safetensors",
                "LoRA": "Ghibli_v6.safetensors",
                "Prompt": prompt,
                "Negative": negative,
                "Steps": steps,
                "CFG": cfg,
                "Width": width,
                "Height": height,
                "Seed": seed,
            }
            st.image(image, caption="Generated Image")

            # GPUメモリ開放
            if pipe.device == "cuda":
                torch.cuda.empty_cache()

        finally:
            st.session_state.is_generating = False
            st.rerun()
    else:
        if st.session_state.image is not None:
            st.image(st.session_state.image)

            with right:
                if st.session_state.image_meta is not None:
                    with st.expander("🔍  詳細情報を表示"):
                        st.json(st.session_state.image_meta)

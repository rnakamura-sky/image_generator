# coding: utf-8
"""
画像生成用アプリケーション
"""
import streamlit as st
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

DEFAULT_PROMPT = """masterpiece, best quality, high detail,
a quiet mountain landscape at sunrise,
misty valley, soft light, natural colors,
cinematic composition
"""

DEFAULT_NEGATIVE_PROMPT = """worst quality, low quality, blurry,
overexposed, underexposed,
jpeg artifacts, distorted
"""

# --- UI ---
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
    "Steps", 10, 50, 25,
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

if  st.button("Generate", disabled=st.session_state.is_generating):
    st.session_state.is_generating = True
    st.session_state.do_generate = True
    st.rerun()

if st.session_state.do_generate:
    st.session_state.do_generate = False

    try:
        with st.spinner("Generating..."):
            generator = None
            if seed != -1:
                import torch
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
        st.image(image)

        # GPUメモリ開放
        if pipe.device == "cuda":
            torch.cuda.empty_cache()

    finally:
        st.session_state.is_generating = False
        st.rerun()
else:
    if st.session_state.image is not None:
        st.image(st.session_state.image)

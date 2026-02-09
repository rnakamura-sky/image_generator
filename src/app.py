# coding: utf-8
"""
画像生成用アプリケーション
"""
import streamlit as st
import torch
from PIL import Image

from core.manager import PipelineManager
from core.generator import ImageGenerator
from core.params import GenerateParams


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


# パイプライン（マネージャー）設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_core():
    """
    load_core
    """
    pipes = PipelineManager(device=DEVICE)
    return ImageGenerator(pipelines=pipes)

generator = load_core()

# アプリケーションページ初期設定
st.set_page_config(page_title="Stable Diffusion", layout="wide")
st.title("  Stable Diffusion")

# --- UI ---
left, center, right = st.columns(
    [0.3, 0.4, 0.3],
)

with left:
    mode = st.radio("Mode", ["text2img", "img2img"])

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
    seed_input = st.text_input(
        "Seed （空欄 = ランダム）",
        disabled=st.session_state.is_generating
    )
    seed = int(seed_input) if seed_input.isdigit() else None

    strength = 0.45
    init_image = None

    if mode == "img2img":
        strength = st.slider("Strength", 0.1, 0.9, 0.45,
        disabled=st.session_state.is_generating)
        uploaded = st.file_uploader("Init Image", type=["png", "jpg"],
        disabled=st.session_state.is_generating)
        if uploaded:
            init_image = Image.open(uploaded).convert("RGB")
            st.image(init_image, width=256)

params = GenerateParams(
    prompt=prompt,
    negative_prompt=negative,
    steps=steps,
    cfg=cfg,
    seed=seed,
    strength=strength,
)

with center:
    generate = st.button("Generate", disabled=st.session_state.is_generating)

    if generate and mode == "img2img" and init_image is None:
        st.error("img2img には元画像が必要です")
        st.stop()
    elif generate:
        st.session_state.is_generating = True
        st.session_state.do_generate = True
        st.rerun()

    if st.session_state.do_generate:
        st.session_state.do_generate = False

        try:
            with st.spinner("Generating..."):
                if mode == "text2img":
                    image, used_seed = generator.text2img(params)
                else:
                    image, used_seed = generator.img2img(params, init_image)

            st.session_state.image = image
            st.session_state.image_meta = {
                "mode": mode,
                "model": "v1-5-pruned-emaonly.safetensors",
                "LoRA": "Ghibli_v6.safetensors",
                "Prompt": prompt,
                "Negative": negative,
                "Steps": steps,
                "CFG": cfg,
                "Width": width,
                "Height": height,
                "Seed": used_seed,
            }
            st.image(image, caption="Generated Image")

            # GPUメモリ開放
            if DEVICE == "cuda":
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

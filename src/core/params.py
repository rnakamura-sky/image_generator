# coding: utf-8
"""
generate_params
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class GenerateParams:
    """
    GenerateParams
    """
    prompt: str
    negative_prompt: str = ""
    steps: int = 25
    cfg: float = 7.5
    seed: Optional[int] = None
    width: int = 512
    height: int = 512

    # img2img用
    strength: float = 0.45

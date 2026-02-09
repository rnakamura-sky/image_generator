# coding: utf-8
"""
generator
"""
import torch
from PIL import Image

from core.manager import PipelineManager
from core.params import GenerateParams

class ImageGenerator:
    """
    ImageGenerator
    """
    def __init__(self, pipelines: PipelineManager):
        self.pipes = pipelines
        self.device = pipelines.device

    def _get_generator(self, seed):
        if seed is None:
            seed = torch.seed()
        gen = torch.Generator(self.device).manual_seed(seed)
        return gen, seed

    def text2img(self, params: GenerateParams):
        """
        text2img
        
        :param self: 説明
        :param params: 説明
        :type params: GenerateParams
        """
        gen, seed = self._get_generator(params.seed)

        image = self.pipes.txt2img(
                    prompt=params.prompt,
                    negative_prompt=params.negative_prompt,
                    num_inference_steps=params.steps,
                    guidance_scale=params.cfg,
                    width=params.width,
                    height=params.height,
                    generator=gen,
                ).images[0]

        return image, seed

    def img2img(self, params: GenerateParams, init_image: Image.Image):
        """
        img2img
        
        :param self: 説明
        :param params: 説明
        :type params: GenerateParams
        :param init_image: 説明
        :type init_image: Image.Image
        """
        gen, seed = self._get_generator(params.seed)

        init_image = init_image.resize((params.width, params.height))

        image = self.pipes.img2img(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            image=init_image,
            strength=params.strength,
            num_inference_steps=params.steps,
            guidance_scale=params.cfg,
            generator=gen,
        ).images[0]

        return image, seed

import math
import json
import torch
import gradio as gr
import numpy as np
from copy import copy
from PIL import Image
from modules import scripts, shared, processing, sd_samplers, rng, images, devices, prompt_parser, sd_models, extra_networks, ui_components, sd_schedulers

quote_swap = str.maketrans('\'"', '"\'')


class I2IHiresFix(scripts.Script):
    def __init__(self):
        super().__init__()
        self.p = None
        self.sampler_name = None
        self.cond = None
        self.uncond = None
        self.ratio = 2.0
        self.width = 0
        self.height = 0
        self.prompt = ""
        self.negative_prompt = ""
        self.steps = 0
        self.upscaler = 'R-ESRGAN 4x+'
        self.denoise_strength = 0.33
        self.cfg = 5

    def title(self):
        return "img2img Hires Fix"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        sampler_names = [x.name for x in sd_samplers.visible_samplers()]
        scheduler_names = [x.label for x in sd_schedulers.schedulers]

        with ui_components.InputAccordion(False, label='img2img Hires Fix') as enable:
            with gr.Row():
                upscaler = gr.Dropdown([x.name for x in shared.sd_upscalers], label='Upscaler', value=self.upscaler)
                steps = gr.Slider(minimum=0, maximum=25, step=1, label="Hires steps", value=self.steps)
                denoise_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Denoising strength", value=self.denoise_strength)

            with gr.Row():
                ratio = gr.Slider(minimum=0, maximum=4.0, step=0.05, label="Upscale by", value=self.ratio)
                width = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize width to", value=self.width)
                height = gr.Slider(minimum=0, maximum=2048, step=8, label="Resize height to", value=self.height)

            with gr.Row():
                sampler = gr.Dropdown(sampler_names, label='Sampler', value=sampler_names[0])
                scheduler = gr.Dropdown(label='Schedule type', elem_id=f"{self.tabname}_scheduler", choices=scheduler_names, value=scheduler_names[0])
                cfg = gr.Slider(minimum=1, maximum=30, step=1, label="CFG Scale", value=self.cfg)

            with gr.Row():
                prompt = gr.Textbox(label='Prompt', placeholder='Leave empty to use the same prompt as in first pass.', value=self.prompt)
                negative_prompt = gr.Textbox(label='Negative prompt', placeholder='Leave empty to use the same prompt as in first pass.', value=self.negative_prompt)

        return [enable, ratio, width, height, steps, upscaler, prompt, negative_prompt, denoise_strength, sampler, cfg, scheduler]

    def postprocess_image(self, p, pp, enable, ratio, width, height, steps, upscaler, prompt, negative_prompt, denoise_strength, sampler, cfg, scheduler):
        if not enable:
            return

        _, loras_act = extra_networks.parse_prompt(self.prompt)
        extra_networks.activate(p, loras_act)
        _, loras_deact = extra_networks.parse_prompt(self.negative_prompt)
        extra_networks.deactivate(p, loras_deact)

        with devices.autocast():
            shared.state.nextjob()
            self.p = copy(p)
            x = self._generate_image(pp.image)
            self.p = None

        sd_models.apply_token_merging(p.sd_model, p.get_token_merging_ratio())
        pp.image = x
        extra_networks.deactivate(p, loras_act)

    def process(self, p, *args, **kwargs):
        enable, *args = args
        if not enable:
            return
        self._update_internal_state(p, *args)

    def before_process_batch(self, p, *args, **kwargs):
        enable, *args = args
        if not enable:
            return
        p.extra_generation_params['img2img Hires Fix'] = self.create_infotext

    def create_infotext(self, p, *args, **kwargs):

        parameters = {
            'scale': f'{self.width}x{self.height}' if self.width and self.height else self.ratio,
            'upscaler': self.upscaler,
            'denoise': self.denoise_strength,
        }

        if self.steps != p.steps:
            parameters['steps'] = self.steps
        if self.sampler_name != p.sampler_name:
            parameters['sampler'] = self.sampler_name
        if self.scheduler != p.scheduler:
            parameters['scheduler'] = self.scheduler
        if self.cfg != p.cfg_scale:
            parameters['cfg'] = self.cfg
        if self.prompt:
            parameters['prompt'] = self.prompt
        if self.negative_prompt:
            parameters['negative_prompt'] = self.negative_prompt

        return json.dumps(parameters, ensure_ascii=False).translate(quote_swap)

    def _update_internal_state(self, p, ratio, width, height, steps, upscaler, prompt, negative_prompt, denoise_strength, sampler, cfg, scheduler):
        """Update internal state variables."""
        assert ratio > 0 or (width > 0 and height > 0), 'Either Upscale by or Resize to width x height muse be none 0'

        self.ratio = ratio
        self.width = width
        self.height = height
        self.prompt = prompt.strip()
        self.negative_prompt = negative_prompt.strip()
        self.steps = steps or p.steps
        self.upscaler = upscaler
        self.denoise_strength = denoise_strength
        self.sampler_name = sampler
        self.cfg = cfg
        self.scheduler = scheduler

    def _process_prompt(self, width, height):
        """Process the prompt and negative prompt for conditioning."""
        prompt = self.prompt or self.p.prompt.strip()
        negative_prompt = self.negative_prompt or self.p.negative_prompt.strip()

        with devices.autocast():
            if width and height and hasattr(prompt_parser, 'SdConditioning'):
                c = prompt_parser.SdConditioning([prompt], False, width, height)
                uc = prompt_parser.SdConditioning([negative_prompt], False, width, height)
            else:
                c, uc = [prompt], [negative_prompt]
            self.cond = prompt_parser.get_multicond_learned_conditioning(shared.sd_model, c, self.steps)
            self.uncond = prompt_parser.get_learned_conditioning(shared.sd_model, uc, self.steps)

    def _generate_image(self, x):
        """Generate the processed image based on the set parameters."""
        if not (self.width > 0 and self.height > 0) and self.ratio > 0:
            width = int(x.width * self.ratio)
            height = int(x.height * self.ratio)
        else:
            width, height = self.width, self.height

        sd_models.apply_token_merging(self.p.sd_model, self.p.get_token_merging_ratio(for_hr=True) / 2)

        with devices.autocast(), torch.inference_mode():
            self._process_prompt(width, height)

        x = images.resize_image(0, x, width, height, upscaler_name=self.upscaler)
        image = np.array(x).astype(np.float32) / 255.0
        image = np.moveaxis(image, 2, 0)
        decoded_sample = torch.from_numpy(image).to(shared.device).to(devices.dtype_vae)
        decoded_sample = 2.0 * decoded_sample - 1.0
        encoded_sample = shared.sd_model.encode_first_stage(decoded_sample.unsqueeze(0).to(devices.dtype_vae))
        sample = shared.sd_model.get_first_stage_encoding(encoded_sample)
        image_conditioning = self.p.img2img_image_conditioning(decoded_sample, sample)

        noise = torch.randn_like(sample)
        self.p.denoising_strength = self.denoise_strength
        self.p.cfg_scale = self.cfg
        self.p.batch_size = 1
        self.p.rng = rng.ImageRNG(sample.shape[1:], self.p.seeds, subseeds=self.p.subseeds, subseed_strength=self.p.subseed_strength, seed_resize_from_h=self.p.seed_resize_from_h, seed_resize_from_w=self.p.seed_resize_from_w)
        self.p.scheduler = self.scheduler

        sampler = sd_samplers.create_sampler(self.sampler_name, self.p.sd_model)
        sample = sampler.sample_img2img(self.p, sample.to(devices.dtype), noise, self.cond, self.uncond, steps=self.steps, image_conditioning=image_conditioning).to(devices.dtype_vae)

        self.cond = None
        self.uncond = None

        devices.torch_gc()
        decoded_sample = processing.decode_first_stage(shared.sd_model, sample)

        if math.isnan(decoded_sample.min()):
            devices.torch_gc()
            sample = torch.clamp(sample, -3, 3)
            decoded_sample = processing.decode_first_stage(shared.sd_model, sample)

        decoded_sample = torch.clamp((decoded_sample + 1.0) / 2.0, min=0.0, max=1.0).squeeze()
        x_sample = 255.0 * np.moveaxis(decoded_sample.to(torch.float32).cpu().numpy(), 0, 2)
        image = Image.fromarray(x_sample.astype(np.uint8))
        return image

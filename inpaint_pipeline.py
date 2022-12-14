import inspect
from typing import List, Optional, Union

import PIL
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
import torchvision as tv
to_pil = tv.transforms.ToPILImage()
to_tensor = tv.transforms.ToTensor()

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.configuration_utils import FrozenDict
from diffusers.utils import logging, deprecate

from fastprogress import progress_bar, master_bar
from delegation import delegates

from pathlib import Path

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def prepare_mask_and_masked_img(image, mask, strength):
    if isinstance(mask,str) or isinstance(mask,Path): mask = Image.open(mask)
    if isinstance(image,str) or isinstance(image,Path): image = Image.open(image)
    image = to_tensor(image.convert("RGB"))[None]

    mask = to_tensor(mask.convert("L"))[None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    
    masked_img = image*(1-strength*(1-mask)) 

    return mask, masked_img

class UnifiedInpaintPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        #scheduler = scheduler.set_format("pt")
        self.trained_betas = None
        
        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)
        
        
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.height = 512
        self.width = 512
        self.guidance_scale = 7.5
        self.steps = 90
        self.eta = 0.0
        self.generator = None
    
    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        if slice_size == "auto":
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        self.enable_attention_slicing(None)

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# ------------------------------------------ inpaint -------------------------------
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
    @torch.no_grad()
    def inpaint(
        self,
        prompt: Union[str, List[str]],
        img: Union[torch.FloatTensor, PIL.Image.Image],
        mask: Union[torch.FloatTensor, PIL.Image.Image],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        strength: float = 0.8,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        latents: Optional[torch.FloatTensor] = None,
        mb: Optional[master_bar] = None
    ):
        eta = 0.0
        generator = None
        steps, guidance_scale = self._get_defaults_nohw(steps,guidance_scale)
        num_imgs_per_prompt = 1
        
        if type(img) == torch.FloatTensor:
            height, width = img.shape[-2:]
        else:
            height, width = img.height, img.width
        
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        
        if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
            removed_text = self.tokenizer.batch_decode(text_input_ids[:, self.tokenizer.model_max_length :])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]
        
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_imgs_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_imgs_per_prompt, seq_len, -1)

        do_classifier_free_guidance = guidance_scale > 1.0
        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""]
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(batch_size, num_imgs_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_imgs_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            

            
        num_channels_latents = self.vae.config.latent_channels
        latents_shape = (batch_size * num_imgs_per_prompt, num_channels_latents, height // 8, width // 8)
        latents_dtype = text_embeddings.dtype
        if latents is None:
            latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=latents_dtype)
        else:
            latents = latents.to(self.device)

            
        initial_latents = latents.detach().clone().cpu()
        
        original_image = to_tensor(img).clone()

        mask, masked_img = prepare_mask_and_masked_img(img, mask, strength)
        mask = mask.to(device=self.device, dtype=text_embeddings.dtype)
        masked_img = masked_img.to(device=self.device, dtype=text_embeddings.dtype)

        # resize the mask to latents shape as we concatenate the mask to the latents
        mask = F.interpolate(mask, size=(height // 8, width // 8))
        
        masked_img_latents = self.vae.encode(masked_img).latent_dist.sample(generator=generator)
        masked_img_latents = 0.18215 * masked_img_latents

        mask = mask.repeat(num_imgs_per_prompt, 1, 1, 1)
        masked_img_latents = masked_img_latents.repeat(num_imgs_per_prompt, 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_img_latents = (
            torch.cat([masked_img_latents] * 2) if do_classifier_free_guidance else masked_img_latents
        )
        
        num_channels_mask = mask.shape[1]
        num_channels_masked_img = masked_img_latents.shape[1]
        
        if num_channels_latents + num_channels_mask + num_channels_masked_img != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_img`: {num_channels_masked_img}"
                f" = {num_channels_latents+num_channels_masked_img+num_channels_mask}. Please verify the config of"
                " `pipeline.unet` or your `mask` or `img` input."
            )

        # set timesteps
        self.scheduler.set_timesteps(steps)
        
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (??) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to ?? in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        for i, t in enumerate(progress_bar(timesteps_tensor, parent=mb)):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            # concat latents, mask, masked_img_latents in the channel dimension
            latent_model_input = torch.cat([latent_model_input, mask, masked_img_latents], dim=1)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample.float()

        image = (image/2+0.5).clamp(0,1)
        #original_image = (original_image/2 + 0.5).clamp(0,1)
        original_mask = F.interpolate(mask.float(),size=image.shape[2:],mode='bilinear')[:,:3]

        om = gaussian_blur(original_mask.repeat(1,3,1,1),kernel_size=63)[:1]
        
        out = torch.lerp(original_image[None,:3].to(self.device),image,om)
        #out = image

        return {'sample': [to_pil(i) for i in out], 'latents': initial_latents, 'steps':steps, 'guidance_scale':guidance_scale, 'prompt': prompt, 'negative_prompt': negative_prompt, 'extra': [om, original_mask, mask, image]}


    def _get_defaults(self,height=None,width=None,steps=None,guidance_scale=None):
        if height is None: height = self.height
        if width is None: width = self.width
        if steps is None: steps = self.steps
        if guidance_scale is None: guidance_scale = self.guidance_scale
        #if eta is None: eta = self.eta
        #if generator is None: generator = self.generator
        return height,width,steps,guidance_scale
    
    def _get_defaults_nohw(self, steps=None, guidance_scale=None):
        if steps is None: steps = self.steps
        if guidance_scale is None: guidance_scale = self.guidance_scale
        return steps,guidance_scale

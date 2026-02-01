import torch
from typing import Literal, Optional

from toolkit.basic import value_map
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.stable_diffusion_model import StableDiffusion
from toolkit.train_tools import get_torch_dtype
from toolkit.config_modules import TrainConfig

GuidanceType = Literal["targeted", "polarity", "targeted_polarity", "direct"]

DIFFERENTIAL_SCALER = 0.2


# targeted
def get_targeted_guidance_loss(
        noisy_latents: torch.Tensor,
        conditional_embeds: 'PromptEmbeds',
        match_adapter_assist: bool,
        network_weight_list: list,
        timesteps: torch.Tensor,
        pred_kwargs: dict,
        batch: 'DataLoaderBatchDTO',
        noise: torch.Tensor,
        sd: 'StableDiffusion',
        **kwargs
):
    with torch.no_grad():
        dtype = get_torch_dtype(sd.torch_dtype)
        device = sd.device_torch

        conditional_latents = batch.latents.to(device, dtype=dtype).detach()
        unconditional_latents = batch.unconditional_latents.to(device, dtype=dtype).detach()

        # Encode the unconditional image into latents
        unconditional_noisy_latents = sd.noise_scheduler.add_noise(
            unconditional_latents,
            noise,
            timesteps
        )
        conditional_noisy_latents = sd.noise_scheduler.add_noise(
            conditional_latents,
            noise,
            timesteps
        )

        # was_network_active = self.network.is_active
        sd.network.is_active = False
        sd.unet.eval()

        target_differential = unconditional_latents - conditional_latents
        # scale our loss by the differential scaler
        target_differential_abs = target_differential.abs()
        target_differential_abs_min = \
        target_differential_abs.min(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        target_differential_abs_max = \
            target_differential_abs.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]

        min_guidance = 1.0
        max_guidance = 2.0

        differential_scaler = value_map(
            target_differential_abs,
            target_differential_abs_min,
            target_differential_abs_max,
            min_guidance,
            max_guidance
        ).detach()


        # With LoRA network bypassed, predict noise to get a baseline of what the network
        # wants to do with the latents + noise. Pass our target latents here for the input.
        target_unconditional = sd.predict_noise(
            latents=unconditional_noisy_latents.to(device, dtype=dtype).detach(),
            conditional_embeddings=conditional_embeds.to(device, dtype=dtype).detach(),
            timestep=timesteps,
            guidance_scale=1.0,
            **pred_kwargs  # adapter residuals in here
        ).detach()
        prior_prediction_loss = torch.nn.functional.mse_loss(
            target_unconditional.float(),
            noise.float(),
            reduction="none"
        ).detach().clone()

    # turn the LoRA network back on.
    sd.unet.train()
    sd.network.is_active = True
    sd.network.multiplier = network_weight_list + [x + -1.0 for x in network_weight_list]

    # with LoRA active, predict the noise with the scaled differential latents added. This will allow us
    # the opportunity to predict the differential + noise that was added to the latents.
    prediction = sd.predict_noise(
        latents=torch.cat([conditional_noisy_latents, unconditional_noisy_latents], dim=0).to(device, dtype=dtype).detach(),
        conditional_embeddings=concat_prompt_embeds([conditional_embeds, conditional_embeds]).to(device, dtype=dtype).detach(),
        timestep=torch.cat([timesteps, timesteps], dim=0),
        guidance_scale=1.0,
        **pred_kwargs  # adapter residuals in here
    )

    prediction_conditional, prediction_unconditional = torch.chunk(prediction, 2, dim=0)

    conditional_loss = torch.nn.functional.mse_loss(
        prediction_conditional.float(),
        noise.float(),
        reduction="none"
    )

    unconditional_loss = torch.nn.functional.mse_loss(
        prediction_unconditional.float(),
        noise.float(),
        reduction="none"
    )

    positive_loss = torch.abs(
        conditional_loss.float() - prior_prediction_loss.float(),
    )
    # scale our loss by the differential scaler
    positive_loss = positive_loss * differential_scaler

    positive_loss = positive_loss.mean([1, 2, 3])

    polar_loss = torch.abs(
        conditional_loss.float() - unconditional_loss.float(),
    ).mean([1, 2, 3])


    positive_loss = positive_loss.mean() + polar_loss.mean()


    positive_loss.backward()
    # loss = positive_loss.detach() + negative_loss.detach()
    loss = positive_loss.detach()

    # add a grad so other backward does not fail
    loss.requires_grad_(True)

    # restore network
    sd.network.multiplier = network_weight_list

    return loss


# this processes all guidance losses based on the batch information
def get_guidance_loss(
        noisy_latents: torch.Tensor,
        conditional_embeds: 'PromptEmbeds',
        match_adapter_assist: bool,
        network_weight_list: list,
        timesteps: torch.Tensor,
        pred_kwargs: dict,
        batch: 'DataLoaderBatchDTO',
        noise: torch.Tensor,
        sd: 'StableDiffusion',
        unconditional_embeds: Optional[PromptEmbeds] = None,
        mask_multiplier=None,
        prior_pred=None,
        scaler=None,
        train_config=None,
        **kwargs
):
    # TODO add others and process individual batch items separately
    guidance_type: GuidanceType = batch.file_items[0].dataset_config.guidance_type

    if guidance_type == "targeted":
        assert unconditional_embeds is None, "Unconditional embeds are not supported for targeted guidance"
        return get_targeted_guidance_loss(
            noisy_latents,
            conditional_embeds,
            match_adapter_assist,
            network_weight_list,
            timesteps,
            pred_kwargs,
            batch,
            noise,
            sd,
            **kwargs
        )
    else:
        raise NotImplementedError(f"Guidance type {guidance_type} is not implemented")

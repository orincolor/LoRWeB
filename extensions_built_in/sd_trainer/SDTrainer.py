# Copyright (C) 2026 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.


import os
import random
from collections import OrderedDict
from typing import Union, List, Optional

import numpy as np
from diffusers import T2IAdapter, AutoencoderTiny, ControlNetModel

from toolkit.basic import value_map
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.guidance import get_guidance_loss, GuidanceType
from toolkit.image_utils import show_tensors
from toolkit.print import print_acc
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.stable_diffusion_model import BlankNetwork
from toolkit.train_tools import get_torch_dtype, apply_snr_weight, add_all_snr_to_noise_scheduler, \
    apply_learnable_snr_gos
import gc
import torch
from jobs.process import BaseSDTrainProcess
from torchvision import transforms
from PIL import Image
import time

def flush():
    torch.cuda.empty_cache()
    gc.collect()


adapter_transforms = transforms.Compose([
    transforms.ToTensor(),
])


class SDTrainer(BaseSDTrainProcess):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.assistant_adapter: Union['T2IAdapter', 'ControlNetModel', None]
        self.do_prior_prediction = False
        self.do_long_prompts = False
        self.do_guided_loss = False
        self.taesd: Optional[AutoencoderTiny] = None

        self._clip_image_embeds_unconditional: Union[List[str], None] = None
        self.negative_prompt_pool: Union[List[str], None] = None
        self.batch_negative_prompt: Union[List[str], None] = None

        self.is_bfloat = self.train_config.dtype == "bfloat16" or self.train_config.dtype == "bf16"

        self.external_query = self.network_config.network_kwargs.get("external_query", False)
        self.query_mode = self.network_config.network_kwargs.get("query_mode", "aa'bb'")

        self.do_grad_scale = True
        if self.is_fine_tuning and self.is_bfloat:
            self.do_grad_scale = False

        self.cached_blank_embeds: Optional[PromptEmbeds] = None
        self.cached_trigger_embeds: Optional[PromptEmbeds] = None
        self.diff_output_preservation_embeds: Optional[PromptEmbeds] = None
        self.unconditional_embeds = None

        if self.train_config.diff_output_preservation:
            if self.trigger_word is None:
                raise ValueError("diff_output_preservation requires a trigger_word to be set")
            if self.network_config is None:
                raise ValueError("diff_output_preservation requires a network to be set")
            if self.train_config.train_text_encoder:
                raise ValueError("diff_output_preservation is not supported with train_text_encoder")

            # always do a prior prediction when doing diff output preservation
            self.do_prior_prediction = True

        # store the loss target for a batch so we can use it in a loss
        self._guidance_loss_target_batch: float = 0.0
        if isinstance(self.train_config.guidance_loss_target, (int, float)):
            self._guidance_loss_target_batch = float(self.train_config.guidance_loss_target)
        elif isinstance(self.train_config.guidance_loss_target, list):
            self._guidance_loss_target_batch = float(self.train_config.guidance_loss_target[0])
        else:
            raise ValueError(f"Unknown guidance loss target type {type(self.train_config.guidance_loss_target)}")

    def before_model_load(self):
        pass

    def before_dataset_load(self):
        self.assistant_adapter = None
        # get adapter assistant if one is set
        if self.train_config.adapter_assist_name_or_path is not None:
            adapter_path = self.train_config.adapter_assist_name_or_path

            if self.train_config.adapter_assist_type == "t2i":
                # dont name this adapter since we are not training it
                self.assistant_adapter = T2IAdapter.from_pretrained(
                    adapter_path, torch_dtype=get_torch_dtype(self.train_config.dtype)
                ).to(self.device_torch)
            elif self.train_config.adapter_assist_type == "control_net":
                self.assistant_adapter = ControlNetModel.from_pretrained(
                    adapter_path, torch_dtype=get_torch_dtype(self.train_config.dtype)
                ).to(self.device_torch, dtype=get_torch_dtype(self.train_config.dtype))
            else:
                raise ValueError(f"Unknown adapter assist type {self.train_config.adapter_assist_type}")

            self.assistant_adapter.eval()
            self.assistant_adapter.requires_grad_(False)
            flush()
        if self.train_config.train_turbo and self.train_config.show_turbo_outputs:
            if self.model_config.is_xl:
                self.taesd = AutoencoderTiny.from_pretrained("madebyollin/taesdxl",
                                                             torch_dtype=get_torch_dtype(self.train_config.dtype))
            else:
                self.taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd",
                                                             torch_dtype=get_torch_dtype(self.train_config.dtype))
            self.taesd.to(dtype=get_torch_dtype(self.train_config.dtype), device=self.device_torch)
            self.taesd.eval()
            self.taesd.requires_grad_(False)

    def hook_before_train_loop(self):
        super().hook_before_train_loop()

        # cache unconditional embeds (blank prompt)
        with torch.no_grad():
            self.unconditional_embeds = self.sd.encode_prompt(
                [self.train_config.unconditional_prompt],
                long_prompts=self.do_long_prompts
            ).to(
                self.device_torch,
                dtype=self.sd.torch_dtype
            ).detach()

        if self.train_config.do_prior_divergence:
            self.do_prior_prediction = True
        # move vae to device if we did not cache latents
        self.sd.vae.eval()
        self.sd.vae.to(self.device_torch)
        add_all_snr_to_noise_scheduler(self.sd.noise_scheduler, self.device_torch)

        if self.train_config.negative_prompt is not None:
            if os.path.exists(self.train_config.negative_prompt):
                with open(self.train_config.negative_prompt, 'r') as f:
                    self.negative_prompt_pool = f.readlines()
                    # remove empty
                    self.negative_prompt_pool = [x.strip() for x in self.negative_prompt_pool if x.strip() != ""]
            else:
                # single prompt
                self.negative_prompt_pool = [self.train_config.negative_prompt]

        # handle unload text encoder
        if self.train_config.unload_text_encoder:
            with torch.no_grad():
                if self.train_config.train_text_encoder:
                    raise ValueError("Cannot unload text encoder if training text encoder")
                # cache embeddings

                print_acc("\n***** UNLOADING TEXT ENCODER *****")
                print_acc("This will train only with a blank prompt or trigger word, if set")
                print_acc("If this is not what you want, remove the unload_text_encoder flag")
                print_acc("***********************************")
                print_acc("")
                self.sd.text_encoder_to(self.device_torch)
                self.cached_blank_embeds = self.sd.encode_prompt("")
                if self.trigger_word is not None:
                    self.cached_trigger_embeds = self.sd.encode_prompt(self.trigger_word)
                if self.train_config.diff_output_preservation:
                    self.diff_output_preservation_embeds = self.sd.encode_prompt(self.train_config.diff_output_preservation_class)

                # move back to cpu
                self.sd.text_encoder_to('cpu')
                flush()

    def process_output_for_turbo(self, pred, noisy_latents, timesteps, noise, batch):
        # to process turbo learning, we make one big step from our current timestep to the end
        # we then denoise the prediction on that remaining step and target our loss to our target latents
        # this currently only works on euler_a (that I know of). Would work on others, but needs to be coded to do so.
        # needs to be done on each item in batch as they may all have different timesteps
        batch_size = pred.shape[0]
        pred_chunks = torch.chunk(pred, batch_size, dim=0)
        noisy_latents_chunks = torch.chunk(noisy_latents, batch_size, dim=0)
        timesteps_chunks = torch.chunk(timesteps, batch_size, dim=0)
        latent_chunks = torch.chunk(batch.latents, batch_size, dim=0)
        noise_chunks = torch.chunk(noise, batch_size, dim=0)

        with torch.no_grad():
            # set the timesteps to 1000 so we can capture them to calculate the sigmas
            self.sd.noise_scheduler.set_timesteps(
                self.sd.noise_scheduler.config.num_train_timesteps,
                device=self.device_torch
            )
            train_timesteps = self.sd.noise_scheduler.timesteps.clone().detach()

            train_sigmas = self.sd.noise_scheduler.sigmas.clone().detach()

            # set the scheduler to one timestep, we build the step and sigmas for each item in batch for the partial step
            self.sd.noise_scheduler.set_timesteps(
                1,
                device=self.device_torch
            )

        denoised_pred_chunks = []
        target_pred_chunks = []

        for i in range(batch_size):
            pred_item = pred_chunks[i]
            noisy_latents_item = noisy_latents_chunks[i]
            timesteps_item = timesteps_chunks[i]
            latents_item = latent_chunks[i]
            noise_item = noise_chunks[i]
            with torch.no_grad():
                timestep_idx = [(train_timesteps == t).nonzero().item() for t in timesteps_item][0]
                single_step_timestep_schedule = [timesteps_item.squeeze().item()]
                # extract the sigma idx for our midpoint timestep
                sigmas = train_sigmas[timestep_idx:timestep_idx + 1].to(self.device_torch)

                end_sigma_idx = random.randint(timestep_idx, len(train_sigmas) - 1)
                end_sigma = train_sigmas[end_sigma_idx:end_sigma_idx + 1].to(self.device_torch)

                # add noise to our target

                # build the big sigma step. The to step will now be to 0 giving it a full remaining denoising half step
                # self.sd.noise_scheduler.sigmas = torch.cat([sigmas, torch.zeros_like(sigmas)]).detach()
                self.sd.noise_scheduler.sigmas = torch.cat([sigmas, end_sigma]).detach()
                # set our single timstep
                self.sd.noise_scheduler.timesteps = torch.from_numpy(
                    np.array(single_step_timestep_schedule, dtype=np.float32)
                ).to(device=self.device_torch)

                # set the step index to None so it will be recalculated on first step
                self.sd.noise_scheduler._step_index = None

            denoised_latent = self.sd.noise_scheduler.step(
                pred_item, timesteps_item, noisy_latents_item.detach(), return_dict=False
            )[0]

            residual_noise = (noise_item * end_sigma.flatten()).detach().to(self.device_torch, dtype=get_torch_dtype(
                self.train_config.dtype))
            # remove the residual noise from the denoised latents. Output should be a clean prediction (theoretically)
            denoised_latent = denoised_latent - residual_noise

            denoised_pred_chunks.append(denoised_latent)

        denoised_latents = torch.cat(denoised_pred_chunks, dim=0)
        # set the scheduler back to the original timesteps
        self.sd.noise_scheduler.set_timesteps(
            self.sd.noise_scheduler.config.num_train_timesteps,
            device=self.device_torch
        )

        output = (denoised_latents / self.sd.vae.config['scaling_factor'])
        output = self.sd.vae.decode(output).sample

        if self.train_config.show_turbo_outputs:
            # since we are completely denoising, we can show them here
            with torch.no_grad():
                show_tensors(output)

        # we return our big partial step denoised latents as our pred and our untouched latents as our target.
        # you can do mse against the two here  or run the denoised through the vae for pixel space loss against the
        # input tensor images.

        return output, batch.tensor.to(self.device_torch, dtype=get_torch_dtype(self.train_config.dtype))

    # you can expand these in a child class to make customization easier
    def calculate_loss(
            self,
            noise_pred: torch.Tensor,
            noise: torch.Tensor,
            noisy_latents: torch.Tensor,
            timesteps: torch.Tensor,
            batch: 'DataLoaderBatchDTO',
            mask_multiplier: Union[torch.Tensor, float] = 1.0,
            prior_pred: Union[torch.Tensor, None] = None,
            **kwargs
    ):
        loss_target = self.train_config.loss_target
        is_reg = any(batch.get_is_reg_list())
        additional_loss = 0.0

        prior_mask_multiplier = None
        target_mask_multiplier = None
        dtype = get_torch_dtype(self.train_config.dtype)

        has_mask = batch.mask_tensor is not None

        with torch.no_grad():
            loss_multiplier = torch.tensor(batch.loss_multiplier_list).to(self.device_torch, dtype=torch.float32)

        if self.train_config.match_noise_norm:
            # match the norm of the noise
            noise_norm = torch.linalg.vector_norm(noise, ord=2, dim=(1, 2, 3), keepdim=True)
            noise_pred_norm = torch.linalg.vector_norm(noise_pred, ord=2, dim=(1, 2, 3), keepdim=True)
            noise_pred = noise_pred * (noise_norm / noise_pred_norm)

        if self.train_config.pred_scaler != 1.0:
            noise_pred = noise_pred * self.train_config.pred_scaler

        target = None

        if self.train_config.target_noise_multiplier != 1.0:
            noise = noise * self.train_config.target_noise_multiplier

        if self.train_config.correct_pred_norm or (self.train_config.inverted_mask_prior and prior_pred is not None and has_mask):
            if self.train_config.correct_pred_norm and not is_reg:
                with torch.no_grad():
                    # this only works if doing a prior pred
                    if prior_pred is not None:
                        prior_mean = prior_pred.mean([2,3], keepdim=True)
                        prior_std = prior_pred.std([2,3], keepdim=True)
                        noise_mean = noise_pred.mean([2,3], keepdim=True)
                        noise_std = noise_pred.std([2,3], keepdim=True)

                        mean_adjust = prior_mean - noise_mean
                        std_adjust = prior_std - noise_std

                        mean_adjust = mean_adjust * self.train_config.correct_pred_norm_multiplier
                        std_adjust = std_adjust * self.train_config.correct_pred_norm_multiplier

                        target_mean = noise_mean + mean_adjust
                        target_std = noise_std + std_adjust

                        eps = 1e-5
                        # match the noise to the prior
                        noise = (noise - noise_mean) / (noise_std + eps)
                        noise = noise * (target_std + eps) + target_mean
                        noise = noise.detach()

            if self.train_config.inverted_mask_prior and prior_pred is not None and has_mask:
                assert not self.train_config.train_turbo
                with torch.no_grad():
                    prior_mask = batch.mask_tensor.to(self.device_torch, dtype=dtype)
                    # resize to size of noise_pred
                    prior_mask = torch.nn.functional.interpolate(prior_mask, size=(noise_pred.shape[2], noise_pred.shape[3]), mode='bicubic')
                    # stack first channel to match channels of noise_pred
                    prior_mask = torch.cat([prior_mask[:1]] * noise_pred.shape[1], dim=1)

                    prior_mask_multiplier = 1.0 - prior_mask

                    # scale so it is a mean of 1
                    prior_mask_multiplier = prior_mask_multiplier / prior_mask_multiplier.mean()
                if self.sd.is_flow_matching:
                    target = (noise - batch.latents).detach()
                else:
                    target = noise
        elif prior_pred is not None and not self.train_config.do_prior_divergence:
            assert not self.train_config.train_turbo
            # matching adapter prediction
            target = prior_pred
        elif self.sd.prediction_type == 'v_prediction':
            # v-parameterization training
            target = self.sd.noise_scheduler.get_velocity(batch.tensor, noise, timesteps)

        elif hasattr(self.sd, 'get_loss_target'):
            target = self.sd.get_loss_target(
                noise=noise,
                batch=batch,
                timesteps=timesteps,
            ).detach()

        elif self.sd.is_flow_matching:
            # forward ODE
            target = (noise - batch.latents).detach()
        else:
            target = noise

        if self.train_config.do_guidance_loss:
            with torch.no_grad():
                # we make cached blank prompt embeds that match the batch size
                unconditional_embeds = concat_prompt_embeds(
                    [self.unconditional_embeds] * noisy_latents.shape[0],
                )
                cfm_pred = self.predict_noise(
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    conditional_embeds=unconditional_embeds,
                    unconditional_embeds=None,
                    batch=batch,
                )

                # zero cfg

                # ref https://github.com/WeichenFan/CFG-Zero-star/blob/cdac25559e3f16cb95f0016c04c709ea1ab9452b/wan_pipeline.py#L557
                batch_size = target.shape[0]
                positive_flat = target.view(batch_size, -1)
                negative_flat = cfm_pred.view(batch_size, -1)
                # Calculate dot production
                dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
                # Squared norm of uncondition
                squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
                # st_star = v_cond^T * v_uncond / ||v_uncond||^2
                st_star = dot_product / squared_norm

                alpha = st_star

                is_video = len(target.shape) == 5

                alpha = alpha.view(batch_size, 1, 1, 1) if not is_video else alpha.view(batch_size, 1, 1, 1, 1)

                guidance_scale = self._guidance_loss_target_batch
                if isinstance(guidance_scale, list):
                    guidance_scale = torch.tensor(guidance_scale).to(target.device, dtype=target.dtype)
                    guidance_scale = guidance_scale.view(-1, 1, 1, 1) if not is_video else guidance_scale.view(-1, 1, 1, 1, 1)

                unconditional_target = cfm_pred * alpha
                target = unconditional_target + guidance_scale * (target - unconditional_target)


        if target is None:
            target = noise

        pred = noise_pred

        if self.train_config.train_turbo:
            pred, target = self.process_output_for_turbo(pred, noisy_latents, timesteps, noise, batch)

        ignore_snr = False

        if loss_target == 'source' or loss_target == 'unaugmented':
            assert not self.train_config.train_turbo
            # ignore_snr = True
            if batch.sigmas is None:
                raise ValueError("Batch sigmas is None. This should not happen")

            # src https://github.com/huggingface/diffusers/blob/324d18fba23f6c9d7475b0ff7c777685f7128d40/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L1190
            denoised_latents = noise_pred * (-batch.sigmas) + noisy_latents
            weighing = batch.sigmas ** -2.0
            if loss_target == 'source':
                # denoise the latent and compare to the latent in the batch
                target = batch.latents
            elif loss_target == 'unaugmented':
                # we have to encode images into latents for now
                # we also denoise as the unaugmented tensor is not a noisy diffirental
                with torch.no_grad():
                    unaugmented_latents = self.sd.encode_images(batch.unaugmented_tensor).to(self.device_torch, dtype=dtype)
                    unaugmented_latents = unaugmented_latents * self.train_config.latent_multiplier
                    target = unaugmented_latents.detach()

                # Get the target for loss depending on the prediction type
                if self.sd.noise_scheduler.config.prediction_type == "epsilon":
                    target = target  # we are computing loss against denoise latents
                elif self.sd.noise_scheduler.config.prediction_type == "v_prediction":
                    target = self.sd.noise_scheduler.get_velocity(target, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {self.sd.noise_scheduler.config.prediction_type}")

            # mse loss without reduction
            loss_per_element = (weighing.float() * (denoised_latents.float() - target.float()) ** 2)
            loss = loss_per_element
        else:
            loss = torch.nn.functional.mse_loss(pred.float(), target.float(), reduction="none")

            do_weighted_timesteps = False
            if self.sd.is_flow_matching:
                if self.train_config.linear_timesteps or self.train_config.linear_timesteps2:
                    do_weighted_timesteps = True
                if self.train_config.timestep_type == "weighted":
                    # use the noise scheduler to get the weights for the timesteps
                    do_weighted_timesteps = True

            # handle linear timesteps and only adjust the weight of the timesteps
            if do_weighted_timesteps:
                # calculate the weights for the timesteps
                timestep_weight = self.sd.noise_scheduler.get_weights_for_timesteps(
                    timesteps,
                    v2=self.train_config.linear_timesteps2,
                    timestep_type=self.train_config.timestep_type
                ).to(loss.device, dtype=loss.dtype)
                if len(loss.shape) == 4:
                    timestep_weight = timestep_weight.view(-1, 1, 1, 1).detach()
                elif len(loss.shape) == 5:
                    timestep_weight = timestep_weight.view(-1, 1, 1, 1, 1).detach()
                loss = loss * timestep_weight

        if self.train_config.do_prior_divergence and prior_pred is not None:
            loss = loss + (torch.nn.functional.mse_loss(pred.float(), prior_pred.float(), reduction="none") * -1.0)

        if self.train_config.train_turbo:
            mask_multiplier = mask_multiplier[:, 3:, :, :]
            # resize to the size of the loss
            mask_multiplier = torch.nn.functional.interpolate(mask_multiplier, size=(pred.shape[2], pred.shape[3]), mode='nearest')

        # multiply by our mask

        # print("loss.shape", loss.shape)
        # print("mask_multiplier.shape", mask_multiplier.shape)
        # exit(0)
        # target = (noise - batch.latents)[:,:,batch.latents.size(2)//2:,batch.latents.size(3)//2:].detach()
        try:
            loss = loss * mask_multiplier
        except:
            # todo handle mask with video models
            pass

        prior_loss = None
        if self.train_config.inverted_mask_prior and prior_pred is not None and prior_mask_multiplier is not None:
            assert not self.train_config.train_turbo
            prior_loss = torch.nn.functional.mse_loss(pred.float(), prior_pred.float(), reduction="none")

            prior_loss = prior_loss * prior_mask_multiplier * self.train_config.inverted_mask_prior_multiplier
            if torch.isnan(prior_loss).any():
                print_acc("Prior loss is nan")
                prior_loss = None
            else:
                prior_loss = prior_loss.mean([1, 2, 3])
                # loss = loss + prior_loss
                # loss = loss + prior_loss
            # loss = loss + prior_loss
        loss = loss.mean([1, 2, 3])
        # apply loss multiplier before prior loss
        # multiply by our mask
        try:
            loss = loss * loss_multiplier
        except:
            # todo handle mask with video models
            pass
        if prior_loss is not None:
            loss = loss + prior_loss

        if not self.train_config.train_turbo:
            if self.train_config.learnable_snr_gos:
                # add snr_gamma
                loss = apply_learnable_snr_gos(loss, timesteps, self.snr_gos)
            elif self.train_config.snr_gamma is not None and self.train_config.snr_gamma > 0.000001 and not ignore_snr:
                # add snr_gamma
                loss = apply_snr_weight(loss, timesteps, self.sd.noise_scheduler, self.train_config.snr_gamma,
                                        fixed=True)
            elif self.train_config.min_snr_gamma is not None and self.train_config.min_snr_gamma > 0.000001 and not ignore_snr:
                # add min_snr_gamma
                loss = apply_snr_weight(loss, timesteps, self.sd.noise_scheduler, self.train_config.min_snr_gamma)

        loss = loss.mean()

        if self.train_config.target_norm_std:
            # seperate out the batch and channels
            pred_std = noise_pred.std([2, 3], keepdim=True)
            norm_std_loss = torch.abs(self.train_config.target_norm_std_value - pred_std).mean()
            loss = loss + norm_std_loss


        return loss + additional_loss

    def preprocess_batch(self, batch: 'DataLoaderBatchDTO'):
        return batch

    def get_guided_loss(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            unconditional_embeds: Optional[PromptEmbeds] = None,
            **kwargs
    ):
        loss = get_guidance_loss(
            noisy_latents=noisy_latents,
            conditional_embeds=conditional_embeds,
            match_adapter_assist=match_adapter_assist,
            network_weight_list=network_weight_list,
            timesteps=timesteps,
            pred_kwargs=pred_kwargs,
            batch=batch,
            noise=noise,
            sd=self.sd,
            unconditional_embeds=unconditional_embeds,
            train_config=self.train_config,
            **kwargs
        )

        return loss


    # ------------------------------------------------------------------
    #  Mean-Flow loss (Geng et al., "Mean Flows for One-step Generative
    #  Modelling", 2025 – see Alg. 1 + Eq. (6) of the paper)
    # This version avoids jvp / double-back-prop issues with Flash-Attention
    # adapted from the work of lodestonerock
    # ------------------------------------------------------------------

    def get_prior_prediction(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            unconditional_embeds: Optional[PromptEmbeds] = None,
            conditioned_prompts=None,
            **kwargs
    ):
        # todo for embeddings, we need to run without trigger words
        was_unet_training = self.sd.unet.training
        was_network_active = False
        if self.network is not None:
            was_network_active = self.network.is_active
            self.network.is_active = False
        can_disable_adapter = False
        was_adapter_active = False

        # do a prediction here so we can match its output with network multiplier set to 0.0
        with torch.no_grad():
            dtype = get_torch_dtype(self.train_config.dtype)

            embeds_to_use = conditional_embeds.clone().detach()
            # handle clip vision adapter by removing triggers from prompt and replacing with the class name
            if self.embedding is not None:
                prompt_list = batch.get_caption_list()
                class_name = ''

                triggers = ['[trigger]', '[name]']
                remove_tokens = []

                if self.embed_config is not None:
                    triggers.append(self.embed_config.trigger)
                    for i in range(1, self.embed_config.tokens):
                        remove_tokens.append(f"{self.embed_config.trigger}_{i}")
                    if self.embed_config.trigger_class_name is not None:
                        class_name = self.embed_config.trigger_class_name

                for idx, prompt in enumerate(prompt_list):
                    for remove_token in remove_tokens:
                        prompt = prompt.replace(remove_token, '')
                    for trigger in triggers:
                        prompt = prompt.replace(trigger, class_name)
                    prompt_list[idx] = prompt

                embeds_to_use = self.sd.encode_prompt(
                    prompt_list,
                    long_prompts=self.do_long_prompts).to(
                    self.device_torch,
                    dtype=dtype).detach()

            # dont use network on this
            # self.network.multiplier = 0.0
            self.sd.unet.eval()

            if unconditional_embeds is not None:
                unconditional_embeds = unconditional_embeds.to(self.device_torch, dtype=dtype).detach()

            guidance_embedding_scale = self.train_config.cfg_scale
            if self.train_config.do_guidance_loss:
                guidance_embedding_scale = self._guidance_loss_target_batch

            prior_pred = self.sd.predict_noise(
                latents=noisy_latents.to(self.device_torch, dtype=dtype).detach(),
                conditional_embeddings=embeds_to_use.to(self.device_torch, dtype=dtype).detach(),
                unconditional_embeddings=unconditional_embeds,
                timestep=timesteps,
                guidance_scale=self.train_config.cfg_scale,
                guidance_embedding_scale=guidance_embedding_scale,
                rescale_cfg=self.train_config.cfg_rescale,
                batch=batch,
                **pred_kwargs  # adapter residuals in here
            )
            if was_unet_training:
                self.sd.unet.train()
            prior_pred = prior_pred.detach()
            # remove the residuals as we wont use them on prediction when matching control
            if match_adapter_assist and 'down_intrablock_additional_residuals' in pred_kwargs:
                del pred_kwargs['down_intrablock_additional_residuals']
            if match_adapter_assist and 'down_block_additional_residuals' in pred_kwargs:
                del pred_kwargs['down_block_additional_residuals']
            if match_adapter_assist and 'mid_block_additional_residual' in pred_kwargs:
                del pred_kwargs['mid_block_additional_residual']

            if can_disable_adapter:
                self.adapter.is_active = was_adapter_active
            # restore network
            # self.network.multiplier = network_weight_list
            if self.network is not None:
                self.network.is_active = was_network_active
        return prior_pred

    def before_unet_predict(self, batch: DataLoaderBatchDTO, **kwargs):
        external_query = None
        if self.external_query:
            external_query = self.sd.pipeline.get_external_query(batch.control_tensor/2+0.5, self.device_torch, self.query_mode)
        return external_query

    def after_unet_predict(self):
        pass

    def end_of_training_loop(self):
        pass

    def log_batch_files(self, batch: DataLoaderBatchDTO):
        """Log the files in each batch to a file in the output directory."""
        if not hasattr(self, 'batch_log_file'):
            # Create batch logging file in output directory
            batch_log_path = os.path.join(self.save_root, 'batch_files_log.txt')
            self.batch_log_file = batch_log_path

            if not os.path.exists(self.batch_log_file):
                # Write header
                with open(self.batch_log_file, 'w') as f:
                    f.write("Step\tBatch_Size\tFile_Paths\n")
                    f.write("-" * 80 + "\n")

        # Extract file paths from batch
        file_paths = [file_item.path for file_item in batch.file_items]
        batch_size = len(batch.file_items)

        # Log to file
        with open(self.batch_log_file, 'a') as f:
            f.write(f"{self.step_num}\t{batch_size}\t{file_paths}\n")


    def save_target_images_for_debugging(self, batch: DataLoaderBatchDTO, step: int = None):
        """
        Save target images to the output directory for debugging purposes.
        This helps verify that the training data is what you expect.
        """
        if not hasattr(self, 'debug_output_dir'):
            self.debug_output_dir = os.path.join(self.save_root, 'debug_target_images')
            os.makedirs(self.debug_output_dir, exist_ok=True)
            print_acc(f"Created debug output directory: {self.debug_output_dir}")

        print_acc(f"Debug: Saving target images for step {step}")
        print_acc(f"Debug: Batch has {len(batch.file_items)} items")

        # Save original tensor images (if available)
        if batch.tensor is not None:
            tensor_images = batch.tensor
            print_acc(f"Debug: batch.tensor shape: {tensor_images.shape}")
            if len(tensor_images.shape) == 4:  # (batch, channels, height, width)
                for i in range(tensor_images.shape[0]):
                    # Convert from [-1, 1] to [0, 1] range
                    img = tensor_images[i].detach().cpu()
                    img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
                    img = img.clamp(0, 1)

                    # Convert to PIL and save
                    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_np)

                    # Create filename
                    step_str = f"_step_{step}" if step is not None else ""
                    timestamp = int(time.time())
                    filename = f"target_tensor_{timestamp}{step_str}_batch_{i}.png"
                    filepath = os.path.join(self.debug_output_dir, filename)
                    pil_img.save(filepath)
                    print_acc(f"Saved target tensor image: {filepath}")
        else:
            print_acc("Debug: batch.tensor is None")

        # Save decoded latents (if available)
        if batch.latents is not None:
            latents = batch.latents
            print_acc(f"Debug: batch.latents shape: {latents.shape}")
            if len(latents.shape) == 4:  # (batch, channels, height, width)
                # Move VAE to device if needed
                if self.sd.vae.device == 'cpu':
                    self.sd.vae.to(self.device_torch)

                with torch.no_grad():
                    for i in range(latents.shape[0]):
                        # Decode the latent
                        latent = latents[i:i+1].to(self.device_torch, dtype=self.sd.vae.dtype)

                        # Apply VAE scaling
                        scaling_factor = getattr(self.sd.vae.config, "scaling_factor", 0.18215)
                        shift_factor = getattr(self.sd.vae.config, "shift_factor", 0)
                        latent = (latent / scaling_factor) + shift_factor

                        # Decode
                        decoded = self.sd.vae.decode(latent.to(self.sd.vae.device, dtype=self.sd.vae.dtype)).sample

                        # Convert from [-1, 1] to [0, 1] range
                        decoded = (decoded + 1) / 2
                        decoded = decoded.clamp(0, 1)

                        # Convert to PIL and save
                        decoded_np = (decoded.to(torch.float32).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        pil_img = Image.fromarray(decoded_np)

                        # Create filename
                        step_str = f"_step_{step}" if step is not None else ""
                        timestamp = int(time.time())
                        filename = f"target_decoded_{timestamp}{step_str}_batch_{i}.png"
                        filepath = os.path.join(self.debug_output_dir, filename)
                        pil_img.save(filepath)
                        print_acc(f"Saved decoded target image: {filepath}")
        else:
            print_acc("Debug: batch.latents is None")

        # Save unaugmented images (if available)
        if batch.unaugmented_tensor is not None:
            unaugmented_images = batch.unaugmented_tensor
            print_acc(f"Debug: batch.unaugmented_tensor shape: {unaugmented_images.shape}")
            if len(unaugmented_images.shape) == 4:  # (batch, channels, height, width)
                for i in range(unaugmented_images.shape[0]):
                    # Convert from [-1, 1] to [0, 1] range
                    img = unaugmented_images[i].detach().cpu()
                    img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
                    img = img.clamp(0, 1)

                    # Convert to PIL and save
                    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img_np)

                    # Create filename
                    step_str = f"_step_{step}" if step is not None else ""
                    timestamp = int(time.time())
                    filename = f"target_unaugmented_{timestamp}{step_str}_batch_{i}.png"
                    filepath = os.path.join(self.debug_output_dir, filename)
                    pil_img.save(filepath)
                    print_acc(f"Saved unaugmented target image: {filepath}")
        else:
            print_acc("Debug: batch.unaugmented_tensor is None")

        # Save control images (if available)
        if batch.control_tensor is not None:
            control_images = batch.control_tensor
            print_acc(f"Debug: batch.control_tensor shape: {control_images.shape}")
            if len(control_images.shape) == 4:  # (batch, channels, height, width)
                for i in range(control_images.shape[0]):
                    # Convert from [-1, 1] to [0, 1] range
                    img = control_images[i].detach().cpu()
                    img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
                    img = img.clamp(0, 1)

                    # Handle different channel configurations
                    if img.shape[0] == 1:
                        # Single channel (depth map, etc.) - convert to grayscale
                        img_np = (img.squeeze(0).numpy() * 255).astype(np.uint8)
                        pil_img = Image.fromarray(img_np, mode='L')

                        # Create filename
                        step_str = f"_step_{step}" if step is not None else ""
                        timestamp = int(time.time())
                        filename = f"control_image_{timestamp}{step_str}_batch_{i}.png"
                        filepath = os.path.join(self.debug_output_dir, filename)
                        pil_img.save(filepath)
                        print_acc(f"Saved control image: {filepath}")

                    elif img.shape[0] == 3:
                        # RGB image
                        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        pil_img = Image.fromarray(img_np)

                        # Create filename
                        step_str = f"_step_{step}" if step is not None else ""
                        timestamp = int(time.time())
                        filename = f"control_image_{timestamp}{step_str}_batch_{i}.png"
                        filepath = os.path.join(self.debug_output_dir, filename)
                        pil_img.save(filepath)
                        print_acc(f"Saved control image: {filepath}")

                    elif img.shape[0] == 9:
                        # 3 concatenated 3-channel images - split them up
                        print_acc(f"Debug: Splitting 9-channel control image into 3 separate images")
                        # Create list to store the 3 image parts
                        pil_images = []

                        for j in range(3):
                            # Extract each 3-channel image
                            img_part = img[j*3:(j+1)*3]  # channels 0-2, 3-5, 6-8
                            img_np = (img_part.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                            pil_images.append(Image.fromarray(img_np))

                        # Calculate total width and max height
                        total_width = sum(img.width for img in pil_images)
                        max_height = max(img.height for img in pil_images)

                        # Create new image with combined width
                        combined_img = Image.new('RGB', (total_width, max_height))

                        # Paste images side by side
                        x_offset = 0
                        for img in pil_images:
                            combined_img.paste(img, (x_offset, 0))
                            x_offset += img.width

                        # Save combined image
                        step_str = f"_step_{step}" if step is not None else ""
                        timestamp = int(time.time())
                        filename = f"control_image_combined_{timestamp}{step_str}_batch_{i}.png"
                        filepath = os.path.join(self.debug_output_dir, filename)
                        combined_img.save(filepath)
                        print_acc(f"Saved combined control image: {filepath}")

                    else:
                        # Multi-channel - take first 3 channels or convert to grayscale
                        if img.shape[0] > 3:
                            img = img[:3]  # Take first 3 channels
                            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                            pil_img = Image.fromarray(img_np)
                        else:
                            # Convert to grayscale by averaging channels
                            img_np = (img.mean(dim=0).numpy() * 255).astype(np.uint8)
                            pil_img = Image.fromarray(img_np, mode='L')

                        # Create filename
                        step_str = f"_step_{step}" if step is not None else ""
                        timestamp = int(time.time())
                        filename = f"control_image_{timestamp}{step_str}_batch_{i}.png"
                        filepath = os.path.join(self.debug_output_dir, filename)
                        pil_img.save(filepath)
                        print_acc(f"Saved control image: {filepath}")
        else:
            print_acc("Debug: batch.control_tensor is None")

        # Save clip images (if available)
        if batch.clip_image_tensor is not None:
            clip_images = batch.clip_image_tensor
            print_acc(f"Debug: batch.clip_image_tensor shape: {clip_images.shape}")
            if len(clip_images.shape) == 4:  # (batch, channels, height, width)
                for i in range(clip_images.shape[0]):
                    # Convert from [-1, 1] to [0, 1] range
                    img = clip_images[i].detach().cpu()
                    img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
                    img = img.clamp(0, 1)

                    # Handle different channel configurations
                    if img.shape[0] == 1:
                        # Single channel - convert to grayscale
                        img_np = (img.squeeze(0).numpy() * 255).astype(np.uint8)
                        pil_img = Image.fromarray(img_np, mode='L')
                    elif img.shape[0] == 3:
                        # RGB image
                        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        pil_img = Image.fromarray(img_np)
                    else:
                        # Multi-channel - take first 3 channels or convert to grayscale
                        if img.shape[0] > 3:
                            img = img[:3]  # Take first 3 channels
                            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                            pil_img = Image.fromarray(img_np)
                        else:
                            # Convert to grayscale by averaging channels
                            img_np = (img.mean(dim=0).numpy() * 255).astype(np.uint8)
                            pil_img = Image.fromarray(img_np, mode='L')

                    # Create filename
                    step_str = f"_step_{step}" if step is not None else ""
                    timestamp = int(time.time())
                    filename = f"clip_image_{timestamp}{step_str}_batch_{i}.png"
                    filepath = os.path.join(self.debug_output_dir, filename)
                    pil_img.save(filepath)
                    print_acc(f"Saved clip image: {filepath}")
        else:
            print_acc("Debug: batch.clip_image_tensor is None")

        # Print file paths for reference
        for i, file_item in enumerate(batch.file_items):
            print_acc(f"Debug: File {i}: {file_item.path}")
            if hasattr(file_item, 'caption'):
                print_acc(f"Debug: Caption {i}: {file_item.caption}")

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timesteps: Union[int, torch.Tensor] = 1,
        conditional_embeds: Union[PromptEmbeds, None] = None,
        unconditional_embeds: Union[PromptEmbeds, None] = None,
        batch: Optional['DataLoaderBatchDTO'] = None,
        is_primary_pred: bool = False,
        **kwargs,
    ):
        dtype = get_torch_dtype(self.train_config.dtype)
        guidance_embedding_scale = self.train_config.cfg_scale
        if self.train_config.do_guidance_loss:
            guidance_embedding_scale = self._guidance_loss_target_batch
        return self.sd.predict_noise(
            latents=noisy_latents.to(self.device_torch, dtype=dtype),
            conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype),
            unconditional_embeddings=unconditional_embeds,
            timestep=timesteps,
            guidance_scale=self.train_config.cfg_scale,
            guidance_embedding_scale=guidance_embedding_scale,
            detach_unconditional=False,
            rescale_cfg=self.train_config.cfg_rescale,
            bypass_guidance_embedding=self.train_config.bypass_guidance_embedding,
            batch=batch,
            **kwargs
        )


    def train_single_accumulation(self, batch: DataLoaderBatchDTO):
        with torch.no_grad():
            self.timer.start('preprocess_batch')
            batch = self.preprocess_batch(batch)
            dtype = get_torch_dtype(self.train_config.dtype)
            # sanity check
            if self.sd.vae.dtype != self.sd.vae_torch_dtype:
                self.sd.vae = self.sd.vae.to(self.sd.vae_torch_dtype)
            if isinstance(self.sd.text_encoder, list):
                for encoder in self.sd.text_encoder:
                    if encoder.dtype != self.sd.te_torch_dtype:
                        encoder.to(self.sd.te_torch_dtype)
            else:
                if self.sd.text_encoder.dtype != self.sd.te_torch_dtype:
                    self.sd.text_encoder.to(self.sd.te_torch_dtype)

            noisy_latents, noise, timesteps, conditioned_prompts, imgs = self.process_general_training_batch(batch)

            try:
                self.log_batch_files(batch)
            except Exception as e:
                pass


            # Save target images for debugging (only on first few steps to avoid spam)
            if (getattr(self.train_config, 'debug_save_target_images', False) and
                (not hasattr(self, '_debug_images_saved') or (self.step_num % 1000 == 0))):
                if not hasattr(self, '_debug_images_saved'):
                    self._debug_images_saved = 0
                self.save_target_images_for_debugging(batch, step=self.step_num)
                self._debug_images_saved += 1

            if self.train_config.do_cfg or self.train_config.do_random_cfg:
                # pick random negative prompts
                if self.negative_prompt_pool is not None:
                    negative_prompts = []
                    for i in range(noisy_latents.shape[0]):
                        num_neg = random.randint(1, self.train_config.max_negative_prompts)
                        this_neg_prompts = [random.choice(self.negative_prompt_pool) for _ in range(num_neg)]
                        this_neg_prompt = ', '.join(this_neg_prompts)
                        negative_prompts.append(this_neg_prompt)
                    self.batch_negative_prompt = negative_prompts
                else:
                    self.batch_negative_prompt = ['' for _ in range(batch.latents.shape[0])]

            network_weight_list = batch.get_network_weight_list()
            if self.train_config.single_item_batching:
                network_weight_list = network_weight_list + network_weight_list

            has_adapter_img = batch.control_tensor is not None
            has_clip_image = batch.clip_image_tensor is not None
            has_clip_image_embeds = batch.clip_image_embeds is not None
            # force it to be true if doing regs as we handle those differently
            if any([batch.file_items[idx].is_reg for idx in range(len(batch.file_items))]):
                has_clip_image = True
                if self._clip_image_embeds_unconditional is not None:
                    has_clip_image_embeds = True  # we are caching embeds, handle that differently
                    has_clip_image = False

            match_adapter_assist = False

            # check if we are matching the adapter assistant
            if self.assistant_adapter:
                if self.train_config.match_adapter_chance == 1.0:
                    match_adapter_assist = True
                elif self.train_config.match_adapter_chance > 0.0:
                    match_adapter_assist = torch.rand(
                        (1,), device=self.device_torch, dtype=dtype
                    ) < self.train_config.match_adapter_chance

            self.timer.stop('preprocess_batch')

            is_reg = False
            loss_multiplier = torch.ones((noisy_latents.shape[0], 1, 1, 1), device=self.device_torch, dtype=dtype)
            for idx, file_item in enumerate(batch.file_items):
                if file_item.is_reg:
                    loss_multiplier[idx] = loss_multiplier[idx] * self.train_config.reg_weight
                    is_reg = True

            adapter_images = None
            sigmas = None
            if has_adapter_img and (self.adapter or self.assistant_adapter):
                with self.timer('get_adapter_images'):
                    # todo move this to data loader
                    if batch.control_tensor is not None:
                        adapter_images = batch.control_tensor.to(self.device_torch, dtype=dtype).detach()
                        # match in channels
                        if self.assistant_adapter is not None:
                            in_channels = self.assistant_adapter.config.in_channels
                            if adapter_images.shape[1] != in_channels:
                                # we need to match the channels
                                adapter_images = adapter_images[:, :in_channels, :, :]
                    else:
                        raise NotImplementedError("Adapter images now must be loaded with dataloader")

            clip_images = None
            if has_clip_image:
                with self.timer('get_clip_images'):
                    # todo move this to data loader
                    if batch.clip_image_tensor is not None:
                        clip_images = batch.clip_image_tensor.to(self.device_torch, dtype=dtype).detach()

            mask_multiplier = torch.ones((noisy_latents.shape[0], 1, 1, 1), device=self.device_torch, dtype=dtype)
            if batch.mask_tensor is not None:
                with self.timer('get_mask_multiplier'):
                    # upsampling no supported for bfloat16
                    mask_multiplier = batch.mask_tensor.to(self.device_torch, dtype=torch.float16).detach()
                    # scale down to the size of the latents, mask multiplier shape(bs, 1, width, height), noisy_latents shape(bs, channels, width, height)
                    mask_multiplier = torch.nn.functional.interpolate(
                        mask_multiplier, size=(noisy_latents.shape[2], noisy_latents.shape[3])
                    )
                    # expand to match latents
                    mask_multiplier = mask_multiplier.expand(-1, noisy_latents.shape[1], -1, -1)
                    mask_multiplier = mask_multiplier.to(self.device_torch, dtype=dtype).detach()
                    # make avg 1.0
                    mask_multiplier = mask_multiplier / mask_multiplier.mean()

        def get_adapter_multiplier():
            if self.adapter and isinstance(self.adapter, T2IAdapter):
                # training a t2i adapter, not using as assistant.
                return 1.0
            elif match_adapter_assist:
                # training a texture. We want it high
                adapter_strength_min = 0.9
                adapter_strength_max = 1.0
            else:
                # training with assistance, we want it low
                # adapter_strength_min = 0.4
                # adapter_strength_max = 0.7
                adapter_strength_min = 0.5
                adapter_strength_max = 1.1

            adapter_conditioning_scale = torch.rand(
                (1,), device=self.device_torch, dtype=dtype
            )

            adapter_conditioning_scale = value_map(
                adapter_conditioning_scale,
                0.0,
                1.0,
                adapter_strength_min,
                adapter_strength_max
            )
            return adapter_conditioning_scale

        # flush()
        with self.timer('grad_setup'):

            # text encoding
            grad_on_text_encoder = False
            if self.train_config.train_text_encoder:
                grad_on_text_encoder = True

            if self.embedding is not None:
                grad_on_text_encoder = True

            # have a blank network so we can wrap it in a context and set multipliers without checking every time
            if self.network is not None:
                network = self.network
            else:
                network = BlankNetwork()

            # set the weights
            network.multiplier = network_weight_list

        # activate network if it exits

        prompts_1 = conditioned_prompts
        prompts_2 = None
        if self.train_config.short_and_long_captions_encoder_split and self.sd.is_xl:
            prompts_1 = batch.get_caption_short_list()
            prompts_2 = conditioned_prompts

            # make the batch splits
        if self.train_config.single_item_batching:
            if self.model_config.refiner_name_or_path is not None:
                raise ValueError("Single item batching is not supported when training the refiner")
            batch_size = noisy_latents.shape[0]
            # chunk/split everything
            noisy_latents_list = torch.chunk(noisy_latents, batch_size, dim=0)
            noise_list = torch.chunk(noise, batch_size, dim=0)
            timesteps_list = torch.chunk(timesteps, batch_size, dim=0)
            conditioned_prompts_list = [[prompt] for prompt in prompts_1]
            if imgs is not None:
                imgs_list = torch.chunk(imgs, batch_size, dim=0)
            else:
                imgs_list = [None for _ in range(batch_size)]
            if adapter_images is not None:
                adapter_images_list = torch.chunk(adapter_images, batch_size, dim=0)
            else:
                adapter_images_list = [None for _ in range(batch_size)]
            if clip_images is not None:
                clip_images_list = torch.chunk(clip_images, batch_size, dim=0)
            else:
                clip_images_list = [None for _ in range(batch_size)]
            mask_multiplier_list = torch.chunk(mask_multiplier, batch_size, dim=0)
            if prompts_2 is None:
                prompt_2_list = [None for _ in range(batch_size)]
            else:
                prompt_2_list = [[prompt] for prompt in prompts_2]

        else:
            noisy_latents_list = [noisy_latents]
            noise_list = [noise]
            timesteps_list = [timesteps]
            conditioned_prompts_list = [prompts_1]
            imgs_list = [imgs]
            adapter_images_list = [adapter_images]
            clip_images_list = [clip_images]
            mask_multiplier_list = [mask_multiplier]
            if prompts_2 is None:
                prompt_2_list = [None]
            else:
                prompt_2_list = [prompts_2]

        for noisy_latents, noise, timesteps, conditioned_prompts, imgs, adapter_images, clip_images, mask_multiplier, prompt_2 in zip(
                noisy_latents_list,
                noise_list,
                timesteps_list,
                conditioned_prompts_list,
                imgs_list,
                adapter_images_list,
                clip_images_list,
                mask_multiplier_list,
                prompt_2_list
        ):

            # if self.train_config.negative_prompt is not None:
            #     # add negative prompt
            #     conditioned_prompts = conditioned_prompts + [self.train_config.negative_prompt for x in
            #                                                  range(len(conditioned_prompts))]
            #     if prompt_2 is not None:
            #         prompt_2 = prompt_2 + [self.train_config.negative_prompt for x in range(len(prompt_2))]

            with (network):
                # do the custom adapter after the prior prediction
                with self.timer('encode_prompt'):
                    unconditional_embeds = None
                    if self.train_config.unload_text_encoder:
                        with torch.set_grad_enabled(False):
                            embeds_to_use = self.cached_blank_embeds.clone().detach().to(
                                self.device_torch, dtype=dtype
                            )
                            if self.cached_trigger_embeds is not None and not is_reg:
                                embeds_to_use = self.cached_trigger_embeds.clone().detach().to(
                                    self.device_torch, dtype=dtype
                                )
                            conditional_embeds = concat_prompt_embeds(
                                [embeds_to_use] * noisy_latents.shape[0]
                            )
                            if self.train_config.do_cfg:
                                unconditional_embeds = self.cached_blank_embeds.clone().detach().to(
                                    self.device_torch, dtype=dtype
                                )
                                unconditional_embeds = concat_prompt_embeds(
                                    [unconditional_embeds] * noisy_latents.shape[0]
                                )

                    elif grad_on_text_encoder:
                        with torch.set_grad_enabled(True):
                            conditional_embeds = self.sd.encode_prompt(
                                conditioned_prompts, prompt_2,
                                dropout_prob=self.train_config.prompt_dropout_prob,
                                long_prompts=self.do_long_prompts).to(
                                self.device_torch,
                                dtype=dtype)

                            if self.train_config.do_cfg:
                                # todo only do one and repeat it
                                unconditional_embeds = self.sd.encode_prompt(
                                    self.batch_negative_prompt,
                                    self.batch_negative_prompt,
                                    dropout_prob=self.train_config.prompt_dropout_prob,
                                    long_prompts=self.do_long_prompts).to(
                                    self.device_torch,
                                    dtype=dtype)
                    else:
                        with torch.set_grad_enabled(False):
                            # make sure it is in eval mode
                            if isinstance(self.sd.text_encoder, list):
                                for te in self.sd.text_encoder:
                                    te.eval()
                            else:
                                self.sd.text_encoder.eval()
                            conditional_embeds = self.sd.encode_prompt(
                                conditioned_prompts, prompt_2,
                                dropout_prob=self.train_config.prompt_dropout_prob,
                                long_prompts=self.do_long_prompts).to(
                                self.device_torch,
                                dtype=dtype)
                            if self.train_config.do_cfg:
                                unconditional_embeds = self.sd.encode_prompt(
                                    self.batch_negative_prompt,
                                    dropout_prob=self.train_config.prompt_dropout_prob,
                                    long_prompts=self.do_long_prompts).to(
                                    self.device_torch,
                                    dtype=dtype)

                            if self.train_config.diff_output_preservation:
                                dop_prompts = [p.replace(self.trigger_word, self.train_config.diff_output_preservation_class) for p in conditioned_prompts]
                                dop_prompts_2 = None
                                if prompt_2 is not None:
                                    dop_prompts_2 = [p.replace(self.trigger_word, self.train_config.diff_output_preservation_class) for p in prompt_2]
                                self.diff_output_preservation_embeds = self.sd.encode_prompt(
                                    dop_prompts, dop_prompts_2,
                                    dropout_prob=self.train_config.prompt_dropout_prob,
                                    long_prompts=self.do_long_prompts).to(
                                    self.device_torch,
                                    dtype=dtype)
                        # detach the embeddings
                        conditional_embeds = conditional_embeds.detach()
                        if self.train_config.do_cfg:
                            unconditional_embeds = unconditional_embeds.detach()

                # flush()
                pred_kwargs = {}

                if has_adapter_img:
                    if (self.adapter and isinstance(self.adapter, T2IAdapter)) or (
                            self.assistant_adapter and isinstance(self.assistant_adapter, T2IAdapter)):
                        with torch.set_grad_enabled(self.adapter is not None):
                            adapter = self.assistant_adapter if self.assistant_adapter is not None else self.adapter
                            adapter_multiplier = get_adapter_multiplier()
                            with self.timer('encode_adapter'):
                                down_block_additional_residuals = adapter(adapter_images)
                                if self.assistant_adapter:
                                    # not training. detach
                                    down_block_additional_residuals = [
                                        sample.to(dtype=dtype).detach() * adapter_multiplier for sample in
                                        down_block_additional_residuals
                                    ]
                                else:
                                    down_block_additional_residuals = [
                                        sample.to(dtype=dtype) * adapter_multiplier for sample in
                                        down_block_additional_residuals
                                    ]

                                pred_kwargs['down_intrablock_additional_residuals'] = down_block_additional_residuals

                prior_pred = None

                do_reg_prior = False
                # if is_reg and (self.network is not None or self.adapter is not None):
                #     # we are doing a reg image and we have a network or adapter
                #     do_reg_prior = True

                do_inverted_masked_prior = False
                if self.train_config.inverted_mask_prior and batch.mask_tensor is not None:
                    do_inverted_masked_prior = True

                do_guidance_prior = False

                if batch.unconditional_latents is not None:
                    # for this not that, we need a prior pred to normalize
                    guidance_type: GuidanceType = batch.file_items[0].dataset_config.guidance_type
                    if guidance_type == 'tnt':
                        do_guidance_prior = True

                if ((
                        has_adapter_img and self.assistant_adapter and match_adapter_assist) or self.do_prior_prediction or do_guidance_prior or do_reg_prior or do_inverted_masked_prior or self.train_config.correct_pred_norm):
                    with self.timer('prior predict'):
                        prior_embeds_to_use = conditional_embeds
                        # use diff_output_preservation embeds if doing dfe
                        if self.train_config.diff_output_preservation:
                            prior_embeds_to_use = self.diff_output_preservation_embeds.expand_to_batch(noisy_latents.shape[0])

                        prior_pred = self.get_prior_prediction(
                            noisy_latents=noisy_latents,
                            conditional_embeds=prior_embeds_to_use,
                            match_adapter_assist=match_adapter_assist,
                            network_weight_list=network_weight_list,
                            timesteps=timesteps,
                            pred_kwargs=pred_kwargs,
                            noise=noise,
                            batch=batch,
                            unconditional_embeds=unconditional_embeds,
                            conditioned_prompts=conditioned_prompts
                        )
                        if prior_pred is not None:
                            prior_pred = prior_pred.detach()

                # do the custom adapter after the prior prediction
                if has_adapter_img:
                    if (self.adapter and isinstance(self.adapter, ControlNetModel)) or (
                            self.assistant_adapter and isinstance(self.assistant_adapter, ControlNetModel)):
                        if self.train_config.do_cfg:
                            raise ValueError("ControlNetModel is not supported with CFG")
                        with torch.set_grad_enabled(self.adapter is not None):
                            adapter: ControlNetModel = self.assistant_adapter if self.assistant_adapter is not None else self.adapter
                            adapter_multiplier = get_adapter_multiplier()
                            with self.timer('encode_adapter'):
                                # add_text_embeds is pooled_prompt_embeds for sdxl
                                added_cond_kwargs = {}
                                if self.sd.is_xl:
                                    added_cond_kwargs["text_embeds"] = conditional_embeds.pooled_embeds
                                    added_cond_kwargs['time_ids'] = self.sd.get_time_ids_from_latents(noisy_latents)
                                down_block_res_samples, mid_block_res_sample = adapter(
                                    noisy_latents,
                                    timesteps,
                                    encoder_hidden_states=conditional_embeds.text_embeds,
                                    controlnet_cond=adapter_images,
                                    conditioning_scale=1.0,
                                    guess_mode=False,
                                    added_cond_kwargs=added_cond_kwargs,
                                    return_dict=False,
                                )
                                pred_kwargs['down_block_additional_residuals'] = down_block_res_samples
                                pred_kwargs['mid_block_additional_residual'] = mid_block_res_sample

                if self.train_config.do_guidance_loss and isinstance(self.train_config.guidance_loss_target, list):
                    batch_size = noisy_latents.shape[0]
                    # update the guidance value, random float between guidance_loss_target[0] and guidance_loss_target[1]
                    self._guidance_loss_target_batch = [
                        random.uniform(
                            self.train_config.guidance_loss_target[0],
                            self.train_config.guidance_loss_target[1]
                        ) for _ in range(batch_size)
                    ]

                external_query = self.before_unet_predict(batch)

                if unconditional_embeds is not None:
                    unconditional_embeds = unconditional_embeds.to(self.device_torch, dtype=dtype).detach()
                with self.timer('condition_noisy_latents'):
                    # do it for the model
                    noisy_latents = self.sd.condition_noisy_latents(noisy_latents, batch)

                if self.train_config.timestep_type == 'next_sample':
                    with self.timer('next_sample_step'):
                        with torch.no_grad():

                            stepped_timestep_indicies = [self.sd.noise_scheduler.index_for_timestep(t) + 1 for t in timesteps]
                            stepped_timesteps = [self.sd.noise_scheduler.timesteps[x] for x in stepped_timestep_indicies]
                            stepped_timesteps = torch.stack(stepped_timesteps, dim=0)

                            # do a sample at the current timestep and step it, then determine new noise
                            next_sample_pred = self.predict_noise(
                                noisy_latents=noisy_latents.to(self.device_torch, dtype=dtype),
                                timesteps=timesteps,
                                conditional_embeds=conditional_embeds.to(self.device_torch, dtype=dtype),
                                unconditional_embeds=unconditional_embeds,
                                batch=batch,
                                **pred_kwargs
                            )
                            stepped_latents = self.sd.step_scheduler(
                                next_sample_pred,
                                noisy_latents,
                                timesteps,
                                self.sd.noise_scheduler
                            )
                            # stepped latents is our new noisy latents. Now we need to determine noise in the current sample
                            noisy_latents = stepped_latents
                            original_samples = batch.latents.to(self.device_torch, dtype=dtype)
                            # todo calc next timestep, for now this may work as it
                            t_01 = (stepped_timesteps / 1000).to(original_samples.device)
                            if len(stepped_latents.shape) == 4:
                                t_01 = t_01.view(-1, 1, 1, 1)
                            elif len(stepped_latents.shape) == 5:
                                t_01 = t_01.view(-1, 1, 1, 1, 1)
                            else:
                                raise ValueError("Unknown stepped latents shape", stepped_latents.shape)
                            next_sample_noise = (stepped_latents - (1.0 - t_01) * original_samples) / t_01
                            noise = next_sample_noise
                            timesteps = stepped_timesteps
                # do a prior pred if we have an unconditional image, we will swap out the giadance later
                if batch.unconditional_latents is not None or self.do_guided_loss:
                    # do guided loss
                    loss = self.get_guided_loss(
                        noisy_latents=noisy_latents,
                        conditional_embeds=conditional_embeds,
                        match_adapter_assist=match_adapter_assist,
                        network_weight_list=network_weight_list,
                        timesteps=timesteps,
                        pred_kwargs=pred_kwargs,
                        batch=batch,
                        noise=noise,
                        unconditional_embeds=unconditional_embeds,
                        mask_multiplier=mask_multiplier,
                        prior_pred=prior_pred,
                    )
                else:
                    with self.timer('predict_unet'):
                        if self.external_query:
                            pred_kwargs['external_query'] = external_query
                        noise_pred = self.predict_noise(
                            noisy_latents=noisy_latents.to(self.device_torch, dtype=dtype),
                            timesteps=timesteps,
                            conditional_embeds=conditional_embeds.to(self.device_torch, dtype=dtype),
                            unconditional_embeds=unconditional_embeds,
                            batch=batch,
                            is_primary_pred=True,
                            **pred_kwargs
                        )
                    self.after_unet_predict()

                    with self.timer('calculate_loss'):
                        noise = noise.to(self.device_torch, dtype=dtype).detach()
                        prior_to_calculate_loss = prior_pred
                        # if we are doing diff_output_preservation and not noing inverted masked prior
                        # then we need to send none here so it will not target the prior
                        if self.train_config.diff_output_preservation and not do_inverted_masked_prior:
                            prior_to_calculate_loss = None

                        loss = self.calculate_loss(
                            noise_pred=noise_pred,
                            noise=noise,
                            noisy_latents=noisy_latents,
                            timesteps=timesteps,
                            batch=batch,
                            mask_multiplier=mask_multiplier,
                            prior_pred=prior_to_calculate_loss,
                        )

                    if self.train_config.diff_output_preservation:
                        # send the loss backwards otherwise checkpointing will fail
                        self.accelerator.backward(loss)
                        normal_loss = loss.detach() # dont send backward again

                        dop_embeds = self.diff_output_preservation_embeds.expand_to_batch(noisy_latents.shape[0])
                        dop_pred = self.predict_noise(
                            noisy_latents=noisy_latents.to(self.device_torch, dtype=dtype),
                            timesteps=timesteps,
                            conditional_embeds=dop_embeds.to(self.device_torch, dtype=dtype),
                            unconditional_embeds=unconditional_embeds,
                            batch=batch,
                            **pred_kwargs
                        )
                        dop_loss = torch.nn.functional.mse_loss(dop_pred, prior_pred) * self.train_config.diff_output_preservation_multiplier
                        self.accelerator.backward(dop_loss)

                        loss = normal_loss + dop_loss
                        loss = loss.clone().detach()
                        # require grad again so the backward wont fail
                        loss.requires_grad_(True)

                # check if nan
                if torch.isnan(loss):
                    print_acc("loss is nan")
                    loss = torch.zeros_like(loss).requires_grad_(True)

                with self.timer('backward'):
                    # todo we have multiplier seperated. works for now as res are not in same batch, but need to change
                    loss = loss * loss_multiplier.mean()
                    # IMPORTANT if gradient checkpointing do not leave with network when doing backward
                    # it will destroy the gradients. This is because the network is a context manager
                    # and will change the multipliers back to 0.0 when exiting. They will be
                    # 0.0 for the backward pass and the gradients will be 0.0
                    # I spent weeks on fighting this. DON'T DO IT
                    # with fsdp_overlap_step_with_backward():
                    # if self.is_bfloat:
                    # loss.backward()
                    # else:
                    self.accelerator.backward(loss)

        return loss.detach()
        # flush()

    def hook_train_loop(self, batch: Union[DataLoaderBatchDTO, List[DataLoaderBatchDTO]]):
        if isinstance(batch, list):
            batch_list = batch
        else:
            batch_list = [batch]
        total_loss = None
        self.optimizer.zero_grad()
        for batch in batch_list:
            loss = self.train_single_accumulation(batch)
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
            if len(batch_list) > 1 and self.model_config.low_vram:
                torch.cuda.empty_cache()


        if not self.is_grad_accumulation_step:
            # fix this for multi params
            if self.train_config.optimizer != 'adafactor':
                if isinstance(self.params[0], dict):
                    for i in range(len(self.params)):
                        self.accelerator.clip_grad_norm_(self.params[i]['params'], self.train_config.max_grad_norm)
                else:
                    self.accelerator.clip_grad_norm_(self.params, self.train_config.max_grad_norm)
            # only step if we are not accumulating
            with self.timer('optimizer_step'):
                self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
        else:
            # gradient accumulation. Just a place for breakpoint
            pass

        # TODO Should we only step scheduler on grad step? If so, need to recalculate last step
        with self.timer('scheduler_step'):
            self.lr_scheduler.step()

        if self.embedding is not None:
            with self.timer('restore_embeddings'):
                # Let's make sure we don't update any embedding weights besides the newly added token
                self.embedding.restore_embeddings()

        loss_dict = OrderedDict(
            {'loss': loss.item()}
        )

        self.end_of_training_loop()

        return loss_dict

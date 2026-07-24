#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fine-tuning script for Stable Video Diffusion with support for LoRA."""
import argparse
from datetime import datetime
import logging
import math
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import cv2
import shutil
from pathlib import Path
from urllib.parse import urlparse

import accelerate
import equilib
import numpy as np
import PIL
from PIL import Image
import torch
from torch.utils.data import RandomSampler
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange

import diffusers
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _resize_with_antialiasing,
)
from diffusers import AutoencoderKLTemporalDecoder
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from conver_equi_cube import equirectangular_to_cubemap
from cube_temporal_alignment import apply_custom_cube_temporal_fusion_w_cube_fea_for_kv_processors_for_unet
from pvdepth.dataset import H5PanoramicVideoDataset
from decord import VideoReader, cpu
from pvdepth.unet_pvdepth import DiffusersUNetSpatioTemporalConditionModelPVDepth

from pano_utils import get_spherical_gaussian_noise

# See https://github.com/huggingface/accelerate/issues/3481
import contextlib
from accelerate import Accelerator
from accelerate.utils import DistributedType

# =================================================================
# Monkey patch: Work around accelerate.accumulate errors with DeepSpeed ZeRO 2/3
# =================================================================
original_no_sync = Accelerator.no_sync

def patched_no_sync(self, model):
    # Skip no_sync when using DeepSpeed ZeRO Stage 2 or 3.
    if self.distributed_type == DistributedType.DEEPSPEED and self.state.deepspeed_plugin.zero_stage >= 2:
        return contextlib.nullcontext()
    
    # Preserve the original behavior for other configurations, such as DDP and ZeRO Stage 1.
    return original_no_sync(self, model)

Accelerator.no_sync = patched_no_sync


logger = get_logger(__name__, log_level="INFO")

# copy from https://github.com/crowsonkb/k-diffusion.git
def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()

def tensor_to_vae_latent(t, vae, chunk_size=8):
    # t: (B, F, C, H, W)
    batch_size, num_frames = t.shape[:2]
    
    t = rearrange(t, "b f c h w -> (b f) c h w")

    latents_chunks = []

    for i in range(0, t.shape[0], chunk_size):
        chunk = t[i : i + chunk_size]
        latents_chunk = vae.encode(chunk).latent_dist.sample() # vae.encode(t).latent_dist.sample()
        latents_chunks.append(latents_chunk)
    
    latents = torch.cat(latents_chunks, dim=0)
    
    latents = rearrange(latents, "(b f) c h w -> b f c h w", b=batch_size, f=num_frames)
    
    latents = latents * vae.config.scaling_factor

    return latents

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Video Diffusion."
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="stabilityai/stable-video-diffusion-img2vid-xt",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--unet_path", 
                        type=str, 
                        default="tencent/DepthCrafter", 
                        help="Path to pretrained model from depthcrafter")
    parser.add_argument(
        "--num_frames",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sft_pano_depthcrafter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=24,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=3,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )

    parser.add_argument(
        "--pretrain_unet",
        type=str,
        default=None,
        help="use weight for unet block",
    )
    
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")

    parser.add_argument("--data_root", type=str,
                    default=None,
                    help="Root directory of the dataset containing JSON files and data folders.")
    
    parser.add_argument("--h5_data_root", type=str,
                    default=None,
                    help="Root directory of the dataset containing h5 format data.")
    
    parser.add_argument("--original_fps", type=int,
                    default=20,
                    help="Frame rate of the data.")
    
    parser.add_argument("--noise_type", type=str,
                    default="normal",
                    choices=["normal", "distortion_noise", "distortion_noise_w_weighting", "distortion_noise_w_weighting_normal_0_1",
                             "distortion_noise_annealed_weighting"],
                    help="noise type"
                )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    # Use DeepSpeed optimizer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.adam_weight_decay,
    )

    return optimizer

def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    original_output_dir = args.output_dir

    from accelerate.utils import DistributedDataParallelKwargs
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,  # Logging backend, e.g. "tensorboard" or "wandb".
        # project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )

    if args.resume_from_checkpoint:
        # Resume in the run directory that contains the selected checkpoint.
        # For example, "./output/run/checkpoint-500" resolves to "./output/run".
        args.output_dir = Path(args.resume_from_checkpoint).parent.as_posix()
        if accelerator.is_main_process:
            print(f"--- Resuming training in existing directory: {args.output_dir} ---")
    else:
        import torch.distributed as dist
        if accelerator.is_main_process:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            objects_to_broadcast = [timestamp]
        else:
            objects_to_broadcast = [None]
        dist.broadcast_object_list(objects_to_broadcast, src=0)
        timestamp = objects_to_broadcast[0]
        args.output_dir = os.path.join(original_output_dir, timestamp)
        if accelerator.is_main_process:
            print(f"--- Starting new run, saving to: {args.output_dir} ---")
    accelerator.wait_for_everyone()  # Ensure every process has resolved output_dir before continuing.
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(logging_dir, exist_ok=True)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator.project_configuration = accelerator_project_config

    generator = torch.Generator(
        device=accelerator.device).manual_seed(args.seed)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load img encoder, tokenizer and models.
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.base_model_path, subfolder="feature_extractor",
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.base_model_path, subfolder="image_encoder",
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.base_model_path, subfolder="vae")
    # A base DepthCrafter/PSNI checkpoint legitimately has no cube weights.
    # Disable low-memory loading so those missing layers retain their zero initialization.
    unet, loading_info = DiffusersUNetSpatioTemporalConditionModelPVDepth.from_pretrained(
        args.unet_path,
        low_cpu_mem_usage=False,
        output_loading_info=True,
    )

    missing_keys = loading_info["missing_keys"]
    missing_cube_keys = [key for key in missing_keys if "cube_zero_" in key]
    missing_non_cube_keys = [key for key in missing_keys if "cube_zero_" not in key]
    load_errors = {
        key: loading_info[key]
        for key in ("unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info[key]
    }
    cube_keys = [key for key in unet.state_dict() if "cube_zero_" in key]

    if (
        len(cube_keys) != 76
        or missing_non_cube_keys
        or len(missing_cube_keys) not in (0, 76)
        or load_errors
    ):
        raise RuntimeError(
            "UNet checkpoint is incompatible with PVDepth training: "
            f"model_cube_keys={len(cube_keys)}, "
            f"missing_cube_keys={len(missing_cube_keys)}, "
            f"missing_non_cube_keys={missing_non_cube_keys[:5]}, "
            f"load_errors={load_errors}"
        )

    if missing_cube_keys:
        named_parameters = dict(unet.named_parameters())
        with torch.no_grad():
            for key in missing_cube_keys:
                named_parameters[key].zero_()
        logger.info("Initialized all 76 missing Cube projection parameters to zero.")

    apply_custom_cube_temporal_fusion_w_cube_fea_for_kv_processors_for_unet(unet)

    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    # Move image_encoder and vae to gpu and cast to weight_dtype
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    unet.requires_grad_(True)

    logger.info("--- Freezing temporal layers ---")

    frozen_params_count = 0
    trainable_params_count = 0
    cube_zero_params_count = 0

    for name, param in unet.named_parameters():
        if "cube_zero" in name:
            param.requires_grad = True
            cube_zero_params_count += param.numel()
            trainable_params_count += param.numel()
        else:
            param.requires_grad = False
            frozen_params_count += param.numel()

    # import pdb; pdb.set_trace()
    logger.info(f"-> Cube Adapter params (Active): {cube_zero_params_count}")
    logger.info(f"freeze {frozen_params_count} params.")
    logger.info(f"fintune {trainable_params_count} parmas.")

    from diffusers.training_utils import EMAModel
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=DiffusersUNetSpatioTemporalConditionModelPVDepth, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    # # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                if hasattr(model, 'module'):
                    model = model.module
                model.save_pretrained(os.path.join(output_dir, "unet"))
                logger.info(f"Saved full UNet weights to {output_dir}")
                # model.save_pretrained(output_dir)
                # logger.info(f"Saved LoRA weights to {output_dir}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(
                    input_dir, "unet_ema"), DiffusersUNetSpatioTemporalConditionModelPVDepth)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                
                if hasattr(model, 'module'):
                    train_model = model.module
                else:
                    train_model = model

                unet_path = os.path.join(input_dir, "unet")

                # The custom class constructs Cube layers before loading the checkpoint.
                load_model = DiffusersUNetSpatioTemporalConditionModelPVDepth.from_pretrained(
                    unet_path,
                    low_cpu_mem_usage=True,
                )

                train_model.load_state_dict(load_model.state_dict(), strict=True)
                logger.info(f"Loaded full PVDepth UNet weights from {input_dir}")
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
        logger.info("Registered checkpoint hooks")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps *
            args.per_gpu_batch_size * accelerator.num_processes
        )

    # lora_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))
    # unet_params = list(unet.parameters())
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))

    from accelerate.utils import DummyOptim, DummyScheduler

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    # DataLoaders creation:
    args.global_batch_size = args.per_gpu_batch_size * accelerator.num_processes

    json_files_train = [
        'setting_dynamic.json'
    ]
    print(f"json_files_train: {json_files_train}")

    train_dataset = H5PanoramicVideoDataset(
        json_files=json_files_train, 
        json_root=args.data_root,  
        h5_root=args.h5_data_root,
        resolution=(args.width, args.height),
        clip_len=args.num_frames,
        original_fps=args.original_fps,
        target_fps_range=(1, 20)
    )
    
    sampler = RandomSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=args.per_gpu_batch_size, 
                                                sampler=sampler,
                                                num_workers=args.num_workers)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
        )
        
    # Prepare everything with our `accelerator`.
    unet, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        unet, optimizer, lr_scheduler, train_dataloader
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # attribute handling for models using DDP
    if isinstance(unet, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        unet = unet.module
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("PVDepth-stage2", config=vars(args))

    # Train!
    total_batch_size = args.per_gpu_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    def encode_video(
        video: torch.Tensor,
        chunk_size: int = 8,
    ) -> torch.Tensor:
        """
        :param video: [b, c, h, w] in range [-1, 1], the b may contain multiple videos or frames
        :param chunk_size: Number of frames to encode per batch, limiting peak GPU memory use.
        :return: image_embeddings in shape of [b, 1024]
        """
        video_224 = _resize_with_antialiasing(video.float(), (224, 224))
        video_224 = (video_224 + 1.0) / 2.0  # [-1, 1] -> [0, 1]

        embeddings = []
        for i in range(0, video_224.shape[0], chunk_size):
            tmp = feature_extractor(
                images=video_224[i : i + chunk_size],
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values.to(video.device, dtype=video.dtype)
            embeddings.append(image_encoder(tmp).image_embeds)  # [chunk_size, 1024]

        embeddings = torch.cat(embeddings, dim=0)  # [b, 1024]

        M = 6
        video_cube = equirectangular_to_cubemap(video, mode="bilinear")  # [b, M, c, h, w]
        video_cube = rearrange(video_cube, "b m c h w -> (b m) c h w")
        video_cube_224 = _resize_with_antialiasing(video_cube.float(), (224, 224))
        video_cube_224 = (video_cube_224 + 1.0) / 2.0  # [-1, 1] -> [0, 1]

        cube_embeddings = []
        for i in range(0, video_cube_224.shape[0], chunk_size):
            tmp = feature_extractor(
                images=video_cube_224[i : i + chunk_size],
                do_normalize=True,
                do_center_crop=False,
                do_resize=False,
                do_rescale=False,
                return_tensors="pt",
            ).pixel_values.to(video.device, dtype=video.dtype)
            cube_embeddings.append(image_encoder(tmp).image_embeds)
        cube_embeddings = torch.cat(cube_embeddings, dim=0)
        cube_embeddings = rearrange(cube_embeddings, "(b m) c -> b m c", m=M)

        return {
            "embeddings": embeddings,
            "cube_embeddings": cube_embeddings,
        }
    
    def _get_add_time_ids(
        fps,
        motion_bucket_id,
        noise_aug_strength,  # [B]
        dtype,
        batch_size,
    ):
        # Repeat fixed FPS and motion bucket values across the batch.
        fps_tensor = torch.full((batch_size, 1), fps, dtype=dtype, device=noise_aug_strength.device)
        motion_tensor = torch.full((batch_size, 1), motion_bucket_id, dtype=dtype, device=noise_aug_strength.device)
        noise_tensor = noise_aug_strength.unsqueeze(1).to(dtype)

        add_time_ids = torch.cat([fps_tensor, motion_tensor, noise_tensor], dim=1)  # [B, 3]
        return add_time_ids

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        # If a checkpoint path is provided, use its parent as output_dir.
        if os.path.basename(checkpoint_path).startswith("checkpoint-"):
            args.output_dir = os.path.dirname(checkpoint_path)
            path = os.path.basename(checkpoint_path)
        else:
            # If a run directory is provided, resume from its latest checkpoint.
            args.output_dir = checkpoint_path
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            if dirs:
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1]
            else:
                path = None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            checkpoint_full_path = os.path.join(args.output_dir, path)
            accelerator.print(f"Refsuming from checkpoint {checkpoint_full_path}")
            accelerator.load_state(checkpoint_full_path)
            global_step = int(path.split("-")[1])

            # resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            logger.info(f"Resuming from global_step {global_step} and epoch {first_epoch}.")
            # resume_step = resume_global_step % (
            #     num_update_steps_per_epoch * args.gradient_accumulation_steps)
    else:
        global_step = 0
        first_epoch = 0
        # resume_step = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        initial=global_step,
                        total=args.max_train_steps,
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # # Skip steps until we reach the resumed step
            # if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            #     if step % args.gradient_accumulation_steps == 0:
            #         progress_bar.update(1)
            #     continue

            with accelerator.accumulate(unet):
                # first, convert images to latent space.
                video_frames = batch["video"].to(weight_dtype).to(accelerator.device, non_blocking=True)
                depth_frames = batch["depth"].to(weight_dtype).to(accelerator.device, non_blocking=True)
                bsz = video_frames.shape[0]
                video_frames = video_frames * 2.0 - 1.0  # [0,1] -> [-1,1], in [t, c, h, w]
                depth_frames = depth_frames * 2.0 - 1.0  # [0,1] -> [-1,1]

                # video_latents = tensor_to_vae_latent(video_frames, vae)
                depth_latents = tensor_to_vae_latent(depth_frames, vae)

                # pixel_values = batch["pixel_values"].to(weight_dtype).to(
                #     accelerator.device, non_blocking=True
                # )
                conditional_pixel_values = video_frames

                # Sample noise that we'll add to the latents
                #  spherical gaussian noise
                if args.noise_type == "normal":
                    noise = torch.randn_like(depth_latents)
                elif args.noise_type == "distortion_noise":
                    noise = get_spherical_gaussian_noise(
                        shape=depth_latents.shape, 
                        device=depth_latents.device, 
                        dtype=depth_latents.dtype,
                        generator=generator
                    )
                elif args.noise_type == "distortion_noise_w_weighting":
                    noise = get_spherical_gaussian_noise(
                        shape=depth_latents.shape, 
                        device=depth_latents.device, 
                        dtype=depth_latents.dtype,
                        generator=generator
                    )
                    noise_1 = torch.randn_like(noise)
                    noise = (noise + noise_1) / math.sqrt(2)
                
                elif args.noise_type == "distortion_noise_w_weighting_normal_0_1":
                    noise = get_spherical_gaussian_noise(
                        shape=depth_latents.shape, 
                        device=depth_latents.device, 
                        dtype=depth_latents.dtype,
                        generator=generator
                    )
                    noise_1 = torch.randn_like(noise)

                    w_normal = 0.1
                    w_spherical = (1.0 - w_normal**2)**0.5  # Approximately 0.994987.
                    noise = (w_spherical * noise) + (w_normal * noise_1)

                elif args.noise_type == "distortion_noise_annealed_weighting":
                    ANNEAL_END_STEP = 5000.0
                    FINAL_NORMAL_WEIGHT = 0.1
                    progress = min(1.0, global_step / ANNEAL_END_STEP)
                    w_normal = 1.0 - progress * (1.0 - FINAL_NORMAL_WEIGHT)
                    w_spherical = math.sqrt(max(0.0, 1.0 - w_normal**2))
                    noise_spherical = get_spherical_gaussian_noise(
                            shape=depth_latents.shape, 
                            device=depth_latents.device, 
                            dtype=depth_latents.dtype,
                            generator=generator
                        )
                    noise_normal = torch.randn_like(depth_latents)
                   
                    noise = (w_spherical * noise_spherical) + (w_normal * noise_normal)
                    if global_step == 0 or global_step == 1000 or global_step == 5000:
                        logger.info(f"Step {global_step}: w_spherical={w_spherical:.4f}, w_normal={w_normal:.4f}")

                # Add noise to conditional pixels for robustness
                cond_sigmas = rand_log_normal(shape=[bsz,], loc=-3.0, scale=0.5).to(depth_latents)
                noise_aug_strength = cond_sigmas  # [B]

                cond_sigmas = cond_sigmas[:, None, None, None, None]
                conditional_pixel_values = \
                    torch.randn_like(conditional_pixel_values) * cond_sigmas + conditional_pixel_values
                # conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae)[:, 0, :, :, :]
                conditional_latents = tensor_to_vae_latent(conditional_pixel_values, vae) # [B, T, C, H, W]
                conditional_latents = conditional_latents / vae.config.scaling_factor

                sigmas_1d = rand_log_normal(shape=[bsz,], loc=0.7, scale=1.6).to(depth_latents.device)

                timesteps = 0.25 * sigmas_1d.log()

                sigmas = sigmas_1d[:, None, None, None, None]

                noisy_latents = depth_latents + noise * sigmas

                inp_noisy_latents = noisy_latents / ((sigmas**2 + 1) ** 0.5)

                # Get the video embeddings for conditioning (Cross-attn in U-Net).
                # encoder_hidden_states = encode_image(
                #     pixel_values[:, 0, :, :, :].float())
                video_frames_flat = rearrange(video_frames, "b f c h w -> (b f) c h w")
                video_embeddings = encode_video(video_frames_flat)
                if isinstance(video_embeddings, dict):
                    video_embeddings["embeddings"] = rearrange(video_embeddings["embeddings"], "(b f) c -> b f c", b=bsz)
                    video_embeddings["cube_embeddings"] = rearrange(video_embeddings["cube_embeddings"], "(b f) m c -> b f m c", b=bsz)
                else:
                    video_embeddings = rearrange(video_embeddings, "(b f) c -> b f c", b=bsz)
                # video_embeddings = rearrange(video_embeddings, "(b f) c-> b f c", b=bsz)

                # Here I input a fixed numerical value for 'motion_bucket_id', which is not reasonable.
                # However, I am unable to fully align with the calculation method of the motion score,
                # so I adopted this approach. The same applies to the 'fps' (frames per second).
                added_time_ids = _get_add_time_ids(
                    7, # fixed
                    127, # motion_bucket_id = 127, fixed
                    noise_aug_strength, # noise_aug_strength == cond_sigmas
                    depth_latents.dtype,
                    bsz,
                )
                added_time_ids = added_time_ids.to(depth_latents.device)

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(
                        bsz, device=depth_latents.device, generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    null_conditioning = torch.zeros_like(encoder_hidden_states)
                    encoder_hidden_states = torch.where(
                        prompt_mask, null_conditioning.unsqueeze(1), encoder_hidden_states.unsqueeze(1))
                    # Sample masks for the original images.
                    image_mask_dtype = conditional_latents.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(
                            image_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    conditional_latents = image_mask * conditional_latents

                # Concatenate the `conditional_latents` with the `noisy_latents`.
                inp_noisy_latents = torch.cat(
                    [inp_noisy_latents, conditional_latents], dim=2) #[B, T, 8, H, W]

                # check https://arxiv.org/abs/2206.00364(the EDM-framework) for more details.
                target = depth_latents
                model_pred = unet(
                    inp_noisy_latents, 
                    timesteps, 
                    encoder_hidden_states=video_embeddings, 
                    added_time_ids=added_time_ids).sample

                # Compute loss weights for EDM formulation
                c_out = -sigmas / ((sigmas**2 + 1)**0.5)
                c_skip = 1 / (sigmas**2 + 1)
                denoised_latents = model_pred * c_out + c_skip * noisy_latents
                weighing = (1 + sigmas ** 2) * (sigmas**-2.0)

                # MSE loss
                loss = torch.mean(
                    (weighing.float() * (denoised_latents.float() -
                     target.float()) ** 2).reshape(target.shape[0], -1),
                    dim=1,
                )
                loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(
                    loss.repeat(args.per_gpu_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                lr_scheduler.step()
                if accelerator.state.deepspeed_plugin is None:
                    optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    # save checkpoints!
                    if global_step % args.checkpointing_steps == 0:
                        accelerator.wait_for_everyone()
                        if args.checkpoints_total_limit is not None:
                            if accelerator.is_main_process:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # Remove the oldest checkpoints before saving to enforce the storage limit.
                                
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints pre-emptively to save space"
                                    )
                                    logger.info(
                                        f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                        try:
                                            shutil.rmtree(removing_checkpoint)
                                        except OSError as e:
                                            logger.warning(f"Error removing checkpoint {removing_checkpoint}: {e}")

                        accelerator.wait_for_everyone()

                        # Save the new checkpoint.
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        accelerator.wait_for_everyone()
                        logger.info(f"Saved state to {save_path}")

                        if global_step > 10000 and global_step % 20000 == 0 and accelerator.is_main_process:
                            if os.path.exists(save_path):
                                permanent_path = os.path.join(args.output_dir, f"milestone-{global_step}")
                                try:
                                    if os.path.exists(permanent_path):
                                        shutil.rmtree(permanent_path)
                                        logger.info(f"Removed existing {permanent_path} to replace it.")
                                        
                                    os.rename(save_path, permanent_path)
                                    logger.info(f"Renamed checkpoint {save_path} to {permanent_path} to prevent deletion.")
                                
                                except OSError as e:
                                    logger.warning(f"Could not rename {save_path} to {permanent_path}: {e}")
                            else:
                                logger.warning(f"Tried to rename {save_path}, but it does not exist.")


                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            accelerator.wait_for_everyone()
            
            
            logs = {"step_loss": loss.detach().item(
            ), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # Check if we've reached max_train_steps
            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)

        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(args.output_dir)
        logger.info(f"Saved final LoRA adapter to {args.output_dir}")

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
    accelerator.end_training()


if __name__ == "__main__":
    main()

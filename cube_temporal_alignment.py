from typing import Dict, Optional, Union

from einops import rearrange
import torch
import torch.nn as nn

# Temporal-layer classes used by the SVD UNet.
from diffusers.models.attention import TemporalBasicTransformerBlock
from diffusers.models.transformers.transformer_temporal import TransformerSpatioTemporalModel, TransformerTemporalModelOutput
from diffusers.models.resnet import SpatioTemporalResBlock

from conver_equi_cube import equirectangular_to_cubemap, safe_cube2equi_wrapper

# --- A. Zero-initialized projection modules ---
class ZeroLayer(nn.Module):
    """
    Zero-initialized convolution for ResNet tensors shaped [B, C, H, W].
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        # Initialize both parameters to zero.
        nn.init.constant_(self.conv.weight, 0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)

class ZeroLinear(nn.Module):
    """
    Zero-initialized linear layer for attention tensors shaped [BT, L, C].
    """
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        # Initialize both parameters to zero.
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        return self.linear(x)

# --- B. Cube-branch layer injection ---
def inject_cube_branch_layers(model):
    """
    Run this before ``load_state_dict`` in both training and inference.
    It extends the model structure with the cube-branch projection layers.
    """
    print("--> Injecting Zero-Conv layers for Cube Branch...")
    
    for name, module in model.named_modules():
        # 1. ResNet blocks (SpatioTemporalResBlock).
        if isinstance(module, SpatioTemporalResBlock):
            # Avoid injecting the same layer more than once.
            if hasattr(module, "cube_zero_conv"): continue
            
            dim = module.spatial_res_block.out_channels
            ref_param = next(module.parameters())
            # Register the injected layer in the PyTorch module tree.
            module.add_module("cube_zero_conv", ZeroLayer(dim, dim).to(ref_param.device, dtype=ref_param.dtype))

        # 2. Transformer blocks (TemporalBasicTransformerBlock).
        elif isinstance(module, TemporalBasicTransformerBlock):
            # Avoid injecting the same layer more than once.
            if hasattr(module, "cube_zero_linear"): continue
            
            dim = module.norm1.normalized_shape[0]
            ref_param = next(module.parameters())
            # Attention outputs are shaped [batch, sequence, channels].
            module.add_module("cube_zero_linear", ZeroLinear(dim).to(ref_param.device, dtype=ref_param.dtype))
            
    return model




'''
# cube_temporal_fusion
'''

def cube_aware_fusion_spatio_temporal_transformer_forward(self):

    def forward(
        hidden_states: torch.Tensor, #[bt, c, h, w]
        encoder_hidden_states: Optional[torch.Tensor] = None, #[bt, 1, c]
        image_only_indicator: Optional[torch.Tensor] = None, #[b, t]
        return_dict: bool = True,
    ):
        # 1. Input
        batch_frames, _, height, width = hidden_states.shape # b*t c h w
        num_frames = image_only_indicator.shape[-1]
        batch_size = batch_frames // num_frames

        M = 6
        face_w = width // 4

        time_context = encoder_hidden_states # [b*t, 1, c] 
        time_context_first_timestep = time_context[None, :].reshape(
            batch_size, num_frames, -1, time_context.shape[-1]
        )[:, 0] # [b*t, 1, c] -> [1, bt, 1, c] -> [b, t, 1, c] -> [b, 1, 1, c]

        time_context = time_context_first_timestep[:, None].broadcast_to(
            batch_size, height * width, time_context.shape[-2], time_context.shape[-1]
        ) # [b 1 1 1 c] -> [b h*w 1 c]
        time_context = time_context.reshape(batch_size * height * width, -1, time_context.shape[-1]) # [bhw 1 c]


        residual = hidden_states # [b*t c h w]

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim) # [b*t, hw, c]
        hidden_states = self.proj_in(hidden_states)

        num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
        num_frames_emb = num_frames_emb.repeat(batch_size, 1)
        num_frames_emb = num_frames_emb.reshape(-1)
        t_emb = self.time_proj(num_frames_emb)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)

        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        # 2. Blocks
        for block, temporal_block in zip(self.transformer_blocks, self.temporal_transformer_blocks):
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states, # [b*t, hw, c]
                    None,
                    encoder_hidden_states, # [b*t, 1, c] 
                    None,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                )

            hidden_states_mix = hidden_states # [b*t, hw, c]
            hidden_states_mix = hidden_states_mix + emb

            def run_cube_branch(x_in, t_context_small):
                # 1. Build the large context inside the checkpoint scope so it is released afterward.
                # Expand t_context_small from [B, 1, 1, C] to [B * M * FaceW * FaceW, 1, C].
                t_context_cube_inner = t_context_small[:, None].expand(
                    batch_size, M * face_w * face_w, -1, -1
                ).reshape(batch_size * M * face_w * face_w, -1, t_context_small.shape[-1])

                # 2. ERP -> cubemap (the main memory-intensive operation).
                x_cube = rearrange(x_in, 'bt (h w) c -> bt c h w', h=height, w=width)
                cubemap_faces = equirectangular_to_cubemap(x_cube, mode="bilinear")
                x_cube = rearrange(cubemap_faces, '(b t) m c f1 f2 -> (b m t) (f1 f2) c', b=batch_size, t=num_frames)

                # 3. Temporal Attention
                x_cube = temporal_block(
                    x_cube,
                    num_frames=num_frames,
                    encoder_hidden_states=t_context_cube_inner, # Use the context created above.
                )

                # 4. Cube -> Equi
                x_cube = rearrange(x_cube, '(b m t) (h w) c -> (b t) c h (m w)', b=batch_size, m=M, t=num_frames, h=face_w, w=face_w)
                x_cube = safe_cube2equi_wrapper(cubemap=x_cube, height=height, width=width, mode='bilinear')
                x_cube = rearrange(x_cube, 'bt c h w -> bt (h w) c')
                
                return x_cube

            # Run the cubemap temporal branch with gradient checkpointing.
            if self.training and self.gradient_checkpointing:
                hidden_states_mix_cube = torch.utils.checkpoint.checkpoint(
                    run_cube_branch, 
                    hidden_states_mix, 
                    time_context_first_timestep, # Pass the compact [B, 1, 1, C] context.
                    use_reentrant=False
                )
            else:
                hidden_states_mix_cube = run_cube_branch(hidden_states_mix, time_context_first_timestep) # [bt hw c]


            # Process the ERP features through the temporal layer.
            hidden_states_mix = temporal_block(
                hidden_states_mix, # [bt hw c]
                num_frames=num_frames,
                encoder_hidden_states=time_context,
            )

            # Fuse the cubemap and ERP features.
            # inject_cube_branch_layers should have attached cube_zero_linear.
            if hasattr(temporal_block, "cube_zero_linear"):
                # Learn how much cubemap information to add.
                # hidden_states_mix_cube shape: [BT, HW, C]
                gate = temporal_block.cube_zero_linear(hidden_states_mix_cube)
                hidden_states_mix.add_(gate)
            else:
                # Fall back to averaging if the projection layer was not injected.
                hidden_states_mix.add_(hidden_states_mix_cube).mul_(0.5)


            hidden_states = self.time_mixer(
                x_spatial=hidden_states,
                x_temporal=hidden_states_mix,
                image_only_indicator=image_only_indicator,
            )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)
    return forward

def cube_aware_fusion_spatio_temporal_resblock_forward(self):

    def forward(
        hidden_states: torch.Tensor, #  [b*t, c,,h, w]
        temb: Optional[torch.Tensor] = None, # [b*t, c]
        image_only_indicator: Optional[torch.Tensor] = None, #  [b, t]
    ):
        M=6
        num_frames = image_only_indicator.shape[-1]

        # 1. Spatial Block
        # hidden_states: [b*t, c, h, w]
        hidden_states = self.spatial_res_block(hidden_states, temb) #  [b*t, c,,h, w]

        batch_frames, channels, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        # Prepare the 5D mixer input once: [b, c, t, h, w].
        hidden_states_mix = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        ) #  [b*t, c,,h, w] -> [b t c h w] -> [b c t h w]
        hidden_states = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        ) #  [b*t, c,,h, w] -> [b t c h w] -> [b c t h w]


        # === A. Cube Branch ===
        # [b c t h w] ->  [b*m c t h w]
        hidden_states_input = rearrange(hidden_states, 'b c t h w -> (b t) c h w')

        cubemap_faces = equirectangular_to_cubemap(hidden_states_input, mode="bilinear")

        hidden_states_cube = rearrange(
            cubemap_faces, 
            '(b t) m c h w -> (b m) c t h w', 
            b=batch_size, 
            t=num_frames
        )

        if temb is not None:
            temb_cube = temb.reshape(batch_size, num_frames, -1) #[bt, c] -> [b t c]
            temb_cube = temb_cube.repeat_interleave(M, dim=0)  # [b*m, t, c]

        # Cube Temporal Layer
        hidden_states_cube = self.temporal_res_block(hidden_states_cube, temb_cube) # [b*m c t h w]

        # Project the cubemap hidden states back to ERP features.
        # [b*m c t h w] -> [m*b*t c h w] -> [b*t c h w] -> [b c t h w]
        hidden_states_cube = rearrange(hidden_states_cube, 
                                  '(b m) c t h w -> (b t) c h (m w)', m=M
                                  )# [b*t c face_w face_w*6]
        

        hidden_states_cube = safe_cube2equi_wrapper(
            cubemap=hidden_states_cube,
            height=height,
            width=width,
            mode="bilinear"
        )

        # === B. Equi Branch ===
        if temb is not None:
            temb = temb.reshape(batch_size, num_frames, -1)
        hidden_states_equi_out = self.temporal_res_block(hidden_states, temb) # Output: [b, c, t, h, w]

        # === C. Fusion
        if hasattr(self, "cube_zero_conv"):
            gate = self.cube_zero_conv(hidden_states_cube)  # [b*t, c, h, w]

            # Restore the gate to [b, c, t, h, w] before addition.
            gate = rearrange(gate, '(b t) c h w -> b c t h w', t=num_frames)

            # Residual fusion: ERP features + gated cubemap features.
            hidden_states_equi_out.add_(gate)
            hidden_states_temporal = hidden_states_equi_out
        else:
            # Fallback
            hidden_states_cube_5d = rearrange(hidden_states_cube, '(b t) c h w -> b c t h w', t=num_frames)
            hidden_states_equi_out.add_(hidden_states_cube_5d).mul_(0.5)
            hidden_states_temporal = hidden_states_equi_out


        # === D. Mixer ===
        hidden_states = self.time_mixer(
            x_spatial=hidden_states_mix,
            x_temporal=hidden_states_temporal,
            image_only_indicator=image_only_indicator,
        )

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)
        return hidden_states
    
    return forward


def apply_custom_cube_temporal_fusion_processors_for_unet(
    model,
):
    """
    Apply forward functions with zero-initialized Conv/Linear fusion.
    Call ``inject_cube_branch_layers(model)`` first to create the required
    projection parameters.
    """
    print("--> Applying Cube-Aware Fusion Processors (with Zero Layers)...")
    for name, module in model.named_modules():
        # print(f"name: {name},,,, module: {module}")
        # 
        if isinstance(module, TransformerSpatioTemporalModel):
            if not hasattr(module, 'original_forward'):
                module.original_forward = module.forward
            module.forward = cube_aware_fusion_spatio_temporal_transformer_forward(module)

            module.forward_w_precube = module.forward
        if isinstance(module, SpatioTemporalResBlock):
            if not hasattr(module, 'original_forward'):
                module.original_forward = module.forward
            module.forward = cube_aware_fusion_spatio_temporal_resblock_forward(module)
            module.forward_w_precube = module.forward


# #=========================================================================================================================================================== ###
'''
# Delayed cube-temporal fusion for joint training with the noise strategy.
'''

def cube_aware_fusion_w_delay_spatio_temporal_transformer_forward(self):

    def forward(
        hidden_states: torch.Tensor, #[bt, c, h, w]
        encoder_hidden_states: Optional[torch.Tensor] = None, #[bt, 1, c]
        image_only_indicator: Optional[torch.Tensor] = None, #[b, t]
        return_dict: bool = True,
    ):
        # Read the cube-fusion feature flag.
        # It remains disabled until the caller explicitly enables it.
        ENABLE_CUBE = getattr(self, "enable_cube_fusion", False)

        # 1. Input
        batch_frames, _, height, width = hidden_states.shape # b*t c h w
        num_frames = image_only_indicator.shape[-1]
        batch_size = batch_frames // num_frames
        
        M = 6
        face_w = width // 4

        time_context = encoder_hidden_states # [b*t, 1, c] 
        time_context_first_timestep = time_context[None, :].reshape(
            batch_size, num_frames, -1, time_context.shape[-1]
        )[:, 0] # [b*t, 1, c] -> [1, bt, 1, c] -> [b, t, 1, c] -> [b, 1, 1, c]

        time_context = time_context_first_timestep[:, None].broadcast_to(
            batch_size, height * width, time_context.shape[-2], time_context.shape[-1]
        ) # [b 1 1 1 c] -> [b h*w 1 c]
        time_context = time_context.reshape(batch_size * height * width, -1, time_context.shape[-1]) # [bhw 1 c]


        residual = hidden_states # [b*t c h w]

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim) # [b*t, hw, c]
        hidden_states = self.proj_in(hidden_states)

        num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
        num_frames_emb = num_frames_emb.repeat(batch_size, 1)
        num_frames_emb = num_frames_emb.reshape(-1)
        t_emb = self.time_proj(num_frames_emb)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)

        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        # 2. Blocks
        for block, temporal_block in zip(self.transformer_blocks, self.temporal_transformer_blocks):
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states, # [b*t, hw, c]
                    None,
                    encoder_hidden_states, # [b*t, 1, c] 
                    None,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                )

            hidden_states_mix = hidden_states # [b*t, hw, c]
            hidden_states_mix = hidden_states_mix + emb

            # Compute the cubemap branch only when cube fusion is enabled.
            hidden_states_mix_cube = None

            if ENABLE_CUBE:
                def run_cube_branch(x_in, t_context_small):
                    # 1. Build the large context inside the checkpoint scope so it is released afterward.
                    # Expand t_context_small from [B, 1, 1, C] to [B * M * FaceW * FaceW, 1, C].
                    t_context_cube_inner = t_context_small[:, None].expand(
                        batch_size, M * face_w * face_w, -1, -1
                    ).reshape(batch_size * M * face_w * face_w, -1, t_context_small.shape[-1])

                    # 2. ERP -> cubemap (the main memory-intensive operation).
                    x_cube = rearrange(x_in, 'bt (h w) c -> bt c h w', h=height, w=width)
                    cubemap_faces = equirectangular_to_cubemap(x_cube, mode="bilinear")
                    x_cube = rearrange(cubemap_faces, '(b t) m c f1 f2 -> (b m t) (f1 f2) c', b=batch_size, t=num_frames)

                    # 3. Temporal Attention
                    x_cube = temporal_block(
                        x_cube,
                        num_frames=num_frames,
                        encoder_hidden_states=t_context_cube_inner, # Use the context created above.
                    )

                    # 4. Cube -> Equi
                    x_cube = rearrange(x_cube, '(b m t) (h w) c -> (b t) c h (m w)', b=batch_size, m=M, t=num_frames, h=face_w, w=face_w)
                    x_cube = safe_cube2equi_wrapper(cubemap=x_cube, height=height, width=width, mode='bilinear')
                    x_cube = rearrange(x_cube, 'bt c h w -> bt (h w) c')

                    return x_cube
                
                # Run the cubemap temporal branch with gradient checkpointing.
                if self.training and self.gradient_checkpointing:
                    hidden_states_mix_cube = torch.utils.checkpoint.checkpoint(
                        run_cube_branch,
                        hidden_states_mix,
                        time_context_first_timestep, # Pass the compact [B, 1, 1, C] context.
                        use_reentrant=False
                    )
                else:
                    hidden_states_mix_cube = run_cube_branch(hidden_states_mix, time_context_first_timestep) # [bt hw c]
            

            # Process the ERP features through the temporal layer.
            hidden_states_mix = temporal_block(
                hidden_states_mix, # [bt hw c]
                num_frames=num_frames,
                encoder_hidden_states=time_context,
            )

            # Fuse the cubemap branch only when cube fusion is enabled.
            if ENABLE_CUBE and hidden_states_mix_cube is not None:
                # Fuse the cubemap and ERP features.
                # inject_cube_branch_layers should have attached cube_zero_linear.
                if hasattr(temporal_block, "cube_zero_linear"):
                    # Learn how much cubemap information to add.
                    # hidden_states_mix_cube shape: [BT, HW, C]
                    gate = temporal_block.cube_zero_linear(hidden_states_mix_cube)
                    hidden_states_mix.add_(gate)
                else:
                    # Fall back to averaging if the projection layer was not injected.
                    hidden_states_mix.add_(hidden_states_mix_cube).mul_(0.5)


            hidden_states = self.time_mixer(
                x_spatial=hidden_states,
                x_temporal=hidden_states_mix,
                image_only_indicator=image_only_indicator,
            )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)
    return forward

def cube_aware_fusion_w_delay_spatio_temporal_resblock_forward(self):

    def forward(
        hidden_states: torch.Tensor, #  [b*t, c,,h, w]
        temb: Optional[torch.Tensor] = None, # [b*t, c]
        image_only_indicator: Optional[torch.Tensor] = None, #  [b, t]
    ):
        # Read the cube-fusion feature flag.
        ENABLE_CUBE = getattr(self, "enable_cube_fusion", False)

        M=6
        num_frames = image_only_indicator.shape[-1]

        # 1. Spatial Block
        # hidden_states: [b*t, c, h, w]
        hidden_states = self.spatial_res_block(hidden_states, temb) #  [b*t, c,,h, w]

        batch_frames, channels, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        # Prepare the 5D mixer input once: [b, c, t, h, w].
        hidden_states_mix = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        ) #  [b*t, c,,h, w] -> [b t c h w] -> [b c t h w]
        hidden_states = (
            hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
        ) #  [b*t, c,,h, w] -> [b t c h w] -> [b c t h w]


        # Reserve storage for the cubemap branch output.
        hidden_states_cube = None
        # Compute the cubemap branch only when enabled.
        if ENABLE_CUBE:
            # === A. Cube Branch ===
            # [b c t h w] ->  [b*m c t h w]
            hidden_states_input = rearrange(hidden_states, 'b c t h w -> (b t) c h w')

            cubemap_faces = equirectangular_to_cubemap(hidden_states_input, mode="bilinear")

            hidden_states_cube = rearrange(
                cubemap_faces,
                '(b t) m c h w -> (b m) c t h w',
                b=batch_size,
                t=num_frames
            )

            if temb is not None:
                temb_cube = temb.reshape(batch_size, num_frames, -1) #[bt, c] -> [b t c]
                temb_cube = temb_cube.repeat_interleave(M, dim=0)  # [b*m, t, c]

            # Cube Temporal Layer
            hidden_states_cube = self.temporal_res_block(hidden_states_cube, temb_cube) # [b*m c t h w]

            # Project the cubemap hidden states back to ERP features.
            # [b*m c t h w] -> [m*b*t c h w] -> [b*t c h w] -> [b c t h w]
            hidden_states_cube = rearrange(hidden_states_cube,
                                    '(b m) c t h w -> (b t) c h (m w)', m=M
                                    )# [b*t c face_w face_w*6]


            hidden_states_cube = safe_cube2equi_wrapper(
                cubemap=hidden_states_cube,
                height=height,
                width=width,
                mode="bilinear"
            )

        # === B. Equi Branch ===
        if temb is not None:
            temb = temb.reshape(batch_size, num_frames, -1)
        hidden_states_equi_out = self.temporal_res_block(hidden_states, temb) # Output: [b, c, t, h, w]

        hidden_states_temporal = hidden_states_equi_out
        # Apply fusion only when the cubemap branch is enabled.
        if ENABLE_CUBE and hidden_states_cube is not None:
            # === C. Fusion
            if hasattr(self, "cube_zero_conv"):
                gate = self.cube_zero_conv(hidden_states_cube)  # [b*t, c, h, w]

                # Restore the gate to [b, c, t, h, w] before addition.
                gate = rearrange(gate, '(b t) c h w -> b c t h w', t=num_frames)

                # Residual fusion: ERP features + gated cubemap features.
                hidden_states_equi_out.add_(gate)
                hidden_states_temporal = hidden_states_equi_out
            else:
                # Fallback
                hidden_states_cube_5d = rearrange(hidden_states_cube, '(b t) c h w -> b c t h w', t=num_frames)
                hidden_states_equi_out.add_(hidden_states_cube_5d).mul_(0.5)
                hidden_states_temporal = hidden_states_equi_out


        # === D. Mixer ===
        hidden_states = self.time_mixer(
            x_spatial=hidden_states_mix,
            x_temporal=hidden_states_temporal,
            image_only_indicator=image_only_indicator,
        )

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)
        return hidden_states
    
    return forward


def apply_custom_cube_temporal_fusion_w_delay_processors_for_unet(
    model,
):
    """
    Apply forward functions with zero-initialized Conv/Linear fusion.
    Call ``inject_cube_branch_layers(model)`` first to create the required
    projection parameters.
    """
    print("--> Applying Cube-Aware Fusion Processors (with Zero Layers)...")
    for name, module in model.named_modules():
        # print(f"name: {name},,,, module: {module}")
        # 
        if isinstance(module, TransformerSpatioTemporalModel):
            if not hasattr(module, 'original_forward'):
                module.original_forward = module.forward
            module.forward = cube_aware_fusion_w_delay_spatio_temporal_transformer_forward(module)

            module.forward_w_precube = module.forward
        if isinstance(module, SpatioTemporalResBlock):
            if not hasattr(module, 'original_forward'):
                module.original_forward = module.forward
            module.forward = cube_aware_fusion_w_delay_spatio_temporal_resblock_forward(module)
            module.forward_w_precube = module.forward


# #=========================================================================================================================================================== ###
'''
# cube_temporal_fusion with cube_fea as kv of cross-attn of cube temporal layers
'''

def cube_aware_fusion_w_cube_fea_for_kv_spatio_temporal_transformer_forward(self):

    def forward(
        hidden_states: torch.Tensor, #[bt, c, h, w]
        encoder_hidden_states: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None, # dict
        image_only_indicator: Optional[torch.Tensor] = None, #[b, t]
        return_dict: bool = True,
    ):
        # 1. Input
        batch_frames, _, height, width = hidden_states.shape # b*t c h w
        num_frames = image_only_indicator.shape[-1]
        batch_size = batch_frames // num_frames
        
        M = 6
        face_w = width // 4

        # equi fea for the kv of cross-attn
        time_context = encoder_hidden_states["embeddings"] # [b*t, 1, c]
        time_context_first_timestep = time_context[None, :].reshape(
            batch_size, num_frames, -1, time_context.shape[-1]
        )[:, 0] # [b*t, 1, c] -> [1, bt, 1, c] -> [b, t, 1, c] -> [b, 1, c]

        time_context = time_context_first_timestep[:, None].broadcast_to(
            batch_size, height * width, time_context.shape[-2], time_context.shape[-1]
        ) # [b 1 1 c] -> [b h*w 1 c]
        time_context = time_context.reshape(batch_size * height * width, -1, time_context.shape[-1]) # [bhw 1 c]

        # cube fea for the kv of cross-attn
        # import pdb; pdb.set_trace()
        cube_time_context = encoder_hidden_states["cube_embeddings"] # [b*t, m, c]
        cube_time_context_first_timestep = cube_time_context[None, :].reshape(
            batch_size, num_frames, -1, time_context.shape[-1]
        )[:, 0].unsqueeze(1) # [b*t, m, c] -> [1, bt, m, c] -> [b, t, m, c] -> [b, 1, m, c]

        residual = hidden_states # [b*t c h w]

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim) # [b*t, hw, c]
        hidden_states = self.proj_in(hidden_states)

        num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
        num_frames_emb = num_frames_emb.repeat(batch_size, 1)
        num_frames_emb = num_frames_emb.reshape(-1)
        t_emb = self.time_proj(num_frames_emb)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)

        emb = self.time_pos_embed(t_emb)
        emb = emb[:, None, :]

        # 2. Blocks
        for block, temporal_block in zip(self.transformer_blocks, self.temporal_transformer_blocks):
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states, # [b*t, hw, c]
                    None,
                    encoder_hidden_states["embeddings"], # [b*t, 1, c]
                    None,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states["embeddings"],
                )

            hidden_states_mix = hidden_states # [b*t, hw, c]
            hidden_states_mix = hidden_states_mix + emb

            def run_cube_branch(x_in, t_context_small):
                # 1. Build the large context inside the checkpoint scope so it is released afterward.
                # Expand t_context_small from [B, 1, M, C] to [B * M * FaceW * FaceW, 1, C].
                t_context_cube_inner = rearrange(t_context_small, 'B T M C -> B M T C') #[B 1 M C]->[B M 1 C]
                # [B, M, 1, C] -> [B, M, FaceW * FaceW,  C] -> [B * M * FaceW * FaceW, 1, C]
                t_context_cube_inner = t_context_cube_inner.expand(
                    batch_size, M, face_w * face_w,  t_context_small.shape[-1]
                ).reshape(batch_size * M * face_w * face_w, -1, t_context_small.shape[-1])

                # 2. ERP -> cubemap (the main memory-intensive operation).
                x_cube = rearrange(x_in, 'bt (h w) c -> bt c h w', h=height, w=width)
                cubemap_faces = equirectangular_to_cubemap(x_cube, mode="bilinear")
                x_cube = rearrange(cubemap_faces, '(b t) m c f1 f2 -> (b m t) (f1 f2) c', b=batch_size, t=num_frames)

                # 3. Temporal Attention
                # import pdb; pdb.set_trace()
                # with sdpa_kernel(backends=[SDPBackend.MATH]):
                # with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                x_cube = temporal_block(
                    x_cube,
                    num_frames=num_frames,
                    encoder_hidden_states=t_context_cube_inner, # Use the context created above.
                )

                # 4. Cube -> Equi
                x_cube = rearrange(x_cube, '(b m t) (h w) c -> (b t) c h (m w)', b=batch_size, m=M, t=num_frames, h=face_w, w=face_w)
                x_cube = safe_cube2equi_wrapper(cubemap=x_cube, height=height, width=width, mode='bilinear')
                x_cube = rearrange(x_cube, 'bt c h w -> bt (h w) c')
                
                return x_cube
            
            # Run the cubemap temporal branch with gradient checkpointing.
            if self.training and self.gradient_checkpointing:
                hidden_states_mix_cube = torch.utils.checkpoint.checkpoint(
                    run_cube_branch, 
                    hidden_states_mix, 
                    cube_time_context_first_timestep, # Pass the compact [B, 1, M, C] context.
                    use_reentrant=False
                )
            else:
                hidden_states_mix_cube = run_cube_branch(hidden_states_mix, cube_time_context_first_timestep) # [bt hw c]
            

            # Process the ERP features through the temporal layer.
            # with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            hidden_states_mix = temporal_block(
                hidden_states_mix, # [bt hw c]
                num_frames=num_frames,
                encoder_hidden_states=time_context,
            )

            # Fuse the cubemap and ERP features.
            # inject_cube_branch_layers should have attached cube_zero_linear.
            if hasattr(temporal_block, "cube_zero_linear"):
                # Learn how much cubemap information to add.
                # hidden_states_mix_cube shape: [BT, HW, C]
                gate = temporal_block.cube_zero_linear(hidden_states_mix_cube)
                hidden_states_mix.add_(gate)
            else:
                # Fall back to averaging if the projection layer was not injected.
                hidden_states_mix.add_(hidden_states_mix_cube).mul_(0.5)


            hidden_states = self.time_mixer(
                x_spatial=hidden_states,
                x_temporal=hidden_states_mix,
                image_only_indicator=image_only_indicator,
            )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)
    return forward


def apply_custom_cube_temporal_fusion_w_cube_fea_for_kv_processors_for_unet(
    model,
):
    """
    Apply forward functions with zero-initialized Conv/Linear fusion.
    Call ``inject_cube_branch_layers(model)`` first to create the required
    projection parameters.
    """
    print("--> Applying Cube-Aware Fusion Processors (with Zero Layers)...")
    for name, module in model.named_modules():
        # print(f"name: {name},,,, module: {module}")
        #
        if isinstance(module, TransformerSpatioTemporalModel):
            if not hasattr(module, 'original_forward'):
                module.original_forward = module.forward
            module.forward = cube_aware_fusion_w_cube_fea_for_kv_spatio_temporal_transformer_forward(module)

            module.forward_w_precube = module.forward
        if isinstance(module, SpatioTemporalResBlock):
            if not hasattr(module, 'original_forward'):
                module.original_forward = module.forward
            module.forward = cube_aware_fusion_spatio_temporal_resblock_forward(module)
            module.forward_w_precube = module.forward

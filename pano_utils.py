import torch
import torch.nn.functional as F
from functools import lru_cache
import math
import os
# ------------------------------------------------------------------
# 1. Precomputed sampling grid (created on first use).
# ------------------------------------------------------------------
GRID_CACHE_DIR = "spherical_grids"  # Directory for cached sampling grids.
os.makedirs(GRID_CACHE_DIR, exist_ok=True)

@torch.no_grad()
@lru_cache(maxsize=8)
def get_spherical_sampling_grid(erp_h, erp_w, device, dtype):
    """
    Return the 3D sampling grid for an ERP image of shape (H, W).
    The grid is computed and saved on first use, then loaded from disk.
    """
    # file_path = os.path.join(GRID_CACHE_DIR, f"grid_{erp_h}x{erp_w}_{dtype}.pt")
    dtype_str = str(dtype).replace("torch.", "")
    file_path = os.path.join(GRID_CACHE_DIR, f"grid_{erp_h}x{erp_w}_{dtype_str}.pt")

    # Load a previously computed grid when available.
    if os.path.exists(file_path):
        print(f"[Spherical Grid] Loading cached grid: {file_path}")
        grid = torch.load(file_path, map_location=device)
        return grid.to(device=device, dtype=dtype)

    # Otherwise, compute and cache the grid.
    print(f"[Spherical Grid] Precomputing {erp_h}x{erp_w} grid...")
    
    theta_1d = torch.linspace(-math.pi / 2, math.pi / 2, erp_h, device=device, dtype=dtype)
    phi_1d = torch.linspace(-math.pi, math.pi, erp_w, device=device, dtype=dtype)

    cos_theta = torch.cos(theta_1d)
    sin_theta = torch.sin(theta_1d)
    cos_phi = torch.cos(phi_1d)
    sin_phi = torch.sin(phi_1d)

    x = cos_theta.unsqueeze(-1) * cos_phi.unsqueeze(0)
    y = cos_theta.unsqueeze(-1) * sin_phi.unsqueeze(0)
    z = sin_theta.unsqueeze(-1).expand(-1, erp_w)

    grid = torch.stack([x, y, z], dim=-1).unsqueeze(0)  # (1, H, W, 3)

    # Save the CPU copy and move it to the target device when loading.
    torch.save(grid.cpu(), file_path)
    print(f"[Spherical Grid] Saved grid to: {file_path}")

    return grid


def get_spherical_gaussian_noise(shape, generator=None, device=None, dtype=None):
    """
    Generate ERP Gaussian noise sampled uniformly over the sphere.
    
    Args:
        shape (tuple): Target ERP latent shape, such as (B, T, C, H, W)
            or (B, C, H, W).
        generator (torch.Generator, optional): Generator for reproducible noise.
        device: Device for the output tensor, such as ``"cuda"``.
        dtype: Output tensor dtype, such as ``torch.float32``.

    Returns:
        A noise tensor sampled according to the spherical distribution.
    """
    
    # 1. Extract the ERP dimensions from the input shape.
    full_shape = shape
    if len(full_shape) < 3:
        raise ValueError(
            f"Expected at least three dimensions (C, H, W), but received "
            f"{len(full_shape)} dimensions: {full_shape}"
        )
    
    c, erp_h, erp_w = full_shape[-3:]  # C, H, W are always the final dimensions.
    
    # batch_dims contains every dimension before C, H, W: (B, T), (B,), or ().
    batch_dims = full_shape[:-3]
    
    # Flatten all batch dimensions into N. math.prod(()) correctly returns 1.
    n = math.prod(batch_dims)

    # 2. Set the 3D noise-volume side length to the cubemap face width.
    d = max(1, erp_w // 4)

    # 3. Generate one continuous i.i.d. 3D Gaussian noise volume.
    # This avoids discontinuities at cubemap seams. Shape: (N, C, D, D, D).
    noise_3d_source = torch.randn(
        n, c, d, d, d, 
        generator=generator,
        device=device, 
        dtype=dtype
    )

    # 4. Get the precomputed 3D sampling-coordinate lookup table.
    sampling_grid = get_spherical_sampling_grid(erp_h, erp_w, device=device, dtype=dtype) # (1, erp_h, erp_w, 3)

    # 5. Expand the lookup table to the current batch size.
    batch_grid = sampling_grid.expand(n, -1, -1, -1)
    batch_grid = batch_grid.unsqueeze(1)  # (N, 1, H, W, 3)

    # import pdb; pdb.set_trace()

    # 6. Sample the 3D noise volume using the ERP coordinate grid.
    noise_erp_flat = F.grid_sample(
        noise_3d_source,       # Input volume.
        batch_grid,            # Sampling coordinates.
        mode='bilinear',       # Performs trilinear interpolation for 5D input.
        padding_mode='border', # Clamp coordinates outside [-1, 1] to the border.
        align_corners=False    # Match the pixel-center convention used by linspace.
    )

    noise_erp_flat = noise_erp_flat.squeeze(2)  # (N, C, H, W)


    # 7. Restore the original batch dimensions: (N, C, H, W) -> (..., C, H, W).
    output_shape = (*batch_dims, c, erp_h, erp_w)
    noise_erp = noise_erp_flat.view(output_shape)

    return noise_erp

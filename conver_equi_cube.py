import torch
import numpy as np
from einops import rearrange
import equilib
import math
from typing import List, Dict, Union

def equirectangular_to_cubemap(equi, mode="nearest"):
    """
    Convert an equirectangular image or video batch to cubemap faces.

    Supports torch tensors and NumPy arrays with shape ``(B, C, H, W)`` or
    ``(B, T, C, H, W)``. The output preserves the input type and dtype, adds
    a six-face dimension, and uses ``H // 2`` as the face size.
    """

    is_torch = isinstance(equi, torch.Tensor)
    backend = torch if is_torch else np
    original_dtype = equi.dtype

    def to_float32(x):
        if is_torch:
            return x.to(torch.float32)
        return x.astype(np.float32)

    def to_original_dtype(x):
        if is_torch:
            return x.to(original_dtype)
        return x.astype(original_dtype)

    if equi.ndim == 5:
        is_video = True
        bsz, num_frames, C, equi_h, equi_w = equi.shape
        batch_frames = bsz * num_frames

        equi_input = rearrange(equi, 'b t c h w -> (b t) c h w')

    elif equi.ndim == 4:
        is_video = False
        bsz, C, equi_h, equi_w = equi.shape
        batch_frames = bsz
        equi_input = equi

    else:
        raise ValueError(
            f"Expected a 4D (B, C, H, W) or 5D (B, T, C, H, W) input, "
            f"but received {equi.ndim}D"
        )

    if not equi_w == 2 * equi_h:
        print(
            "Warning: input does not appear to use a standard 2:1 "
            f"equirectangular aspect ratio (W={equi_w}, H={equi_h})"
        )

    M = 6
    face_w = equi_h // 2

    default_rots_list = [{'roll': 0., 'pitch': 0., 'yaw': 0.}] * batch_frames

    list_of_face_lists = equilib.equi2cube(
        equi=to_float32(equi_input),
        rots=default_rots_list,
        w_face=face_w,
        cube_format='list',
        mode=mode
    )
    
    all_faces_tensor_list = []
    for face_list in list_of_face_lists:
        all_faces_tensor_list.extend(face_list)

    output_cube = backend.stack(all_faces_tensor_list, axis=0)
    output_cube = to_original_dtype(output_cube)

    if is_video:
        output_cube = rearrange(
            output_cube, '(b t m) c h w -> b t m c h w',
            b=bsz, m=M
        )
    else:
        output_cube = rearrange(
            output_cube, '(b m) c h w -> b m c h w',
            b=bsz, m=M
        )

    return output_cube


def safe_cube2equi_wrapper(
    cubemap: Union[torch.Tensor, np.ndarray],
    height: int,
    width: int,
    mode: str = 'nearest',
    **kwargs
) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert a horizon-format cubemap to an equirectangular projection.

    Accepts ``(C, F, 6F)``, ``(B, C, F, 6F)``, or
    ``(B, T, C, F, 6F)`` input. For target sizes not divisible by eight, the
    result is rendered at a padded size and then resampled geometrically.
    """

    is_video = False
    input_for_equilib = cubemap
    bsz = 1
    original_dtype = cubemap.dtype

    if cubemap.ndim == 5:
        is_video = True
        bsz, num_frames = cubemap.shape[:2]
        input_for_equilib = rearrange(cubemap, 'b t c h w -> (b t) c h w')
    elif cubemap.ndim == 4:
        bsz = cubemap.shape[0]
        input_for_equilib = cubemap
    elif cubemap.ndim == 3:
        unbatched_output = True
        input_for_equilib = rearrange(cubemap, 'c h w -> 1 c h w')
    else:
        raise ValueError(
            "Expected a 3D, 4D, or 5D horizon-format cubemap, "
            f"but received {cubemap.ndim}D"
        )

    is_multiple_of_8 = (height % 8 == 0) and (width % 8 == 0)

    if isinstance(input_for_equilib, torch.Tensor):
        input_for_equilib = input_for_equilib.to(torch.float32)
    else:
        input_for_equilib = input_for_equilib.astype(np.float32)

    if is_multiple_of_8:
        equi_final = equilib.cube2equi(
            cubemap=input_for_equilib,
            height=height,
            width=width,
            cube_format='horizon',
            mode=mode,
            **kwargs
        )

    else:
        padded_height = math.ceil(height / 8) * 8
        padded_width = padded_height * 2

        equi_padded = equilib.cube2equi(
            cubemap=input_for_equilib,
            height=padded_height,
            width=padded_width,
            cube_format='horizon',
            mode=mode,
            **kwargs
        )

        rots: Union[Dict, List[Dict]]
        if len(equi_padded.shape) == 3:
            rots = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        else:
            batch_size = equi_padded.shape[0]
            rots = [{"roll": 0.0, "pitch": 0.0, "yaw": 0.0}] * batch_size

        equi_final = equilib.equi2equi(
            src=equi_padded,
            rots=rots,
            height=height,
            width=width,
            mode=mode,
            **kwargs
        )

    # equilib may squeeze a singleton batch dimension.
    if equi_final.ndim == 3:
        if isinstance(equi_final, torch.Tensor):
            equi_final = equi_final.unsqueeze(0)
        else:
            equi_final = equi_final[np.newaxis, ...]
    output = equi_final
    if is_video:
        output = rearrange(equi_final, '(b t) c h w -> b t c h w', b=bsz)
    if isinstance(output, torch.Tensor):
        output = output.to(original_dtype)
    else:
        output = output.astype(original_dtype)

    return output


def safe_equi2equi_resize(
    equi_img: Union[torch.Tensor, np.ndarray],
    height: int,
    width: int,
    mode: str = 'nearest',
    **kwargs
) -> Union[torch.Tensor, np.ndarray]:
    """
    Resize a batched equirectangular image with zero rotation.

    Supports torch tensors and NumPy arrays with shape ``(B, C, H, W)`` or
    ``(B, T, C, H, W)`` and preserves the input type and dtype.
    """

    is_video = False
    input_for_equilib = equi_img
    bsz = 1
    original_dtype = equi_img.dtype

    if equi_img.ndim == 5:
        is_video = True
        bsz, num_frames = equi_img.shape[:2]
        input_for_equilib = rearrange(equi_img, 'b t c h w -> (b t) c h w')
    elif equi_img.ndim == 4:
        bsz = equi_img.shape[0]
        input_for_equilib = equi_img
    else:
        raise ValueError(
            f"Expected a 4D (B, C, H, W) or 5D (B, T, C, H, W) input, "
            f"but received {equi_img.ndim}D"
        )

    if isinstance(input_for_equilib, torch.Tensor):
        input_for_equilib = input_for_equilib.to(torch.float32)
    else:
        input_for_equilib = input_for_equilib.astype(np.float32)

    batch_size = input_for_equilib.shape[0]
    rots = [{"roll": 0.0, "pitch": 0.0, "yaw": 0.0}] * batch_size

    equi_final = equilib.equi2equi(
        src=input_for_equilib,
        rots=rots,
        height=height,
        width=width,
        mode=mode,
        **kwargs
    )

    output = equi_final
    if is_video:
        output = rearrange(equi_final, '(b t) c h w -> b t c h w', b=bsz)

    if isinstance(output, torch.Tensor):
        output = output.to(original_dtype)
    else:
        output = output.astype(original_dtype)

    return output

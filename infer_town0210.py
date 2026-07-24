import torch
import json
import os
import time
import numpy as np
from PIL import Image
import cv2
import argparse
import gc
import multiprocessing as mp

from cube_temporal_alignment import apply_custom_cube_temporal_fusion_w_cube_fea_for_kv_processors_for_unet
from pvdepth.pipeline_base import DepthCrafterPipeline
from pvdepth.pipeline_pvdepth import PVDepthPipeline
from pvdepth.pipeline_w_psni import DepthCrafterPipeline_w_distortion_noise_annealed_weighting
from pvdepth.unet_base import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from pvdepth.unet_pvdepth import DiffusersUNetSpatioTemporalConditionModelPVDepth
from pvdepth.utils import vis_sequence_depth, save_video
from conver_equi_cube import safe_equi2equi_resize

dtype = torch.float16

def resize_erp_frame(frame_np, target_h, target_w, mode):
    """Resize an HxW or HxWxC frame with ERP-aware interpolation."""
    if frame_np.ndim == 2:
        original_h, original_w = frame_np.shape
        if original_h == target_h and original_w == target_w:
            return frame_np
        resized = safe_equi2equi_resize(
            frame_np[None, None, :, :],
            height=target_h,
            width=target_w,
            mode=mode,
        )
        return resized[0, 0]

    if frame_np.ndim == 3:
        original_h, original_w = frame_np.shape[:2]
        if original_h == target_h and original_w == target_w:
            return frame_np
        chw = np.transpose(frame_np, (2, 0, 1))
        resized = safe_equi2equi_resize(
            chw[None, :, :, :],
            height=target_h,
            width=target_w,
            mode=mode,
        )[0]
        return np.transpose(resized, (1, 2, 0))

    raise ValueError(f"Unsupported frame dimensions: {frame_np.ndim}")


def process_image_worker(args):
    """Load, resize, and normalize one image in a worker process."""
    image_path, target_w, target_h = args
    
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Warning: cv2.imread could not read {image_path}")
            return None
            
        # Convert OpenCV's BGR output to the RGB format expected by the model.
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        original_h, original_w, _ = img_rgb.shape
        if original_h != target_h or original_w != target_w:
            img_resized = resize_erp_frame(
                img_rgb,
                target_h=target_h,
                target_w=target_w,
                mode="bilinear",
            )
        else:
            img_resized = img_rgb
            
        # Convert to float32 and normalize to [0, 1].
        frame_np = (img_resized.astype(np.float32) / 255.0)
        
        return frame_np
        
    except Exception as e:
        print(f"Error: failed to process {image_path}: {e}")
        return None


def validate_pvdepth_unet_load(unet, loading_info):
    """Fail fast when a PVDepth checkpoint does not fully provide its cube weights."""
    cube_load_issues = {}
    for issue_type in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        issue_keys = []
        for issue in loading_info.get(issue_type, []):
            key = issue[0] if isinstance(issue, tuple) else issue
            if "cube_zero_" in key:
                issue_keys.append(key)
        if issue_keys:
            cube_load_issues[issue_type] = issue_keys

    cube_keys = [key for key in unet.state_dict() if "cube_zero_" in key]
    if len(cube_keys) != 76 or cube_load_issues:
        raise RuntimeError(
            "PVDepth UNet cube weights were not loaded completely: "
            f"model_cube_keys={len(cube_keys)}, load_issues={cube_load_issues}"
        )

def run_inference(json_path, image_base_dir, output_root_dir, args):
    """Run inference for each town/path/clip entry in a benchmark JSON file."""

    try:
        with open(json_path, 'r') as f:
            all_data = json.load(f)
        print(f"Loaded JSON file: {json_path}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: failed to load or parse JSON file {json_path}: {e}")
        return

    print("Loading UNet...")
    if args.ppl_type == "pvdepth":
        print("Loading PVDepth UNet with Cube Temporal Fusion...")
        unet, loading_info = DiffusersUNetSpatioTemporalConditionModelPVDepth.from_pretrained(
            args.unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            output_loading_info=True,
        )
        validate_pvdepth_unet_load(unet, loading_info)
        apply_custom_cube_temporal_fusion_w_cube_fea_for_kv_processors_for_unet(unet)
        unet.to(dtype=torch.float16, device=device)

    else:
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            args.unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
    print("UNet loaded.")

    print("Loading pipeline...")
    if args.ppl_type == "depthcrafter":
        pipe = DepthCrafterPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )
    elif args.ppl_type == "distortion_noise_annealed_weighting":
        pipe = DepthCrafterPipeline_w_distortion_noise_annealed_weighting.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )
    elif args.ppl_type == "pvdepth":
        pipe = PVDepthPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )

    else:
        raise ValueError(f"Unknown pipeline type: {args.ppl_type}")

    print("Pipeline loaded.")

    # Configure CPU offloading.
    if args.cpu_offload is not None:
        if args.cpu_offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif args.cpu_offload == "model":
            pipe.enable_model_cpu_offload()
        else:
            raise ValueError(f"Unknown cpu offload option: {args.cpu_offload}")
    else:
        pipe.to("cuda")

    # Enable memory optimizations.
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print(e)
        print("Xformers is not enabled")
    pipe.enable_attention_slicing()

    clip_counter = 0
    total_clips = sum(
        len(path_data)
        for town_data in all_data.values()
        for path_data in town_data.values()
    )
    print(f"Found {total_clips} clips across all towns and paths.")

    # Generate at most one MP4 preview for each benchmark JSON file.
    max_visualizations = 1
    visualize_interval = (total_clips + max_visualizations - 1) // max_visualizations
    visualize_interval = max(1, visualize_interval)
    print(f"Generating an MP4 preview every {visualize_interval} clips.")

    for town_name, town_data in all_data.items():
        for path_name, path_data in town_data.items():
            for clip_name, frames_data in path_data.items():
                
                clip_counter += 1
                print(f"\n--- Processing clip {clip_counter}/{total_clips}: {clip_name} ---")

                output_dir = output_root_dir
                os.makedirs(output_dir, exist_ok=True)

                npy_save_path = os.path.join(output_dir, f"{clip_name}_disparity.npy")
                vis_save_path = os.path.join(output_dir, f"{clip_name}_combined_animation.mp4")


                should_visualize = (clip_counter - 1) % visualize_interval == 0
                
                if os.path.exists(npy_save_path):
                    print(" - Disparity output already exists. Skipping clip.")
                    continue
                
                image_paths = []
                for frame_info in frames_data:
                    relative_path = frame_info.get("rgb_path")
                    if not relative_path:
                        print(f"Error: a frame in clip {clip_name} is missing 'rgb_path'.")
                        continue
                    full_path = os.path.join(image_base_dir, relative_path)
                    if not os.path.exists(full_path):
                        print(f"Warning: image file not found: {full_path}. Skipping clip.")
                        image_paths = []
                        break
                    image_paths.append(full_path)
                
                if not image_paths:
                    continue

                try:
                    with Image.open(image_paths[0]) as first_img:
                        original_width, original_height = first_img.size
                except Exception as e:
                    print(f"Error: could not read the size of {image_paths[0]}: {e}")
                    continue
                
                scale = args.resolution / original_width
                target_h = round(original_height * scale / 64) * 64
                target_w = round(original_width * scale / 64) * 64

                if target_h == 0 or target_w == 0:
                    print(f"Warning: invalid target size ({target_h}, {target_w}). Skipping clip.")
                    continue

                print(
                    f"Resizing frames from ({original_width}, {original_height}) "
                    f"to ({target_w}, {target_h})."
                )

                tasks = [(path, target_w, target_h) for path in image_paths]
                print(f"Loading {len(tasks)} frames with {args.num_workers} workers...")
                start_load_time = time.time()
                
                frames_list = []
                with mp.Pool(processes=args.num_workers) as pool:
                    frames_list = pool.map(process_image_worker, tasks)
                
                frames_list = [f for f in frames_list if f is not None]
                
                if not frames_list or len(frames_list) != len(image_paths):
                    failed_frames = len(image_paths) - len(frames_list)
                    print(
                        f"Warning: failed to load {failed_frames}/{len(image_paths)} frames. "
                        "Skipping clip."
                    )
                    continue
                
                end_load_time = time.time()
                print(f"Frame loading and preprocessing: {end_load_time - start_load_time:.3f}s")

                frames_np = np.stack(frames_list, axis=0)
                print(f"Frame tensor shape: {frames_np.shape}")

                print("Starting inference...")
                start_infer_time = time.time()
                with torch.inference_mode():
                    res_3channel = pipe(
                        frames_np,
                        height=target_h,
                        width=target_w,
                        output_type="np",
                        guidance_scale=1.0,
                        num_inference_steps=5,
                        window_size=110,
                        overlap=25,
                        track_time=False,
                        decode_chunk_size=args.decode_chunk_size,
                    ).frames[0]
                end_infer_time = time.time()
                print(f" - Pipeline inference: {end_infer_time - start_infer_time:.3f}s")

                start_post_time = time.time()
                res_1channel = (res_3channel.sum(-1) / res_3channel.shape[-1])
                clip_min, clip_max = res_1channel.min(), res_1channel.max()
                
                if clip_max <= clip_min:
                    normalized_disparity_stack = np.zeros_like(res_1channel)
                else:
                    normalized_disparity_stack = (res_1channel - clip_min) / (clip_max - clip_min)
                
                np.save(npy_save_path, normalized_disparity_stack)
                print(f" - Saved normalized disparity: {npy_save_path}")
                end_post_time = time.time()
                print(f" - NumPy postprocessing: {end_post_time - start_post_time:.3f}s")

                vis_frames_array = None
                
                if should_visualize:
                    print(f" - Generating MP4 preview for clip {clip_counter}...")
                    vis_start_time = time.time()
                    vis_frames_array = vis_sequence_depth(normalized_disparity_stack)
                    
                    pil_frames_list = []
                    
                    try:
                        for frame_idx in range(frames_np.shape[0]):
                            rgb_frame_np = (frames_np[frame_idx] * 255).astype(np.uint8)
                            depth_frame_np = (vis_frames_array[frame_idx] * 255).astype(np.uint8)
                            
                            rgb_pil = Image.fromarray(rgb_frame_np)
                            depth_pil = Image.fromarray(depth_frame_np)
                            
                            if rgb_pil.size != depth_pil.size:
                                depth_frame_np = resize_erp_frame(
                                    depth_frame_np,
                                    target_h=rgb_frame_np.shape[0],
                                    target_w=rgb_frame_np.shape[1],
                                    mode="nearest",
                                )
                                depth_pil = Image.fromarray(depth_frame_np)
                            
                            total_height = rgb_pil.height + depth_pil.height
                            combined_image = Image.new('RGB', (rgb_pil.width, total_height))
                            combined_image.paste(rgb_pil, (0, 0))
                            combined_image.paste(depth_pil, (0, rgb_pil.height))
                            
                            pil_frames_list.append(combined_image)
                    
                        save_video(pil_frames_list, vis_save_path, fps=10)
                    
                        vis_end_time = time.time()
                        print(f" - Saved MP4 preview: {vis_save_path}")
                        print(f" - Visualization: {vis_end_time - vis_start_time:.3f}s")

                    except Exception as e:
                        print(f" - Error: failed to generate MP4 preview: {e}")
                    finally:
                        del pil_frames_list
                
                else:
                    print(
                        f" - Clip {clip_counter} is outside the 1/{visualize_interval} "
                        "preview interval. Skipping MP4 generation."
                    )

                del frames_np, frames_list, res_3channel, res_1channel, normalized_disparity_stack
                if vis_frames_array is not None:
                    del vis_frames_array
                
                if clip_counter % 10 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                gc.collect()

    print("\n--- All clips processed ---")

def extract_file_parameters(json_filename):
    """Extract setting, frame-rate, and clip-length labels from a JSON filename."""
    setting = "unknown"
    if "dynamic" in json_filename:
        setting = "dynamic"
    elif "static" in json_filename:
        setting = "static"

    fps_map = {
        "fps01": "fps01",
        "fps10": "fps10",
        "fps20": "fps20",
        "fps02": "fps02"
    }
    fps_str = "fps_unknown"
    for key, value in fps_map.items():
        if key in json_filename:
            fps_str = value
            break

    len_map = {
        "len20": "len20",
        "len50": "len50",
        "len90": "len90",
        "len110": "len110"
    }
    len_str = "len_unknown"
    for key, value in len_map.items():
        if key in json_filename:
            len_str = value
            break

    return setting, fps_str, len_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Town0210 depth inference on video clips.")

    parser.add_argument("--json_path", type=str, 
                        default=None,
                        help="Path to the JSON file containing clip information.")
    parser.add_argument("--image_base_dir", type=str, 
                        default=None,
                        help="Base directory where image files are located.")
    
    parser.add_argument("--output_root_dir", type=str, 
                        default="carla_benchmark_results/zeroshot",
                        help="Base directory to save inference results (e.g., 'carla_benchmark_results/zeroshot').")
    
    parser.add_argument("--resolution", type=int, 
                        default=1024,
                        help="Target width for resizing.")
    
    parser.add_argument("--unet_path", type=str, 
                        default="tencent/DepthCrafter", 
                        help="Path to the UNet checkpoint.")
    parser.add_argument("--num_workers", type=int, 
                        default=8,
                        help="Number of worker processes for parallel image loading.")
    
    parser.add_argument(
        "--decode_chunk_size", type=int, default=24,
        help="Chunk size for decoding during inference."
    )

    parser.add_argument("--cpu_offload", type=str, default=None,
                        help="CPU offload strategy: None, 'model', or 'sequential'.")
    
    parser.add_argument(
        "--ppl_type",
        type=str,
        default="depthcrafter",
        choices=[
            "depthcrafter",
            "distortion_noise_annealed_weighting",
            "pvdepth",
        ],
        help="Pipeline type to use."
    )

    args = parser.parse_args()

    if args.cpu_offload == "None":
        args.cpu_offload = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    json_filename = os.path.basename(args.json_path)
    setting, fps_str, len_str = extract_file_parameters(json_filename)
    town_name = "town0210"
        
    output_root_dir = os.path.join(args.output_root_dir, f"{town_name}_{args.resolution}_{setting}_{fps_str}_{len_str}")

    print(f"Saving all outputs under: {output_root_dir}")

    run_inference(args.json_path, args.image_base_dir, output_root_dir, args)

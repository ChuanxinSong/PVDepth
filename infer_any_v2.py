import torch
import os
import numpy as np
from PIL import Image
import cv2
import argparse
import multiprocessing as mp

# --- 1. Import model components ---
from cube_temporal_alignment import (
    apply_custom_cube_temporal_fusion_w_cube_fea_for_kv_processors_for_unet,
    )
from pvdepth.pipeline_pvdepth import PVDepthPipeline
from pvdepth.unet_pvdepth import DiffusersUNetSpatioTemporalConditionModelPVDepth
from pvdepth.utils import vis_sequence_depth, save_video


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

def process_image_worker(args):
    """
    Load, convert (BGR to RGB), resize, and normalize an image in a worker process.
    """
    image_path, target_w, target_h = args
    
    try:
        # 1. Load the image in BGR format with OpenCV.
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"Warning: cv2.imread could not read {image_path}")
            return None
            
        # 2. Convert BGR to the RGB format expected by the model.
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 3. Resize the image if necessary.
        original_h, original_w, _ = img_rgb.shape
        if original_h != target_h or original_w != target_w:
            # OpenCV expects the target size in (W, H) order.
            # INTER_LANCZOS4 corresponds to PIL's LANCZOS filter.
            img_resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            img_resized = img_rgb
            
        # 4. Convert to float32 and normalize to [0, 1].
        frame_np = (img_resized.astype(np.float32) / 255.0)
        
        return frame_np
        
    except Exception as e:
        print(f"Error while processing {image_path}: {e}")
        return None

def load_input_any(input_path, target_res, num_workers):
    """
    Support three input types:
    1. A single image.
    2. A folder containing an image sequence.
    3. A video file.
    """
    frames_np = None
    fps = 15
    original_size = (0, 0)

    if os.path.isdir(input_path):
        # Image-folder input.
        print(f"==> Loading image folder: {input_path}")
        img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        image_paths = sorted([os.path.join(input_path, f) for f in os.listdir(input_path) 
                             if os.path.splitext(f)[1].lower() in img_exts])
        if not image_paths:
            raise ValueError(f"No images found in folder: {input_path}")
        
        with Image.open(image_paths[0]) as first_img:
            original_width, original_height = first_img.size
            original_size = (original_height, original_width)
        
        # Use a 2:1 aspect ratio and dimensions divisible by 64.
        target_w = round(target_res / 64) * 64
        target_h = round((target_w / 2) / 64) * 64
        
        tasks = [(path, target_w, target_h) for path in image_paths]
        with mp.Pool(processes=num_workers) as pool:
            frames_list = pool.map(process_image_worker, tasks)
        frames_list = [f for f in frames_list if f is not None]
        frames_np = np.stack(frames_list, axis=0)

    elif os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            # Single-image input.
            print(f"==> Loading image: {input_path}")
            with Image.open(input_path) as img:
                original_width, original_height = img.size
                original_size = (original_height, original_width)
            
            # Use a 2:1 aspect ratio and dimensions divisible by 64.
            target_w = round(target_res / 64) * 64
            target_h = round((target_w / 2) / 64) * 64
            
            frame_np = process_image_worker((input_path, target_w, target_h))
            frames_np = np.expand_dims(frame_np, axis=0)
        else:
            # Video input.
            print(f"==> Loading video: {input_path}")
            from decord import VideoReader, cpu
            vid = VideoReader(input_path, ctx=cpu(0))
            fps = vid.get_avg_fps()
            original_height, original_width = vid.get_batch([0]).shape[1:3]
            original_size = (original_height, original_width)
            
            # Use a 2:1 aspect ratio and dimensions divisible by 64.
            target_w = round(target_res / 64) * 64
            target_h = round((target_w / 2) / 64) * 64
            
            # Decode the video again at the target resolution.
            vid = VideoReader(input_path, ctx=cpu(0), width=target_w, height=target_h)
            frames_np = vid.get_batch(range(len(vid))).asnumpy().astype(np.float32) / 255.0
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    return frames_np, fps, original_size

def main():
    parser = argparse.ArgumentParser(description="PVDepth inference for images, folders, and videos")
    parser.add_argument("--input_path", type=str, required=True, help="Path to image, folder, or video")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--resolution", type=int, default=1024, help="Target width for resizing.")
    parser.add_argument(
        "--unet_path",
        type=str,
        default="Soon122/PVDepth",
        help="Path or Hugging Face model ID of the PVDepth UNet.",
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of CPU cores for parallel image loading.")
    parser.add_argument("--decode_chunk_size", type=int, default=8, help="Chunk size for decoding during inference.")
    parser.add_argument("--cpu_offload", type=str, default=None, help="CPU offload strategy: None, 'model', or 'sequential'.")
    parser.add_argument("--num_denoising_steps", type=int, default=5)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--window_size", type=int, default=110)
    parser.add_argument("--overlap", type=int, default=25)
    parser.add_argument("--fps", type=int, default=15, help="Output FPS for image/folder inputs")
    parser.add_argument("--save_frames", action="store_true", help="Save individual visualization frames")

    args = parser.parse_args()
    if args.cpu_offload == "None": args.cpu_offload = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Initialize the UNet ---
    print("Loading PVDepth UNet...")
    unet, loading_info = DiffusersUNetSpatioTemporalConditionModelPVDepth.from_pretrained(
        args.unet_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        output_loading_info=True,
    )
    validate_pvdepth_unet_load(unet, loading_info)
    apply_custom_cube_temporal_fusion_w_cube_fea_for_kv_processors_for_unet(unet)
    unet.to(dtype=torch.float16, device=device)

    # --- 2. Initialize the pipeline ---
    print("Loading pipeline...")
    svd_path = "stabilityai/stable-video-diffusion-img2vid-xt"
    pipe = PVDepthPipeline.from_pretrained(
        svd_path,
        unet=unet,
        torch_dtype=torch.float16,
        variant="fp16",
    )

    if args.cpu_offload:
        if args.cpu_offload == "sequential": pipe.enable_sequential_cpu_offload()
        else: pipe.enable_model_cpu_offload()
    else: pipe.to("cuda")
    pipe.enable_attention_slicing()

    # --- 3. Load the input ---
    frames_np, input_fps, (orig_h, orig_w) = load_input_any(args.input_path, args.resolution, args.num_workers)
    T, H, W, C = frames_np.shape
    fps = args.fps if os.path.isdir(args.input_path) or (os.path.isfile(args.input_path) and os.path.splitext(args.input_path)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']) else input_fps

    # --- 4. Run inference ---
    print(f"Running inference on {T} frame(s) at {W}x{H}...")
    with torch.inference_mode():
        res_3channel = pipe(frames_np, height=H, width=W, output_type="np", guidance_scale=args.guidance_scale, num_inference_steps=args.num_denoising_steps, window_size=min(args.window_size, T), overlap=0 if T <= args.window_size else args.overlap, decode_chunk_size=args.decode_chunk_size).frames[0]

    # --- 5. Post-process and save the results ---
    res_1channel = res_3channel.sum(-1) / res_3channel.shape[-1]
    
    os.makedirs(args.output_dir, exist_ok=True)
    input_name = os.path.splitext(os.path.basename(args.input_path.rstrip('/')))[0]
    # Save the relative inverse-depth prediction.
    np.save(
        os.path.join(args.output_dir, f"{input_name}_disparity.npy"),
        res_1channel,
    )
    
    # Normalize only for visualization.
    res_min, res_max = res_1channel.min(), res_1channel.max()
    res_norm = (res_1channel - res_min) / (res_max - res_min + 1e-6) if res_max > res_min else np.zeros_like(res_1channel)
    
    vis_frames = vis_sequence_depth(res_norm)
    pil_frames = []
    for i in range(T):
        rgb = (frames_np[i] * 255).astype(np.uint8)
        depth = (vis_frames[i] * 255).astype(np.uint8)
        if rgb.shape[:2] != depth.shape[:2]: depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        combined = np.concatenate([rgb, depth], axis=0)
        pil_frames.append(Image.fromarray(combined))
    
    # Determine whether the input is a single image.
    is_single_image = os.path.isfile(args.input_path) and os.path.splitext(args.input_path)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    if is_single_image:
        out_path = os.path.join(args.output_dir, f"{input_name}_vis.png")
        pil_frames[0].save(out_path)
        print(f"Saved the visualization image to: {out_path}")
    else:
        out_path = os.path.join(args.output_dir, f"{input_name}_vis.mp4")
        save_video(pil_frames, out_path, fps=fps)
        print(f"Saved the visualization video to: {out_path}")
        
        if args.save_frames:
            frames_dir = os.path.join(args.output_dir, f"{input_name}_frames")
            os.makedirs(frames_dir, exist_ok=True)
            print(f"Saving individual frames to: {frames_dir}")
            for i, frame in enumerate(pil_frames):
                frame.save(os.path.join(frames_dir, f"{i:05d}.png"))
    
    print(f"Done. Results are available in: {args.output_dir}")

if __name__ == "__main__":
    main()

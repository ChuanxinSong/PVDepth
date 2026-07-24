import os
import os.path as osp
import numpy as np
import cv2
import json
import logging
import csv
import sys
from typing import Any, Dict
import argparse
from concurrent.futures import ThreadPoolExecutor
from depth import depth_evaluation
import re
from tqdm import tqdm

CURRENT_DIR = osp.dirname(osp.abspath(__file__))
PROJECT_ROOT = osp.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from conver_equi_cube import safe_equi2equi_resize

# Global evaluation settings.
EVAL_KWARGS = {
    "max_depth": 80.0,
    "use_gpu": True,
}


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run batch evaluation for video depth estimation."
    )

    parser.add_argument('--json_files', required=True, nargs='+', 
                        help='One or more benchmark JSON filenames to process.')

    parser.add_argument('--json_base_dir', type=str, required=True, 
                        help='Directory containing the benchmark JSON files.')
    parser.add_argument('--pred_base_dir', type=str, required=True, 
                        help='Prediction root matching the inference output directory.')
    parser.add_argument('--gt_root', type=str, required=True, 
                        help='Ground-truth data root.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory for evaluation CSV files. '
                             'Defaults to depth_eval/results/pvdepth.')
    parser.add_argument('--town_name', type=str, default="town0210", 
                        help="Town label used in prediction paths. Defaults to 'town0210'.")
    parser.add_argument('--resolution', type=int, default=1024, 
                        help='Resolution used in prediction paths. Defaults to 1024.')

    parser.add_argument('--align_method', type=str, default="median", 
                        choices=['median', 'scale&shift', 'scale', 'metric'],
                        help='Depth alignment method. Defaults to "median".')
    
    parser.add_argument('--save_error_maps', action='store_true', 
                        help='Save mean and per-frame error maps for each clip.')
    
    parser.add_argument('--fixed_error_max', type=float, default=None,
                        help='Fixed error-map maximum for cross-model visualization. '
                             'Defaults to the 98th percentile of each sequence.')

    parser.add_argument('--max_depth', type=float, default=999.0,
                        help='Maximum valid depth used for evaluation. Defaults to 999.0.')
    
    return parser.parse_args()


def extract_file_parameters(json_filename):
    """Extract setting, FPS, and clip-length labels from a JSON filename."""

    setting = "unknown"
    if "dynamic" in json_filename:
        setting = "dynamic"
    elif "static" in json_filename:
        setting = "static"

    fps_match = re.search(r'fps(\d+)', json_filename)
    fps_str = fps_match.group(0) if fps_match else "fps_unknown"

    len_match = re.search(r'len(\d+)', json_filename)
    len_str = len_match.group(0) if len_match else "len_unknown"

    return setting, fps_str, len_str


def depth2disparity(depth, return_mask=False):
    """Convert valid depth values to disparity."""
    if isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity
    

def write_csv(filename: str, metrics: Dict[str, Any]):
    """Write metrics to CSV with floating-point values rounded to three decimals."""
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            v = v.item()
        if isinstance(v, float):
            v = round(v, 3)
        elif not isinstance(v, (int, float, str)):
            v = str(v)
        metrics[k] = v

    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(metrics.keys())
            writer.writerow(metrics.values())
        print(f"Saved metrics to: {filename}")
    except Exception as e:
        print(f"Failed to write CSV: {e}")

def load_npy_file(path: str) -> np.ndarray:
    """Load a NumPy file in a worker thread."""
    return np.load(path)

def run_evaluation(json_split_file: str, pred_root_dir: str, output_csv_file: str, args: argparse.Namespace):
    """Evaluate one benchmark JSON split against its prediction directory."""
    
    logger = logging.getLogger("videodepth-eval-standalone")
    if not osp.exists(json_split_file):
        logger.error(f"JSON split file not found: {json_split_file}")
        return False 

    logger.info(
        f"Starting PVDepth evaluation (JSON: {osp.basename(json_split_file)})"
    )
    logger.info(f"Loading clips from: {json_split_file}")
    logger.info(f"Using prediction directory: {pred_root_dir}")
    
    with open(json_split_file, 'r') as f:
        all_data = json.load(f)

    seq_list = []
    gt_paths = {} 
    pred_paths = {}

    for town_name, town_data in all_data.items():
        for path_name, path_data in town_data.items():
            for clip_name, frames_data in path_data.items():
                
                seq_name = clip_name
                
                clip_gt_paths = []
                for frame in frames_data:
                    gt_p = frame.get('depth_path')
                    if not gt_p:
                        err_msg = (
                            f"Evaluation aborted: a frame in clip '{seq_name}' "
                            f"is missing 'depth_path'. Frame data: {frame}"
                        )
                        logger.error(err_msg)
                        raise KeyError(err_msg)
                    
                    clip_gt_paths.append(osp.join(args.gt_root, gt_p))
                
                if not clip_gt_paths:
                    continue

                pred_suffix = f"{seq_name}_disparity.npy"
                pred_path = osp.join(pred_root_dir, pred_suffix)

                if not osp.exists(pred_path):
                    err_msg = f"Evaluation aborted: prediction file not found: {pred_path}"
                    logger.error(err_msg)
                    raise FileNotFoundError(err_msg)

                missing_gts = [p for p in clip_gt_paths if not osp.exists(p)]
                if missing_gts:
                    err_msg = (
                        f"Evaluation aborted: {len(missing_gts)} ground-truth "
                        f"files are missing in clip '{seq_name}'. "
                        f"First missing file: {missing_gts[0]}"
                    )
                    logger.error(err_msg)
                    raise FileNotFoundError(err_msg)
                
                seq_list.append(seq_name)
                gt_paths[seq_name] = clip_gt_paths
                pred_paths[seq_name] = pred_path

    if not seq_list:
        logger.error(f"No valid sequences found in {json_split_file}.")
        return False 

    depth_evaluation_kwargs = EVAL_KWARGS.copy()
    depth_evaluation_kwargs["max_depth"] = args.max_depth
    
    if args.align_method == "scale&shift":
        depth_evaluation_kwargs["align_with_lad2"] = True
    elif args.align_method == "scale":
        depth_evaluation_kwargs["align_with_scale"] = True
    elif args.align_method == "metric":
        depth_evaluation_kwargs["metric_scale"] = True
    else:
        logger.info(f"Using {args.align_method} alignment.")

    gathered_depth_metrics = [] 
    

    error_maps_base_dir = None
    if args.save_error_maps:
        # Derive the error-map directory from the CSV path.
        run_output_name = osp.splitext(osp.basename(output_csv_file))[0]
        csv_dir = osp.dirname(output_csv_file) 
        error_maps_base_dir = osp.join(csv_dir, run_output_name)
        
        os.makedirs(error_maps_base_dir, exist_ok=True)
        logger.info(f"Error maps will be saved to: {error_maps_base_dir}")

    logger.info(f"Evaluating {len(seq_list)} sequences.")
    
    for idx_seq, seq in enumerate(tqdm(seq_list, desc=f"Evaluating {osp.basename(json_split_file)}"), start=1):
        seq_gt_paths = gt_paths[seq]
        seq_pd_path = pred_paths[seq] 
        
        gt_depth_list = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            # executor.map preserves the input path order.
            gt_depth_list = list(executor.map(load_npy_file, seq_gt_paths))
            
        gt_depth = np.stack(gt_depth_list, axis=0)
            
        pr_depth_stack = np.load(seq_pd_path) 
        depth_evaluation_kwargs["disp_input"] = True

        assert len(pr_depth_stack) == len(gt_depth), \
            f"Frame count mismatch for {seq}: Pred ({len(pr_depth_stack)}) vs GT ({len(gt_depth)})"
            
        pred_h, pred_w = pr_depth_stack.shape[1], pr_depth_stack.shape[2]

        gt_depth = safe_equi2equi_resize(
            gt_depth[:, None, :, :],
            height=pred_h,
            width=pred_w,
            mode="nearest",
        )[:, 0, :, :]
        pr_depth = pr_depth_stack
        
        depth_results, depth_error_parity_map_full, predict_depth_map_full, gt_depth_map_full = depth_evaluation(
            pr_depth, 
            gt_depth, 
            **depth_evaluation_kwargs
        )

        # Restore the frame dimension if depth_evaluation flattened
        # [N, H, W] into [N * H, W].
        if depth_error_parity_map_full.dim() == 2 and len(gt_depth.shape) == 3:
            N, H, W = gt_depth.shape
            if depth_error_parity_map_full.shape[0] == N * H:
                 depth_error_parity_map_full = depth_error_parity_map_full.view(N, H, W)

        gathered_depth_metrics.append(depth_results)

        if args.save_error_maps and error_maps_base_dir is not None:
            error_maps_save_dir = error_maps_base_dir 
            error_map_np = depth_error_parity_map_full.cpu().numpy()

            eps = np.finfo(float).eps
            if args.fixed_error_max is not None:
                clip_max_error = args.fixed_error_max
            else:
                clip_max_error = np.percentile(error_map_np, 98) 
            
            if clip_max_error == 0:
                clip_max_error = eps

            # Save one temporally averaged error map for the clip.
            mean_error_map = np.mean(error_map_np, axis=0)
            mean_error_map_vis = (np.clip(mean_error_map / (clip_max_error + eps), 0, 1) * 255).astype(np.uint8)
            mean_error_map_vis_color = cv2.applyColorMap(mean_error_map_vis, cv2.COLORMAP_VIRIDIS)

            cv2.imwrite(osp.join(error_maps_save_dir, f"{seq}_error_map.png"), mean_error_map_vis_color)

            # Save per-frame error maps.
            frame_dir = osp.join(error_maps_save_dir, seq)
            os.makedirs(frame_dir, exist_ok=True)
            for i in range(error_map_np.shape[0]):
                frame_err = error_map_np[i]

                frame_err_vis = (np.clip(frame_err / (clip_max_error + eps), 0, 1) * 255).astype(np.uint8)
                frame_err_vis_color = cv2.applyColorMap(frame_err_vis, cv2.COLORMAP_VIRIDIS)
                cv2.imwrite(osp.join(frame_dir, f"frame_{i:04d}.png"), frame_err_vis_color)


    if not gathered_depth_metrics:
        logger.error("No sequences were evaluated successfully.")
        return False

    total_average_metrics = {
        key: np.average(
            [metrics[key] for metrics in gathered_depth_metrics],
            weights=[
                metrics["valid_pixels"] for metrics in gathered_depth_metrics
            ],
        )
        for key in gathered_depth_metrics[0].keys()
        if key != "valid_pixels"
    }

    logger.info(
        f"PVDepth evaluation complete (JSON: {osp.basename(json_split_file)})"
    )
    logger.info(f"Average metrics: {total_average_metrics}")
    
    write_csv(output_csv_file, total_average_metrics)


def main_loop(args: argparse.Namespace):

    logger = logging.getLogger("videodepth-eval-standalone")

    json_filenames_to_run = args.json_files
    
    logger.info("Starting batch evaluation.")
    logger.info(f"JSON directory: {args.json_base_dir}")
    logger.info(f"Prediction directory: {args.pred_base_dir}")
    logger.info(f"JSON files to process: {len(json_filenames_to_run)}")

    for json_filename in json_filenames_to_run:
        
        json_full_path = osp.join(args.json_base_dir, json_filename)
                
        setting, fps_str, len_str = extract_file_parameters(json_filename)
        
        assert setting != "unknown" and fps_str != "fps_unknown" and len_str != "len_unknown", \
            f"Failed to parse benchmark parameters from {json_filename}"
            
        pred_dir_suffix = f"{args.town_name}_{args.resolution}_{setting}_{fps_str}_{len_str}"
        pred_full_path = osp.join(args.pred_base_dir, pred_dir_suffix)
        
        current_max_depth = args.max_depth
        output_csv_filename = (
            f"{setting}_{fps_str}_{len_str}_res{args.resolution}"
            f"-metric-{args.align_method}_maxdepth{current_max_depth}.csv"
        )
        output_csv_full_path = osp.join(args.output_dir, output_csv_filename)
        
        print("\n" + "="*80)
        
        run_evaluation(
            json_full_path, 
            pred_full_path, 
            output_csv_full_path, 
            args
        )

    print("="*80)
    logger.info("Batch evaluation complete.")


def main():
    """Parse arguments, configure logging, and run batch evaluation."""
    args = get_args()
    
    logger = logging.getLogger("videodepth-eval-standalone")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.output_dir is None:
        args.output_dir = osp.join(CURRENT_DIR, "results", "pvdepth")
    
    if not osp.exists(args.output_dir):
        logger.info(f"Creating output directory: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        
    main_loop(args)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compute error maps between images in two folder trees (GT vs PRED), saving to an output folder.

Usage:
  CUDA_VISIBLE_DEVICES=3 python compute_error_maps.py \
    --gt   /data2/kongsujie/3dgrut/runs/nerf_debug/lego_3dgrt_multi_debug/lego-1709_155324/ours_30000/gt \
    --pred /data2/kongsujie/3dgrut/runs/nerf_debug/lego_3dgrt_multi_debug/lego-1709_155324/ours_30000/renders \
    --out  /data2/kongsujie/3dgrut/runs/nerf_debug/lego_3dgrt_multi_debug/lego-1709_155324/ours_30000/_error_maps \
    --colormap \
    --workers 8
    [--mode abs|l2] [--colormap] [--resize] [--suffix _err] [--workers 8]

Details:
- Images are matched by *relative path* under --gt and --pred.
  Example: if --gt contains "mic/test/0001.png", we look for "mic/test/0001.png" under --pred.
- Loads images as RGB float32 in [0,1].
- Error map per pixel (H x W):
    abs -> mean(|a-b|) across channels
    l2  -> sqrt(mean((a-b)^2)) across channels
- Visual normalization: robust scaling by 99th percentile before saving.
- Heatmap if --colormap (matplotlib), otherwise 8-bit grayscale.
"""
import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image, ImageOps

# Optional matplotlib, only needed for colormap
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images_rel(dir_path: Path) -> Dict[str, Path]:
    """
    Recursively list images under dir_path, keyed by POSIX-style relative path (e.g., 'scene/test/0001.png').
    """
    mapping = {}
    base = dir_path.resolve()
    for p in base.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            rel = p.resolve().relative_to(base).as_posix()
            mapping[rel] = p
    return mapping


def load_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float('inf')
    import math
    return -10.0 * math.log10(mse)


def compute_error_map(a: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
    """
    a, b: H x W x 3, float32 in [0,1]
    returns: H x W float32 in [0,1]
    """
    # NOTE: 保留你当前的误差放大（若不需要，可改为 diff = a - b）
    diff = a - b
    if mode == "abs":
        emap = np.mean(np.abs(diff), axis=2)
    elif mode == "l2":
        emap = np.sqrt(np.mean(diff**2, axis=2))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if emap.size > 0:
        p99 = float(np.percentile(emap, 99))
        mmax = max(1e-8, p99)
        emap = np.clip(emap, 0.0, mmax) / mmax
    return emap


def save_map_gray(emap: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(emap * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(out_path)


def save_map_colormap(emap: np.ndarray, out_path: Path, cmap_name: str = "magma") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not _HAS_MPL:
        warnings.warn("matplotlib not available; falling back to grayscale.")
        save_map_gray(emap, out_path)
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6), dpi=150)
    plt.axis("off")
    plt.imshow(emap, cmap=cmap_name, vmin=0.0, vmax=1.0)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def process_one(args_tuple) -> Tuple[str, bool, str, float]:
    rel, pathA, pathB, out_root, mode, use_colormap, resize, suffix = args_tuple
    try:
        imA = load_rgb(pathA)
        imB = load_rgb(pathB)
        if imA.size != imB.size:
            if resize:
                imB = imB.resize(imA.size, Image.BICUBIC)
            else:
                return (rel, False, f"size mismatch {imA.size} vs {imB.size}", float('nan'))
        a = np.asarray(imA, dtype=np.float32) / 255.0
        b = np.asarray(imB, dtype=np.float32) / 255.0
        psnr = compute_psnr(a, b)
        emap = compute_error_map(a, b, mode=mode)

        # 输出路径：保持与 GT 相同的相对目录结构
        stem = Path(rel).stem
        parent_rel = Path(rel).parent
        out_dir = out_root / parent_rel
        out_name = f"{stem}{suffix}.png"
        out_path = out_dir / out_name

        if use_colormap:
            save_map_colormap(emap, out_path)
        else:
            save_map_gray(emap, out_path)
        return (rel, True, out_path.as_posix(), psnr)
    except Exception as e:
        return (rel, False, str(e), float('nan'))


def run_pair(gt_dir: Path, pred_dir: Path, out_dir: Path, mode: str, use_colormap: bool, resize: bool, suffix: str, workers: int) -> Tuple[int, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mapA = list_images_rel(gt_dir)
    mapB = list_images_rel(pred_dir)
    common = sorted(set(mapA.keys()) & set(mapB.keys()))
    if not common:
        print(f"[WARN] No same-relative-path images found in:\n  gt:   {gt_dir}\n  pred: {pred_dir}", file=sys.stderr)

    tasks = [
        (rel, mapA[rel], mapB[rel], out_dir, mode, use_colormap, resize, suffix)
        for rel in common
    ]

    results: List[Tuple[str, bool, str, float]] = []
    if workers and workers > 0 and len(tasks) > 1:
        try:
            import multiprocessing as mp
            with mp.Pool(processes=workers) as pool:
                for res in pool.imap_unordered(process_one, tasks):
                    results.append(res)
        except Exception as e:
            warnings.warn(f"Multiprocessing failed ({e}); falling back to single process.")
            results = [process_one(t) for t in tasks]
    else:
        results = [process_one(t) for t in tasks]

    report_lines, ok = [], 0
    header = "STATUS\tREL_PATH\tOUTPUT\tPSNR_dB"
    report_lines.append(header)
    for rel, success, info, psnr in results:
        status = "OK" if success else "SKIP/ERR"
        if success:
            ok += 1
        psnr_str = (f"{psnr:.4f}" if np.isfinite(psnr) else "NA")
        report_lines.append(f"{status}\t{rel}\t{info}\t{psnr_str}")

    report_path = out_dir / "error_map_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    return ok, report_path


def main():
    parser = argparse.ArgumentParser(description="Compute error maps (GT vs PRED) by matching relative paths, output to a single folder.")
    parser.add_argument("--gt", type=str, required=True, help="GT root directory.")
    parser.add_argument("--pred", type=str, required=True, help="PRED/Renders root directory.")
    parser.add_argument("--out", type=str, required=True, help="Output directory to save error maps.")
    parser.add_argument("--mode", type=str, default="abs", choices=["abs", "l2"], help="Error mode: abs (MAE) or l2 (RMSE).")
    parser.add_argument("--colormap", action="store_true", help="Save colored heatmap (requires matplotlib).")
    parser.add_argument("--resize", action="store_true", help="Resize PRED images to match GT sizes if mismatched.")
    parser.add_argument("--suffix", type=str, default="_err", help="Suffix for output filenames before extension.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (0 for single-process).")
    args = parser.parse_args()

    gt_dir = Path(args.gt).expanduser().resolve()
    pred_dir = Path(args.pred).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    if not gt_dir.is_dir() or not pred_dir.is_dir():
        print(f"[ERROR] Invalid input directories:\n  --gt:   {gt_dir}\n  --pred: {pred_dir}", file=sys.stderr)
        sys.exit(2)

    ok, report_path = run_pair(gt_dir, pred_dir, out_dir, args.mode, args.colormap, args.resize, args.suffix, args.workers)
    print(f"Done. Saved {ok} error maps to: {out_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

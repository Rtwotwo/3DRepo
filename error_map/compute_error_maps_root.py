#!/usr/bin/env python3
'''
CUDA_VISIBLE_DEVICES=1 python compute_error_maps.py \
  --batch-root /data/kongsujie/gaussian-splatting/outputs/nerf_mtmt \
  --a-name ours_30000/gt \
  --b-name ours_30000/renders \
  --midpath test \
  --scenes "ship drums ficus hotdog lego materials mic chair" \
  --output-root /data/kongsujie/gaussian-splatting/outputs/nerf_mtmt/_error_maps \
  --mode abs --colormap --workers 8
    '''
"""
Compute error maps between same-named images in two folders.

Two modes:
1) Single-pair mode (backward compatible):
   python compute_error_maps.py DIR_A DIR_B OUT_DIR [--mode abs|l2] [--colormap] [--resize] [--suffix _err] [--workers 8]

2) Batch mode over scenes under a root (auto-detect layout):
   python compute_error_maps.py --batch-root ROOT --a-name gt --b-name pred \
       --scenes "ship drums ficus hotdog lego materials mic chair" \
       [--output-root ROOT/_error_maps] [--mode abs|l2] [--colormap] [--resize] [--suffix _err] [--workers 8]

Layout auto-detection (tries both):
  A) ROOT/a-name/scene   vs ROOT/b-name/scene
  B) ROOT/scene/a-name   vs ROOT/scene/b-name

Details:
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
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageOps

# Optional matplotlib, only needed for colormap
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(dir_path: Path) -> dict:
    mapping = {}
    for p in dir_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            mapping[p.name] = p
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
        # mmax = p99
        emap = np.clip(emap, 0.0, mmax) / mmax
    return emap


def save_map_gray(emap: np.ndarray, out_path: Path) -> None:
    arr = np.clip(emap * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(out_path)


def save_map_colormap(emap: np.ndarray, out_path: Path, cmap_name: str = "magma") -> None: # magma\inferno\plasma
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
    fname, pathA, pathB, out_dir, mode, use_colormap, resize, suffix = args_tuple
    try:
        imA = load_rgb(pathA)
        imB = load_rgb(pathB)
        if imA.size != imB.size:
            if resize:
                imB = imB.resize(imA.size, Image.BICUBIC)
            else:
                return (fname, False, f"size mismatch {imA.size} vs {imB.size}", float('nan'))
        a = np.asarray(imA, dtype=np.float32) / 255.0
        b = np.asarray(imB, dtype=np.float32) / 255.0
        psnr = compute_psnr(a, b)
        emap = compute_error_map(a, b, mode=mode)
        stem = Path(fname).stem
        out_name = f"{stem}{suffix}.png"
        out_path = out_dir / out_name
        if use_colormap:
            save_map_colormap(emap, out_path)
        else:
            save_map_gray(emap, out_path)
        return (fname, True, out_name, psnr)
    except Exception as e:
        return (fname, False, str(e), float('nan'))


def run_single(dirA: Path, dirB: Path, out_dir: Path, mode: str, use_colormap: bool, resize: bool, suffix: str, workers: int) -> Tuple[int, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mapA = list_images(dirA)
    mapB = list_images(dirB)
    common = sorted(set(mapA.keys()) & set(mapB.keys()))
    if not common:
        print(f"[WARN] No same-named images found in {dirA} vs {dirB}.", file=sys.stderr)

    tasks = [
        (fname, mapA[fname], mapB[fname], out_dir, mode, use_colormap, resize, suffix)
        for fname in common
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
    header = "STATUS\tFILENAME\tOUTPUT\tPSNR_dB"
    report_lines.append(header)
    for fname, success, info, psnr in results:
        status = "OK" if success else "SKIP/ERR"
        if success:
            ok += 1
        psnr_str = (f"{psnr:.4f}" if np.isfinite(psnr) else "NA")
        report_lines.append(f"{status}\t{fname}\t{info}\t{psnr_str}")

    report_path = out_dir / "error_map_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    return ok, report_path


def resolve_scene_paths(batch_root: Path, scene: str, a_name: str, b_name: str, midpath: str = "") -> Tuple[Path, Path]:
    """
    Try two common layouts:
      1) ROOT/a_name/scene , ROOT/b_name/scene
      2) ROOT/scene/(midpath)/a_name , ROOT/scene/(midpath)/b_name
    """
    cand1_a = batch_root / a_name / scene
    cand1_b = batch_root / b_name / scene
    if midpath:
        cand2_a = batch_root / scene / midpath / a_name
        cand2_b = batch_root / scene / midpath / b_name
    else:
        cand2_a = batch_root / scene / a_name
        cand2_b = batch_root / scene / b_name
    if cand1_a.is_dir() and cand1_b.is_dir():
        return cand1_a, cand1_b
    if cand2_a.is_dir() and cand2_b.is_dir():
        return cand2_a, cand2_b
    raise FileNotFoundError(f"Cannot resolve scene '{scene}'. Tried: {cand1_a} vs {cand1_b}, and {cand2_a} vs {cand2_b}.")


def parse_scenes_list(s: str) -> List[str]:
    if not s:
        return []
    return [t for t in s.replace(',', ' ').split() if t.strip()]


def main():
    parser = argparse.ArgumentParser(description="Compute error maps for same-named images in two folders, with optional batch mode over scenes.")
    # Single-pair positional (optional if using batch mode)
    parser.add_argument("dirA", nargs="?", type=str, help="First directory (reference).")
    parser.add_argument("dirB", nargs="?", type=str, help="Second directory (comparison).")
    parser.add_argument("out_dir", nargs="?", type=str, help="Output directory to save error maps.")

    # Batch mode
    parser.add_argument("--batch-root", type=str, default=None, help="Root directory containing scene subfolders.")
    parser.add_argument("--scenes", type=str, default=None, help="Scene names (comma/space separated).")
    parser.add_argument("--a-name", type=str, default="A", help="Name of the A subfolder (e.g., gt, ref, A).")
    parser.add_argument("--b-name", type=str, default="B", help="Name of the B subfolder (e.g., pred, out, B).")
    parser.add_argument("--output-root", type=str, default=None, help="Output root for batch mode. Defaults to <batch-root>/_error_maps")
    parser.add_argument("--midpath", type=str, default="", help="Optional middle subpath between <scene> and <a/b> in scene-first layout, e.g., 'test'.")

    # Common options
    parser.add_argument("--mode", type=str, default="abs", choices=["abs", "l2"], help="Error mode: abs (MAE) or l2 (RMSE).")
    parser.add_argument("--colormap", action="store_true", help="Save colored heatmap (requires matplotlib).")
    parser.add_argument("--resize", action="store_true", help="Resize images from dirB to match dirA sizes if mismatched.")
    parser.add_argument("--suffix", type=str, default="_err", help="Suffix for output filenames before extension.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (0 for single-process).")

    args = parser.parse_args()

    # Batch mode
    if args.batch_root:
        batch_root = Path(args.batch_root).expanduser().resolve()
        if not batch_root.is_dir():
            print(f"[ERROR] Invalid --batch-root: {batch_root}", file=sys.stderr)
            sys.exit(2)
        scenes = parse_scenes_list(args.scenes) or ["ship", "drums", "ficus", "hotdog", "lego", "materials", "mic", "chair"]
        output_root = Path(args.output_root).expanduser().resolve() if args.output_root else (batch_root / "_error_maps")
        output_root.mkdir(parents=True, exist_ok=True)

        total_ok = 0
        reports: List[str] = []
        for sc in scenes:
            try:
                dirA, dirB = resolve_scene_paths(batch_root, sc, args.a_name, args.b_name, args.midpath)
            except FileNotFoundError as e:
                print(f"[WARN] {e}", file=sys.stderr)
                continue
            out_dir = output_root / sc
            ok, report_path = run_single(dirA, dirB, out_dir, args.mode, args.colormap, args.resize, args.suffix, args.workers)
            total_ok += ok
            reports.append(str(report_path))
            print(f"[INFO] Scene '{sc}': saved {ok} error maps to {out_dir}")

        index_path = output_root / "index.txt"
        with open(index_path, "w", encoding="utf-8") as f:
            f.write("\n".join(reports))
        print(f"Done. Total saved {total_ok} error maps under: {output_root}")
        print(f"Reports listed in: {index_path}")
        return

    # Single mode
    if not args.dirA or not args.dirB or not args.out_dir:
        print("[ERROR] Single mode requires: dirA dirB out_dir (or use --batch-root)", file=sys.stderr)
        sys.exit(2)

    dirA = Path(args.dirA).expanduser().resolve()
    dirB = Path(args.dirB).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not dirA.is_dir() or not dirB.is_dir():
        print(f"[ERROR] Invalid input directories:\n  dirA: {dirA}\n  dirB: {dirB}", file=sys.stderr)
        sys.exit(2)

    ok, report_path = run_single(dirA, dirB, out_dir, args.mode, args.colormap, args.resize, args.suffix, args.workers)
    print(f"Done. Saved {ok} error maps to: {out_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

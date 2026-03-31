import os
import re
import ants
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import clear_output
from nibabel.processing import resample_from_to
from nilearn import plotting
from scipy.ndimage import binary_fill_holes
import subprocess
import shutil


# ──────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────


def metabolite_name(filename: str):

    m = re.search(r"desc-([^_]+)_mrsi", filename)
    return m.group(1) if m else filename

def get_nonzero_com(img: nib.Nifti1Image):

    data = img.get_fdata()
    mask = np.isfinite(data) & (data > 0)
    if not mask.any():
        return (0.0, 0.0, 0.0)
    ijk = np.argwhere(mask).mean(axis=0)
    xyz = nib.affines.apply_affine(img.affine, ijk)
    return tuple(float(v) for v in xyz[:3])

def estimate_coverage(
    mrs_resampled_img: nib.Nifti1Image,
    t1_img: nib.Nifti1Image):
    
    mrs = mrs_resampled_img.get_fdata()
    t1  = t1_img.get_fdata()
    brain   = np.isfinite(t1) & (t1 > 0)
    covered = np.isfinite(mrs) & (mrs > 0)
    frac = float(np.mean(covered[brain])) if brain.any() else 0.0
    print(f"  Brain coverage: {frac * 100:.1f}%")
    return frac

def _nib_to_ants(data: np.ndarray, ref_img: nib.Nifti1Image):
 
    aff       = ref_img.affine
    spacing   = np.sqrt(np.sum(np.square(aff[:3, :3]), axis=0))
    origin    = aff[:3, 3].copy()
    direction = aff[:3, :3] / spacing
    direction[:2, :] *= -1   # RAS → LPS
    origin[:2]       *= -1   # RAS → LPS
    return ants.from_numpy(
        data,
        origin=origin.tolist(),
        spacing=spacing.tolist(),
        direction=direction.tolist(),
    )

def _mutual_information(x: np.ndarray, y: np.ndarray, bins: int = 32):
    
    hist_2d, _, _ = np.histogram2d(x, y, bins=bins)
    hist_2d = hist_2d + 1e-10
    pxy = hist_2d / hist_2d.sum()
    px  = pxy.sum(axis=1)
    py  = pxy.sum(axis=0)
    hx  = -np.sum(px  * np.log(px))
    hy  = -np.sum(py  * np.log(py))
    hxy = -np.sum(pxy * np.log(pxy))
    return float(2.0 * (hx + hy - hxy) / (hx + hy + 1e-10))


def _fsl_reorient2std(in_path: str) -> nib.Nifti1Image:
    """Reorient a NIfTI to standard (RAS) orientation using fslreorient2std.

    Preferred over nibabel.as_closest_canonical because FSL also corrects the
    qform/sform codes and matches the behaviour expected by antsRegistration.
    """
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    # Copy the input into the temp dir so FSL only sees one version of the
    # file (avoids the "Could not find image" error that occurs when both a
    # .nii and a .nii.gz with the same base name coexist in the source dir).
    tmp_in  = os.path.join(tmp_dir, "input.nii.gz")
    tmp_out = os.path.join(tmp_dir, "reoriented.nii.gz")
    try:
        shutil.copy2(in_path, tmp_in)
        subprocess.run(
            ["fslreorient2std", tmp_in, tmp_out],
            check=True, capture_output=True,
        )
        img = nib.load(tmp_out)
        # Detach from the temp file before cleanup
        img = nib.Nifti1Image(img.get_fdata(), img.affine, img.header)
        return img
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"fslreorient2std failed for {in_path}:\n{e.stderr.decode()}"
        ) from e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────────
# Data utilities
# ──────────────────────────────────────────────────────────────────────────

def img_info(img: nib.Nifti1Image, label: str):

    vox = np.sqrt(np.sum(img.affine[:3, :3] ** 2, axis=0))
    print(f"{label}")
    print(f"  shape      : {img.shape}")
    print(f"  voxel size : {vox.round(3)} mm")
    print(f"  dtype      : {img.get_data_dtype()}")
    print()

def fill_mask_holes(
    water_img_path: str,
    out_mask_path: str | None = None,
    overwrite: bool = False):

    water_img = nib.load(water_img_path)
    water_data = np.asarray(water_img.get_fdata(), dtype=np.float32)

    # Cut the bottom 5% of positive water-signal values (noise floor) and
    # keep the top 95% → threshold at the 5th percentile of positive voxels.
    finite_vals = water_data[np.isfinite(water_data) & (water_data > 0)]
    threshold = float(np.percentile(finite_vals, 5)) if finite_vals.size > 0 else 0.0
    base_mask = np.isfinite(water_data) & (water_data > threshold)
    # Fill holes slice-by-slice (axial = last axis) so that a hole open in
    # one slice but closed in a neighbouring slice is not erroneously filled.
    filled_mask = np.stack(
        [binary_fill_holes(base_mask[..., z]) for z in range(base_mask.shape[2])],
        axis=2,
    )

    mask_data = filled_mask.astype(np.uint8)
    mask_img = nib.Nifti1Image(mask_data, water_img.affine, water_img.header)
    mask_img.set_data_dtype(np.uint8)

    if out_mask_path and (overwrite or not os.path.exists(out_mask_path)):
        nib.save(mask_img, out_mask_path)

    return mask_img

def rank_metabolites_by_snr(
    mrs_dir: str,
    subj: str = "sub-01",
    ses: str = "ses-01",
    top_n: int | None = None):
    
    snr_candidates = [
        f for f in os.listdir(mrs_dir)
        if "VoxelSNR" in f and f.endswith(".nii.gz")
    ]
    
    snr_map = nib.load(os.path.join(mrs_dir, snr_candidates[0])).get_fdata()

    # collect all individual OrigRes concentration maps
    conc_files = sorted(
        f for f in os.listdir(mrs_dir)
        if f.endswith(".nii.gz")
        and "acq-OrigRes" in f
        and "AllMetabSum" not in f)

    records = []
    for fname in conc_files:
        data = nib.load(os.path.join(mrs_dir, fname)).get_fdata()
        # support mask: voxels with positive concentration AND positive SNR
        mask = (data > 0) & (snr_map > 0)
        if not mask.any():
            continue
        pos_data = data[mask]
        snr_vals = snr_map[mask]
        records.append({
            "filename":   fname,
            "metabolite": metabolite_name(fname),
            "mean_snr":   float(np.mean(snr_vals)),
            "median_snr": float(np.median(snr_vals)),
            "n_voxels":   int(mask.sum()),
            "cv":         float(np.std(pos_data) / (np.mean(pos_data) + 1e-12)),
            "subj":       subj,
            "ses":        ses})

    records.sort(key=lambda r: r["mean_snr"], reverse=True)
    for i, r in enumerate(records):
        r["rank"] = i + 1

    return records[:top_n] if top_n is not None else records

def save_metabolite_sum(
    bids_dir: str,
    ses: str = "ses-01",
    overwrite: bool = False,
    out_dir: str | None = None):
   
    saved = {}

    for subj in sorted(os.listdir(bids_dir)):
        if not subj.startswith("sub-"):
            continue

        mrs_dir  = os.path.join(bids_dir, subj, ses, "mrs")
        if not os.path.isdir(mrs_dir):
            print(f"  [sum] {subj}: mrs/ folder not found, skipping.")
            continue

        out_name = f"{subj}_{ses}_acq-OrigRes_desc-AllMetabSum_mrsi.nii.gz"
        save_dir = out_dir if out_dir is not None else mrs_dir
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, out_name)

        if os.path.exists(out_path) and not overwrite:
            print(f"  [sum] {subj}: already exists  {out_name}")
            saved[subj] = out_path
            continue

        # collect all individual metabolite maps 
        maps = sorted(
            f for f in os.listdir(mrs_dir)
            if f.endswith(".nii.gz")
            and "acq-OrigRes" in f
            and "AllMetabSum" not in f
        )
        if not maps:
            print(f"  [sum] {subj}: no OrigRes maps found, skipping.")
            continue

        # use first file as the affine/header reference
        ref_img = nib.load(os.path.join(mrs_dir, maps[0]))
        accum   = np.zeros(ref_img.shape, dtype=np.float32)

        n_used = 0
        for fname in maps:
            data = nib.load(os.path.join(mrs_dir, fname)).get_fdata().astype(np.float32)
            accum += np.nan_to_num(data, nan=0.0)
            n_used += 1

        sum_img = nib.Nifti1Image(accum, ref_img.affine, ref_img.header)
        sum_img.set_data_dtype(np.float32)
        nib.save(sum_img, out_path)
        print(f"  [sum] {subj}: saved {out_name}  ({n_used} maps summed)")
        saved[subj] = out_path

    return saved

def save_reoriented_metabolite_sum(
    bids_dir: str,
    ses: str = "ses-01",
    overwrite: bool = False,
    out_dir: str | None = None):
    
    saved = {}

    for subj in sorted(os.listdir(bids_dir)):
        if not subj.startswith("sub-"):
            continue

        mrs_dir = os.path.join(bids_dir, subj, ses, "mrs")
        if not os.path.isdir(mrs_dir):
            print(f"  [sum-ras] {subj}: mrs/ folder not found, skipping.")
            continue

        out_name = f"{subj}_{ses}_acq-OrigRes_desc-AllMetabSumRAS_mrsi.nii.gz"
        save_dir = out_dir if out_dir is not None else mrs_dir
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, out_name)

        if os.path.exists(out_path) and not overwrite:
            print(f"  [sum-ras] {subj}: already exists  {out_name}")
            saved[subj] = out_path
            continue

        maps = sorted(
            f for f in os.listdir(mrs_dir)
            if f.endswith(".nii.gz")
            and "acq-OrigRes" in f
            and "AllMetab" not in f
        )
        if not maps:
            print(f"  [sum-ras] {subj}: no OrigRes maps found, skipping.")
            continue

        ref_img = _fsl_reorient2std(os.path.join(mrs_dir, maps[0]))
        accum   = np.zeros(ref_img.shape, dtype=np.float32)

        n_used = 0
        for fname in maps:
            img  = _fsl_reorient2std(os.path.join(mrs_dir, fname))
            data = img.get_fdata().astype(np.float32)
            if data.shape != accum.shape:
                print(f"  [sum-ras] {subj}: shape mismatch in {fname}, skipping.")
                continue
            accum += np.nan_to_num(data, nan=0.0)
            n_used += 1

        sum_img = nib.Nifti1Image(accum, ref_img.affine, ref_img.header)
        sum_img.set_data_dtype(np.float32)
        nib.save(sum_img, out_path)
        print(f"  [sum-ras] {subj}: saved {out_name}  ({n_used} maps reoriented and summed)")
        saved[subj] = out_path

    return saved

def downsample_t1w_to_mrs(
    bids_dir: str,
    ses: str = "ses-01",
    t1w_acq: str = "UNIDEN",
    overwrite: bool = False,
    out_dir: str | None = None):
    
    saved = {}

    for subj in sorted(os.listdir(bids_dir)):
        if not subj.startswith("sub-"):
            continue

        anat_dir = os.path.join(bids_dir, subj, ses, "anat")
        mrs_dir  = os.path.join(bids_dir, subj, ses, "mrs")

        if not os.path.isdir(anat_dir):
            continue
        if not os.path.isdir(mrs_dir):
            print(f"  [t1w-ds] {subj}: no mrs/ folder, skipping.")
            continue

        out_name = f"{subj}_{ses}_acq-MRSIres_T1w.nii.gz"
        save_dir = out_dir if out_dir is not None else anat_dir
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, out_name)

        if os.path.exists(out_path) and not overwrite:
            print(f"  [t1w-ds] {subj}: already exists  {out_name}")
            saved[subj] = out_path
            continue

        # locate the T1w source
        t1w_name = f"{subj}_{ses}_acq-{t1w_acq}_T1w.nii"
        t1w_path = os.path.join(anat_dir, t1w_name)
        if not os.path.exists(t1w_path):
                print(f"  [t1w-ds] {subj}: T1w not found ({t1w_name}), skipping.")
                continue

        # pick the first OrigRes MRSI map as the target grid
        mrs_maps = sorted(
            f for f in os.listdir(mrs_dir)
            if f.endswith(".nii.gz") and "acq-OrigRes" in f and "AllMetabSum" not in f
        )
        if not mrs_maps:
            print(f"  [t1w-ds] {subj}: no OrigRes MRSI maps found, skipping.")
            continue

        mrs_ref_img = nib.load(os.path.join(mrs_dir, mrs_maps[0]))
        t1w_img     = nib.load(t1w_path)

        # resample T1w to MRSI grid (order=1: linear interpolation)
        t1w_ds = resample_from_to(t1w_img, mrs_ref_img, order=1)
        t1w_ds = nib.Nifti1Image(
            np.array(t1w_ds.dataobj, dtype=np.float32),
            t1w_ds.affine,
            t1w_ds.header,
        )
        t1w_ds.set_data_dtype(np.float32)
        nib.save(t1w_ds, out_path)
        print(
            f"  [t1w-ds] {subj}: saved {out_name}  "
            f"(T1w {t1w_img.shape} to MRS grid {mrs_ref_img.shape})"
        )
        saved[subj] = out_path

    return saved


def skull_strip_t1w(
    in_path: str,
    out_path: str,
    frac: float = 0.5,
    overwrite: bool = False):

    if os.path.exists(out_path) and not overwrite:
        print(f"  [bet] already exists: {out_path}")
        return nib.load(out_path)

    cmd = ["bet", in_path, out_path, "-f", str(frac), "-g", "0", "-m"]
    print(f"  [bet] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"  [bet] saved: {out_path}")
    return nib.load(out_path)


def segment_t1w(
    brain_path: str,
    out_prefix: str,
    n_classes: int = 3,
    overwrite: bool = False,
) -> dict:
    """Segment a skull-stripped T1w image into CSF / GM / WM using FSL FAST.

    Parameters
    ----------
    brain_path : str
        Path to the skull-stripped T1w NIfTI (e.g. output of skull_strip_t1w).
    out_prefix : str
        Base path for FAST output files (no extension).
        FAST writes e.g. ``<out_prefix>_pve_0.nii.gz`` (CSF),
        ``_pve_1.nii.gz`` (GM), ``_pve_2.nii.gz`` (WM),
        ``_seg.nii.gz`` (hard segmentation).
    n_classes : int
        Number of tissue classes (default 3: CSF/GM/WM).
    overwrite : bool
        Re-run FAST even if outputs already exist.

    Returns
    -------
    dict with keys ``csf``, ``gm``, ``wm``, ``seg`` — each a loaded
    ``nib.Nifti1Image``, or ``None`` if that file is missing.
    """
    seg_path = f"{out_prefix}_seg.nii.gz"
    pve0     = f"{out_prefix}_pve_0.nii.gz"

    if os.path.exists(seg_path) and os.path.exists(pve0) and not overwrite:
        print(f"  [fast] already exists: {os.path.basename(seg_path)}")
    else:
        cmd = [
            "fast",
            "-n", str(n_classes),
            "-t", "1",          # T1-weighted
            "-o", out_prefix,
            brain_path,
        ]
        print(f"  [fast] running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"  [fast] done → {out_prefix}_pve_{{0,1,2}}.nii.gz")

    def _load(path):
        return nib.load(path) if os.path.exists(path) else None

    return {
        "csf": _load(f"{out_prefix}_pve_0.nii.gz"),
        "gm":  _load(f"{out_prefix}_pve_1.nii.gz"),
        "wm":  _load(f"{out_prefix}_pve_2.nii.gz"),
        "seg": _load(f"{out_prefix}_seg.nii.gz"),
    }


def segment_t1w_atlas(
    brain_path: str,
    out_prefix: str,
    nonlinear: bool = False,
    overwrite: bool = False,
) -> dict:
    """Register a skull-stripped T1w to MNI152 and backproject Harvard-Oxford atlas labels.

    Uses FLIRT (12 DOF affine, normmi cost) to register the brain image to the
    MNI152 2mm template, inverts the transform with ``convert_xfm``, then
    applies the inverse to the Harvard-Oxford cortical and subcortical max-prob
    atlases (25% probability threshold, 2 mm resolution), bringing both
    parcellations into native T1w space.

    If ``nonlinear=True`` a subsequent FNIRT warp is estimated and inverted with
    ``invwarp``/``applywarp`` for improved accuracy.

    Parameters
    ----------
    brain_path : str
        Path to the skull-stripped T1w NIfTI (output of :func:`skull_strip_t1w`).
    out_prefix : str
        Base path for output files.  Results are written as
        ``<out_prefix>_atlas_cort.nii.gz``, ``_atlas_sub.nii.gz``,
        ``_T1w_in_MNI.nii.gz``, ``_t1w2mni.mat``.
    nonlinear : bool
        If ``True``, run FNIRT after FLIRT for a nonlinear warp. Default False.
    overwrite : bool
        Re-run registration even if outputs already exist.

    Returns
    -------
    dict with keys:
        ``atlas_cort``  — Harvard-Oxford cortical labels in native T1w space.
        ``atlas_sub``   — Harvard-Oxford subcortical labels in native T1w space.
        ``t1w_in_mni``  — T1w registered to MNI152 (QC image).
        ``labels_cort`` — list[str] of region names, index = label value.
        ``labels_sub``  — same for subcortical atlas.
    """
    import xml.etree.ElementTree as ET

    fsldir    = os.environ.get("FSLDIR", "/usr/local/fsl")
    mni_brain = os.path.join(fsldir, "data", "standard", "MNI152_T1_2mm_brain.nii.gz")
    mni_head  = os.path.join(fsldir, "data", "standard", "MNI152_T1_2mm.nii.gz")
    atlas_cort_mni = os.path.join(fsldir, "data", "atlases", "HarvardOxford",
                                  "HarvardOxford-cort-maxprob-thr25-2mm.nii.gz")
    atlas_sub_mni  = os.path.join(fsldir, "data", "atlases", "HarvardOxford",
                                  "HarvardOxford-sub-maxprob-thr25-2mm.nii.gz")
    xml_cort = os.path.join(fsldir, "data", "atlases", "HarvardOxford-Cortical.xml")
    xml_sub  = os.path.join(fsldir, "data", "atlases", "HarvardOxford-Subcortical.xml")

    out_cort    = f"{out_prefix}_atlas_cort.nii.gz"
    out_sub     = f"{out_prefix}_atlas_sub.nii.gz"
    out_t1w_mni = f"{out_prefix}_T1w_in_MNI.nii.gz"
    out_xfm     = f"{out_prefix}_t1w2mni.mat"
    out_inv_xfm = f"{out_prefix}_mni2t1w.mat"

    if (os.path.exists(out_cort) and os.path.exists(out_sub)
            and os.path.exists(out_t1w_mni) and not overwrite):
        print(f"  [atlas] already exists: {os.path.basename(out_cort)}")
    else:
        print("  [atlas] FLIRT: registering T1w to MNI152 (12 DOF, normmi)…")
        subprocess.run([
            "flirt",
            "-in",     brain_path,
            "-ref",    mni_brain,
            "-out",    out_t1w_mni,
            "-omat",   out_xfm,
            "-dof",    "12",
            "-cost",   "normmi",
            "-interp", "spline",
        ], check=True, capture_output=True)

        if nonlinear:
            out_warp     = f"{out_prefix}_t1w2mni_warp.nii.gz"
            out_inv_warp = f"{out_prefix}_mni2t1w_warp.nii.gz"
            print("  [atlas] FNIRT: nonlinear warp to MNI152…")
            subprocess.run([
                "fnirt",
                f"--in={brain_path}",
                f"--aff={out_xfm}",
                f"--cout={out_warp}",
                f"--ref={mni_head}",
                "--config=T1_2_MNI152_2mm",
            ], check=True, capture_output=True)
            print("  [atlas] invwarp: inverting nonlinear warp…")
            subprocess.run([
                "invwarp",
                f"-w={out_warp}",
                f"-o={out_inv_warp}",
                f"-r={brain_path}",
            ], check=True, capture_output=True)
            for atlas_in, atlas_out in [
                (atlas_cort_mni, out_cort),
                (atlas_sub_mni,  out_sub),
            ]:
                subprocess.run([
                    "applywarp",
                    f"-i={atlas_in}",
                    f"-r={brain_path}",
                    f"-w={out_inv_warp}",
                    f"-o={atlas_out}",
                    "--interp=nn",
                ], check=True, capture_output=True)
        else:
            subprocess.run([
                "convert_xfm",
                "-omat",    out_inv_xfm,
                "-inverse", out_xfm,
            ], check=True, capture_output=True)
            for atlas_in, atlas_out in [
                (atlas_cort_mni, out_cort),
                (atlas_sub_mni,  out_sub),
            ]:
                subprocess.run([
                    "flirt",
                    "-in",    atlas_in,
                    "-ref",   brain_path,
                    "-out",   atlas_out,
                    "-init",  out_inv_xfm,
                    "-applyxfm",
                    "-interp", "nearestneighbour",
                ], check=True, capture_output=True)

        print(f"  [atlas] done → {os.path.basename(out_cort)}, {os.path.basename(out_sub)}")

    def _parse_labels(xml_path):
        """Return list[str] indexed from 0 (background) through N (last region)."""
        labels = ["Background"]
        try:
            tree = ET.parse(xml_path)
            for el in sorted(tree.iter("label"), key=lambda e: int(e.get("index", 0))):
                labels.append(el.text.strip() if el.text else f"Region {el.get('index')}")
        except Exception:
            pass
        return labels

    def _load(p):
        return nib.load(p) if os.path.exists(p) else None

    return {
        "atlas_cort":  _load(out_cort),
        "atlas_sub":   _load(out_sub),
        "t1w_in_mni":  _load(out_t1w_mni),
        "labels_cort": _parse_labels(xml_cort),
        "labels_sub":  _parse_labels(xml_sub),
    }


# ──────────────────────────────────────────────────────────────────────────
# Registration utilities
# ──────────────────────────────────────────────────────────────────────────

def register_mrsi_to_t1w(
    mrsi_img: nib.Nifti1Image,
    t1w_img: nib.Nifti1Image,
    mask: np.ndarray,
    out_path: str | None = None,
    overwrite: bool = False,
    init_transforms: list | None = None):

    # Sidecar file that persists the forward transform next to the NIfTI
    transform_sidecar = (
        out_path.replace(".nii.gz", "_fwdtransform.mat") if out_path else None
    )

    if out_path and os.path.exists(out_path) and not overwrite:
        print(f"  [reg] already exists: {out_path}")
        saved_t = (
            [transform_sidecar]
            if transform_sidecar and os.path.exists(transform_sidecar)
            else None
        )
        return nib.load(out_path), saved_t

    mrsi_data   = mrsi_img.get_fdata().astype(np.float32)
    mrsi_masked = np.where(mask, mrsi_data, 0.0)
    t1w_data    = t1w_img.get_fdata().astype(np.float32)

    fixed_ants  = _nib_to_ants(t1w_data, t1w_img)
    moving_ants = _nib_to_ants(mrsi_masked, mrsi_img)

    if init_transforms is not None:
        fwd_transforms = init_transforms
        print(f"  [reg] applying pre-computed transforms (skipping optimisation)")
    else:
        result = ants.registration(
            fixed=fixed_ants,
            moving=moving_ants,
            type_of_transform="Rigid",
            verbose=False,
        )
        fwd_transforms = result["fwdtransforms"]

    if transform_sidecar and fwd_transforms:
        src = os.path.abspath(fwd_transforms[0])
        dst = os.path.abspath(transform_sidecar)
        if src != dst:
            shutil.copy(src, dst)
        used_transforms = [transform_sidecar]
    else:
        used_transforms = fwd_transforms

    # Apply transform to the  MRSI 
    warped_ants = ants.apply_transforms(
        fixed=fixed_ants,
        moving=_nib_to_ants(mrsi_data, mrsi_img),
        transformlist=fwd_transforms,
        interpolator="linear")

    # Warp the mask into T1w space so we can zero out voxels 
    warped_mask_ants = ants.apply_transforms(
        fixed=fixed_ants,
        moving=_nib_to_ants(mask.astype(np.float32), mrsi_img),
        transformlist=fwd_transforms,
        interpolator="nearestNeighbor")
    brain_mask_t1w = warped_mask_ants.numpy() > 0.5

    reg_data = warped_ants.numpy().astype(np.float32)
    reg_data[~brain_mask_t1w] = 0.0

    reg_img = nib.Nifti1Image(reg_data, t1w_img.affine, t1w_img.header)
    reg_img.set_data_dtype(np.float32)

    if out_path:
        nib.save(reg_img, out_path)
        print(f"  [reg] saved registered MRSI: {out_path}")

    return reg_img, used_transforms

def register_t1w_to_mrsi(
    t1w_img: nib.Nifti1Image,
    mrsi_img: nib.Nifti1Image,
    out_path: str | None = None,
    overwrite: bool = False,
    init_transforms: list | None = None):

    transform_sidecar = (
        out_path.replace(".nii.gz", "_fwdtransform.mat") if out_path else None
    )

    if out_path and os.path.exists(out_path) and not overwrite:
        print(f"  [inv-reg] already exists: {out_path}")
        saved_t = (
            [transform_sidecar]
            if transform_sidecar and os.path.exists(transform_sidecar)
            else None
        )
        return nib.load(out_path), saved_t

    t1w_data  = t1w_img.get_fdata().astype(np.float32)
    mrsi_data = mrsi_img.get_fdata().astype(np.float32)

    fixed_ants  = _nib_to_ants(mrsi_data, mrsi_img)
    moving_ants = _nib_to_ants(t1w_data,  t1w_img)

    if init_transforms is not None:
        fwd_transforms = init_transforms
        print("  [inv-reg] applying pre-computed transforms (skipping optimisation)")
    else:
        result = ants.registration(
            fixed=fixed_ants,
            moving=moving_ants,
            type_of_transform="Rigid",
            verbose=False,
        )
        fwd_transforms = result["fwdtransforms"]

    if transform_sidecar and fwd_transforms:
        src = os.path.abspath(fwd_transforms[0])
        dst = os.path.abspath(transform_sidecar)
        if src != dst:
            shutil.copy(src, dst)
        used_transforms = [transform_sidecar]
    else:
        used_transforms = fwd_transforms

    warped_ants = ants.apply_transforms(
        fixed=fixed_ants,
        moving=moving_ants,
        transformlist=fwd_transforms,
        interpolator="linear",
    )

    reg_data = warped_ants.numpy().astype(np.float32)
    reg_img  = nib.Nifti1Image(reg_data, mrsi_img.affine, mrsi_img.header)
    reg_img.set_data_dtype(np.float32)

    if out_path:
        nib.save(reg_img, out_path)
        print(f"  [inv-reg] saved T1w in MRSI space: {out_path}")

    return reg_img, used_transforms

def register_t1w_to_mrsi_weighted(
    fixed_path: str,
    moving_path: str,
    mask_path: str,
    out_path: str,
    transform_path: str,
    overwrite: bool = False,
    init_transforms: list | None = None,
    moving_mask_path: str | None = None,
    init_from_path: str | None = None):

    if os.path.exists(out_path) and not overwrite:
        print(f"[weighted-reg] already exists: {out_path}")
        transforms = [transform_path] if os.path.exists(transform_path) else None
        return nib.load(out_path), transforms

    if init_transforms is not None:
        cmd = [
            "antsApplyTransforms",
            "--dimensionality", "3",
            "--input", moving_path,
            "--reference-image", fixed_path,
            "--output", out_path,
            "--interpolation", "Linear",
        ]
        for t in init_transforms:
            cmd.extend(["--transform", t])

        print("[weighted-reg] applying pre-computed transforms...")
        subprocess.run(cmd, check=True)
        reg_img = nib.load(out_path)
        print(f"[weighted-reg] saved image: {out_path}")
        return reg_img, init_transforms

    out_prefix = out_path.replace(".nii.gz", "_ants_")

    cmd = [
        "antsRegistration",
        "--dimensionality", "3",
        "--float", "1",
        "--output", f"[{out_prefix},{out_prefix}Warped.nii.gz]",
        "--interpolation", "Linear",
        # ANTs does not accept "NULL" as a placeholder — omit the second
        # entry entirely when no moving-image mask is available.
        "--masks", "[{},{}]".format(mask_path, moving_mask_path) if moving_mask_path else "[{}]".format(mask_path),
        # Seed from a pre-computed transform when provided (avoids CoM
        # instability for skull-stripped images whose CoM differs from
        # the MRSI signal CoM), otherwise fall back to CoM alignment.
        "--initial-moving-transform",
            init_from_path if init_from_path else f"[{fixed_path},{moving_path},1]",

        "--transform", "Rigid[0.1]",
        # Mattes MI is voxel-wise (no neighborhood) so it works correctly
        # at MRSI resolution (~10 mm voxels, ~9 brain voxels total) where
        # CC always fails with "No valid points".  32 histogram bins is
        # standard for same-modality; 80% random sampling balances speed/accuracy.
        "--metric", f"Mattes[{fixed_path},{moving_path},1,32,Random,0.8]",
        "--convergence", "[1000x500,1e-6,10]",
        "--shrink-factors", "2x1",
        "--smoothing-sigmas", "1x0vox",
    ]

    print("[weighted-reg] running antsRegistration...")
    subprocess.run(cmd, check=True)

    warped_path = f"{out_prefix}Warped.nii.gz"
    affine_path = f"{out_prefix}0GenericAffine.mat"

    shutil.move(warped_path, out_path)
    shutil.move(affine_path, transform_path)

    reg_img = nib.load(out_path)
    print(f"[weighted-reg] saved image: {out_path}")

    return reg_img, [transform_path]

def apply_transform_to_metabolite(
    mrsi_path: str,
    transform_path: str,
    t1w_ref_path: str,
    out_path: str,
    overwrite: bool = False): 

    if os.path.exists(out_path) and not overwrite:
        print(f"  [reg17] already exists: {out_path}")
        return nib.load(out_path)

    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    label   = metabolite_name(mrsi_path)
    ras_tmp = os.path.join(out_dir, f"_tmp_{label}_ras.nii.gz")
    nib.save(nib.as_closest_canonical(nib.load(mrsi_path)), ras_tmp)

    cmd = [
        "antsApplyTransforms", "-d", "3",
        "-i", ras_tmp,
        "-r", t1w_ref_path,
        "-t", f"[{transform_path},1]",  
        "-n", "Linear",
        "-o", out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"  [reg17] {label}: saved to {os.path.basename(out_path)}")
        return nib.load(out_path)
    except subprocess.CalledProcessError as e:
        print(f"  [reg17] {label}: FAILED\n{e.stderr.decode()}")
        return None
    finally:
        if os.path.exists(ras_tmp):
            os.remove(ras_tmp)

def apply_transform_all_metabolites(
    mrs_dir: str,
    transform_path: str,
    t1w_ref_path: str,
    out_dir: str,
    subj: str = "sub-01",
    ses: str = "ses-01",
    overwrite: bool = False):
   
    maps = sorted(
        f for f in os.listdir(mrs_dir)
        if f.endswith(".nii.gz") and "acq-OrigRes" in f and "AllMetab" not in f
    )
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for fname in maps:
        label    = metabolite_name(fname)
        out_name = f"{subj}_{ses}_acq-MRSIres_desc-{label}Reg17_T1w.nii.gz"
        img = apply_transform_to_metabolite(
            mrsi_path=os.path.join(mrs_dir, fname),
            transform_path=transform_path,
            t1w_ref_path=t1w_ref_path,
            out_path=os.path.join(out_dir, out_name),
            overwrite=overwrite,
        )
        if img is not None:
            results[label] = img

    return results

def compute_registration_metrics(
    t1w_img: nib.Nifti1Image,
    reg_mrsi_imgs: dict):
    
    t1w_data   = t1w_img.get_fdata().astype(np.float32)
    brain_mask = t1w_data > 0
    n_brain    = int(brain_mask.sum())

    records = []
    for label, img in reg_mrsi_imgs.items():
        mrsi_data   = img.get_fdata().astype(np.float32)
        signal_mask = mrsi_data > 0
        overlap     = brain_mask & signal_mask
        coverage    = float(overlap.sum()) / n_brain if n_brain > 0 else 0.0

        if overlap.sum() > 20:
            t1w_vals  = t1w_data[overlap]
            mrsi_vals = mrsi_data[overlap]
            # NCC
            t1w_c = t1w_vals  - t1w_vals.mean()
            mrs_c = mrsi_vals - mrsi_vals.mean()
            denom = np.sqrt((t1w_c ** 2).sum() * (mrs_c ** 2).sum())
            ncc   = float((t1w_c * mrs_c).sum() / (denom + 1e-10))
            # NMI
            nmi = _mutual_information(t1w_vals, mrsi_vals)
        else:
            ncc = 0.0
            nmi = 0.0

        records.append({"label": label, "coverage": coverage, "ncc": ncc, "nmi": nmi})

    # Sort by |NCC| descending: strongest correlation magnitude = best structural
    # coherence with T1w, regardless of sign (anti-correlation is expected here).
    records.sort(key=lambda r: abs(r["ncc"]), reverse=True)
    return records

def run_total_pipeline(
    bids_dir: str,
    mrs_dir: str,
    water_path: str,
    t1w_ds_brain_path: str,
    t1w_ds_brain_img: "nib.Nifti1Image",
    subj: str,
    ses: str,
    output_dir: str,
    overwrite: bool = False,
    t1w_brain_mask_ds_path: str | None = None):

    # Step 1 – RAS canonical reoriented metabolite sum
    sum_ras_name = f"{subj}_{ses}_acq-OrigRes_desc-AllMetabSumRAS_mrsi.nii.gz"
    sum_ras_path = os.path.join(output_dir, sum_ras_name)
    save_reoriented_metabolite_sum(bids_dir, ses=ses, out_dir=output_dir, overwrite=overwrite)
    sum_ras_img = nib.load(sum_ras_path) if os.path.exists(sum_ras_path) else None

    # Step 2 – RAS canonical reoriented water signal
    water_ras_name = f"{subj}_{ses}_desc-WaterSignalRAS_mrsi.nii.gz"
    water_ras_path = os.path.join(output_dir, water_ras_name)
    if overwrite or not os.path.exists(water_ras_path):
        water_ras_img = _fsl_reorient2std(water_path)
        nib.save(water_ras_img, water_ras_path)
        print(f"  [ras] saved reoriented water: {water_ras_name}")
    # Binary mask of the RAS water signal — ANTs --masks expects 0/1 values,
    # not the raw continuous water signal intensity.
    water_ras_mask_name = f"{subj}_{ses}_desc-WaterSignalRASMask_mrsi.nii.gz"
    water_ras_mask_path = os.path.join(output_dir, water_ras_mask_name)
    fill_mask_holes(water_ras_path, out_mask_path=water_ras_mask_path, overwrite=overwrite)

    # Step 3  Reg-17: skull stripped DS T1w to reoriented sum, water-masked
    t1w_brain_in_sum_ras_name = f"{subj}_{ses}_acq-MRSIres_desc-BrainT1wInSumRAS_T1w.nii.gz"
    t1w_brain_in_sum_ras_path = os.path.join(output_dir, t1w_brain_in_sum_ras_name)
    t1w_brain_in_sum_ras_xfm  = t1w_brain_in_sum_ras_path.replace(".nii.gz", "_fwdtransform.mat")
    if sum_ras_img is not None:
        t1w_brain_in_sum_ras_img, brain_sum_ras_transforms = register_t1w_to_mrsi_weighted(
            fixed_path=sum_ras_path,
            moving_path=t1w_ds_brain_path,
            mask_path=water_ras_mask_path,
            out_path=t1w_brain_in_sum_ras_path,
            transform_path=t1w_brain_in_sum_ras_xfm,
            overwrite=overwrite,
            moving_mask_path=t1w_brain_mask_ds_path,
        )
    else:
        t1w_brain_in_sum_ras_img, brain_sum_ras_transforms = None, None

    # Step 4  Apply Reg-17 transform to every individual metabolite map
    final_reg_dir = os.path.join(output_dir, "final_reg")
    if brain_sum_ras_transforms is not None:
        final_reg_imgs = apply_transform_all_metabolites(
            mrs_dir=mrs_dir,
            transform_path=brain_sum_ras_transforms[0],
            t1w_ref_path=t1w_ds_brain_path,
            out_dir=final_reg_dir,
            subj=subj,
            ses=ses,
            overwrite=overwrite,
        )
    else:
        final_reg_imgs = {}
        print("  [pipeline] Step 4 skipped – Reg-17 transform not available.")

    # Step 5  Per-metabolite quality metrics in T1w space
    if final_reg_imgs:
        metrics = compute_registration_metrics(t1w_ds_brain_img, final_reg_imgs)
        print("\nRegistration quality metrics (Reg-17 total pipeline):")
        for r in metrics:
            print(f"  {r['label']:<28}  coverage={r['coverage']:.3f}"
                  f"  NCC={r['ncc']:+.4f}  NMI={r['nmi']:.4f}")
    else:
        metrics = []

    return t1w_brain_in_sum_ras_img, brain_sum_ras_transforms, final_reg_imgs, metrics



# ──────────────────────────────────────────────────────────────────────────
# Plot utilities
# ──────────────────────────────────────────────────────────────────────────

def plot_single_overlay(
    mrs_filename: str,
    t1w_path: str,
    mrs_dir: str,
    subj: str = "sub-01",
    cmap: str = "hot",
    threshold=None,
    vmax=None,
    verbose: bool = False):
    
    mrs_path = os.path.join(mrs_dir, mrs_filename)
    label    = metabolite_name(mrs_filename)

    t1_img  = nib.as_closest_canonical(nib.load(t1w_path))
    mrs_img = nib.as_closest_canonical(nib.load(mrs_path))

    if verbose:
        print(f"T1  orient: {nib.orientations.aff2axcodes(t1_img.affine)}")
        print(f"MRS orient: {nib.orientations.aff2axcodes(mrs_img.affine)}")

    mrs_res  = resample_from_to(mrs_img, t1_img, order=1)
    mrs_data = mrs_res.get_fdata()
    valid    = np.isfinite(mrs_data) & (mrs_data > 0)

    if vmax is None and valid.any():
        vmax = float(np.nanpercentile(mrs_data[valid], 99))
    if threshold is None and valid.any():
        threshold = float(np.nanpercentile(mrs_data[valid], 20))

    plotting.plot_stat_map(
        stat_map_img=mrs_res,
        bg_img=t1_img,
        title=f"{subj} - {label}",
        display_mode="ortho",
        cut_coords=get_nonzero_com(mrs_res),
        cmap=cmap,
        colorbar=True,
        threshold=threshold,
        vmax=vmax,
        resampling_interpolation="continuous",
    )
    plt.show()
    return estimate_coverage(mrs_res, t1_img)

def plot_support_mask(
    mrs_filename: str,
    t1w_path: str,
    mrs_dir: str,
    subj: str = "sub-01"):
    
    mrs_path = os.path.join(mrs_dir, mrs_filename)

    t1_img  = nib.as_closest_canonical(nib.load(t1w_path))
    mrs_img = nib.as_closest_canonical(nib.load(mrs_path))

    mask_data = (mrs_img.get_fdata() > 0).astype(np.float32)
    mask_img  = nib.Nifti1Image(mask_data, mrs_img.affine, mrs_img.header)
    mask_res  = resample_from_to(mask_img, t1_img, order=0)

    plotting.plot_roi(
        roi_img=mask_res,
        bg_img=t1_img,
        title=f"{subj} - MRSI support mask ({metabolite_name(mrs_filename)})",
        display_mode="ortho",
        cut_coords=get_nonzero_com(mask_res),
    )
    plt.show()
    return estimate_coverage(mask_res, t1_img)

def plot_coverage_grid(
    coverage_maps: list,
    mrs_dir: str,
    t1w_path: str,
    subj: str = "sub-01"):
    
    t1w_img  = nib.load(t1w_path)
    t1w_data = t1w_img.get_fdata()

    centre_ijk = [s // 2 for s in t1w_data.shape]
    centre_xyz = nib.affines.apply_affine(t1w_img.affine, centre_ijk)
    cut_coords = tuple(float(v) for v in centre_xyz[:3])

    for mrs_filename in coverage_maps:
        mrs_path = os.path.join(mrs_dir, mrs_filename)
        mrs_img  = nib.load(mrs_path)
        label    = metabolite_name(mrs_filename)
        data     = mrs_img.get_fdata()
        vmax     = float(np.nanpercentile(data[data > 0], 99)) if np.any(data > 0) else 1.0

        plotting.plot_stat_map(
            stat_map_img=mrs_img,
            bg_img=t1w_path,
            title=f"{subj} - {label}",
            display_mode="ortho",
            cut_coords=cut_coords,
            cmap="hot",
            colorbar=True,
            vmax=vmax,
            threshold=1e-6,
            resampling_interpolation="continuous",
        )
        plt.show()

def plot_snr_ranking(
    records: list[dict],
    top_n: int = 20,
    highlight_best: int = 3):
   
    shown   = records[:top_n]
    labels  = [r["metabolite"] for r in shown]
    means   = [r["mean_snr"]   for r in shown]
    medians = [r["median_snr"] for r in shown]
    n_vox   = [r["n_voxels"]   for r in shown]

    n = len(shown)
    y = np.arange(n)

    colours = [
        "#f4a261" if i < highlight_best else "#457b9d"
        for i in range(n)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n * 0.45)),
                             facecolor="#111111")

    # --- left panel: SNR bars ---
    ax = axes[0]
    ax.set_facecolor("#111111")
    bars = ax.barh(y, means, color=colours, edgecolor="none", height=0.6)
    ax.errorbar(medians, y, fmt="D", color="white", markersize=4,
                linewidth=0, label="median SNR", zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color="white", fontsize=9)
    ax.set_xlabel("Mean VoxelSNR (over metabolite support)", color="white", fontsize=10)
    ax.set_title("Metabolite ranking by SNR", color="white", fontsize=11, pad=8)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#444444")
    ax.invert_yaxis()
    ax.legend(facecolor="#333333", labelcolor="white", fontsize=8)

    # annotate bars with mean value
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", ha="left", color="white", fontsize=8,
        )

    # --- right panel: number of valid voxels ---
    ax2 = axes[1]
    ax2.set_facecolor("#111111")
    ax2.barh(y, n_vox, color=colours, edgecolor="none", height=0.6)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, color="white", fontsize=9)
    ax2.set_xlabel("Number of valid voxels (coverage)", color="white", fontsize=10)
    ax2.set_title("Spatial coverage per metabolite", color="white", fontsize=11, pad=8)
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#444444")
    ax2.invert_yaxis()

    subj = shown[0]["subj"] if shown else ""
    ses  = shown[0]["ses"]  if shown else ""
    fig.suptitle(
        f"{subj} {ses} – Metabolite SNR ranking  "
        f"({'all' if top_n >= len(records) else f'top {top_n}'} maps shown)",
        color="white", fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.show()

    # print the top-3 best
    print(f"\nTop-{highlight_best} metabolites by mean VoxelSNR:")
    for r in records[:highlight_best]:
        print(
            f"  #{r['rank']:2d}  {r['metabolite']:<25s}"
            f"  mean_SNR={r['mean_snr']:.2f}"
            f"  median_SNR={r['median_snr']:.2f}"
            f"  n_voxels={r['n_voxels']:,}"
            f"  CV={r['cv']:.3f}"
        )

def plot_t1w_mrs_comparison(
    t1w_ds_img: nib.Nifti1Image,
    mrs_img: nib.Nifti1Image,
    subj: str = "sub-01",
    mrs_label: str = "MRSI",
    t1w_cmap: str = "gray",
    mrs_cmap: str = "hot"):
   
    t1_data = t1w_ds_img.get_fdata()
    mrs_data = mrs_img.get_fdata()

    # normalise
    t1_norm  = np.clip(t1_data  / (np.nanpercentile(t1_data[t1_data  > 0], 99) or 1), 0, 1) \
               if np.any(t1_data > 0) else t1_data
    mrs_norm = np.clip(mrs_data / (np.nanpercentile(mrs_data[mrs_data > 0], 99) or 1), 0, 1) \
               if np.any(mrs_data > 0) else mrs_data

    mid = [s // 2 for s in t1_data.shape]

    def _slices(vol):
        return [
            vol[:, :, mid[2]],   # axial
            vol[:, mid[1], :],   # coronal
            vol[mid[0], :, :],   # sagittal
        ]

    t1_slices  = _slices(t1_norm)
    mrs_slices = _slices(mrs_norm)
    view_labels = ["Axial (z)", "Coronal (y)", "Sagittal (x)"]

    fig, axes = plt.subplots(3, 3, figsize=(13, 11), facecolor="black")
    row_labels = [f"T1w @ MRS res", mrs_label, "Overlay"]

    cmap_t1  = plt.get_cmap(t1w_cmap)
    cmap_mrs = plt.get_cmap(mrs_cmap)

    for col, (t1s, ms, vl) in enumerate(zip(t1_slices, mrs_slices, view_labels)):
        # Row 0 – T1w
        axes[0, col].imshow(t1s.T, origin="lower", cmap=t1w_cmap,
                            vmin=0, vmax=1, interpolation="nearest")
        axes[0, col].set_title(vl, color="white", fontsize=10)

        # Row 1 – MRSI
        masked_mrs = np.ma.masked_where(ms < 1e-6, ms)
        axes[1, col].imshow(t1s.T, origin="lower", cmap=t1w_cmap,
                            vmin=0, vmax=1, interpolation="nearest")
        axes[1, col].imshow(masked_mrs.T, origin="lower", cmap=mrs_cmap,
                            vmin=0, vmax=1, alpha=0.85, interpolation="nearest")

        # Row 2 – Overlay
        axes[2, col].imshow(t1s.T, origin="lower", cmap=t1w_cmap,
                            vmin=0, vmax=1, interpolation="nearest")
        axes[2, col].imshow(masked_mrs.T, origin="lower", cmap=mrs_cmap,
                            vmin=0, vmax=1, alpha=0.6, interpolation="nearest")

    for row, rl in enumerate(row_labels):
        for col in range(3):
            axes[row, col].set_facecolor("black")
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(rl, color="white", fontsize=10, rotation=90,
                                labelpad=6)

    fig.suptitle(
        f"{subj} – Downsampled T1w vs {mrs_label} (native MRS space, centre slices)",
        color="white", fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.show()

def plot_registration_comparison(
    t1w_img: nib.Nifti1Image,
    mrsi_img_unreg: nib.Nifti1Image,
    mrsi_reg_img: nib.Nifti1Image,
    subj: str = "sub-01",
    ses: str = "ses-01",
    mrsi_label: str = "MRSI",
    t1w_cmap: str = "gray",
    mrs_cmap: str = "hot"):
  
    def _norm(data: np.ndarray) -> np.ndarray:
        pos = data[data > 0]
        if pos.size == 0:
            return data
        return np.clip(data / float(np.nanpercentile(pos, 99)), 0, 1)

    t1w_norm = _norm(t1w_img.get_fdata().astype(np.float32))
    mid = [s // 2 for s in t1w_norm.shape]

    def _slices(vol):
        return [
            vol[:, :, mid[2]],   # axial
            vol[:, mid[1], :],   # coronal
            vol[mid[0], :, :],   # sagittal
        ]

    # Resample unregistered MRSI to T1w grid for the "before" column
    mrsi_unreg_on_t1w = resample_from_to(mrsi_img_unreg, t1w_img, order=1)
    cols_data = [
        (_norm(mrsi_unreg_on_t1w.get_fdata().astype(np.float32)), "MRSI\n(no registration)"),
        (_norm(mrsi_reg_img.get_fdata().astype(np.float32)),       "MRSI\n+ registered"),
    ]

    view_labels = ["Axial (z)", "Coronal (y)", "Sagittal (x)"]
    t1w_slices  = _slices(t1w_norm)

    fig, axes = plt.subplots(3, 2, figsize=(10, 11), facecolor="black")

    for row, (t1s, vl) in enumerate(zip(t1w_slices, view_labels)):
        for col, (mrs_norm_col, col_lbl) in enumerate(cols_data):
            ax = axes[row, col]
            ax.set_facecolor("black")
            ax.imshow(t1s.T, origin="lower", cmap=t1w_cmap,
                      vmin=0, vmax=1, interpolation="nearest")
            masked_mrs = np.ma.masked_where(_slices(mrs_norm_col)[row] < 1e-6,
                                            _slices(mrs_norm_col)[row])
            ax.imshow(masked_mrs.T, origin="lower", cmap=mrs_cmap,
                      vmin=0, vmax=1, alpha=0.6, interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(col_lbl, color="white", fontsize=10)
        axes[row, 0].set_ylabel(vl, color="white", fontsize=10,
                                rotation=90, labelpad=6)

    fig.suptitle(
        f"{subj} {ses} – MRSI→T1w registration: {mrsi_label}  "
        f"(rigid ANTs, T1w background)",
        color="white", fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.show()

def plot_registration_methods_comparison(
    t1w_ds_img: nib.Nifti1Image,
    t1w_fullres_img: nib.Nifti1Image,
    mrsi_reg_ds_img: nib.Nifti1Image,
    mrsi_reg_fullres_img: nib.Nifti1Image,
    subj: str = "sub-01",
    ses: str = "ses-01",
    mrsi_label: str = "MRSI",
    t1w_cmap: str = "gray",
    mrs_cmap: str = "hot",
) -> None:
    """
    Side-by-side comparison of two registration strategies.

    Left column : MRSI registered to the downsampled T1w (DS T1w as fixed).
    Right column: MRSI registered to the full-resolution T1w (full T1w as fixed).

    Each column shows the registered MRSI overlaid on its respective T1w background.
    """
    def _norm(data: np.ndarray) -> np.ndarray:
        pos = data[data > 0]
        if pos.size == 0:
            return data
        return np.clip(data / float(np.nanpercentile(pos, 99)), 0, 1)

    view_labels = ["Axial (z)", "Coronal (y)", "Sagittal (x)"]

    def _slices(vol, mid):
        return [
            vol[:, :, mid[2]],
            vol[:, mid[1], :],
            vol[mid[0], :, :],
        ]

    cols_data = [
        (t1w_ds_img,      mrsi_reg_ds_img,      "MRSI → DS T1w\n(MRSI-res fixed)"),
        (t1w_fullres_img, mrsi_reg_fullres_img,  "MRSI → Full-res T1w\n(full-res fixed)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(10, 11), facecolor="black")

    for col, (t1w_ref, mrsi_reg, col_lbl) in enumerate(cols_data):
        t1w_norm = _norm(t1w_ref.get_fdata().astype(np.float32))
        mrs_norm = _norm(mrsi_reg.get_fdata().astype(np.float32))
        mid = [s // 2 for s in t1w_norm.shape]
        t1w_slices = _slices(t1w_norm, mid)
        mrs_slices = _slices(mrs_norm, mid)
        for row, (t1s, ms, vl) in enumerate(zip(t1w_slices, mrs_slices, view_labels)):
            ax = axes[row, col]
            ax.set_facecolor("black")
            ax.imshow(t1s.T, origin="lower", cmap=t1w_cmap,
                      vmin=0, vmax=1, interpolation="nearest")
            masked_mrs = np.ma.masked_where(ms < 1e-6, ms)
            ax.imshow(masked_mrs.T, origin="lower", cmap=mrs_cmap,
                      vmin=0, vmax=1, alpha=0.6, interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(col_lbl, color="white", fontsize=10)
            if col == 0:
                ax.set_ylabel(vl, color="white", fontsize=10, rotation=90, labelpad=6)

    fig.suptitle(
        f"{subj} {ses} – Registration method comparison: {mrsi_label}  "
        f"(DS T1w fixed vs full-res T1w fixed)",
        color="white", fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.show()

def plot_registration_target_comparison(
    t1w_ds_img: nib.Nifti1Image,
    mrsi_reg_bestsnr_img: nib.Nifti1Image,
    mrsi_reg_sum_img: nib.Nifti1Image,
    subj: str = "sub-01",
    ses: str = "ses-01",
    bestsnr_label: str = "best-SNR metabolite",
    sum_label: str = "Metabolite sum",
    t1w_cmap: str = "gray",
    mrs_cmap: str = "hot",
) -> None:
    """
    Compare two MRSI registrations, both displayed in DS T1w space.

    Left column : first registration overlaid on DS T1w.
    Right column: second registration overlaid on DS T1w.

    Both are overlaid on the DS T1w background for direct comparison.
    """
    def _norm(data: np.ndarray) -> np.ndarray:
        pos = data[data > 0]
        if pos.size == 0:
            return data
        return np.clip(data / float(np.nanpercentile(pos, 99)), 0, 1)

    t1w_norm = _norm(t1w_ds_img.get_fdata().astype(np.float32))
    mid = [s // 2 for s in t1w_norm.shape]

    def _slices(vol):
        return [
            vol[:, :, mid[2]],   # axial
            vol[:, mid[1], :],   # coronal
            vol[mid[0], :, :],   # sagittal
        ]

    cols_data = [
        (_norm(mrsi_reg_bestsnr_img.get_fdata().astype(np.float32)),
         f"{bestsnr_label} (moving) to DS T1w (fixed)"),
        (_norm(mrsi_reg_sum_img.get_fdata().astype(np.float32)),
         f"{sum_label} (moving) to DS T1w (fixed)"),
    ]

    view_labels = ["Axial (z)", "Coronal (y)", "Sagittal (x)"]
    t1w_slices  = _slices(t1w_norm)

    fig, axes = plt.subplots(3, 2, figsize=(10, 11), facecolor="black")

    for row, (t1s, vl) in enumerate(zip(t1w_slices, view_labels)):
        for col, (mrs_norm_col, col_lbl) in enumerate(cols_data):
            ax = axes[row, col]
            ax.set_facecolor("black")
            ax.imshow(t1s.T, origin="lower", cmap=t1w_cmap,
                      vmin=0, vmax=1, interpolation="nearest")
            masked_mrs = np.ma.masked_where(_slices(mrs_norm_col)[row] < 1e-6,
                                            _slices(mrs_norm_col)[row])
            ax.imshow(masked_mrs.T, origin="lower", cmap=mrs_cmap,
                      vmin=0, vmax=1, alpha=0.6, interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(col_lbl, color="white", fontsize=10)
        axes[row, 0].set_ylabel(vl, color="white", fontsize=10, rotation=90, labelpad=6)

    fig.suptitle(
        f"{subj} {ses} – Registration comparison: {bestsnr_label} vs {sum_label}\n"
        f"(DS T1w fixed, rigid ANTs)",
        color="white", fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.show()

def plot_coverage_registration_comparison(
    t1w_ds_img: nib.Nifti1Image,
    mrsi_cov_reg_img: nib.Nifti1Image,
    mrsi_gly_reg_img: nib.Nifti1Image,
    mrsi_gly_via_sum_img: nib.Nifti1Image,
    subj: str = "sub-01",
    ses: str = "ses-01",
    cov_label: str = "Best-coverage",
    gly_label: str = "Gly",
    t1w_cmap: str = "gray",
    mrs_cmap: str = "hot",
) -> None:
    """
    Three-column registration comparison in DS T1w space, plus a metrics table.

    Left   – best-coverage metabolite (e.g. Ins) with its own rigid transform.
    Centre – Gly (best-SNR) with its own rigid transform (the reference).
    Right  – Gly with the metabolite-sum's transform applied (no re-optimisation).

    Each column shows axial / coronal / sagittal views overlaid on the DS T1w.
    Below the figure a metrics table prints CC with T1w and mean in-brain
    intensity for each of the three registrations.
    """
    def _norm(data: np.ndarray) -> np.ndarray:
        pos = data[data > 0]
        if pos.size == 0:
            return data
        return np.clip(data / float(np.nanpercentile(pos, 99)), 0, 1)

    t1w_norm = _norm(t1w_ds_img.get_fdata().astype(np.float32))
    mid = [s // 2 for s in t1w_norm.shape]

    def _slices(vol):
        return [vol[:, :, mid[2]], vol[:, mid[1], :], vol[mid[0], :, :]]

    cols_data = [
        (_norm(mrsi_cov_reg_img.get_fdata().astype(np.float32)),
         f"{cov_label}\n(own transform)"),
        (_norm(mrsi_gly_reg_img.get_fdata().astype(np.float32)),
         f"{gly_label}\n(own transform)"),
        (_norm(mrsi_gly_via_sum_img.get_fdata().astype(np.float32)),
         f"{gly_label}\n(sum's transform)"),
    ]

    view_labels = ["Axial (z)", "Coronal (y)", "Sagittal (x)"]
    t1w_slices  = _slices(t1w_norm)

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), facecolor="black")

    for row, (t1s, vl) in enumerate(zip(t1w_slices, view_labels)):
        for col, (mrs_col, col_lbl) in enumerate(cols_data):
            ax = axes[row, col]
            ax.set_facecolor("black")
            ax.imshow(t1s.T, origin="lower", cmap=t1w_cmap,
                      vmin=0, vmax=1, interpolation="nearest")
            sl = _slices(mrs_col)[row]
            masked = np.ma.masked_where(sl < 1e-6, sl)
            ax.imshow(masked.T, origin="lower", cmap=mrs_cmap,
                      vmin=0, vmax=1, alpha=0.6, interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(col_lbl, color="white", fontsize=10)
        axes[row, 0].set_ylabel(vl, color="white", fontsize=10,
                                rotation=90, labelpad=6)

    fig.suptitle(
        f"{subj} {ses} – Registration comparison: {cov_label} vs {gly_label} own vs {gly_label} via sum\n"
        f"(DS T1w fixed, rigid ANTs)",
        color="white", fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.show()

    # Metrics table
    brain = t1w_norm > 0.05
    print(f"\n  {'Strategy':<36}  {'Mean in brain':>14}  {'CC with T1w':>12}")
    print(f"  {'-'*66}")
    for label, d in [
        (f"{cov_label} — own transform",     _norm(mrsi_cov_reg_img.get_fdata().astype(np.float32))),
        (f"{gly_label} — own transform",     _norm(mrsi_gly_reg_img.get_fdata().astype(np.float32))),
        (f"{gly_label} — sum's transform",   _norm(mrsi_gly_via_sum_img.get_fdata().astype(np.float32))),
    ]:
        cc = float(np.corrcoef(d[brain].ravel(), t1w_norm[brain].ravel())[0, 1])
        print(f"  {label:<36}  {d[brain].mean():>14.4f}  {cc:>12.4f}")

def compare_registration_coverage(
    t1w_ds_img: nib.Nifti1Image,
    mrsi_mask: np.ndarray,
    mrsi_img: nib.Nifti1Image,
    mrsi_reg_bestsnr_img: nib.Nifti1Image,
    mrsi_reg_sum_img: nib.Nifti1Image,
    bestsnr_transforms: list,
    sum_transforms: list,
    subj: str = "sub-01",
    ses: str = "ses-01",
    bestsnr_label: str = "best-SNR metabolite",
    sum_label: str = "Sum reg"):
    
    def _norm(data: np.ndarray) :
        pos = data[data > 0]
        if pos.size == 0:
            return data
        return np.clip(data / float(np.nanpercentile(pos, 99)), 0, 1)

    d_bestsnr = _norm(mrsi_reg_bestsnr_img.get_fdata().astype(np.float32))
    d_sum     = _norm(mrsi_reg_sum_img.get_fdata().astype(np.float32))
    diff      = d_bestsnr - d_sum   # positive = more MRSI signal in best-SNR reg

    # Warp the MRSI-space mask into T1w space using the ANTs transforms so
    # the contour follows the actual registration, not just the raw affine.
    t1w_data    = t1w_ds_img.get_fdata().astype(np.float32)
    fixed_ants  = _nib_to_ants(t1w_data, t1w_ds_img)
    mask_f32    = mrsi_mask.astype(np.float32)
    moving_mask = _nib_to_ants(mask_f32, mrsi_img)

    def _warp_mask(transforms):
        warped = ants.apply_transforms(
            fixed=fixed_ants,
            moving=moving_mask,
            transformlist=transforms,
            interpolator="nearestNeighbor",
        )
        return warped.numpy().astype(bool)

    mask_bestsnr = _warp_mask(bestsnr_transforms)
    mask_sum     = _warp_mask(sum_transforms)

    # Metrics use the best-SNR warped mask (the reference registration)
    mask_bool = mask_bestsnr
    out_mask  = ~mask_bool

    # ── metrics ──────────────────────────────────────────────────────────────
    def _metrics(d):
        return {
            "mean_in":        float(d[mask_bool].mean())  if mask_bool.any() else 0.0,
            "mean_out":       float(d[out_mask].mean())   if out_mask.any()  else 0.0,
            "snr_ratio":      (float(d[mask_bool].mean()) / (float(d[out_mask].mean()) + 1e-9))
                              if mask_bool.any() else 0.0,
            "pct_active_in":  float((d[mask_bool] > 0.05).mean() * 100) if mask_bool.any() else 0.0,
            "pct_active_out": float((d[out_mask]  > 0.05).mean() * 100) if out_mask.any()  else 0.0,
        }

    m_best = _metrics(d_bestsnr)
    m_sum  = _metrics(d_sum)

    # ── figure ────────────────────────────────────────────────────────────────
    t1w_norm = _norm(t1w_ds_img.get_fdata().astype(np.float32))
    mid = [s // 2 for s in t1w_norm.shape]

    diff_abs_max = float(np.abs(diff).max()) or 1.0

    fig = plt.figure(figsize=(18, 11), facecolor="black")
    gs  = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1.4], hspace=0.35, wspace=0.25)

    view_labels = ["Axial (z)", "Coronal (y)", "Sagittal (x)"]

    def _slices_at(vol, centres):
        return [
            vol[:, :, centres[2]],
            vol[:, centres[1], :],
            vol[centres[0], :, :],
        ]

    t1w_slices  = _slices_at(t1w_norm, mid)
    diff_slices = _slices_at(diff,     mid)

    # column 0: difference map (3 rows of orthogonal views), T1w background
    for row, (t1s, ds, vl) in enumerate(zip(t1w_slices, diff_slices, view_labels)):
        ax = fig.add_subplot(gs[row, 0])
        ax.set_facecolor("black")
        ax.imshow(t1s.T, origin="lower", cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest")
        masked_diff = np.ma.masked_where(np.abs(ds) < 0.02, ds)
        im = ax.imshow(masked_diff.T, origin="lower", cmap="RdBu_r",
                       vmin=-diff_abs_max, vmax=diff_abs_max,
                       alpha=0.85, interpolation="nearest")
        # Two mask contours: orange = bestsnr_label reg boundary, cyan = sum_label boundary
        ax.contour(_slices_at(mask_bestsnr.astype(np.float32), mid)[row].T,
                   levels=[0.5], colors=["#f4a261"], linewidths=1.0, origin="lower")
        ax.contour(_slices_at(mask_sum.astype(np.float32), mid)[row].T,
                   levels=[0.5], colors=["cyan"], linewidths=1.0, origin="lower")
        ax.axis("off")
        ax.set_ylabel(vl, color="white", fontsize=9, rotation=90, labelpad=4)
        if row == 0:
            ax.set_title(f"Difference map\n({bestsnr_label} − {sum_label})", color="white", fontsize=10)

    cb_ax = fig.add_axes([0.01, 0.08, 0.01, 0.25])
    cbar  = plt.colorbar(im, cax=cb_ax)
    cbar.set_label("ΔMRSI (norm.)", color="white", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)
    cbar.outline.set_edgecolor("gray")

    # column 1: in-mask histogram
    ax_in = fig.add_subplot(gs[:, 1])
    ax_in.set_facecolor("#111111")
    bins = np.linspace(0, 1, 60)
    ax_in.hist(d_bestsnr[mask_bool], bins=bins, color="#f4a261",
               alpha=0.75, label=f"{bestsnr_label} reg", density=True)
    ax_in.hist(d_sum[mask_bool],     bins=bins, color="#4cc9f0",
               alpha=0.75, label=f"{sum_label}", density=True)
    ax_in.set_xlabel("Normalised MRSI intensity", color="white", fontsize=9)
    ax_in.set_ylabel("Density", color="white", fontsize=9)
    ax_in.set_title("Inside MRSI mask\n(brain coverage)", color="white", fontsize=10)
    ax_in.tick_params(colors="white")
    ax_in.spines[:].set_color("#555555")
    ax_in.legend(facecolor="#222222", labelcolor="white", fontsize=8)
    ax_in.axvline(m_best["mean_in"], color="#f4a261", linestyle="--", linewidth=1.2)
    ax_in.axvline(m_sum["mean_in"],  color="#4cc9f0", linestyle="--", linewidth=1.2)

    # column 2: out-of-mask histogram
    ax_out = fig.add_subplot(gs[:, 2])
    ax_out.set_facecolor("#111111")
    ax_out.hist(d_bestsnr[out_mask], bins=bins, color="#f4a261",
                alpha=0.75, label=f"{bestsnr_label} reg", density=True)
    ax_out.hist(d_sum[out_mask],     bins=bins, color="#4cc9f0",
                alpha=0.75, label=f"{sum_label}", density=True)
    ax_out.set_xlabel("Normalised MRSI intensity", color="white", fontsize=9)
    ax_out.set_ylabel("Density", color="white", fontsize=9)
    ax_out.set_title("Outside MRSI mask\n(signal leakage)", color="white", fontsize=10)
    ax_out.tick_params(colors="white")
    ax_out.spines[:].set_color("#555555")
    ax_out.legend(facecolor="#222222", labelcolor="white", fontsize=8)
    ax_out.axvline(m_best["mean_out"], color="#f4a261", linestyle="--", linewidth=1.2)
    ax_out.axvline(m_sum["mean_out"],  color="#4cc9f0", linestyle="--", linewidth=1.2)

    fig.suptitle(
        f"{subj} {ses} – MRSI→T1w registration quality: coverage inside vs outside MRSI mask\n"
        f"(orange contour = {bestsnr_label} reg boundary, cyan contour = {sum_label} boundary, T1w background)",
        color="white", fontsize=12, y=1.01,
    )
    plt.show()

    # ── print metrics table ────────────────────────────────────────────────
    header = f"{'Metric':<28}  {bestsnr_label + ' reg':>14}  {sum_label:>14}  {'Δ (a−b)':>14}"
    print(header)
    print("─" * len(header))
    rows = [
        ("Mean MRSI in-mask",       m_best["mean_in"],        m_sum["mean_in"]),
        ("Mean MRSI out-of-mask",   m_best["mean_out"],       m_sum["mean_out"]),
        ("In/Out SNR ratio",         m_best["snr_ratio"],      m_sum["snr_ratio"]),
        ("% active voxels in-mask",  m_best["pct_active_in"],  m_sum["pct_active_in"]),
        ("% active voxels out",      m_best["pct_active_out"], m_sum["pct_active_out"]),
    ]
    for label, vb, vs in rows:
        print(f"{label:<28}  {vb:>14.4f}  {vs:>14.4f}  {vb - vs:>+14.4f}")

    return {"best_snr": m_best, "sum": m_sum}

def plot_sum_registration_comparison(
    t1w_ds_img: nib.Nifti1Image,
    mrsi_reg_ds_img: nib.Nifti1Image,
    mrsi_gly_via_sum_img: nib.Nifti1Image,
    subj: str = "sub-01",
    ses: str = "ses-01",
    bestsnr_label: str = "best-SNR",
    t1w_cmap: str = "gray",
    mrs_cmap: str = "hot",
) -> None:
    """
    Impact of transform choice on the best-SNR (Gly) metabolite map.

    Both columns show the **same** Gly image, but positioned differently:

    Left  – Gly with its **own** rigid transform (Registration 1): the
            optimizer had Gly as the moving image and found the best alignment
            for it → this is the ground-truth reference.
    Right – Gly with the **sum's** rigid transform (Registration 3): the
            optimizer had the metabolite sum as the moving image; that
            transform is then applied to Gly instead.

    The difference between the two columns is entirely due to how well the
    sum-optimized transform generalises to an individual metabolite map.
    Any shift or rotation visible on the right but not the left means the
    sum registration is introducing a systematic error for that metabolite.
    """
    def _norm(data: np.ndarray) -> np.ndarray:
        pos = data[data > 0]
        if pos.size == 0:
            return data
        return np.clip(data / float(np.nanpercentile(pos, 99)), 0, 1)

    t1w_norm = _norm(t1w_ds_img.get_fdata().astype(np.float32))
    mid = [s // 2 for s in t1w_norm.shape]

    def _slices(vol):
        return [
            vol[:, :, mid[2]],   # axial
            vol[:, mid[1], :],   # coronal
            vol[mid[0], :, :],   # sagittal
        ]

    cols_data = [
        (
            _norm(mrsi_reg_ds_img.get_fdata().astype(np.float32)),
            f"{bestsnr_label} — own transform\n(optimizer used Gly as moving image)",
        ),
        (
            _norm(mrsi_gly_via_sum_img.get_fdata().astype(np.float32)),
            f"{bestsnr_label} — sum's transform\n(optimizer used sum as moving image)",
        ),
    ]

    view_labels = ["Axial (z)", "Coronal (y)", "Sagittal (x)"]
    t1w_slices  = _slices(t1w_norm)

    fig, axes = plt.subplots(3, 2, figsize=(10, 11), facecolor="black")

    for row, (t1s, vl) in enumerate(zip(t1w_slices, view_labels)):
        for col, (mrs_norm_col, col_lbl) in enumerate(cols_data):
            ax = axes[row, col]
            ax.set_facecolor("black")
            ax.imshow(t1s.T, origin="lower", cmap=t1w_cmap,
                      vmin=0, vmax=1, interpolation="nearest")
            sl = _slices(mrs_norm_col)[row]
            masked_mrs = np.ma.masked_where(sl < 1e-6, sl)
            ax.imshow(masked_mrs.T, origin="lower", cmap=mrs_cmap,
                      vmin=0, vmax=1, alpha=0.6, interpolation="nearest")
            ax.axis("off")
            if row == 0:
                ax.set_title(col_lbl, color="white", fontsize=10)
        axes[row, 0].set_ylabel(vl, color="white", fontsize=10,
                                rotation=90, labelpad=6)

    fig.suptitle(
        f"{subj} {ses} – {bestsnr_label}: own transform vs sum's transform\n"
        f"(same image, two different rigid registrations, T1w background)",
        color="white", fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.show()

    # Print alignment quality metrics for both
    t_norm = _norm(t1w_ds_img.get_fdata().astype(np.float32))
    brain  = t_norm > 0.05
    print(f"  {'':30}  {'mean in brain':>14}  {'CC with T1w':>12}")
    print(f"  {'-'*58}")
    for label, img in [
        (f"{bestsnr_label} — own transform",  mrsi_reg_ds_img),
        (f"{bestsnr_label} — sum's transform", mrsi_gly_via_sum_img),
    ]:
        d  = _norm(img.get_fdata().astype(np.float32))
        cc = float(np.corrcoef(d[brain].ravel(), t_norm[brain].ravel())[0, 1])
        print(f"  {label:<30}  {d[brain].mean():>14.4f}  {cc:>12.4f}")

def plot_water_mask_comparison(
    water_img: nib.Nifti1Image,
    filled_mask: np.ndarray,
    subj: str = "sub-01",
    ses: str = "ses-01",
) -> None:
    """
    ortho comparison of the raw water mask, the hole filled mask,
    and the recovered holes (voxels added by binary_fill_holes).
    """
    water_data = water_img.get_fdata()
    _finite = water_data[np.isfinite(water_data) & (water_data > 0)]
    _thr    = float(np.percentile(_finite, 5)) if _finite.size > 0 else 0.0  # cut bottom 5%
    raw_mask   = np.isfinite(water_data) & (water_data > _thr)
    hole_map   = filled_mask & ~raw_mask

    mid = [s // 2 for s in raw_mask.shape]

    def _slices(vol):
        return [
            vol[:, :, mid[2]],   # axial
            vol[:, mid[1], :],   # coronal
            vol[mid[0], :, :],   # sagittal
        ]

    rows_data  = [_slices(raw_mask.astype(np.float32)),
                  _slices(filled_mask.astype(np.float32)),
                  _slices(hole_map.astype(np.float32))]
    row_labels = ["Raw mask", "Filled mask", "Recovered holes"]
    cmaps      = ["Blues",    "Greens",      "Reds"]
    view_labels = ["Axial (z)", "Coronal (y)", "Sagittal (x)"]

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), facecolor="black")

    for row, (slices, cmap, rlabel) in enumerate(zip(rows_data, cmaps, row_labels)):
        for col, (slc, vlabel) in enumerate(zip(slices, view_labels)):
            ax = axes[row, col]
            ax.set_facecolor("black")
            ax.imshow(slc.T, origin="lower", cmap=cmap,
                      vmin=0, vmax=1, interpolation="nearest")
            if row == 0:
                ax.set_title(vlabel, color="white", fontsize=10)
            ax.axis("off")
        axes[row, 0].set_ylabel(rlabel, color="white", fontsize=10,
                                rotation=90, labelpad=6)

    n_raw    = int(raw_mask.sum())
    n_filled = int(filled_mask.sum())
    n_holes  = int(hole_map.sum())
    fig.suptitle(
        f"{subj} {ses} – Water mask comparison  |  "
        f"raw={n_raw:,} vox   filled={n_filled:,} vox   recovered={n_holes:,} vox",
        color="white", fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.show()

    print(f"Raw mask voxels   : {n_raw:,}")
    print(f"Filled mask voxels: {n_filled:,}")
    print(f"Holes recovered   : {n_holes:,}  ({100 * n_holes / max(n_filled, 1):.2f}% of filled mask)")

def plot_mrsi_mosaic(
    mrs_filenames: list,
    mrs_dir: str,
    subj: str = "sub-01",
    cmap: str = "hot",
    alpha: float = 0.6,
    cols: int = 3,
) -> None:
    """
    Overlay multiple MRSI concentration maps in their own native space,
    without any T1w background and without registration.
    """
    n    = len(mrs_filenames)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(5 * cols, 4 * rows),
        facecolor="black",
    )
    axes = np.array(axes).reshape(-1)   # always 1-D

    for ax, fname in zip(axes, mrs_filenames):
        img  = nib.load(os.path.join(mrs_dir, fname))
        data = img.get_fdata()

        # pick the axial slice with the most signal
        signal_per_slice = (data > 0).sum(axis=(0, 1))
        z_mid = int(signal_per_slice.argmax())
        slc   = data[:, :, z_mid]

        # normalise to [0, 1] for display
        vmax = float(np.nanpercentile(slc[slc > 0], 99)) if np.any(slc > 0) else 1.0
        slc_norm = np.clip(slc / vmax, 0, 1)

        ax.set_facecolor("black")
        ax.imshow(
            slc_norm.T,
            origin="lower",
            cmap=cmap,
            alpha=alpha,
            vmin=0, vmax=1,
            interpolation="nearest",
        )
        ax.set_title(metabolite_name(fname), color="white", fontsize=9, pad=3)
        ax.axis("off")

    # hide unused subplot slots
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"{subj} – MRSI concentration maps (native MRS space, axial peak slice)",
        color="white", fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.show()

def plot_mrsi_combined(
    mrs_filenames: list,
    mrs_dir: str,
    subj: str = "sub-01",
    cmaps: list | None = None,
    alpha: float = 0.5):
    
    default_cmaps = [
        "Reds", "Blues", "Greens", "Oranges", "Purples",
        "YlOrBr", "PuRd", "BuGn", "RdPu", "GnBu",
    ]
    if cmaps is None:
        cmaps = [default_cmaps[i % len(default_cmaps)] for i in range(len(mrs_filenames))]

    # load all images
    imgs  = [nib.load(os.path.join(mrs_dir, f)) for f in mrs_filenames]
    datas = [img.get_fdata() for img in imgs]

    # find shared volume shape (use first image as reference)
    ref_shape = datas[0].shape
    mid = [s // 2 for s in ref_shape]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="black")
    view_labels = ["Axial (z)", "Coronal (y)", "Sagittal (x)"]

    for ax in axes:
        ax.set_facecolor("black")

    for data, cmap_name, fname in zip(datas, cmaps, mrs_filenames):
        # pad/crop to reference shape if needed
        d = data
        vmax = float(np.nanpercentile(d[d > 0], 99)) if np.any(d > 0) else 1.0
        d_norm = np.clip(d / vmax, 0, 1)

        slices = [
            d_norm[:, :, mid[2]],   # axial
            d_norm[:, mid[1], :],   # coronal
            d_norm[mid[0], :, :],   # sagittal
        ]

        for ax, slc in zip(axes, slices):
            masked = np.ma.masked_where(slc < 1e-6, slc)
            ax.imshow(
                masked.T,
                origin="lower",
                cmap=cmap_name,
                alpha=alpha,
                vmin=0, vmax=1,
                interpolation="nearest",
            )

    for ax, title in zip(axes, view_labels):
        ax.set_title(title, color="white", fontsize=10)
        ax.axis("off")

    labels = [metabolite_name(f) for f in mrs_filenames]
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1,
                       fc=plt.get_cmap(c)(0.7), alpha=0.8, label=lbl)
        for c, lbl in zip(cmaps, labels)
    ]
    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=min(len(labels), 6), fontsize=8,
        facecolor="#222222", labelcolor="white",
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        f"{subj} – MRSI coverage overlay (native MRS space, centre slices)",
        color="white", fontsize=12,
    )
    plt.tight_layout()
    plt.show()

def build_coverage_widget(
    mrs_dir: str,
    subj: str = "sub-01",
) -> widgets.Widget:
    """
    Interactive widget: select any combination of metabolites via checkboxes
    and view their concentration maps overlaid in native MRS space.
    """
    _PALETTE = [
        "Reds", "Blues", "Greens", "Oranges", "Purples",
        "YlOrBr", "PuRd", "BuGn", "RdPu", "GnBu",
        "copper", "cool", "autumn", "winter", "spring",
    ]

    all_files = sorted(
        f for f in os.listdir(mrs_dir)
        if f.endswith(".nii.gz") and "acq-OrigRes" in f
    )
    all_labels = [metabolite_name(f) for f in all_files]

    # load + cache raw arrays (no resampling needed — native space)
    _cache: dict = {}

    def _get(fname: str) -> np.ndarray:
        if fname not in _cache:
            _cache[fname] = nib.load(os.path.join(mrs_dir, fname)).get_fdata()
        return _cache[fname]

    # reference shape from first file
    ref_shape = _get(all_files[0]).shape
    mid = [s // 2 for s in ref_shape]

    defaults = {"NAA", "Cr", "Glu"}
    n        = len(all_labels)
    col_size = (n + 1) // 2

    def _make_boxes(labels):
        return [
            widgets.Checkbox(
                value=(lbl in defaults),
                description=lbl,
                layout=widgets.Layout(width="190px"),
            )
            for lbl in labels
        ]

    col1_boxes = _make_boxes(all_labels[:col_size])
    col2_boxes = _make_boxes(all_labels[col_size:])
    all_boxes  = col1_boxes + col2_boxes

    btn_all  = widgets.Button(
        description="Select all",
        button_style="info",
        layout=widgets.Layout(width="120px"),
    )
    btn_none = widgets.Button(
        description="Deselect all",
        button_style="warning",
        layout=widgets.Layout(width="120px"),
    )
    btn_plot = widgets.Button(
        description="▶  Plot",
        button_style="success",
        layout=widgets.Layout(width="100px"),
    )
    alpha_slider = widgets.FloatSlider(
        value=0.6, min=0.1, max=1.0, step=0.05,
        description="Alpha:", readout_format=".2f",
        layout=widgets.Layout(width="260px"),
    )
    out = widgets.Output()

    def _on_all(_):
        for cb in all_boxes:
            cb.value = True

    def _on_none(_):
        for cb in all_boxes:
            cb.value = False

    def _on_plot(_):
        chosen_idx = [i for i, cb in enumerate(all_boxes) if cb.value]
        chosen     = [all_files[i] for i in chosen_idx]
        labels     = [all_labels[i] for i in chosen_idx]
        alpha      = alpha_slider.value

        with out:
            clear_output(wait=True)
            if not chosen:
                print("Select at least one metabolite.")
                return

            fig, axes = plt.subplots(
                1, 3,
                figsize=(15, 5),
                facecolor="black",
            )
            view_titles = ["Axial (z)", "Coronal (y)", "Sagittal (x)"]
            for ax, title in zip(axes, view_titles):
                ax.set_facecolor("black")
                ax.set_title(title, color="white", fontsize=10)
                ax.axis("off")

            legend_handles = []
            for k, (fname, label) in enumerate(zip(chosen, labels)):
                data  = _get(fname)
                cmap_name = _PALETTE[k % len(_PALETTE)]
                vmax  = float(np.nanpercentile(data[data > 0], 99)) if np.any(data > 0) else 1.0
                d_norm = np.clip(data / vmax, 0, 1)

                slices = [
                    d_norm[:, :, mid[2]],   # axial
                    d_norm[:, mid[1], :],   # coronal
                    d_norm[mid[0], :, :],   # sagittal
                ]
                for ax, slc in zip(axes, slices):
                    masked = np.ma.masked_where(slc < 1e-6, slc)
                    ax.imshow(
                        masked.T,
                        origin="lower",
                        cmap=cmap_name,
                        alpha=alpha,
                        vmin=0, vmax=1,
                        interpolation="nearest",
                    )
                legend_handles.append(
                    plt.Rectangle(
                        (0, 0), 1, 1,
                        fc=plt.get_cmap(cmap_name)(0.7),
                        alpha=0.85,
                        label=label,
                    )
                )

            fig.legend(
                handles=legend_handles,
                loc="lower center",
                ncol=min(len(labels), 7),
                fontsize=8,
                facecolor="#222222",
                labelcolor="white",
                bbox_to_anchor=(0.5, -0.06),
            )
            fig.suptitle(
                f"{subj} – MRSI overlay (native MRS space, centre slices)",
                color="white", fontsize=12,
            )
            plt.tight_layout()
            plt.show()
            print(f"{len(chosen)} map(s) shown: {labels}")

    btn_all.on_click(_on_all)
    btn_none.on_click(_on_none)
    btn_plot.on_click(_on_plot)

    ui = widgets.VBox([
        widgets.HBox([widgets.VBox(col1_boxes), widgets.VBox(col2_boxes)]),
        widgets.HBox([btn_all, btn_none, alpha_slider, btn_plot]),
        out,
    ])

    _on_plot(None)
    return ui

def plot_inverse_registration_panels(
    mrsi_sum_img: nib.Nifti1Image,
    mrsi_gly_img: nib.Nifti1Image,
    t1w_via_sum_img: nib.Nifti1Image,
    t1w_via_gly_img: nib.Nifti1Image,
    t1w_via_sum_in_gly_img: nib.Nifti1Image,
    subj: str = "sub-01",
    ses: str = "ses-01",
    sum_label: str = "Sum",
    gly_label: str = "Gly",
    t1w_cmap: str = "gray",
    mrs_cmap: str = "hot"):
    
    def _norm(data: np.ndarray) -> np.ndarray:
        pos = data[data > 0]
        if pos.size == 0:
            return data
        return np.clip(data / float(np.nanpercentile(pos, 99)), 0, 1)

    views = [
        (mrsi_sum_img,           t1w_via_sum_img,        f"T1w → {sum_label} space\n(sum-driven reg)",         f"{sum_label} map"),
        (mrsi_gly_img,           t1w_via_gly_img,        f"T1w → {gly_label} space\n({gly_label}-own reg)",    f"{gly_label} map"),
        (mrsi_gly_img,           t1w_via_sum_in_gly_img, f"T1w → {gly_label} space\n(sum transform reused)",   f"{gly_label} map"),
    ]

    view_labels = ["Axial (z)", "Coronal (y)", "Sagittal (x)"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), facecolor="black")
    fig.subplots_adjust(hspace=0.08, wspace=0.04)

    for col, (bg_img, t1w_reg_img, col_title, bg_label) in enumerate(views):
        bg_norm  = _norm(bg_img.get_fdata().astype(np.float32))
        t1w_norm = _norm(t1w_reg_img.get_fdata().astype(np.float32))
        mid = [s // 2 for s in bg_norm.shape]

        slices_bg  = [bg_norm[:, :, mid[2]], bg_norm[:, mid[1], :], bg_norm[mid[0], :, :]]
        slices_t1w = [t1w_norm[:, :, mid[2]], t1w_norm[:, mid[1], :], t1w_norm[mid[0], :, :]]

        for row, (s_bg, s_t1w, vl) in enumerate(zip(slices_bg, slices_t1w, view_labels)):
            ax = axes[row, col]
            ax.set_facecolor("black")
            ax.imshow(s_t1w.T, origin="lower", cmap=t1w_cmap, vmin=0, vmax=1,
                      interpolation="nearest", alpha=1.0)
            ax.imshow(s_bg.T,  origin="lower", cmap=mrs_cmap, vmin=0, vmax=1,
                      interpolation="nearest", alpha=0.6)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(vl, color="white", fontsize=9, rotation=90, labelpad=4)
            if row == 0:
                ax.set_title(col_title, color="white", fontsize=10, pad=6)

    fig.suptitle(
        f"{subj} {ses} – T1w registered into MRSI space (inverse registration)\n"
        f"MRSI overlay ({mrs_cmap}, α=0.6) on registered T1w background",
        color="white", fontsize=12, y=1.01,
    )
    plt.show()

def compare_inverse_registration_pair(
    mrsi_bg_img: nib.Nifti1Image,
    t1w_left_img: nib.Nifti1Image,
    t1w_right_img: nib.Nifti1Image,
    subj: str = "sub-01",
    ses: str = "ses-01",
    label_left: str = "No mask",
    label_right: str = "Water weighted",
    t1w_cmap: str = "gray",
    mrs_cmap: str = "hot",
) -> None:
    
    def _norm(data: np.ndarray) -> np.ndarray:
        pos = data[data > 0]
        if pos.size == 0:
            return data
        return np.clip(data / float(np.nanpercentile(pos, 99)), 0, 1)

    bg_norm = _norm(mrsi_bg_img.get_fdata().astype(np.float32))
    views = [
        (t1w_left_img,  label_left),
        (t1w_right_img, label_right),
    ]
    view_labels = ["Axial (z)", "Coronal (y)", "Sagittal (x)"]
    mid = [s // 2 for s in bg_norm.shape]

    fig, axes = plt.subplots(3, 2, figsize=(10, 12), facecolor="black")
    fig.subplots_adjust(hspace=0.08, wspace=0.04)

    slices_bg = [bg_norm[:, :, mid[2]], bg_norm[:, mid[1], :], bg_norm[mid[0], :, :]]

    for col, (t1w_img, col_title) in enumerate(views):
        t1w_norm = _norm(t1w_img.get_fdata().astype(np.float32))
        slices_t1w = [t1w_norm[:, :, mid[2]], t1w_norm[:, mid[1], :], t1w_norm[mid[0], :, :]]

        for row, (s_bg, s_t1w, vl) in enumerate(zip(slices_bg, slices_t1w, view_labels)):
            ax = axes[row, col]
            ax.set_facecolor("black")
            ax.imshow(s_t1w.T, origin="lower", cmap=t1w_cmap, vmin=0, vmax=1,
                      interpolation="nearest", alpha=1.0)
            ax.imshow(s_bg.T, origin="lower", cmap=mrs_cmap, vmin=0, vmax=1,
                      interpolation="nearest", alpha=0.6)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(vl, color="white", fontsize=9, rotation=90, labelpad=4)
            if row == 0:
                ax.set_title(col_title, color="white", fontsize=10, pad=6)

    fig.suptitle(
        f"{subj} {ses} – Inverse registration comparison\n"
        f"MRSI overlay ({mrs_cmap}, α=0.6) on registered T1w background",
        color="white", fontsize=12, y=1.01,
    )
    plt.show()

def plot_total_pipeline_comparison(
    sum_orig_img: nib.Nifti1Image,
    t1w_in_sum_orig_img: nib.Nifti1Image,
    sum_ras_img: nib.Nifti1Image,
    t1w_in_sum_ras_img: nib.Nifti1Image,
    subj: str = "sub-01",
    ses: str = "ses-01",
    label_left: str = "Reg 15 – skull-stripped T1w\n→ original sum",
    label_right: str = "Reg 17 – skull-stripped T1w\n→ reoriented sum (total pipeline)",
    t1w_cmap: str = "gray",
    mrs_cmap: str = "hot",
) -> None:
    """
    Two-column comparison for the total pipeline visualisation (Section 19).

    Each column is shown on its **own** native MRSI background so that the
    two different voxel spaces (original sum vs reoriented RAS sum) are never
    mixed.

    - Left  : skull-stripped T1w registered to the original (non-reoriented)
              metabolite sum, overlaid on that sum as background.
    - Right : skull-stripped T1w registered to the reoriented RAS sum,
              overlaid on the reoriented sum as background.
    """
    def _norm(data: np.ndarray) -> np.ndarray:
        pos = data[data > 0]
        if pos.size == 0:
            return data
        return np.clip(data / float(np.nanpercentile(pos, 99)), 0, 1)

    panels = [
        (sum_orig_img,  t1w_in_sum_orig_img, label_left),
        (sum_ras_img,   t1w_in_sum_ras_img,  label_right),
    ]
    view_labels = ["Axial", "Coronal", "Sagittal"]

    fig, axes = plt.subplots(3, 2, figsize=(10, 12), facecolor="black")
    fig.subplots_adjust(hspace=0.08, wspace=0.06)

    for col, (bg_img, t1w_reg_img, title) in enumerate(panels):
        if bg_img is None or t1w_reg_img is None:
            for row in range(3):
                axes[row, col].set_facecolor("black")
                axes[row, col].axis("off")
            axes[0, col].set_title(title + "\n(not available)", color="white", fontsize=9)
            continue

        bg  = _norm(bg_img.get_fdata().astype(np.float32))
        t1w = _norm(t1w_reg_img.get_fdata().astype(np.float32))
        mid = [s // 2 for s in bg.shape]
        slices_bg  = [bg[:, :, mid[2]], bg[:, mid[1], :], bg[mid[0], :, :]]
        slices_t1w = [t1w[:, :, mid[2]], t1w[:, mid[1], :], t1w[mid[0], :, :]]

        for row, (s_bg, s_t1w, vl) in enumerate(zip(slices_bg, slices_t1w, view_labels)):
            ax = axes[row, col]
            ax.set_facecolor("black")
            ax.imshow(s_t1w.T, origin="lower", cmap=t1w_cmap, vmin=0, vmax=1,
                      interpolation="nearest", alpha=1.0)
            ax.imshow(s_bg.T,  origin="lower", cmap=mrs_cmap, vmin=0, vmax=1,
                      interpolation="nearest", alpha=0.6)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(vl, color="white", fontsize=9, rotation=90, labelpad=4)
            if row == 0:
                ax.set_title(title, color="white", fontsize=9, pad=6)

    fig.suptitle(
        f"{subj} {ses} – Total pipeline\n"
        f"T1w background ({t1w_cmap}) + MRSI sum overlay ({mrs_cmap}, α=0.6)",
        color="white", fontsize=10, y=1.02,
    )
    plt.tight_layout()
    plt.show()

def plot_registration_metrics(
    metrics: list,
    subj: str = "sub-01",
    ses: str = "ses-01",
) -> None:
    """
    Analytical multi-panel figure showing per-metabolite registration quality
    after the Reg-17 total-pipeline transform.

    Dot colour always encodes coverage (viridis: low=purple, high=yellow).
    """
    if not isinstance(metrics, list) or not metrics:
        print("[metrics] no metrics to plot (expected non-empty list from "
              "compute_registration_metrics()).")
        return
    if not isinstance(metrics[0], dict):
        print(f"[metrics] unexpected format (first element is "
              f"{type(metrics[0]).__name__}), "
              "expected list of dicts from compute_registration_metrics().")
        return

   
    labels_raw = [r["label"]    for r in metrics]
    coverage   = np.array([r["coverage"] for r in metrics], dtype=float)
    ncc        = np.array([r["ncc"]      for r in metrics], dtype=float)
    nmi        = np.array([r["nmi"]      for r in metrics], dtype=float)

    # short display labels (strip common BIDS prefixes)
    def _short(lbl):
        for prefix in ("acq-OrigRes_", "acq-", "desc-"):
            lbl = lbl.replace(prefix, "")
        return lbl
    labels = [_short(l) for l in labels_raw]

    cmap_cov = plt.get_cmap("viridis")
    vmin_c, vmax_c = coverage.min(), max(coverage.max(), 1e-6)
    dot_colors = [cmap_cov((v - vmin_c) / (vmax_c - vmin_c)) for v in coverage]

    BG   = "#0d0d0d"
    PAN  = "#1a1a1a"
    GREY = "#444444"
    TXT  = "white"

    fig, axes = plt.subplots(
        2, 2,
        figsize=(15, 11),
        facecolor=BG,
    )
    fig.subplots_adjust(hspace=0.42, wspace=0.38)

  
    def _lollipop(ax, values, sort_idx, xlabel, title, ref_line=None):
        ax.set_facecolor(PAN)
        y = np.arange(len(sort_idx))
        vlabels = [labels[i] for i in sort_idx]
        vvals   = values[sort_idx]
        vcols   = [dot_colors[i] for i in sort_idx]

        ax.hlines(y, 0, vvals, color=GREY, linewidth=1.2, zorder=1)
        sc = ax.scatter(vvals, y, c=[coverage[i] for i in sort_idx],
                        cmap="viridis", vmin=vmin_c, vmax=vmax_c,
                        s=70, zorder=3, edgecolors="white", linewidths=0.4)

        if ref_line is not None:
            ax.axvline(ref_line, color="white", linestyle="--",
                       linewidth=0.7, alpha=0.5, zorder=2)

        ax.set_yticks(y)
        ax.set_yticklabels(vlabels, color=TXT, fontsize=7.5)
        ax.set_xlabel(xlabel, color=TXT, fontsize=9)
        ax.set_title(title, color=TXT, fontsize=10, pad=6)
        ax.tick_params(colors=TXT, labelsize=8)
        ax.grid(axis="x", color=GREY, linewidth=0.4, zorder=0)
        for spine in ax.spines.values():
            spine.set_edgecolor(GREY)
        return sc

    # panel 1: NCC ranked by |NCC| ascending so strongest anti-correlation is at top.
    # Values are expected to be negative (T1w bright in WM, MRSI higher in GM).
    # Top = most negative = largest |NCC| = best structural coherence.
    order_ncc = np.argsort(np.abs(ncc))
    sc1 = _lollipop(axes[0, 0], ncc, order_ncc,
                    "NCC  (negative = anti-correlated, expected for MRSI vs T1w UNI-DEN)",
                    "NCC  –  ranked by |NCC|  (↑ most anti-correlated = best)",
                    ref_line=0.0)

    # panel 2: NMI ranked (sorted by NMI ascending so best is at top)
    order_nmi = np.argsort(nmi)
    sc2 = _lollipop(axes[0, 1], nmi, order_nmi,
                    "Normalised Mutual Information (NMI)",
                    "NMI  –  ranked by quality  (↑ best)")

    # shared colorbar for both lollipop panels
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(vmin=vmin_c, vmax=vmax_c))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes[0, :], orientation="vertical",
                      fraction=0.015, pad=0.02, shrink=0.8)
    cb.set_label("Coverage", color=TXT, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=TXT, labelcolor=TXT)

    
    def _scatter(ax, xvals, yvals, xlabel, ylabel, title,
                 xref=None, yref=None):
        ax.set_facecolor(PAN)
        sc = ax.scatter(xvals, yvals, c=coverage,
                        cmap="viridis", vmin=vmin_c, vmax=vmax_c,
                        s=65, zorder=3, edgecolors="white", linewidths=0.4)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (xvals[i], yvals[i]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=6.5, color="lightgrey", zorder=4)
        if xref is not None:
            ax.axvline(xref, color="white", linestyle="--",
                       linewidth=0.7, alpha=0.5)
        if yref is not None:
            ax.axhline(yref, color="white", linestyle="--",
                       linewidth=0.7, alpha=0.5)
        ax.set_xlabel(xlabel, color=TXT, fontsize=9)
        ax.set_ylabel(ylabel, color=TXT, fontsize=9)
        ax.set_title(title, color=TXT, fontsize=10, pad=6)
        ax.tick_params(colors=TXT, labelsize=8)
        ax.grid(color=GREY, linewidth=0.4, zorder=0)
        for spine in ax.spines.values():
            spine.set_edgecolor(GREY)
        return sc

    # panel 3: NCC vs NMI scatter
    _scatter(axes[1, 0], ncc, nmi,
             "NCC", "NMI",
             "NCC vs NMI  (colour = coverage)",
             xref=0.0)

    # panel 4: Coverage vs NCC
    _scatter(axes[1, 1], coverage, ncc,
             "Coverage", "NCC",
             "Coverage vs NCC  (colour = coverage)",
             yref=0.0)

    fig.suptitle(
        f"{subj}  {ses}  –  Registration quality metrics  (Reg-17 total pipeline)",
        color=TXT, fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.show()

    # print table 
    print(f"\n{'Label':<35}  {'Coverage':>9}  {'NCC':>8}  {'NMI':>8}")
    print("─" * 65)
    for r in metrics:
        print(f"{r['label']:<35}  {r['coverage']:>9.3f}  {r['ncc']:>8.4f}  {r['nmi']:>8.4f}")

def plot_final_registration_mosaic(
    t1w_img: nib.Nifti1Image,
    reg_mrsi_imgs: dict,
    subj: str = "sub-01",
    ses: str = "ses-01",
    cmap: str = "hot",
    alpha: float = 0.6,
    cols: int = 4):
    if not reg_mrsi_imgs:
        print("[mosaic] no registered metabolite images to display.")
        return

    n    = len(reg_mrsi_imgs)
    rows = (n + cols - 1) // cols

    t1w_data = t1w_img.get_fdata().astype(np.float32)
    pos      = t1w_data[t1w_data > 0]
    t1w_norm = np.clip(t1w_data / (float(np.nanpercentile(pos, 99)) if pos.size else 1.0), 0, 1)

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(5 * cols, 4 * rows),
        facecolor="black",
    )
    axes = np.array(axes).reshape(-1)

    for ax, (label, img) in zip(axes, reg_mrsi_imgs.items()):
        mrsi_data = img.get_fdata().astype(np.float32)
        # pick axial slice with most signal
        sig_per_z = (mrsi_data > 0).sum(axis=(0, 1))
        best_z    = int(sig_per_z.argmax()) if sig_per_z.max() > 0 else t1w_norm.shape[2] // 2
        t1w_sl    = t1w_norm[:, :, best_z]
        mrsi_sl   = mrsi_data[:, :, best_z]
        vmax      = float(np.nanpercentile(mrsi_data[mrsi_data > 0], 99)) \
                    if (mrsi_data > 0).any() else 1.0

        ax.set_facecolor("black")
        ax.imshow(t1w_sl.T,  origin="lower", cmap="gray", vmin=0, vmax=1)
        ax.imshow(mrsi_sl.T, origin="lower", cmap=cmap,   vmin=0, vmax=vmax, alpha=alpha)
        ax.set_title(label, color="white", fontsize=8, pad=3)
        ax.axis("off")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"{subj} {ses}  –  All metabolites in T1w space (Reg-17 total pipeline)\n"
        f"T1w background (gray) + MRSI overlay ({cmap}, α={alpha})",
        color="white", fontsize=10, y=1.01,
    )
    plt.tight_layout()
    plt.show()


def plot_segmentation(
    t1w_img: nib.Nifti1Image,
    seg_imgs: dict,
    subj: str = "sub-01",
    ses: str = "ses-01",
    alpha: float = 0.55,
):
    """Visualise FSL FAST tissue segmentation overlaid on the T1w image.

    Parameters
    ----------
    t1w_img : nib.Nifti1Image
        Background anatomical image (skull-stripped DS T1w recommended).
    seg_imgs : dict
        Output of ``segment_t1w`` — keys ``csf``, ``gm``, ``wm``, ``seg``.
    subj, ses : str
        Subject / session labels for the figure title.
    alpha : float
        Overlay transparency for the PVE maps.
    """
    BG   = "black"
    TXT  = "white"

    tissues = [
        ("csf", "CSF",          "Blues"),
        ("gm",  "Grey Matter",  "Reds"),
        ("wm",  "White Matter", "Greens"),
    ]

    # ── row 1: PVE maps (one column per tissue class) ──────────────────────
    fig, axes = plt.subplots(
        2, 3,
        figsize=(15, 9),
        facecolor=BG,
    )
    fig.suptitle(
        f"{subj}  {ses}  –  FSL FAST segmentation  (skull-stripped DS T1w)",
        color=TXT, fontsize=12,
    )

    cut_coords = (0.0, 0.0, 0.0)
    try:
        import nilearn.image as nlimg
        com = get_nonzero_com(t1w_img)
        cut_coords = com
    except Exception:
        pass

    for col, (key, title, cmap) in enumerate(tissues):
        overlay = seg_imgs.get(key)

        # ── top row: ortho stat_map overlay ────────────────────────────────
        ax_top = axes[0, col]
        if overlay is not None:
            data  = overlay.get_fdata()
            vmax  = float(np.nanpercentile(data[data > 0], 99)) if (data > 0).any() else 1.0
            disp  = plotting.plot_stat_map(
                stat_map_img=overlay,
                bg_img=t1w_img,
                display_mode="ortho",
                cut_coords=cut_coords,
                cmap=cmap,
                colorbar=True,
                threshold=0.05,
                vmax=vmax,
                title=f"{title} (PVE)",
                axes=ax_top,
                black_bg=True,
            )
        else:
            ax_top.set_facecolor(BG)
            ax_top.text(0.5, 0.5, f"{title}\nnot available",
                        ha="center", va="center", color=TXT, fontsize=11)
            ax_top.axis("off")

    # ── row 2: hard segmentation (all three classes in one figure) ─────────
    seg_img = seg_imgs.get("seg")
    ax_bot_mid = axes[1, 1]
    axes[1, 0].set_visible(False)
    axes[1, 2].set_visible(False)

    if seg_img is not None:
        seg_data = seg_img.get_fdata()
        # Encode CSF=1→blue, GM=2→red, WM=3→white in a single label image
        # FAST labels: 1=CSF, 2=GM, 3=WM
        plotting.plot_roi(
            roi_img=seg_img,
            bg_img=t1w_img,
            display_mode="ortho",
            cut_coords=cut_coords,
            title="Hard segmentation  (1=CSF · 2=GM · 3=WM)",
            axes=ax_bot_mid,
            black_bg=True,
            cmap="gist_rainbow",
            alpha=alpha,
        )
    else:
        ax_bot_mid.set_facecolor(BG)
        ax_bot_mid.text(0.5, 0.5, "seg not available",
                        ha="center", va="center", color=TXT, fontsize=11)
        ax_bot_mid.axis("off")

    plt.tight_layout()
    plt.show()

    # ── volume summary ──────────────────────────────────────────────────────
    vox_vol_mm3 = float(np.abs(np.linalg.det(t1w_img.affine[:3, :3])))
    print(f"\n{'Tissue':<18}  {'Mean PVE':>9}  {'Volume (mL)':>12}")
    print("─" * 44)
    for key, title, _ in tissues:
        img = seg_imgs.get(key)
        if img is None:
            print(f"  {title:<16}  {'n/a':>9}  {'n/a':>12}")
            continue
        data    = img.get_fdata()
        mean_pv = float(data[data > 0].mean()) if (data > 0).any() else 0.0
        vol_ml  = float(data.sum()) * vox_vol_mm3 / 1000.0
        print(f"  {title:<16}  {mean_pv:>9.3f}  {vol_ml:>11.1f} mL")


def plot_atlas_segmentation(
    t1w_img: nib.Nifti1Image,
    atlas_imgs: dict,
    subj: str = "sub-01",
    ses: str = "ses-01",
) -> None:
    """Visualise Harvard-Oxford cortical and subcortical atlas labels on T1w.

    Parameters
    ----------
    t1w_img : nib.Nifti1Image
        Background anatomical image (skull-stripped DS T1w, same space as
        atlas labels).
    atlas_imgs : dict
        Output of :func:`segment_t1w_atlas`.  Expected keys: ``atlas_cort``,
        ``atlas_sub``, ``labels_cort``, ``labels_sub``.
    subj, ses : str
        Subject / session labels for figure titles.
    """
    from nilearn import plotting

    BG  = "black"
    TXT = "white"

    atlas_cort  = atlas_imgs.get("atlas_cort")
    atlas_sub   = atlas_imgs.get("atlas_sub")
    labels_cort = atlas_imgs.get("labels_cort", [])
    labels_sub  = atlas_imgs.get("labels_sub",  [])

    cut_coords = (0.0, 0.0, 0.0)
    try:
        cut_coords = get_nonzero_com(t1w_img)
    except Exception:
        pass

    vox_vol_mm3 = float(np.abs(np.linalg.det(t1w_img.affine[:3, :3])))

    # ── Cortical atlas ────────────────────────────────────────────────────
    if atlas_cort is not None:
        fig_c = plt.figure(figsize=(14, 4), facecolor=BG)
        ax_c  = fig_c.add_subplot(111)
        plotting.plot_roi(
            roi_img=atlas_cort,
            bg_img=t1w_img,
            display_mode="ortho",
            cut_coords=cut_coords,
            title=f"{subj}  {ses}  –  Harvard-Oxford Cortical (FLIRT → native T1w)",
            axes=ax_c,
            black_bg=True,
            cmap="gist_ncar",
            alpha=0.65,
        )
        plt.tight_layout()
        plt.show()
    else:
        print("[atlas] atlas_cort not available – run segment_t1w_atlas() first.")

    # ── Subcortical atlas ─────────────────────────────────────────────────
    if atlas_sub is not None:
        fig_s = plt.figure(figsize=(14, 4), facecolor=BG)
        ax_s  = fig_s.add_subplot(111)
        plotting.plot_roi(
            roi_img=atlas_sub,
            bg_img=t1w_img,
            display_mode="ortho",
            cut_coords=cut_coords,
            title=f"{subj}  {ses}  –  Harvard-Oxford Subcortical (FLIRT → native T1w)",
            axes=ax_s,
            black_bg=True,
            cmap="Set1",
            alpha=0.70,
        )
        plt.tight_layout()
        plt.show()
    else:
        print("[atlas] atlas_sub not available – run segment_t1w_atlas() first.")

    # ── Region volume tables ───────────────────────────────────────────────
    for name, img, labs in [
        ("Cortical",    atlas_cort, labels_cort),
        ("Subcortical", atlas_sub,  labels_sub),
    ]:
        if img is None:
            continue
        data = np.asarray(img.dataobj).round().astype(int)
        uniq, counts = np.unique(data[data > 0], return_counts=True)
        print(f"\n{'─'*60}")
        print(f"  Harvard-Oxford {name} regions  "
              f"(native T1w space,  thr25, 2 mm)")
        print(f"{'─'*60}")
        print(f"  {'Idx':>4}  {'Region':<44}  {'Vol (mL)':>9}")
        print(f"{'─'*60}")
        for idx, cnt in zip(uniq, counts):
            lbl = labs[int(idx)] if int(idx) < len(labs) else f"Region {idx}"
            vol = cnt * vox_vol_mm3 / 1000.0
            print(f"  {int(idx):>4}  {lbl:<44}  {vol:>8.2f}")
        print(f"{'─'*60}")

# ──────────────────────────────────────────────────────────────────────────
# Mask diagnostic plots
# ──────────────────────────────────────────────────────────────────────────

def plot_mrsi_sum_mask_contours(
    sum_img:        "nib.Nifti1Image",
    water_mask_img: "nib.Nifti1Image",
    bet_mask_img:   "nib.Nifti1Image | None",
    subj: str = "sub-01",
    ses:  str = "ses-01",
    n_slices: int = 7,
) -> None:
    """Axial multi-slice panel: MRSI sum as greyscale background with three
    contour overlays (all images must share the same voxel grid):

    - Yellow : MRSI sum signal boundary (signal > 0)
    - Blue   : water mask boundary
    - Green  : BET brain mask boundary (omitted when *bet_mask_img* is None)
    """
    sum_data   = sum_img.get_fdata().astype(np.float32)
    water_data = water_mask_img.get_fdata().astype(bool)
    bet_data   = bet_mask_img.get_fdata().astype(bool) if bet_mask_img is not None else None

    nz = sum_data.shape[2]
    z_indices = np.linspace(0, nz - 1, n_slices, dtype=int)

    sum_max = np.percentile(sum_data[sum_data > 0], 99) if sum_data.max() > 0 else 1.0

    fig, axes = plt.subplots(1, n_slices, figsize=(3.2 * n_slices, 4), facecolor="black")
    if n_slices == 1:
        axes = [axes]

    legend_handles = []
    for ax, z in zip(axes, z_indices):
        ax.set_facecolor("black")
        slc = sum_data[:, :, z].T
        ax.imshow(slc, origin="lower", cmap="gray",
                  vmin=0, vmax=sum_max, interpolation="nearest")

        # Sum signal contour (yellow)
        sum_bin = (sum_data[:, :, z] > 0).astype(np.float32).T
        if sum_bin.max() > 0:
            ax.contour(sum_bin, levels=[0.5], colors=["gold"], linewidths=1.2)
            if not legend_handles:
                legend_handles.append(
                    plt.matplotlib.lines.Line2D([], [], color="gold",
                                                linewidth=1.5, label="Sum signal boundary"))

        # Water mask contour (blue)
        water_bin = water_data[:, :, z].astype(np.float32).T
        if water_bin.max() > 0:
            ax.contour(water_bin, levels=[0.5], colors=["deepskyblue"], linewidths=1.2)
            if len(legend_handles) < 2:
                legend_handles.append(
                    plt.matplotlib.lines.Line2D([], [], color="deepskyblue",
                                                linewidth=1.5, label="Water mask boundary"))

        # BET mask contour (green)
        if bet_data is not None:
            bet_bin = bet_data[:, :, z].astype(np.float32).T
            if bet_bin.max() > 0:
                ax.contour(bet_bin, levels=[0.5], colors=["limegreen"], linewidths=1.2)
                if len(legend_handles) < 3:
                    legend_handles.append(
                        plt.matplotlib.lines.Line2D([], [], color="limegreen",
                                                    linewidth=1.5, label="BET brain mask boundary"))

        ax.set_title(f"z={z}", color="white", fontsize=9)
        ax.axis("off")

    fig.suptitle(f"{subj}  {ses}  –  MRSI sum signal vs mask contours",
                 color="white", fontsize=12)
    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
                   frameon=False, labelcolor="white", fontsize=9,
                   bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    plt.show()


def plot_bet_vs_water_mask(
    bet_mask_img:    "nib.Nifti1Image",
    water_mask_img:  "nib.Nifti1Image",
    t1w_ds_img:      "nib.Nifti1Image",
    subj:            str = "sub-01",
    ses:             str = "ses-01",
    active_mask_name: str | None = None,
    n_slices: int = 7,
) -> None:
    """Three-row axial mosaic comparing the BET brain mask and the water mask,
    both on the MRSI voxel grid:

    - Row 1 : BET mask (green overlay) on DS T1w background
    - Row 2 : Water mask (blue overlay) on DS T1w background
    - Row 3 : Comparison — BET-only (red), water-only (blue), overlap (grey)

    Also prints a quantitative table (volumes, Dice, Jaccard).
    """
    bet_data   = bet_mask_img.get_fdata().astype(bool)
    water_data = water_mask_img.get_fdata().astype(bool)
    t1w_data   = t1w_ds_img.get_fdata().astype(np.float32)

    vox_vol_mm3 = float(np.abs(np.linalg.det(bet_mask_img.affine[:3, :3])))

    intersection  = bet_data & water_data
    bet_only      = bet_data & ~water_data
    water_only    = water_data & ~bet_data

    n_bet    = int(bet_data.sum())
    n_water  = int(water_data.sum())
    n_inter  = int(intersection.sum())
    n_bonly  = int(bet_only.sum())
    n_wonly  = int(water_only.sum())

    dice    = 2 * n_inter / (n_bet + n_water + 1e-10)
    jaccard = n_inter / (n_bet + n_water - n_inter + 1e-10)

    # ── Text table ────────────────────────────────────────────────────────
    sep = "─" * 44
    print(sep)
    print(f"  Mask comparison (MRSI grid)")
    print(sep)
    print(f"  BET mask volume   : {n_bet   * vox_vol_mm3/1000:>8.1f} mL  ({n_bet:>6} vox)")
    print(f"  Water mask volume : {n_water * vox_vol_mm3/1000:>8.1f} mL  ({n_water:>6} vox)")
    print(f"  Intersection      : {n_inter * vox_vol_mm3/1000:>8.1f} mL  ({n_inter:>6} vox)")
    print(f"  BET-only voxels   : {n_bonly * vox_vol_mm3/1000:>8.1f} mL  ({n_bonly:>6} vox)")
    print(f"  Water-only voxels : {n_wonly * vox_vol_mm3/1000:>8.1f} mL  ({n_wonly:>6} vox)")
    print(f"  Dice coefficient  : {dice:.4f}  (1.0 = identical)")
    print(f"  Jaccard index     : {jaccard:.4f}")
    print(f"  Masks are identical : {bool(np.array_equal(bet_data, water_data))}")
    print(sep)
    if active_mask_name:
        print(f"\n  ACTIVE mask for Reg 18 : {active_mask_name}")

    # ── Axial mosaic ──────────────────────────────────────────────────────
    nz = bet_data.shape[2]
    z_indices = np.linspace(0, nz - 1, n_slices, dtype=int)

    t1w_max = np.percentile(t1w_data[t1w_data > 0], 99) if t1w_data.max() > 0 else 1.0

    fig, axes = plt.subplots(3, n_slices,
                             figsize=(2.8 * n_slices, 9),
                             facecolor="black")

    row_labels = ["BET mask", "Water mask", "Comparison"]

    for col, z in enumerate(z_indices):
        t1w_slc = t1w_data[:, :, z].T
        bet_slc = bet_data[:, :, z].T.astype(np.float32)
        wat_slc = water_data[:, :, z].T.astype(np.float32)

        # Row 0 – BET mask on T1w
        ax = axes[0, col]
        ax.set_facecolor("black")
        ax.imshow(t1w_slc, origin="lower", cmap="gray",
                  vmin=0, vmax=t1w_max, interpolation="nearest")
        ax.imshow(np.ma.masked_where(bet_slc == 0, bet_slc),
                  origin="lower", cmap="Greens", vmin=0, vmax=1,
                  alpha=0.55, interpolation="nearest")
        ax.set_title(f"z={z}", color="white", fontsize=8)
        ax.axis("off")

        # Row 1 – Water mask on T1w
        ax = axes[1, col]
        ax.set_facecolor("black")
        ax.imshow(t1w_slc, origin="lower", cmap="gray",
                  vmin=0, vmax=t1w_max, interpolation="nearest")
        ax.imshow(np.ma.masked_where(wat_slc == 0, wat_slc),
                  origin="lower", cmap="Blues", vmin=0, vmax=1,
                  alpha=0.55, interpolation="nearest")
        ax.axis("off")

        # Row 2 – Comparison: BET-only red, water-only blue, overlap grey
        ax = axes[2, col]
        ax.set_facecolor("black")
        ax.imshow(t1w_slc, origin="lower", cmap="gray",
                  vmin=0, vmax=t1w_max, alpha=0.25, interpolation="nearest")
        # overlap
        ov = intersection[:, :, z].T.astype(np.float32)
        ax.imshow(np.ma.masked_where(ov == 0, ov),
                  origin="lower", cmap="gray", vmin=0, vmax=1,
                  alpha=0.65, interpolation="nearest")
        # water-only
        wo = water_only[:, :, z].T.astype(np.float32)
        ax.imshow(np.ma.masked_where(wo == 0, wo),
                  origin="lower", cmap="Blues", vmin=0, vmax=1,
                  alpha=0.80, interpolation="nearest")
        # BET-only
        bo = bet_only[:, :, z].T.astype(np.float32)
        ax.imshow(np.ma.masked_where(bo == 0, bo),
                  origin="lower", cmap="Reds", vmin=0, vmax=1,
                  alpha=0.80, interpolation="nearest")
        ax.axis("off")

    for row, lbl in enumerate(row_labels):
        axes[row, 0].set_ylabel(lbl, color="white", fontsize=10,
                                rotation=90, labelpad=6)

    legend_patches = [
        plt.matplotlib.patches.Patch(color="red",        label=f"BET only ({n_bonly:,} vox)"),
        plt.matplotlib.patches.Patch(color="dodgerblue", label=f"Water only ({n_wonly:,} vox)"),
        plt.matplotlib.patches.Patch(color="grey",       label=f"Overlap ({n_inter:,} vox)"),
    ]
    title = (f"BET mask vs water mask (MRSI grid)  |  "
             f"Dice={dice:.3f}  Jaccard={jaccard:.3f}")
    if active_mask_name:
        title += f"  |  Active: {active_mask_name}"
    fig.suptitle(title, color="white", fontsize=10)
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               frameon=False, labelcolor="white", fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.show()

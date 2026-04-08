import os
import re
import ants
import nibabel as nib
import numpy as np
import subprocess
import shutil
from nibabel.processing import resample_from_to
from scipy.ndimage import binary_fill_holes


# Internal helpers

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
  
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    tmp_in  = os.path.join(tmp_dir, "input.nii.gz")
    tmp_out = os.path.join(tmp_dir, "reoriented.nii.gz")
    try:
        shutil.copy2(in_path, tmp_in)
        subprocess.run(
            ["fslreorient2std", tmp_in, tmp_out],
            check=True, capture_output=True,
        )
        img = nib.load(tmp_out)
        img = nib.Nifti1Image(img.get_fdata(), img.affine, img.header)
        return img
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"fslreorient2std failed for {in_path}:\n{e.stderr.decode()}"
        ) from e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)



# Data utilities


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

    # Cut the bottom 5% of positive water signal values (noise floor) and
    # keep the top 95%  threshold at the 5th percentile of positive voxels.
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

    conc_files = sorted(
        f for f in os.listdir(mrs_dir)
        if f.endswith(".nii.gz")
        and "acq-OrigRes" in f
        and "AllMetabSum" not in f)

    records = []
    for fname in conc_files:
        data = nib.load(os.path.join(mrs_dir, fname)).get_fdata()
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
    out_dir: str | None = None,
    mask_dir: str | None = None,
    subjects: list[str] | None = None):
   
    saved = {}

    candidates = subjects if subjects is not None else sorted(os.listdir(bids_dir))
    for subj in candidates:
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

        maps = sorted(
            f for f in os.listdir(mrs_dir)
            if f.endswith(".nii.gz")
            and "acq-OrigRes" in f
            and "AllMetabSum" not in f
        )
        if not maps:
            print(f"  [sum] {subj}: no OrigRes maps found, skipping.")
            continue

        ref_img = nib.load(os.path.join(mrs_dir, maps[0]))
        accum   = np.zeros(ref_img.shape, dtype=np.float32)

        # Build a water-signal coverage mask to suppress lipid/artifact voxels
        support_mask = None
        _mask_path = (
            os.path.join(mask_dir, f"{subj}_{ses}_desc-WaterMask_mrsi.nii.gz")
            if mask_dir else None
        )
        if _mask_path and os.path.exists(_mask_path):
            support_mask = nib.load(_mask_path).get_fdata().astype(bool)
        else:
            _water_path = os.path.join(mrs_dir, f"{subj}_{ses}_desc-WaterSignal_mrsi.nii.gz")
            if os.path.exists(_water_path):
                w = nib.load(_water_path).get_fdata().astype(np.float32)
                finite_pos = w[np.isfinite(w) & (w > 0)]
                thresh = float(np.percentile(finite_pos, 5)) if finite_pos.size > 0 else 0.0
                raw = np.isfinite(w) & (w > thresh)
                support_mask = np.stack(
                    [binary_fill_holes(raw[..., z]) for z in range(raw.shape[2])],
                    axis=2,
                )

        n_used = 0
        for fname in maps:
            data = nib.load(os.path.join(mrs_dir, fname)).get_fdata().astype(np.float32)
            data = np.nan_to_num(data, nan=0.0)
            if support_mask is not None:
                data = np.where(support_mask, data, 0.0)
            accum += data
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
    out_dir: str | None = None,
    subjects: list[str] | None = None):
    
    saved = {}

    candidates = subjects if subjects is not None else sorted(os.listdir(bids_dir))
    for subj in candidates:
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
    out_dir: str | None = None,
    subjects: list[str] | None = None):
    
    saved = {}

    candidates = subjects if subjects is not None else sorted(os.listdir(bids_dir))
    for subj in candidates:
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
    overwrite: bool = False):

    seg_path = f"{out_prefix}_seg.nii.gz"
    pve0     = f"{out_prefix}_pve_0.nii.gz"

    if os.path.exists(seg_path) and os.path.exists(pve0) and not overwrite:
        print(f"  [fast] already exists: {os.path.basename(seg_path)}")
    else:
        cmd = [
            "fast",
            "-n", str(n_classes),
            "-t", "1",         
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
    overwrite: bool = False):
    
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

        print(f"  [atlas] done  {os.path.basename(out_cort)}, {os.path.basename(out_sub)}")

    def _parse_labels(xml_path):
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



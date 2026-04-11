import os
import ants
import nibabel as nib
import numpy as np
import subprocess
import shutil
from data_utils import (
    _nib_to_ants,
    metabolite_name,
    _mutual_information,
    _fsl_reorient2std,
    fill_mask_holes,
    save_reoriented_metabolite_sum,
)

# Registration utilities

def _f32(img: nib.Nifti1Image) :
    return img.get_fdata()


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

    mrsi_data   = _f32(mrsi_img)
    mrsi_masked = np.where(mask, mrsi_data, 0.0)
    t1w_data    = _f32(t1w_img)

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
        moving=_nib_to_ants(mask.astype(float), mrsi_img),
        transformlist=fwd_transforms,
        interpolator="nearestNeighbor")
    brain_mask_t1w = warped_mask_ants.numpy() > 0.5

    reg_data = warped_ants.numpy()
    reg_data[~brain_mask_t1w] = 0.0

    reg_img = nib.Nifti1Image(reg_data, t1w_img.affine, t1w_img.header)

    if out_path:
        nib.save(reg_img, out_path)
        print(f"  [reg] saved registered MRSI: {out_path}")

    return reg_img, used_transforms

def register_t1w_to_mrsi(
    t1w_img: nib.Nifti1Image,
    mrsi_img: nib.Nifti1Image,
    out_path: str | None = None,
    overwrite: bool = False,
    init_transforms: list | None = None,
    mask: np.ndarray | None = None):

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

    t1w_data  = _f32(t1w_img)
    mrsi_data = _f32(mrsi_img)
    mrsi_fixed = np.where(mask, mrsi_data, 0.0) if mask is not None else mrsi_data

    fixed_ants  = _nib_to_ants(mrsi_fixed, mrsi_img)
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

    reg_data = warped_ants.numpy()
    reg_img  = nib.Nifti1Image(reg_data, mrsi_img.affine, mrsi_img.header)

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

def apply_transform(
    in_path: str,
    ref_path: str,
    transform_path: str,
    out_path: str,
    interpolation: str = "Linear",
    overwrite: bool = False) -> "nib.Nifti1Image | None":
    """Apply a pre-computed ANTs transform to any NIfTI image."""

    if os.path.exists(out_path) and not overwrite:
        print(f"  [xfm] already exists: {os.path.basename(out_path)}")
        return nib.load(out_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        "antsApplyTransforms", "-d", "3",
        "-i", in_path,
        "-r", ref_path,
        "-t", transform_path,
        "-n", interpolation,
        "-o", out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"  [xfm] saved: {os.path.basename(out_path)}")
    return nib.load(out_path)


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

    cmd = [
        "antsApplyTransforms", "-d", "3",
        "-i", mrsi_path,
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

def apply_transform_all_metabolites(
    mrs_dir: str,
    transform_path: str,
    t1w_ref_path: str,
    out_dir: str,
    subj: str = "sub-01",
    ses: str = "ses-01",
    overwrite: bool = False,
    out_suffix: str = ""):
   
    maps = sorted(
        f for f in os.listdir(mrs_dir)
        if f.endswith(".nii.gz") and "acq-OrigRes" in f and "AllMetab" not in f
    )
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for fname in maps:
        label    = metabolite_name(fname)
        out_name = f"{subj}_{ses}_acq-MRSIres_desc-{label}Reg17{out_suffix}_T1w.nii.gz"
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
    
    t1w_data   = _f32(t1w_img)
    brain_mask = t1w_data > 0
    n_brain    = int(brain_mask.sum())

    records = []
    for label, img in reg_mrsi_imgs.items():
        mrsi_data   = _f32(img)
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
            # in-region SNR: mean / std within overlap (higher = more uniform signal = better reg)
            snr = float(mrsi_vals.mean()) / (float(mrsi_vals.std()) + 1e-9)
        else:
            ncc = 0.0
            nmi = 0.0
            snr = 0.0

        records.append({"label": label, "coverage": coverage, "ncc": ncc, "nmi": nmi, "snr": snr})
    records.sort(key=lambda r: abs(r["ncc"]), reverse=True)
    return records

def run_total_pipeline(
    mrs_dir: str,
    water_path: str,
    t1w_ds_brain_path: str,
    t1w_ds_brain_img: "nib.Nifti1Image",
    subj: str,
    ses: str,
    output_dir: str,
    sum_ras_path: str,
    overwrite: bool = False,
    t1w_brain_mask_ds_path: str | None = None,
    out_suffix: str = ""):
  

    sum_ras_img = nib.load(sum_ras_path) 

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
    _desc = f"BrainT1wInSumRAS{out_suffix}"
    t1w_brain_in_sum_ras_name = f"{subj}_{ses}_acq-MRSIres_desc-{_desc}_T1w.nii.gz"
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
            out_suffix=out_suffix,
        )
        # Also register the RAS water signal with the same transform (used as reference)
        os.makedirs(final_reg_dir, exist_ok=True)
        water_reg_name = f"{subj}_{ses}_acq-MRSIres_desc-WaterSignalReg17{out_suffix}_T1w.nii.gz"
        water_reg_img = apply_transform_to_metabolite(
            mrsi_path=water_ras_path,
            transform_path=brain_sum_ras_transforms[0],
            t1w_ref_path=t1w_ds_brain_path,
            out_path=os.path.join(final_reg_dir, water_reg_name),
            overwrite=overwrite,
        )
        if water_reg_img is not None:
            final_reg_imgs["WaterSignal"] = water_reg_img
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



def compute_tissue_fractions_in_mrsi(
    pve_gm_path: str,
    pve_wm_path: str,
    pve_csf_path: str,
    mrsi_ref_path: str,
    xfm_path: str,
    out_dir: str,
    subj: str = "sub-01",
    ses: str = "ses-01",
    overwrite: bool = False):
   
    os.makedirs(out_dir, exist_ok=True)

    def _warp(pve_path, tissue_label):
        if not pve_path or not os.path.exists(pve_path):
            return None
        out_name = f"{subj}_{ses}_desc-{tissue_label}PVE_mrsi.nii.gz"
        out_path = os.path.join(out_dir, out_name)
        if os.path.exists(out_path) and not overwrite:
            return nib.load(out_path)
        cmd = [
            "antsApplyTransforms", "-d", "3",
            "-i", pve_path,
            "-r", mrsi_ref_path,
            "-t", xfm_path,
            "-n", "Linear",
            "-o", out_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"  [tissue-pve] warped {tissue_label} → {out_name}")
        return nib.load(out_path)

    return {
        "gm":  _warp(pve_gm_path,  "GM"),
        "wm":  _warp(pve_wm_path,  "WM"),
        "csf": _warp(pve_csf_path, "CSF"),
    }



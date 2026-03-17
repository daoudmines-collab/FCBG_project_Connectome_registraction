import os
import re
import ants
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from nibabel.processing import resample_from_to
from nilearn import plotting
from scipy.ndimage import binary_fill_holes


def img_info(img: nib.Nifti1Image, label: str) -> None:
    """Print shape, voxel size and dtype of a NIfTI image."""
    vox = np.sqrt(np.sum(img.affine[:3, :3] ** 2, axis=0))
    print(f"{label}")
    print(f"  shape      : {img.shape}")
    print(f"  voxel size : {vox.round(3)} mm")
    print(f"  dtype      : {img.get_data_dtype()}")
    print()


def fill_mask_holes(
    water_img_path: str,
    threshold: float = 0.0,
    out_mask_path: str | None = None,
    overwrite: bool = False,
) -> nib.Nifti1Image:
    """
    Build a binary mask from a water signal image and fill internal holes.
    """
    if not os.path.exists(water_img_path):
        raise FileNotFoundError(f"Water image not found: {water_img_path}")

    water_img = nib.load(water_img_path)
    water_data = np.asarray(water_img.get_fdata(), dtype=np.float32)

    base_mask = np.isfinite(water_data) & (water_data > threshold)
    filled_mask = binary_fill_holes(base_mask)

    mask_data = filled_mask.astype(np.uint8)
    mask_img = nib.Nifti1Image(mask_data, water_img.affine, water_img.header)
    mask_img.set_data_dtype(np.uint8)

    if out_mask_path and (overwrite or not os.path.exists(out_mask_path)):
        nib.save(mask_img, out_mask_path)

    return mask_img



def metabolite_name(filename: str) -> str:
    """Extract the BIDS desc label from an MRSI filename."""
    m = re.search(r"desc-([^_]+)_mrsi", filename)
    return m.group(1) if m else filename


def get_nonzero_com(img: nib.Nifti1Image) -> tuple:
    """coords of nonzero finite voxels."""
    data = img.get_fdata()
    mask = np.isfinite(data) & (data > 0)
    if not mask.any():
        return (0.0, 0.0, 0.0)
    ijk = np.argwhere(mask).mean(axis=0)
    xyz = nib.affines.apply_affine(img.affine, ijk)
    return tuple(float(v) for v in xyz[:3])


def estimate_coverage(
    mrs_resampled_img: nib.Nifti1Image,
    t1_img: nib.Nifti1Image,
) -> float:
    """Return fraction of T1 brain voxels (T1>0) covered by the MRSI map."""
    mrs = mrs_resampled_img.get_fdata()
    t1  = t1_img.get_fdata()
    brain   = np.isfinite(t1) & (t1 > 0)
    covered = np.isfinite(mrs) & (mrs > 0)
    frac = float(np.mean(covered[brain])) if brain.any() else 0.0
    print(f"  Brain coverage: {frac * 100:.1f}%")
    return frac

def plot_single_overlay(
    mrs_filename: str,
    t1w_path: str,
    mrs_dir: str,
    subj: str = "sub-01",
    cmap: str = "hot",
    threshold=None,
    vmax=None,
    verbose: bool = False,
) -> float:
    """
    Canonicalize, resample, and plot one MRSI map on the T1w.
    Returns the brain coverage fraction.
    """
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
    subj: str = "sub-01",
) -> float:
    """Binary MRSI support mask overlaid on T1w. Returns coverage fraction."""
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
    subj: str = "sub-01",
):
    """
    Plot each metabolite map overlaid on the T1w, one figure per metabolite.
    """
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


def rank_metabolites_by_snr(
    mrs_dir: str,
    subj: str = "sub-01",
    ses: str = "ses-01",
    top_n: int | None = None,
) -> list[dict]:
    """
    Rank all ``acq-OrigRes`` concentration maps by their effective SNR,
    using the scanner-provided per-voxel SNR map (desc-VoxelSNR_mrsi).
    For each metabolite map the function computes inside the voxels where
    that map has positive signal
    """
    # locate the VoxelSNR map
    snr_candidates = [
        f for f in os.listdir(mrs_dir)
        if "VoxelSNR" in f and f.endswith(".nii.gz")
    ]
    if not snr_candidates:
        raise FileNotFoundError(
            f"No VoxelSNR map found in {mrs_dir}. "
            "Expected a file matching 'VoxelSNRmrsi.nii.gz'."
        )
    snr_map = nib.load(os.path.join(mrs_dir, snr_candidates[0])).get_fdata()

    # collect all individual OrigRes concentration maps
    conc_files = sorted(
        f for f in os.listdir(mrs_dir)
        if f.endswith(".nii.gz")
        and "acq-OrigRes" in f
        and "AllMetabSum" not in f
    )

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
            "ses":        ses,
        })

    # sort descending by mean_snr
    records.sort(key=lambda r: r["mean_snr"], reverse=True)
    for i, r in enumerate(records):
        r["rank"] = i + 1

    return records[:top_n] if top_n is not None else records


def plot_snr_ranking(
    records: list[dict],
    top_n: int = 20,
    highlight_best: int = 3,
) -> None:
    """
    Horizontal bar chart of metabolite maps ranked by mean VoxelSNR.
    """
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


def save_metabolite_sum(
    bids_dir: str,
    ses: str = "ses-01",
    overwrite: bool = False,
    out_dir: str | None = None,
) -> dict:
    """
    For every subject in ``bids_dir``, sum all acq-OrigRes MRSI concentration
    maps in native MRS space and save the result.
    If ``out_dir`` is given, files are written there; otherwise they go into
    each subject's own mrs/ folder.
    """
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
            if data.shape != accum.shape:
                print(f"  [sum] {subj}: shape mismatch in {fname}, skipping that map.")
                continue
            accum += np.nan_to_num(data, nan=0.0)
            n_used += 1

        sum_img = nib.Nifti1Image(accum, ref_img.affine, ref_img.header)
        sum_img.set_data_dtype(np.float32)
        nib.save(sum_img, out_path)
        print(f"  [sum] {subj}: saved {out_name}  ({n_used} maps summed)")
        saved[subj] = out_path

    return saved


def downsample_t1w_to_mrs(
    bids_dir: str,
    ses: str = "ses-01",
    t1w_acq: str = "UNIDEN",
    overwrite: bool = False,
    out_dir: str | None = None,
):
    """
    For every subject in ``bids_dir`` that has an ``anat/`` folder containing
    a T1w UNI-DEN image, resample the T1w to the spatial grid of the first
    ``acq-OrigRes`` MRSI map and save the result as
    <subj>_<ses>_acq-MRSIres_T1w.nii.gz.
    If ``out_dir`` is given, files are written there; otherwise they go into
    each subject's own anat/ folder.
    """
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

        # resample T1w → MRSI grid (order=1: linear interpolation, good for anatomical)
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


def plot_t1w_mrs_comparison(
    t1w_ds_img: nib.Nifti1Image,
    mrs_img: nib.Nifti1Image,
    subj: str = "sub-01",
    mrs_label: str = "MRSI",
    t1w_cmap: str = "gray",
    mrs_cmap: str = "hot",
) -> None:
    """
    Side-by-side ortho comparison of the downsampled T1w and one MRSI map,
    both shown at the same voxel grid (native MRS space).

    Three rows:
      • Downsampled T1w  (axial / coronal / sagittal centre slices)
      • MRSI map (same slices)
      • Overlay (T1w in grey, MRSI in colour)
    """
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


def _nib_to_ants(data: np.ndarray, ref_img: nib.Nifti1Image):
    """Convert a numpy array + nibabel reference image to an ANTsPy image.

    nibabel uses RAS+ convention; ANTs/ITK uses LPS.  Negating the first two
    rows of the direction cosine matrix and the first two origin components
    performs the RAS→LPS conversion, matching ants.from_nibabel internally.
    """
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


def register_mrsi_to_t1w(
    mrsi_img: nib.Nifti1Image,
    t1w_img: nib.Nifti1Image,
    mask: np.ndarray,
    out_path: str | None = None,
    overwrite: bool = False,
    init_transforms: list | None = None,
) -> tuple:
    """
    Register an MRSI map to a T1w image using ANTsPy rigid registration.

    The T1w is the **fixed** (reference) image; the MRSI is the **moving**
    image that gets warped into T1w space.  The result is the MRSI resampled
    onto the T1w voxel grid.

    ``mask`` (boolean array in MRSI voxel space) is applied to the MRSI
    before registration so that only voxels with valid signal drive the cost
    function, and then warped into T1w space to clip the output.  The output
    is additionally clipped to the T1w outer boundary (voxels where T1w == 0
    are true image background and are zeroed in the output).

    ``init_transforms`` – optional list of ANTs transform file paths (as
    returned by a previous call to this function).  When provided the ANTs
    optimisation step is skipped entirely and these transforms are applied
    directly.  Use this when multiple MRSI maps share the same acquisition
    geometry and only one registration needs to be solved.

    Returns ``(reg_img, transform_paths)`` where ``transform_paths`` is a
    list of file paths to the saved ANTs transforms (or ``None`` on cache hit
    with no sidecar).
    """
    import ants, shutil

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

    # _nib_to_ants is defined at module level (RAS→LPS conversion).
    fixed_ants  = _nib_to_ants(t1w_data, t1w_img)
    moving_ants = _nib_to_ants(mrsi_masked, mrsi_img)

    if init_transforms is not None:
        # Skip the optimisation — apply the provided transforms directly.
        # All MRSI maps from the same session share the same voxel geometry,
        # so one solved transform is correct for all of them.
        fwd_transforms = init_transforms
        print(f"  [reg] applying pre-computed transforms (skipping optimisation)")
    else:
        # Single-step rigid registration — T1w fixed, MRSI moving.
        # ANTs' "Rigid" type already performs an internal centre-of-mass
        # initialisation before the gradient-descent optimizer runs, so a
        # separate Translation pre-step is not needed and can hurt convergence
        # when the moving image (masked MRSI) is very sparse or low-contrast.
        result = ants.registration(
            fixed=fixed_ants,
            moving=moving_ants,
            type_of_transform="Rigid",
            verbose=False,
        )
        fwd_transforms = result["fwdtransforms"]

    # Save the transform to a predictable sidecar so it survives caching
    if transform_sidecar and fwd_transforms:
        src = os.path.abspath(fwd_transforms[0])
        dst = os.path.abspath(transform_sidecar)
        if src != dst:
            shutil.copy(src, dst)
        used_transforms = [transform_sidecar]
    else:
        used_transforms = fwd_transforms

    # Apply transform to the original (unmasked) MRSI to preserve true
    # signal values (not clipped by the mask), resampled into T1w space.
    warped_ants = ants.apply_transforms(
        fixed=fixed_ants,
        moving=_nib_to_ants(mrsi_data, mrsi_img),
        transformlist=fwd_transforms,
        interpolator="linear",
    )

    # Warp the mask into T1w space so we can zero out voxels that were
    # outside the brain in MRSI space (otherwise background noise spreads
    # across the whole T1w FOV after resampling).
    warped_mask_ants = ants.apply_transforms(
        fixed=fixed_ants,
        moving=_nib_to_ants(mask.astype(np.float32), mrsi_img),
        transformlist=fwd_transforms,
        interpolator="nearestNeighbor",
    )
    brain_mask_t1w = warped_mask_ants.numpy() > 0.5

    reg_data = warped_ants.numpy().astype(np.float32)
    reg_data[~brain_mask_t1w] = 0.0

    reg_img = nib.Nifti1Image(reg_data, t1w_img.affine, t1w_img.header)
    reg_img.set_data_dtype(np.float32)

    if out_path:
        nib.save(reg_img, out_path)
        print(f"  [reg] saved registered MRSI: {out_path}")

    return reg_img, used_transforms


def plot_registration_comparison(
    t1w_img: nib.Nifti1Image,
    mrsi_img_unreg: nib.Nifti1Image,
    mrsi_reg_img: nib.Nifti1Image,
    subj: str = "sub-01",
    ses: str = "ses-01",
    mrsi_label: str = "MRSI",
    t1w_cmap: str = "gray",
    mrs_cmap: str = "hot",
) -> None:
    """
    Comparison of MRSI/T1w alignment before and after registration.

    ``t1w_img``         – T1w background (fixed reference, DS or full-res).
    ``mrsi_img_unreg``  – original MRSI resampled naively to T1w grid (no reg).
    ``mrsi_reg_img``    – MRSI after rigid registration, already in T1w space.

    Two columns of 3 orthogonal views (axial / coronal / sagittal):
      • unregistered MRSI overlaid on T1w
      • registered MRSI overlaid on T1w
    """
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
    t1w_cmap: str = "gray",
    mrs_cmap: str = "hot",
) -> None:
    """
    Compare two MRSI registrations, both displayed in DS T1w space.

    Left column : best-SNR MRSI registered to the DS T1w.
    Right column: metabolite sum registered to the DS T1w.

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
         "Metabolite sum (moving) to DS T1w (fixed)"),
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
        f"{subj} {ses} – Registration target comparison  "
        f"(MRSI moving, DS T1w fixed, rigid ANTs, T1w background)",
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
) -> dict:
    """
    Quantitative comparison of two MRSI→T1w registrations in DS-T1w space.

    ``t1w_ds_img``          – DS T1w background reference (fixed space).
    ``mrsi_mask``           – boolean mask in MRSI voxel space.
    ``mrsi_img``            – original MRSI image (provides geometry for mask warping).
    ``mrsi_reg_bestsnr_img``– MRSI registered via best-SNR moving image.
    ``mrsi_reg_sum_img``    – MRSI registered via metabolite-sum moving image.
    ``bestsnr_transforms``  – ANTs transform paths from the best-SNR registration.
    ``sum_transforms``      – ANTs transform paths from the sum registration.

    The mask is warped into T1w space using each set of transforms (via ANTs,
    same RAS→LPS conversion as the registration) so the two contours reflect
    where each registration places the MRSI FOV boundary.

    Three-panel figure:
      1. Signed difference map (best-SNR reg − sum reg) overlaid on T1w with
         two mask contours: orange = best-SNR reg boundary, cyan = sum reg boundary.
      2. In-mask MRSI intensity distributions for both registrations.
      3. Out-of-mask MRSI intensity distribution (signal leakage check).

    Returns a dict of scalar metrics for programmatic use.
    """
    import ants

    def _norm(data: np.ndarray) -> np.ndarray:
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
        # Two mask contours: orange = best-SNR reg boundary, cyan = sum reg boundary
        ax.contour(_slices_at(mask_bestsnr.astype(np.float32), mid)[row].T,
                   levels=[0.5], colors=["#f4a261"], linewidths=1.0, origin="lower")
        ax.contour(_slices_at(mask_sum.astype(np.float32), mid)[row].T,
                   levels=[0.5], colors=["cyan"], linewidths=1.0, origin="lower")
        ax.axis("off")
        ax.set_ylabel(vl, color="white", fontsize=9, rotation=90, labelpad=4)
        if row == 0:
            ax.set_title("Difference map\n(best-SNR − sum reg)", color="white", fontsize=10)

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
               alpha=0.75, label="Sum reg", density=True)
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
                alpha=0.75, label="Sum reg", density=True)
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
        f"(orange contour = best-SNR reg boundary, cyan contour = sum reg boundary, T1w background)",
        color="white", fontsize=12, y=1.01,
    )
    plt.show()

    # ── print metrics table ────────────────────────────────────────────────
    header = f"{'Metric':<28}  {'best-SNR reg':>14}  {'Sum reg':>14}  {'Δ (best−sum)':>14}"
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
    raw_mask   = np.isfinite(water_data) & (water_data > 0)
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
    alpha: float = 0.5,
) -> None:
    """
    Overlay multiple MRSI maps on the same axial, coronal, and sagittal
    panels in native MRS space  each map gets its own colour so they can be
    distinguished at a glance.
    """
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

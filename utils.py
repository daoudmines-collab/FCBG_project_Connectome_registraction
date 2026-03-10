import os
import re

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from nibabel.processing import resample_from_to
from nilearn import plotting


def img_info(img: nib.Nifti1Image, label: str) -> None:
    """Print shape, voxel size and dtype of a NIfTI image."""
    vox = np.sqrt(np.sum(img.affine[:3, :3] ** 2, axis=0))
    print(f"{label}")
    print(f"  shape      : {img.shape}")
    print(f"  voxel size : {vox.round(3)} mm")
    print(f"  dtype      : {img.get_data_dtype()}")
    print()



def metabolite_name(filename: str) -> str:
    """Extract the BIDS ``desc`` label from an MRSI filename."""
    m = re.search(r"desc-([^_]+)_mrsi", filename)
    return m.group(1) if m else filename


def get_nonzero_com(img: nib.Nifti1Image) -> tuple:
    """Centre-of-mass (world coords) of nonzero finite voxels."""
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
    Returns the brain-coverage fraction.
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
) -> None:
    """
    Plot each metabolite map overlaid on the T1w, one figure per metabolite,
    using nilearn's ortho view (axial + coronal + sagittal in one figure).
    Avoids passing axes= to nilearn so there are no stray white rectangles.
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

    Parameters
    ----------
    records : list of dict
        Output of :func:`rank_metabolites_by_snr` (already sorted).
    top_n : int
        How many bars to display (default 20).
    highlight_best : int
        The top-N bars are drawn in a distinct colour (default 3).
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
) -> dict:
    """
    For every subject in ``bids_dir``, sum all acq-OrigRes MRSI concentration
    maps in native MRS space and save the result inside that subject's own
    mrs/ folder as a NIfTI file.
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
        out_path = os.path.join(mrs_dir, out_name)

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
) -> dict:
    """
    For every subject in ``bids_dir`` that has an ``anat/`` folder containing
    a T1w UNI-DEN image, resample the T1w to the spatial grid of the first
    ``acq-OrigRes`` MRSI map and save the result as
         anat/<subj>_<ses>_acq-MRSIres_T1w.nii.gz
    """
    saved = {}

    for subj in sorted(os.listdir(bids_dir)):
        if not subj.startswith("sub-"):
            continue

        anat_dir = os.path.join(bids_dir, subj, ses, "anat")
        mrs_dir  = os.path.join(bids_dir, subj, ses, "mrs")

        if not os.path.isdir(anat_dir):
            print(f"  [t1w-ds] {subj}: no anat/ folder, skipping.")
            continue
        if not os.path.isdir(mrs_dir):
            print(f"  [t1w-ds] {subj}: no mrs/ folder, skipping.")
            continue

        out_name = f"{subj}_{ses}_acq-MRSIres_T1w.nii.gz"
        out_path = os.path.join(anat_dir, out_name)

        if os.path.exists(out_path) and not overwrite:
            print(f"  [t1w-ds] {subj}: already exists  {out_name}")
            saved[subj] = out_path
            continue

        # locate the T1w source
        t1w_name = f"{subj}_{ses}_acq-{t1w_acq}_T1w.nii"
        t1w_path = os.path.join(anat_dir, t1w_name)
        if not os.path.exists(t1w_path):
            # also try .nii.gz
            t1w_path_gz = t1w_path + ".gz"
            if os.path.exists(t1w_path_gz):
                t1w_path = t1w_path_gz
            else:
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
            f"(T1w {t1w_img.shape} → MRS grid {mrs_ref_img.shape})"
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

    Each selected map gets its own colour; maps are shown as transparent
    layers on 3 orthogonal views (axial / coronal / sagittal) — no T1w
    background, no registration required.
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

import os
import ants
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import clear_output
from nibabel.processing import resample_from_to
from nilearn import plotting
from data_utils import (
    metabolite_name,
    get_nonzero_com,
    estimate_coverage,
    _nib_to_ants,
    _mutual_information,
)

# Plot utilities


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
    mrs_cmap: str = "hot"):
    
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
    mrs_cmap: str = "hot"):
  
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
    mrs_cmap: str = "hot"):
   
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
    mrs_cmap: str = "hot"):

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
    ses: str = "ses-01"):
   
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
    cols: int = 3):
   
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
    subj: str = "sub-01"):
   
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
    mrs_cmap: str = "hot"):
    
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
    mrs_cmap: str = "hot"):
 
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


def plot_total_pipeline_mask_comparison(
    sum_ras_img:          "nib.Nifti1Image",
    t1w_bet_img:          "nib.Nifti1Image | None" = None,
    t1w_freesurfer_img:   "nib.Nifti1Image | None" = None,
    t1w_water_img:        "nib.Nifti1Image | None" = None,
    subj: str = "sub-01",
    ses:  str = "ses-01",
    t1w_cmap: str = "gray",
    mrs_cmap: str = "hot"):
    """Compare Reg-17 (total pipeline) results using three different moving masks.

    Columns : BET mask | FreeSurfer / atlas mask | Water mask
    Rows    : Axial | Coronal | Sagittal (mid-slice of sum_ras_img)
    """

    def _norm(data: np.ndarray) -> np.ndarray:
        pos = data[data > 0]
        if pos.size == 0:
            return data
        return np.clip(data / float(np.nanpercentile(pos, 99)), 0, 1)

    panels = [
        (t1w_bet_img,        "Reg-17  BET mask"),
        (t1w_freesurfer_img, "Reg-17  FreeSurfer mask"),
        (t1w_water_img,      "Reg-17  Water mask"),
    ]
    view_labels = ["Axial", "Coronal", "Sagittal"]

    bg_norm = _norm(sum_ras_img.get_fdata().astype(np.float32))
    mid     = [s // 2 for s in bg_norm.shape]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), facecolor="black")
    fig.subplots_adjust(hspace=0.06, wspace=0.04)

    for col, (t1w_reg_img, title) in enumerate(panels):
        if t1w_reg_img is None:
            for row in range(3):
                ax = axes[row, col]
                ax.set_facecolor("black")
                ax.axis("off")
            axes[0, col].set_title(title + "\n(not available)", color="white", fontsize=9)
            continue

        t1w_norm = _norm(t1w_reg_img.get_fdata().astype(np.float32))
        slices_bg  = [bg_norm[:, :, mid[2]],  bg_norm[:, mid[1], :],  bg_norm[mid[0], :, :]]
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
                ax.set_ylabel(vl, color="white", fontsize=10, rotation=90, labelpad=4)
            if row == 0:
                ax.set_title(title, color="white", fontsize=10, pad=6)

    fig.suptitle(
        f"{subj} {ses}  –  Reg-17 total pipeline: moving-mask comparison\n"
        f"T1w ({t1w_cmap}) + MRSI sum RAS overlay ({mrs_cmap}, \u03b1=0.6)",
        color="white", fontsize=11, y=1.02,
    )
    plt.tight_layout()
    plt.show()


def plot_registration_metrics(
    metrics: list,
    subj: str = "sub-01",
    ses: str = "ses-01"):
   
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
    coverage   = np.array([r["coverage"]      for r in metrics], dtype=float)
    ncc        = np.array([r["ncc"]           for r in metrics], dtype=float)
    nmi        = np.array([r["nmi"]           for r in metrics], dtype=float)
    snr        = np.array([r.get("snr", 0.0) for r in metrics], dtype=float)

    # short display labels (strip common BIDS prefixes)
    def _short(lbl):
        for prefix in ("acq-OrigRes_", "acq-", "desc-"):
            lbl = lbl.replace(prefix, "")
        return lbl
    labels = [_short(l) for l in labels_raw]

    cmap_snr = plt.get_cmap("plasma")
    vmin_c, vmax_c = snr.min(), max(snr.max(), 1e-6)
    dot_colors = [cmap_snr((v - vmin_c) / (vmax_c - vmin_c)) for v in snr]

    BG   = "#0d0d0d"
    PAN  = "#1a1a1a"
    GREY = "#444444"
    TXT  = "white"

    n      = len(labels)
    fig_h  = max(11, n * 0.32 + 4)   # grow with number of metabolites

    fig, axes = plt.subplots(
        2, 2,
        figsize=(16, fig_h),
        facecolor=BG,
        constrained_layout=True,
    )

  
    def _lollipop(ax, values, sort_idx, xlabel, title, ref_line=None):
        ax.set_facecolor(PAN)
        y = np.arange(len(sort_idx))
        vlabels = [labels[i] for i in sort_idx]
        vvals   = values[sort_idx]
        vcols   = [dot_colors[i] for i in sort_idx]

        ax.hlines(y, 0, vvals, color=GREY, linewidth=1.2, zorder=1)
        sc = ax.scatter(vvals, y, c=[snr[i] for i in sort_idx],
                        cmap="plasma", vmin=vmin_c, vmax=vmax_c,
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

    # shared colorbar – attached to right column so it sits at the outer right edge
    sm = plt.cm.ScalarMappable(cmap="plasma",
                               norm=plt.Normalize(vmin=vmin_c, vmax=vmax_c))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes[:, 1], location="right",
                      shrink=0.6, pad=0.02)
    cb.set_label("In-region SNR (mean/std)", color=TXT, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=TXT, labelcolor=TXT)

    
    def _scatter(ax, xvals, yvals, xlabel, ylabel, title,
                 xref=None, yref=None):
        ax.set_facecolor(PAN)
        sc = ax.scatter(xvals, yvals, c=snr,
                        cmap="plasma", vmin=vmin_c, vmax=vmax_c,
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
             "NCC vs NMI  (colour = SNR)",
             xref=0.0)

    # panel 4: SNR vs NCC
    _scatter(axes[1, 1], snr, ncc,
             "In-region SNR (mean/std)", "NCC",
             "SNR vs NCC  (colour = SNR)",
             yref=0.0)

    fig.suptitle(
        f"{subj}  {ses}  –  Registration quality metrics  (Reg-17 total pipeline)",
        color=TXT, fontsize=12,
    )
    plt.show()

    # print table 
    print(f"\n{'Label':<35}  {'Coverage':>9}  {'NCC':>8}  {'NMI':>8}  {'SNR':>8}")
    print("─" * 75)
    for r in metrics:
        print(f"{r['label']:<35}  {r['coverage']:>9.3f}  {r['ncc']:>8.4f}  {r['nmi']:>8.4f}  {r.get('snr', 0.0):>8.3f}")

def plot_pipeline_metrics_comparison(
    metrics_a: list,
    metrics_b: list,
    label_a: str = "BET mask (Reg-17a)",
    label_b: str = "FreeSurfer mask (Reg-17b)",
    subj: str = "sub-01",
    ses: str = "ses-01"):
    """Side-by-side comparison of registration quality metrics for two pipelines.

    Four panels:
    - Top-left  : NCC lollipop (both pipelines, sorted by mean |NCC|)
    - Top-right : NMI lollipop (both pipelines, sorted by mean NMI)
    - Bottom-left: NCC vs NMI scatter (both pipelines, connected per metabolite)
    - Bottom-right: Δ bar chart (FreeSurfer − BET for NCC and NMI)
    """
    if not metrics_a or not metrics_b:
        print("[pipeline comparison] one or both metrics lists are empty.")
        return

    def _short(lbl):
        for prefix in ("acq-OrigRes_", "acq-", "desc-"):
            lbl = lbl.replace(prefix, "")
        return lbl

    a_by_label = {r["label"]: r for r in metrics_a}
    b_by_label = {r["label"]: r for r in metrics_b}
    common     = sorted(set(a_by_label) & set(b_by_label),
                        key=lambda l: -(abs(a_by_label[l]["ncc"]) + abs(b_by_label[l]["ncc"])) / 2)

    if not common:
        print("[pipeline comparison] no common metabolite labels found between the two pipelines.")
        return

    ncc_a = np.array([a_by_label[l]["ncc"]           for l in common], dtype=float)
    ncc_b = np.array([b_by_label[l]["ncc"]           for l in common], dtype=float)
    nmi_a = np.array([a_by_label[l]["nmi"]           for l in common], dtype=float)
    nmi_b = np.array([b_by_label[l]["nmi"]           for l in common], dtype=float)
    snr_a = np.array([a_by_label[l].get("snr", 0.0) for l in common], dtype=float)
    snr_b = np.array([b_by_label[l].get("snr", 0.0) for l in common], dtype=float)
    labels = [_short(l) for l in common]

    BG      = "#0d0d0d"
    PAN     = "#1a1a1a"
    GREY    = "#444444"
    TXT     = "white"
    COLOR_A = "#f4a261"   # orange  → BET
    COLOR_B = "#4cc9f0"   # cyan    → FreeSurfer

    n       = len(labels)
    order_ncc = np.argsort((np.abs(ncc_a) + np.abs(ncc_b)) / 2)
    order_nmi = np.argsort((nmi_a + nmi_b) / 2)

    fig, axes = plt.subplots(2, 2, figsize=(18, max(10, n * 0.4 + 3)), facecolor=BG)
    fig.subplots_adjust(hspace=0.45, wspace=0.45)

    def _lollipop_pair(ax, vals_a, vals_b, sort_idx, xlabel, title, ref_line=None):
        ax.set_facecolor(PAN)
        y = np.arange(len(sort_idx))
        _la = [labels[i] for i in sort_idx]
        _va = vals_a[sort_idx]
        _vb = vals_b[sort_idx]

        # stagger rows slightly so dots don't overlap
        ax.hlines(y - 0.18, 0, _va, color=COLOR_A, linewidth=1.1, zorder=1, alpha=0.7)
        ax.scatter(_va, y - 0.18, c=COLOR_A, s=55, zorder=3,
                   edgecolors="white", linewidths=0.3, label=label_a)
        ax.hlines(y + 0.18, 0, _vb, color=COLOR_B, linewidth=1.1, zorder=1, alpha=0.7)
        ax.scatter(_vb, y + 0.18, c=COLOR_B, s=55, marker="s", zorder=3,
                   edgecolors="white", linewidths=0.3, label=label_b)

        if ref_line is not None:
            ax.axvline(ref_line, color="white", linestyle="--", linewidth=0.7, alpha=0.4, zorder=2)

        ax.set_yticks(y)
        ax.set_yticklabels(_la, color=TXT, fontsize=7.5)
        ax.set_xlabel(xlabel, color=TXT, fontsize=9)
        ax.set_title(title, color=TXT, fontsize=10, pad=6)
        ax.tick_params(colors=TXT, labelsize=8)
        ax.grid(axis="x", color=GREY, linewidth=0.4, zorder=0)
        for spine in ax.spines.values():
            spine.set_edgecolor(GREY)
        ax.legend(facecolor="#222222", labelcolor="white", fontsize=8, loc="lower right")

    _lollipop_pair(
        axes[0, 0], ncc_a, ncc_b, order_ncc,
        "NCC  (negative = anti-correlated, expected MRSI vs T1w UNI-DEN)",
        f"NCC: {label_a} vs {label_b}",
        ref_line=0.0,
    )
    _lollipop_pair(
        axes[0, 1], nmi_a, nmi_b, order_nmi,
        "Normalised Mutual Information (NMI)",
        f"NMI: {label_a} vs {label_b}",
    )

    # bottom-left: NCC vs NMI scatter, both pipelines with connecting lines
    ax_sc = axes[1, 0]
    ax_sc.set_facecolor(PAN)
    for na, nb, ma, mb in zip(ncc_a, ncc_b, nmi_a, nmi_b):
        ax_sc.plot([na, nb], [ma, mb], color="white", linewidth=0.5, alpha=0.2, zorder=2)
    ax_sc.scatter(ncc_a, nmi_a, c=COLOR_A, s=60, zorder=3,
                  edgecolors="white", linewidths=0.3, label=label_a, alpha=0.9)
    ax_sc.scatter(ncc_b, nmi_b, c=COLOR_B, s=60, marker="s", zorder=3,
                  edgecolors="white", linewidths=0.3, label=label_b, alpha=0.9)
    ax_sc.axvline(0, color="white", linestyle="--", linewidth=0.7, alpha=0.4)
    ax_sc.set_xlabel("NCC", color=TXT, fontsize=9)
    ax_sc.set_ylabel("NMI", color=TXT, fontsize=9)
    ax_sc.set_title("NCC vs NMI  (lines connect same metabolite)", color=TXT, fontsize=10, pad=6)
    ax_sc.tick_params(colors=TXT, labelsize=8)
    ax_sc.grid(color=GREY, linewidth=0.4, zorder=0)
    for spine in ax_sc.spines.values():
        spine.set_edgecolor(GREY)
    ax_sc.legend(facecolor="#222222", labelcolor="white", fontsize=8)

    # bottom-right: delta bar chart (FreeSurfer − BET)
    ax_d = axes[1, 1]
    ax_d.set_facecolor(PAN)
    delta_ncc = ncc_b - ncc_a
    delta_nmi = nmi_b - nmi_a
    delta_snr = snr_b - snr_a
    order_d   = np.argsort(delta_ncc)
    _yd       = np.arange(len(order_d))
    _ld       = [labels[i] for i in order_d]

    ax_d.barh(_yd - 0.25, delta_ncc[order_d], height=0.25, color=COLOR_A,
              label="ΔNCC", alpha=0.9, zorder=2)
    ax_d.barh(_yd,         delta_nmi[order_d] * 10, height=0.25, color=COLOR_B,
              label="ΔNMI ×10", alpha=0.9, zorder=2)
    ax_d.barh(_yd + 0.25, delta_snr[order_d] * 0.01, height=0.25, color="#90e0b0",
              label="ΔSNR ×0.01", alpha=0.9, zorder=2)
    ax_d.axvline(0, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_d.set_yticks(_yd)
    ax_d.set_yticklabels(_ld, color=TXT, fontsize=7.5)
    ax_d.set_xlabel(f"Δ ({label_b} − {label_a})\npositive = FreeSurfer better",
                    color=TXT, fontsize=9)
    ax_d.set_title("Per-metabolite pipeline delta", color=TXT, fontsize=10, pad=6)
    ax_d.tick_params(colors=TXT, labelsize=8)
    ax_d.grid(axis="x", color=GREY, linewidth=0.4, zorder=0)
    for spine in ax_d.spines.values():
        spine.set_edgecolor(GREY)
    ax_d.legend(facecolor="#222222", labelcolor="white", fontsize=8)

    fig.suptitle(
        f"{subj}  {ses}  –  Pipeline comparison: {label_a} vs {label_b}\n"
        f"Reg-17 total pipeline  (rigid ANTs, skull-stripped T1w → RAS MRSI sum, water-weighted)",
        color=TXT, fontsize=12, y=1.01,
    )
    plt.tight_layout()
    plt.show()

    # print summary table
    header = (f"\n{'Label':<35}  {'NCC(A)':>9}  {'NCC(B)':>9}  {'ΔNCC':>8}  "
              f"{'NMI(A)':>9}  {'NMI(B)':>9}  {'ΔNMI':>8}  {'SNR(A)':>8}  {'SNR(B)':>8}")
    print(header)
    print("─" * len(header.strip()))
    for l, ra, rb in zip(common,
                         [a_by_label[l] for l in common],
                         [b_by_label[l] for l in common]):
        sl = _short(l)
        print(
            f"{sl:<35}  {ra['ncc']:>9.4f}  {rb['ncc']:>9.4f}  {rb['ncc'] - ra['ncc']:>+8.4f}  "
            f"{ra['nmi']:>9.4f}  {rb['nmi']:>9.4f}  {rb['nmi'] - ra['nmi']:>+8.4f}  "
            f"{ra.get('snr', 0.0):>8.3f}  {rb.get('snr', 0.0):>8.3f}"
        )

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
    alpha: float = 0.55):
  
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
            plotting.plot_stat_map(
                overlay,
                bg_img=t1w_img,
                display_mode="ortho",
                cut_coords=cut_coords,
                colorbar=True,
                title=title,
                axes=ax_top,
                black_bg=True,
                cmap=cmap,
                vmax=vmax,
                alpha=alpha,
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
    ses: str = "ses-01"):

    BG  = "black"

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
    sum_img:              "nib.Nifti1Image",
    water_mask_img:       "nib.Nifti1Image",
    bet_mask_img:         "nib.Nifti1Image" ,
    freesurfer_mask_img:  "nib.Nifti1Image" ,
    subj: str = "sub-01",
    ses:  str = "ses-01",
    n_slices: int = 7):
   
    sum_data        = sum_img.get_fdata().astype(np.float32)
    water_data      = water_mask_img.get_fdata().astype(bool)
    bet_data        = bet_mask_img.get_fdata().astype(bool)        
    freesurfer_data = freesurfer_mask_img.get_fdata().astype(bool) 

    nz = sum_data.shape[2]
    z_indices = np.linspace(0, nz - 1, n_slices, dtype=int)

    sum_max = np.percentile(sum_data[sum_data > 0], 99) if sum_data.max() > 0 else 1.0

    fig, axes = plt.subplots(1, n_slices, figsize=(3.2 * n_slices, 4), facecolor="black")
    if n_slices == 1:
        axes = [axes]

    # Track which legend entries have already been added (by label)
    legend_seen: set = set()
    legend_handles = []

    def _add_contour(ax, binary_slice, color, label):
        """Draw a contour and, the first time, add a legend entry."""
        bin_f = binary_slice.astype(np.float32)
        if bin_f.max() > 0:
            ax.contour(bin_f, levels=[0.5], colors=[color], linewidths=1.2)
            if label not in legend_seen:
                legend_seen.add(label)
                legend_handles.append(
                    plt.matplotlib.lines.Line2D([], [], color=color,
                                                linewidth=1.5, label=label))

    for ax, z in zip(axes, z_indices):
        ax.set_facecolor("black")
        slc = sum_data[:, :, z].T
        ax.imshow(slc, origin="lower", cmap="gray",
                  vmin=0, vmax=sum_max, interpolation="nearest")

        # Sum signal boundary (gold)
        _add_contour(ax, (sum_data[:, :, z] > 0).T, "gold", "Sum signal boundary")

        # Water mask (blue)
        _add_contour(ax, water_data[:, :, z].T, "deepskyblue", "Water mask")

        # BET brain mask registered to MRSI space (green)
        if bet_data is not None:
            _add_contour(ax, bet_data[:, :, z].T, "limegreen", "BET brain mask (reg.)")

        # FreeSurfer / atlas mask registered to MRSI space (red)
        if freesurfer_data is not None:
            _add_contour(ax, freesurfer_data[:, :, z].T, "tomato", "FreeSurfer mask (reg.)")

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


def plot_mask_coverage_comparaison(
    bet_mask_img:    "nib.Nifti1Image",
    water_mask_img:  "nib.Nifti1Image",
    t1w_ds_img:      "nib.Nifti1Image",
    extra_mask_img:  "nib.Nifti1Image",
    active_mask_name: str | None = None,
    n_slices: int = 7):
    bet_data   = bet_mask_img.get_fdata().astype(bool)
    water_data = water_mask_img.get_fdata().astype(bool)
    t1w_data   = t1w_ds_img.get_fdata().astype(np.float32)
    
    extra_data = extra_mask_img.get_fdata().astype(bool)

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

    all3_inter   = bet_data & water_data & extra_data
    bet_extra_i  = bet_data & extra_data
    wat_extra_i  = water_data & extra_data
    extra_only   = extra_data & ~bet_data & ~water_data
    n_extra      = int(extra_data.sum())
    n_eonly      = int(extra_only.sum())
    n_all3       = int(all3_inter.sum())
    dice_be      = 2 * int(bet_extra_i.sum())  / (n_bet + n_extra + 1e-10)
    dice_we      = 2 * int(wat_extra_i.sum())  / (n_water + n_extra + 1e-10)
    jacc_be      = int(bet_extra_i.sum())  / (n_bet + n_extra - int(bet_extra_i.sum()) + 1e-10)
    jacc_we      = int(wat_extra_i.sum())  / (n_water + n_extra - int(wat_extra_i.sum()) + 1e-10)
    # exclusive-only regions for comparison row
    bet_excl   = bet_data  & ~water_data & ~extra_data
    water_excl = water_data & ~bet_data  & ~extra_data
    any_overlap = (bet_data.astype(np.uint8) + water_data.astype(np.uint8) +
                       extra_data.astype(np.uint8)) >= 2

    # ── Text table ────────────────────────────────────────────────────────
    sep = "─" * 52
    print(sep)
    print(f"  Mask comparison (MRSI grid)")
    print(sep)
    print(f"  BET mask volume      : {n_bet   * vox_vol_mm3/1000:>8.1f} mL  ({n_bet:>6} vox)")
    print(f"  Water mask volume    : {n_water * vox_vol_mm3/1000:>8.1f} mL  ({n_water:>6} vox)")
    
    print(f"  FREESURFER mask volume : {n_extra * vox_vol_mm3/1000:>8.1f} mL  ({n_extra:>6} vox)")
    print(sep)
    print(f"  BET ∩ Water          : {n_inter * vox_vol_mm3/1000:>8.1f} mL  ({n_inter:>6} vox)")
    print(f"  BET-only voxels      : {n_bonly * vox_vol_mm3/1000:>8.1f} mL  ({n_bonly:>6} vox)")
    print(f"  Water-only voxels    : {n_wonly * vox_vol_mm3/1000:>8.1f} mL  ({n_wonly:>6} vox)")
    print(f"  Dice  BET/Water      : {dice:.4f}  Jaccard: {jaccard:.4f}")
  
    print(f"  Dice  BET/FREESURFER : {dice_be:.4f}  Jaccard: {jacc_be:.4f}")
    print(f"  Dice  Water/FREESURFER : {dice_we:.4f}  Jaccard: {jacc_we:.4f}")
    print(f"  All-3 intersection   : {n_all3 * vox_vol_mm3/1000:>8.1f} mL  ({n_all3:>6} vox)")
    print(f"  FREESURFERmask-only voxels : {n_eonly * vox_vol_mm3/1000:>8.1f} mL  ({n_eonly:>6} vox)")
    print(f"  BET ≡ Water          : {bool(np.array_equal(bet_data, water_data))}")
    print(sep)
    if active_mask_name:
        print(f"\n  ACTIVE mask for Reg 18 : {active_mask_name}")

    # ── Axial mosaic ──────────────────────────────────────────────────────
    nz = bet_data.shape[2]
    z_indices = np.linspace(0, nz - 1, n_slices, dtype=int)

    t1w_max = np.percentile(t1w_data[t1w_data > 0], 99) if t1w_data.max() > 0 else 1.0

    n_rows = 4 
    fig_h  = 3.0 * n_rows
    fig, axes = plt.subplots(n_rows, n_slices,
                             figsize=(2.8 * n_slices, fig_h),
                             facecolor="black")

    row_labels = ["BET mask", "Water mask","freeSurfer mask", "Comparison"]

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

       
        # Row 2 – Extra mask on T1w
        ext_slc = extra_data[:, :, z].T.astype(np.float32)
        ax = axes[2, col]
        ax.set_facecolor("black")
        ax.imshow(t1w_slc, origin="lower", cmap="gray",
                      vmin=0, vmax=t1w_max, interpolation="nearest")
        ax.imshow(np.ma.masked_where(ext_slc == 0, ext_slc),
                      origin="lower", cmap="Oranges", vmin=0, vmax=1,
                      alpha=0.55, interpolation="nearest")
        ax.axis("off")

        # Last row – Comparison
        ax = axes[n_rows - 1, col]
        ax.set_facecolor("black")
        ax.imshow(t1w_slc, origin="lower", cmap="gray",
                  vmin=0, vmax=t1w_max, alpha=0.25, interpolation="nearest")

        
        # any pairwise/triple overlap = grey
        ov = any_overlap[:, :, z].T.astype(np.float32)
        ax.imshow(np.ma.masked_where(ov == 0, ov),
                      origin="lower", cmap="gray", vmin=0, vmax=1,
                      alpha=0.55, interpolation="nearest")
        # extra-only (orange)
        eo = extra_only[:, :, z].T.astype(np.float32)
        ax.imshow(np.ma.masked_where(eo == 0, eo),
                      origin="lower", cmap="Oranges", vmin=0, vmax=1,
                      alpha=0.80, interpolation="nearest")
        # water-excl (blue)
        we = water_excl[:, :, z].T.astype(np.float32)
        ax.imshow(np.ma.masked_where(we == 0, we),
                      origin="lower", cmap="Blues", vmin=0, vmax=1,
                      alpha=0.80, interpolation="nearest")
        # bet-excl (red)
        be = bet_excl[:, :, z].T.astype(np.float32)
        ax.imshow(np.ma.masked_where(be == 0, be),
                      origin="lower", cmap="Reds", vmin=0, vmax=1,
                      alpha=0.80, interpolation="nearest")
       
    for row, lbl in enumerate(row_labels):
        axes[row, 0].set_ylabel(lbl, color="white", fontsize=10,
                                rotation=90, labelpad=6)

    legend_patches = [
            plt.matplotlib.patches.Patch(color="red",        label=f"BET only ({int(bet_excl.sum()):,} vox)"),
            plt.matplotlib.patches.Patch(color="dodgerblue", label=f"Water only ({int(water_excl.sum()):,} vox)"),
            plt.matplotlib.patches.Patch(color="darkorange",  label=f"Freesurfer mask only ({n_eonly:,} vox)"),
            plt.matplotlib.patches.Patch(color="grey",        label=f"Any overlap ({int(any_overlap.sum()):,} vox)"),
        ]
    title = (f"BET / Water / Freesurfer mask comparison (MRSI grid)  |  "
                 f"Dice BET-Water={dice:.3f}  BET-Freesurfer={dice_be:.3f}  "
                 f"Water-Freesurfer={dice_we:.3f}")
    
    title = (f"BET mask vs water mask (MRSI grid)  |  "
                 f"Dice={dice:.3f}  Jaccard={jaccard:.3f}")
    if active_mask_name:
        title += f"  |  Active: {active_mask_name}"
    fig.suptitle(title, color="white", fontsize=10)
    fig.legend(handles=legend_patches, loc="lower center", ncol=len(legend_patches),
               frameon=False, labelcolor="white", fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.show()


# Section 25 – Registration methodology comparison (NMI + visual thumbnails)


def _compute_nmi_vs_reference(
    reg_imgs: dict,
    reference_img: nib.Nifti1Image):
   
    ref_data = reference_img.get_fdata().astype(np.float32)
    ref_mask = ref_data > 0
    n_ref   = int(ref_mask.sum())
    records = []
    for label, img in reg_imgs.items():
        if img is None:
            continue
        # Resample to reference grid if geometries differ
        if img.shape != reference_img.shape or not np.allclose(img.affine, reference_img.affine, atol=1e-3):
            img_r = resample_from_to(img, reference_img, order=1)
            mv_data = img_r.get_fdata().astype(np.float32)
        else:
            mv_data = img.get_fdata().astype(np.float32)
        overlap  = ref_mask & (mv_data > 0)
        coverage = float(overlap.sum()) / n_ref if n_ref > 0 else 0.0
        nmi = _mutual_information(ref_data[overlap], mv_data[overlap]) if overlap.sum() > 10 else 0.0
        records.append({"label": label, "nmi": nmi, "coverage": coverage})
    records.sort(key=lambda r: r["nmi"], reverse=True)
    return records


def plot_multi_reg_nmi_comparison(
    reg_imgs: dict,
    water_img: nib.Nifti1Image,
    subj: str = "sub-01",
    ses: str = "ses-01"):
  
    metrics = _compute_nmi_vs_reference(reg_imgs, water_img)
    if not metrics:
        print("[reg-nmi] nothing to plot — reg_imgs is empty or all None.")
        return

    water_data = water_img.get_fdata().astype(np.float32)
    mid_z      = water_data.shape[2] // 2
    wsl        = water_data[:, :, mid_z]
    vmax_w     = float(np.nanpercentile(wsl[wsl > 0], 99)) if (wsl > 0).any() else 1.0

    n_regs = len(metrics)
    BG, PAN, GREY, TXT = "#0d0d0d", "#1a1a1a", "#444444", "white"

    n_cols  = n_regs + 1   # water reference + one col per reg
    fig = plt.figure(figsize=(3.5 * n_cols, 9), facecolor=BG)
    gs  = fig.add_gridspec(2, n_cols,
                           height_ratios=[1.3, 1],
                           hspace=0.5, wspace=0.3)

    # ── Row 0: thumbnail panels ──────────────────────────────────────────
    ax_w = fig.add_subplot(gs[0, 0])
    ax_w.set_facecolor(BG)
    ax_w.imshow(wsl.T, origin="lower", cmap="Blues", vmin=0, vmax=vmax_w,
                interpolation="nearest")
    ax_w.set_title("Water signal\n(reference)", color=TXT, fontsize=9, pad=4)
    ax_w.axis("off")

    for col_i, rec in enumerate(metrics):
        label = rec["label"]
        img   = reg_imgs[label]
        ax    = fig.add_subplot(gs[0, col_i + 1])
        ax.set_facecolor(BG)

        # Resample to water grid for display
        if img.shape != water_img.shape or not np.allclose(img.affine, water_img.affine, atol=1e-3):
            img_r = resample_from_to(img, water_img, order=1)
            t1w_data = img_r.get_fdata().astype(np.float32)
        else:
            t1w_data = img.get_fdata().astype(np.float32)

        t1w_sl = t1w_data[:, :, mid_z]
        vmax_t = float(np.nanpercentile(t1w_sl[t1w_sl > 0], 99)) if (t1w_sl > 0).any() else 1.0
        ax.imshow(t1w_sl.T, origin="lower", cmap="gray", vmin=0, vmax=vmax_t,
                  alpha=0.9, interpolation="nearest")
        # Water signal contour as alignment guide
        ax.contour(wsl.T, levels=[0.15 * vmax_w], colors=["#4fc3f7"],
                   linewidths=0.8, alpha=0.95)

        short = label
        ax.set_title(f"{short}\nNMI={rec['nmi']:.3f}  cov={rec['coverage']:.2f}",
                     color=TXT, fontsize=8, pad=4)
        ax.axis("off")

    # ── Row 1: NMI lollipop (spanning all columns) ───────────────────────
    ax_lol = fig.add_subplot(gs[1, :])
    ax_lol.set_facecolor(PAN)

    labels_all = [r["label"] for r in metrics]
    nmis       = np.array([r["nmi"]      for r in metrics])
    covs       = np.array([r["coverage"] for r in metrics])
    y          = np.arange(len(labels_all))

    vmin_c, vmax_c = covs.min(), max(covs.max(), 1e-6)
    ax_lol.hlines(y, 0, nmis, color=GREY, linewidth=1.8, zorder=1)
    sc = ax_lol.scatter(nmis, y, c=covs, cmap="viridis",
                        vmin=vmin_c, vmax=vmax_c,
                        s=110, zorder=3, edgecolors="white", linewidths=0.5)
    for i, (nv, cv) in enumerate(zip(nmis, covs)):
        ax_lol.annotate(f"  {nv:.3f}  (cov {cv:.2f})", (nv, i),
                        textcoords="offset points", xytext=(4, 0),
                        color="lightgrey", fontsize=8.5, va="center")

    ax_lol.set_yticks(y)
    ax_lol.set_yticklabels(labels_all, color=TXT, fontsize=9)
    ax_lol.set_xlabel("NMI vs water signal  (higher = better alignment)",
                      color=TXT, fontsize=9)
    ax_lol.set_title("Registration quality comparison  –  all methods",
                     color=TXT, fontsize=10, pad=6)
    ax_lol.tick_params(colors=TXT, labelsize=9)
    ax_lol.grid(axis="x", color=GREY, linewidth=0.5, zorder=0)
    for sp in ax_lol.spines.values():
        sp.set_edgecolor(GREY)

    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(vmin=vmin_c, vmax=vmax_c))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_lol, orientation="vertical",
                      fraction=0.015, pad=0.01, shrink=0.8)
    cb.set_label("Coverage", color=TXT, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=TXT, labelcolor=TXT)

    fig.suptitle(
        f"{subj}  {ses}  –  T1w → MRSI  registration methodology comparison\n"
        "Thumbnails: T1w (gray) + water signal contour (blue).  "
        "Bar: NMI vs water signal, colour = coverage fraction.",
        color=TXT, fontsize=10, y=1.01,
    )
    plt.tight_layout()
    plt.show()

    print(f"\n{'Method':<35}  {'NMI':>7}  {'Coverage':>9}")
    print("─" * 56)
    for r in metrics:
        print(f"{r['label']:<35}  {r['nmi']:>7.4f}  {r['coverage']:>9.3f}")




def plot_tissue_metabolite_metrics(
    tissue_fracs: dict,
    mrsi_conc_imgs: dict,
    water_img: nib.Nifti1Image,
    subj: str = "sub-01",
    ses: str = "ses-01"):
  
    gm_img  = tissue_fracs.get("gm")
    wm_img  = tissue_fracs.get("wm")
    csf_img = tissue_fracs.get("csf")

    if gm_img is None and wm_img is None:
        print("[tissue-metrics] no PVE maps available — run compute_tissue_fractions_in_mrsi first.")
        return

    BG, PAN, GREY, TXT = "#0d0d0d", "#1a1a1a", "#444444", "white"

    # ── Common MRSI grid reference ────────────────────────────────────────
    water_data = water_img.get_fdata().astype(np.float32)
    water_mask = water_data > 0
    mid_z      = water_data.shape[2] // 2
    wsl        = water_data[:, :, mid_z]
    vmax_w     = float(np.nanpercentile(wsl[wsl > 0], 99)) if (wsl > 0).any() else 1.0

    def _get_slice(img):
        if img is None:
            return None
        d = img.get_fdata().astype(np.float32)
        # clip negatives from linear interpolation
        d = np.clip(d, 0, None)
        if img.shape != water_img.shape or not np.allclose(img.affine, water_img.affine, atol=1e-3):
            d = resample_from_to(img, water_img, order=1).get_fdata().astype(np.float32)
            d = np.clip(d, 0, None)
        return d[:, :, mid_z]

    gm_sl  = _get_slice(gm_img)
    wm_sl  = _get_slice(wm_img)
    csf_sl = _get_slice(csf_img)

    # ── Flatten PVE data to MRSI voxels ──────────────────────────────────
    def _flat(img):
        if img is None:
            return None
        d = img.get_fdata().astype(np.float32)
        d = np.clip(d, 0, None)
        if img.shape != water_img.shape or not np.allclose(img.affine, water_img.affine, atol=1e-3):
            d = resample_from_to(img, water_img, order=1).get_fdata().astype(np.float32)
            d = np.clip(d, 0, None)
        return d[water_mask]

    gm_flat  = _flat(gm_img)
    wm_flat  = _flat(wm_img)

    # Threshold: voxels where one tissue dominates (>40% PVE)
    THR = 0.40
    gm_vox = (gm_flat > THR) if gm_flat is not None else None
    wm_vox = (wm_flat > THR) if wm_flat is not None else None

    # ── Per-metabolite GM mean / WM mean ─────────────────────────────────
    scatter_pts = []   # list of (label, gm_mean, wm_mean, is_water)
    all_imgs = dict(mrsi_conc_imgs)
    all_imgs["Water"] = water_img   # always include water as reference

    for label, img in all_imgs.items():
        if img is None:
            continue
        if img.shape != water_img.shape or not np.allclose(img.affine, water_img.affine, atol=1e-3):
            d = resample_from_to(img, water_img, order=1).get_fdata().astype(np.float32)
        else:
            d = img.get_fdata().astype(np.float32)
        d = np.clip(d, 0, None)
        vals = d[water_mask]

        # Normalise to [0,1] so all metabolites are on the same scale
        v_max = vals.max()
        if v_max < 1e-9:
            continue
        vals_n = vals / v_max

        gm_mean  = float(vals_n[gm_vox].mean()) if gm_vox is not None and gm_vox.sum() > 0 else np.nan
        wm_mean  = float(vals_n[wm_vox].mean()) if wm_vox is not None and wm_vox.sum() > 0 else np.nan
        scatter_pts.append((label, gm_mean, wm_mean, label == "Water"))

    scatter_pts = [(lbl, gm, wm, iw)
                   for lbl, gm, wm, iw in scatter_pts
                   if not np.isnan(gm) and not np.isnan(wm)]

    # ── Figure ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 10), facecolor=BG)
    gs  = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.35,
                           height_ratios=[1, 1.2])

    # --- Row 0: tissue fraction maps ---
    pve_panels = [
        ("Water signal",  wsl,  "Blues",   vmax_w),
        ("GM fraction",   gm_sl,  "Greens",  1.0),
        ("WM fraction",   wm_sl,  "Oranges", 1.0),
        ("CSF fraction",  csf_sl, "Purples", 1.0),
    ]
    for col, (title, slc, cmap, vmax_panel) in enumerate(pve_panels):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(BG)
        if slc is not None:
            ax.imshow(slc.T, origin="lower", cmap=cmap,
                      vmin=0, vmax=vmax_panel, interpolation="nearest")
            # Water contour reference on non-water panels
            if col > 0:
                ax.contour(wsl.T, levels=[0.15 * vmax_w], colors=["#4fc3f7"],
                           linewidths=0.7, alpha=0.85)
        ax.set_title(title, color=TXT, fontsize=10, pad=4)
        ax.axis("off")

    # --- Row 1: biological plausibility scatter ---
    ax_sc = fig.add_subplot(gs[1, :3])
    ax_sc.set_facecolor(PAN)

    if scatter_pts:
        cmap_sc  = plt.get_cmap("tab20")
        diag_max = max(max(gm for _, gm, _, _ in scatter_pts),
                       max(wm for _, _, wm, _ in scatter_pts)) * 1.05
        ax_sc.plot([0, diag_max], [0, diag_max], "--",
                   color="white", alpha=0.35, linewidth=1.0,
                   label="y = x  (no tissue preference)", zorder=1)

        for i, (lbl, gm_m, wm_m, is_water) in enumerate(scatter_pts):
            color  = "#4fc3f7" if is_water else cmap_sc(i / max(len(scatter_pts), 1))
            marker = "*" if is_water else "o"
            size   = 220 if is_water else 80
            ax_sc.scatter(gm_m, wm_m, color=color, marker=marker,
                          s=size, zorder=4 if is_water else 3,
                          edgecolors="white", linewidths=0.4)
            offset = (4, 5) if not is_water else (4, 8)
            ax_sc.annotate(lbl, (gm_m, wm_m),
                           textcoords="offset points", xytext=offset,
                           fontsize=7.5, color=("cyan" if is_water else "lightgrey"),
                           zorder=5)

        ax_sc.set_xlabel("Mean (normalised conc.)  in GM-dominant voxels  (PVE > 40%)",
                         color=TXT, fontsize=9)
        ax_sc.set_ylabel("Mean (normalised conc.)  in WM-dominant voxels  (PVE > 40%)",
                         color=TXT, fontsize=9)
        ax_sc.set_title(
            "Biological plausibility — GM vs WM metabolite separation\n"
            "Above diagonal = WM-dominant  |  Below = GM-dominant  |  "
            "★ = water (reference)",
            color=TXT, fontsize=10, pad=6,
        )
        ax_sc.tick_params(colors=TXT, labelsize=9)
        ax_sc.grid(color=GREY, linewidth=0.4, zorder=0)
        for sp in ax_sc.spines.values():
            sp.set_edgecolor(GREY)
        ax_sc.legend(frameon=False, labelcolor="white", fontsize=8)

    # --- Row 1 col 3: mean tissue composition bar ---
    ax_bar = fig.add_subplot(gs[1, 3])
    ax_bar.set_facecolor(PAN)
    if gm_flat is not None and wm_flat is not None:
        gm_mean_all  = float(gm_flat.mean())
        wm_mean_all  = float(wm_flat.mean())
        csf_flat = _flat(csf_img)
        csf_mean_all = float(csf_flat.mean()) if csf_flat is not None else 0.0
        total = gm_mean_all + wm_mean_all + csf_mean_all + 1e-9
        fracs  = [gm_mean_all / total, wm_mean_all / total, csf_mean_all / total]
        colors = ["#43a047", "#fb8c00", "#7e57c2"]
        labels_bar = [f"GM\n{100*fracs[0]:.1f}%", f"WM\n{100*fracs[1]:.1f}%",
                      f"CSF\n{100*fracs[2]:.1f}%"]
        bars = ax_bar.bar(labels_bar, fracs, color=colors, edgecolor="#222", width=0.5)
        ax_bar.set_facecolor(PAN)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_ylabel("Mean PVE fraction across MRSI voxels", color=TXT, fontsize=8)
        ax_bar.set_title("Avg tissue composition\nacross water-mask voxels",
                         color=TXT, fontsize=9, pad=4)
        ax_bar.tick_params(colors=TXT, labelsize=9)
        for sp in ax_bar.spines.values():
            sp.set_edgecolor(GREY)

    fig.suptitle(
        f"{subj}  {ses}  –  Tissue-based registration quality\n"
        "Row 1: GM / WM / CSF fraction maps (water contour in blue).  "
        "Row 2: metabolite GM-mean vs WM-mean scatter (normalised concentrations).",
        color=TXT, fontsize=10, y=1.01,
    )
    plt.tight_layout()
    plt.show()

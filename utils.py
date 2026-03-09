"""
utils.py
========
Visualization helpers for the FCBG Connectome Registration project.

All functions operate on paths/images set up by main.py.

Public API
----------
img_info(img, label)
metabolite_name(filename)
get_nonzero_com(img)
estimate_coverage(mrs_resampled_img, t1_img)
plot_single_overlay(mrs_filename, t1w_path, mrs_dir, subj, ...)
plot_support_mask(mrs_filename, t1w_path, mrs_dir, subj)
plot_coverage_grid(coverage_maps, mrs_dir, t1w_path, subj)
build_coverage_widget(mrs_dir, t1w_path, subj)
"""

import os
import re

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from nibabel.processing import resample_from_to
from nilearn import plotting


# ---------------------------------------------------------------------------
# Sanity-check helper
# ---------------------------------------------------------------------------

def img_info(img: nib.Nifti1Image, label: str) -> None:
    """Print shape, voxel size and dtype of a NIfTI image."""
    vox = np.sqrt(np.sum(img.affine[:3, :3] ** 2, axis=0))
    print(f"{label}")
    print(f"  shape      : {img.shape}")
    print(f"  voxel size : {vox.round(3)} mm")
    print(f"  dtype      : {img.get_data_dtype()}")
    print()


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def metabolite_name(filename: str) -> str:
    """Extract the BIDS ``desc`` label from an MRSI filename."""
    m = re.search(r"desc-([^_]+)_mrsi", filename)
    return m.group(1) if m else filename


def get_nonzero_com(img: nib.Nifti1Image) -> tuple[float, float, float]:
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
    mrs  = mrs_resampled_img.get_fdata()
    t1   = t1_img.get_fdata()
    brain   = np.isfinite(t1) & (t1 > 0)
    covered = np.isfinite(mrs) & (mrs > 0)
    frac = float(np.mean(covered[brain])) if brain.any() else 0.0
    print(f"  Brain coverage: {frac * 100:.1f}%")
    return frac


# ---------------------------------------------------------------------------
# Single-metabolite overlay
# ---------------------------------------------------------------------------

def plot_single_overlay(
    mrs_filename: str,
    t1w_path: str,
    mrs_dir: str,
    subj: str = "sub-01",
    cmap: str = "hot",
    threshold: float | None = None,
    vmax: float | None = None,
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
        title=f"{subj} – {label}",
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


# ---------------------------------------------------------------------------
# Binary support-mask overlay
# ---------------------------------------------------------------------------

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
        title=f"{subj} – MRSI support mask ({metabolite_name(mrs_filename)})",
        display_mode="ortho",
        cut_coords=get_nonzero_com(mask_res),
    )
    plt.show()
    return estimate_coverage(mask_res, t1_img)


# ---------------------------------------------------------------------------
# Multi-metabolite coverage grid
# ---------------------------------------------------------------------------

def plot_coverage_grid(
    coverage_maps: list[str],
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

    # world-space centre of the T1 volume
    centre_ijk = [s // 2 for s in t1w_data.shape]
    centre_xyz = nib.affines.apply_affine(t1w_img.affine, centre_ijk)
    cut_coords = tuple(float(v) for v in centre_xyz[:3])

    for mrs_filename in coverage_maps:
        mrs_path = os.path.join(mrs_dir, mrs_filename)
        mrs_img  = nib.load(mrs_path)
        label    = metabolite_name(mrs_filename)
        data     = mrs_img.get_fdata()
        vmax     = float(np.nanpercentile(data[data > 0], 99)) if np.any(data > 0) else 1.0

        # nilearn manages its own figure → no white-rectangle artefacts
        plotting.plot_stat_map(
            stat_map_img=mrs_img,
            bg_img=t1w_path,
            title=f"{subj} – {label}",
            display_mode="ortho",
            cut_coords=cut_coords,
            cmap="hot",
            colorbar=True,
            vmax=vmax,
            threshold=1e-6,
            resampling_interpolation="continuous",
        )
        plt.show()


# ---------------------------------------------------------------------------
# Interactive widget
# ---------------------------------------------------------------------------

def build_coverage_widget(
    mrs_dir: str,
    t1w_path: str,
    subj: str = "sub-01",
) -> widgets.Widget:
    """
    Build and return an ipywidgets UI.
    Call ``display(build_coverage_widget(...))`` in the notebook.

    Checkboxes select metabolites whose maps are summed and overlaid on the T1w.
    Coverage fraction is printed below the figure.
    """
    # load T1 once
    t1_can = nib.as_closest_canonical(nib.load(t1w_path))

    # only OrigRes maps (exclude EIB etc.)
    all_files = sorted(
        f for f in os.listdir(mrs_dir)
        if f.endswith(".nii.gz") and "acq-OrigRes" in f
    )
    all_labels = [metabolite_name(f) for f in all_files]

    # resampling cache
    _cache: dict[str, np.ndarray] = {}

    def _resampled(fname: str) -> np.ndarray:
        if fname not in _cache:
            img = nib.as_closest_canonical(nib.load(os.path.join(mrs_dir, fname)))
            _cache[fname] = resample_from_to(img, t1_can, order=1).get_fdata()
        return _cache[fname]

    # ── widgets ─────────────────────────────────────────────────────────────
    defaults = {"NAA", "Cr", "Glu"}
    n        = len(all_labels)
    col_size = (n + 1) // 2

    def _make_boxes(labels):
        return [
            widgets.Checkbox(
                value=(lbl in defaults),
                description=lbl,
                layout=widgets.Layout(width="200px"),
            )
            for lbl in labels
        ]

    col1_boxes = _make_boxes(all_labels[:col_size])
    col2_boxes = _make_boxes(all_labels[col_size:])
    all_boxes  = col1_boxes + col2_boxes

    btn_all     = widgets.Button(description="Select all",    button_style="info",    layout=widgets.Layout(width="120px"))
    btn_none    = widgets.Button(description="Deselect all",  button_style="warning", layout=widgets.Layout(width="120px"))
    btn_plot    = widgets.Button(description="▶  Plot",       button_style="success", layout=widgets.Layout(width="100px"))
    cmap_drop   = widgets.Dropdown(
        options=["hot", "plasma", "viridis", "RdYlBu_r", "jet"],
        value="hot", description="Colormap:", layout=widgets.Layout(width="200px"),
    )
    out = widgets.Output()

    def _on_all(_):
        for cb in all_boxes: cb.value = True
    def _on_none(_):
        for cb in all_boxes: cb.value = False

    def _on_plot(_):
        chosen = [all_files[i] for i, cb in enumerate(all_boxes) if cb.value]
        with out:
            clear_output(wait=True)
            if not chosen:
                print("⚠  Select at least one metabolite.")
                return

            labels   = [metabolite_name(f) for f in chosen]
            print(f"Combining {len(chosen)} maps: {labels}")
            arrays   = [_resampled(f) for f in chosen]
            combined = np.sum(np.stack(arrays, axis=0), axis=0)
            mask     = combined > 0
            combined[~mask] = 0

            combo_img = nib.Nifti1Image(combined, t1_can.affine)
            valid     = combined[mask]
            vmax      = float(np.nanpercentile(valid, 99)) if valid.size else 1.0
            thr       = float(np.nanpercentile(valid, 5))  if valid.size else 0.0

            plotting.plot_stat_map(
                stat_map_img=combo_img,
                bg_img=t1_can,
                title=f"{subj} – {' + '.join(labels)}",
                display_mode="ortho",
                cut_coords=get_nonzero_com(combo_img),
                cmap=cmap_drop.value,
                colorbar=True,
                threshold=thr,
                vmax=vmax,
                resampling_interpolation="nearest",
            )
            plt.show()

            t1_data = t1_can.get_fdata()
            brain   = np.isfinite(t1_data) & (t1_data > 0)
            frac    = mask[brain].mean() if brain.any() else 0.0
            print(f"Brain coverage: {frac*100:.1f}%  ({mask.sum()} voxels with signal)")

    btn_all.on_click(_on_all)
    btn_none.on_click(_on_none)
    btn_plot.on_click(_on_plot)

    ui = widgets.VBox([
        widgets.HBox([widgets.VBox(col1_boxes), widgets.VBox(col2_boxes)]),
        widgets.HBox([btn_all, btn_none, cmap_drop, btn_plot]),
        out,
    ])

    # auto-plot defaults on first display
    _on_plot(None)
    return ui

"""
utils.py
========
Visualization helpers for the FCBG Connectome Registration project.

All functions here operate on pre-loaded nibabel images or BIDS paths
that are set up by main.py.

Public API
----------
img_info(img, label)
    Print shape, voxel size and dtype of a NIfTI image.

metabolite_name(filename)
    Extract the BIDS ``desc`` label from an MRSI filename.

get_nonzero_com(img)
    Centre-of-mass (world coords) of nonzero voxels.

estimate_coverage(mrs_resampled_img, t1_img)
    Fraction of T1 brain voxels covered by the MRSI map.

plot_single_overlay(mrs_filename, t1w_path, mrs_dir, subj, cmap, threshold, vmax)
    Overlay one metabolite map on the T1w and show the coverage fraction.

plot_support_mask(mrs_filename, t1w_path, mrs_dir, subj)
    Binary support-mask overlay (for pure spatial coverage inspection).

plot_coverage_grid(coverage_maps, mrs_dir, t1w_path, subj)
    Multi-row figure: one row per metabolite, three orthogonal views each.

build_coverage_widget(mrs_dir, t1w_path, subj)
    Build and return an ipywidgets UI for interactive multi-metabolite overlay.
"""

import os
import re

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from nibabel.orientations import aff2axcodes
from nibabel.processing import resample_from_to
from nilearn import plotting

_MP2RAGE_SERIES_MAP: dict[int, dict] = {
    3:  {"suffix": "T1map",   "acq": None,          "inv": None},
    5:  {"suffix": "UNIT1",   "acq": "UNI",         "inv": None},
    6:  {"suffix": "T1w",     "acq": "UNIDEN",      "inv": None},
    7:  {"suffix": "MP2RAGE", "acq": None,          "inv": "1"},
    8:  {"suffix": "MP2RAGE", "acq": None,          "inv": "2"},
    9:  {"suffix": "T1w",     "acq": "MPRtra",      "inv": None},
    10: {"suffix": "T1w",     "acq": "MPRcor",      "inv": None},
    11: {"suffix": "T1w",     "acq": "UNIDEND",     "inv": None},
    12: {"suffix": "T1w",     "acq": "MPRcorND",    "inv": None},
    13: {"suffix": "T1w",     "acq": "MPRtraND",    "inv": None},
}

_ENTITY_ORDER = ["sub", "ses", "task", "acq", "ce", "rec", "run", "echo", "inv", "part"]


def _build_bids_filename(entities: dict, suffix: str, ext: str) -> str:
    """Assemble a BIDS-compliant filename from an entity dict, suffix, and extension."""
    parts = []
    for key in _ENTITY_ORDER:
        if key in entities:
            parts.append(f"{key}-{entities[key]}")
    # Append any extra keys not in the canonical order
    for key, val in entities.items():
        if key not in _ENTITY_ORDER:
            parts.append(f"{key}-{val}")
    return "_".join(parts) + f"_{suffix}{ext}"


def _parse_metabolite_label(filename: str) -> str:
    """
    Extract a BIDS-safe metabolite label from a filename such as
    ``OrigRes_NAA+NAAG_conc_reo.nii.gz``  →  ``NAAaddNAAG``
    ``EIB_conc_reo.nii.gz``               →  ``EIB``
    ``OrigRes_Cr+PCr_conc_reo.nii.gz``    →  ``CraddPCr``
    ``OrigRes_-CrCH2_conc_reo.nii.gz``    →  ``minusCrCH2``
    """
    stem = filename.replace(".nii.gz", "").replace(".nii", "")

    # Strip known prefix / suffix tokens
    stem = re.sub(r"^OrigRes_", "", stem)
    stem = re.sub(r"_conc_reo$", "", stem)
    stem = re.sub(r"_conc$", "", stem)

    # Make BIDS-safe: replace '+' with 'add', leading '-' with 'minus'
    label = stem
    label = re.sub(r"^\-", "minus", label)
    label = label.replace("+", "add")
    # Remove any remaining non-alphanumeric characters
    label = re.sub(r"[^a-zA-Z0-9]", "", label)

    return label


def _parse_acq_label(filename: str) -> str | None:
    """
    Return ``OrigRes`` if the filename starts with ``OrigRes_``, else ``None``
    (used for the BIDS ``acq`` entity on MRS maps).
    """
    if filename.startswith("OrigRes_"):
        return "OrigRes"
    return None


def convert_to_bids(
    data_dir: str | Path,
    output_dir: str | Path,
    subject_t1w: str = "01",
    session: str | None = None,
    overwrite: bool = False,
) -> None:
    """
    Convert the raw FCBG dataset into a BIDS-compliant directory tree.
    Output structure:

        <output_dir>/
        ├── dataset_description.json
        ├── participants.tsv
        ├── sub-01/
            ├── anat/
            │   ├── sub-01_T1map.nii
            │   ├── sub-01_T1map.json
            │   ├── sub-01_acq-UNI_UNIT1.nii
            │   ├── sub-01_T1w.nii           ← UNI-DEN (acq-UNIDEN)
            │   ├── sub-01_inv-1_MP2RAGE.nii
            │   ├── sub-01_inv-2_MP2RAGE.nii
            │   └── ...
            └── mrs/
                ├── sub-01_acq-OrigRes_metabolite-NAA_mrsmap.nii.gz
                └── ...
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    t1w_dir = data_dir / "T1w_NIFTI"
    mrs_root = data_dir / "met_conc_NIFTI"

    if not t1w_dir.exists():
        raise FileNotFoundError(f"T1w directory not found: {t1w_dir}")
    if not mrs_root.exists():
        raise FileNotFoundError(f"MRS directory not found: {mrs_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    _write_dataset_description(output_dir)
    _convert_t1w(t1w_dir, output_dir, subject_t1w, session, overwrite)

    participants: list[str] = [f"sub-{subject_t1w}"]
    for sub_dir in sorted(mrs_root.iterdir()):
        if not sub_dir.is_dir():
            continue
        # e.g. "sub01" → "01"
        label = re.sub(r"^sub0*", "", sub_dir.name) or sub_dir.name
        label = label.zfill(2)
        _convert_mrs(sub_dir, output_dir, label, session, overwrite)
        bids_label = f"sub-{label}"
        if bids_label not in participants:
            participants.append(bids_label)

    _write_participants_tsv(output_dir, participants)
    print(f"[convert_to_bids] Done. BIDS dataset written to: {output_dir}")



def _transfer(src: Path, dst: Path, overwrite: bool) -> None:
    """Copy ``src`` → ``dst``, respecting the overwrite flag."""
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _convert_t1w(
    t1w_dir: Path,
    bids_root: Path,
    subject: str,
    session: str | None,
    overwrite: bool,
) -> None:
    """Convert T1w / MP2RAGE NIfTI files to BIDS ``anat/`` layout."""
    base_entities = {"sub": subject}
    if session:
        base_entities["ses"] = session

    ses_part = f"ses-{session}/" if session else ""
    anat_dir = bids_root / f"sub-{subject}" / (f"ses-{session}/" if session else "") / "anat"
    anat_dir = (
        bids_root / f"sub-{subject}" / f"ses-{session}" / "anat"
        if session
        else bids_root / f"sub-{subject}" / "anat"
    )

    # Collect all NIfTI files together with their JSON sidecar
    nii_files = sorted(t1w_dir.glob("_t1_mp2rage*.nii"))
    for nii in nii_files:
        json_path = nii.with_suffix(".json")
        if not json_path.exists():
            print(f"  [T1w] No JSON sidecar for {nii.name}, skipping.")
            continue

        with open(json_path) as fh:
            meta = json.load(fh)

        series_num = meta.get("SeriesNumber")
        if series_num not in _MP2RAGE_SERIES_MAP:
            print(f"  [T1w] Series {series_num} ({nii.name}) not in mapping, skipping.")
            continue

        mapping = _MP2RAGE_SERIES_MAP[series_num]
        entities = {**base_entities, **mapping["entities"]}
        suffix = mapping["suffix"]

        bids_stem = _build_bids_filename(entities, suffix, "")
        dst_nii = anat_dir / (bids_stem + nii.suffix)  # preserves .nii or .nii.gz
        dst_json = anat_dir / (bids_stem + ".json")

        _transfer(nii, dst_nii, overwrite)
        _transfer(json_path, dst_json, overwrite)
        print(f"  [T1w] {nii.name}  →  anat/{dst_nii.name}")


def _convert_mrs(
    sub_dir: Path,
    bids_root: Path,
    subject: str,
    session: str | None,
    overwrite: bool,
) -> None:
    """Convert MRS metabolite-concentration maps to BIDS ``mrs/`` layout."""
    mrs_out = (
        bids_root / f"sub-{subject}" / f"ses-{session}" / "mrs"
        if session
        else bids_root / f"sub-{subject}" / "mrs"
    )

    for nii in sorted(sub_dir.glob("*.nii.gz")):
        metabolite = _parse_metabolite_label(nii.name)
        acq = _parse_acq_label(nii.name)

        entities: dict[str, str] = {"sub": subject}
        if session:
            entities["ses"] = session
        if acq:
            entities["acq"] = acq
        entities["metabolite"] = metabolite

        bids_name = _build_bids_filename(entities, "mrsmap", ".nii.gz")
        dst = mrs_out / bids_name
        _transfer(nii, dst, overwrite)
        print(f"  [MRS] sub-{subject}: {nii.name}  →  mrs/{bids_name}")


def _write_dataset_description(bids_root: Path) -> None:
    """Write a minimal ``dataset_description.json`` if one does not exist."""
    dst = bids_root / "dataset_description.json"
    if dst.exists():
        return
    description = {
        "Name": "FCBG Connectome Registration Dataset",
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "Authors": ["FCBG"],
        "License": "CC0",
    }
    with open(dst, "w") as fh:
        json.dump(description, fh, indent=2)


def _write_participants_tsv(bids_root: Path, participants: list[str]) -> None:
    """Write / update ``participants.tsv``."""
    dst = bids_root / "participants.tsv"
    with open(dst, "w") as fh:
        fh.write("participant_id\n")
        for p in participants:
            fh.write(f"{p}\n")

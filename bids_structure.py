import json
import os
import re
import shutil
from pathlib import Path

_MP2RAGE_SERIES_MAP: dict[int, dict] = {
    3:  {"suffix": "T1map",   "acq": None,        "inv": None},
    5:  {"suffix": "UNIT1",   "acq": "UNI",       "inv": None},
    6:  {"suffix": "T1w",     "acq": "UNIDEN",    "inv": None},
    7:  {"suffix": "MP2RAGE", "acq": None,        "inv": "1"},
    8:  {"suffix": "MP2RAGE", "acq": None,        "inv": "2"},
    9:  {"suffix": "T1w",     "acq": "MPRtra",    "inv": None},
    10: {"suffix": "T1w",     "acq": "MPRcor",    "inv": None},
    11: {"suffix": "T1w",     "acq": "UNIDEND",   "inv": None},
    12: {"suffix": "T1w",     "acq": "MPRcorND",  "inv": None},
    13: {"suffix": "T1w",     "acq": "MPRtraND",  "inv": None},
}

_ENTITY_ORDER = ["sub", "ses", "task", "acq", "ce", "rec", "run", "echo", "inv", "part", "desc"]

class BIDSStructure:

    def __init__(
        self,
        subj: str,
        sess: int | str,
        datatype: str,
        root: str | Path,
    ):
        if isinstance(sess, int):
            sess_label = f"ses-{sess:02d}"
        elif sess.startswith("ses-"):
            sess_label = sess
        else:
            sess_label = f"ses-{sess}"

        self.subj      = subj if subj.startswith("sub-") else f"sub-{subj}"
        self.sess      = sess_label
        self.datatype  = datatype
        self.root      = Path(root)
        self.path      = self.root / self.subj / self.sess / self.datatype

    def __repr__(self) -> str:
        return f"BIDSStructure({self.path})"


def create_bids_structure(
    subj: str,
    sess: int | str,
    datatype: str,
    root: str | Path,
) -> BIDSStructure:
    
    return BIDSStructure(subj=subj, sess=sess, datatype=datatype, root=root)


def find_files_with_pattern(
    bids_structure: BIDSStructure,
    pattern: str,
    suffix: str | None = None,
) -> list[str]:
    folder = bids_structure.path
    if not folder.exists():
        raise FileNotFoundError(f"BIDS folder not found: {folder}")

    results = []
    for fname in sorted(os.listdir(folder)):
        if suffix and not fname.endswith(suffix):
            continue
        if re.search(pattern, fname):
            results.append(str(folder / fname))
    return results


def run(
    data_dir: str | Path,
    output_dir: str | Path,
    subject_t1w: str = "01",
    session: int = 1,
    overwrite: bool = False,
) -> None:
    """
    Convert the raw FCBG dataset to a BIDS directory tree.
    """
    data_dir   = Path(data_dir)
    output_dir = Path(output_dir)
    t1w_dir    = data_dir / "T1w_NIFTI"
    mrs_root   = data_dir / "met_conc_NIFTI"

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
        label = re.sub(r"^sub0*", "", sub_dir.name) or sub_dir.name
        label = label.zfill(2)
        _convert_mrs(sub_dir, output_dir, label, session, overwrite)
        bids_label = f"sub-{label}"
        if bids_label not in participants:
            participants.append(bids_label)

    _write_participants_tsv(output_dir, participants)
    

def _bids_filename(entities: dict, suffix: str, ext: str) -> str:
    """Assemble a BIDS filename from an entity dict, suffix, and extension."""
    parts = []
    for key in _ENTITY_ORDER:
        val = entities.get(key)
        if val is not None:
            parts.append(f"{key}-{val}")
    # any extra entity not in the canonical order goes at the end
    for key, val in entities.items():
        if key not in _ENTITY_ORDER and val is not None:
            parts.append(f"{key}-{val}")
    return "_".join(parts) + f"_{suffix}{ext}"


def _transfer(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _convert_t1w(
    t1w_dir: Path,
    bids_root: Path,
    subject: str,
    session: int,
    overwrite: bool,
):
    ses_label = f"ses-{session:02d}"
    anat_dir  = bids_root / f"sub-{subject}" / ses_label / "anat"

    for nii in sorted(t1w_dir.glob("_t1_mp2rage*.nii")):
        json_path = nii.with_suffix(".json")
        if not json_path.exists():
            print(f"  [T1w] no JSON sidecar for {nii.name}, skipping.")
            continue

        with open(json_path) as fh:
            meta = json.load(fh)

        series_num = meta.get("SeriesNumber")
        if series_num not in _MP2RAGE_SERIES_MAP:
            print(f"  [T1w] series {series_num} not in map ({nii.name}), skipping.")
            continue

        mapping = _MP2RAGE_SERIES_MAP[series_num]
        entities: dict[str, str | None] = {
            "sub": subject,
            "ses": f"{session:02d}",
            "acq": mapping["acq"],
            "inv": mapping["inv"],
        }
        suffix    = mapping["suffix"]
        bids_stem = _bids_filename(entities, suffix, "")
        dst_nii   = anat_dir / (bids_stem + nii.suffix)
        dst_json  = anat_dir / (bids_stem + ".json")

        _transfer(nii, dst_nii, overwrite)
        _transfer(json_path, dst_json, overwrite)
    


def _metabolite_label(filename: str) -> str:
    stem = filename.replace(".nii.gz", "").replace(".nii", "")
    stem = re.sub(r"^OrigRes_", "", stem)
    stem = re.sub(r"_conc_reo$", "", stem)
    stem = re.sub(r"_conc$", "", stem)
    stem = re.sub(r"^\-", "minus", stem)
    stem = stem.replace("+", "add")
    stem = re.sub(r"[^a-zA-Z0-9]", "", stem)
    return stem


def _convert_mrs(
    sub_dir: Path,
    bids_root: Path,
    subject: str,
    session: int,
    overwrite: bool,
) -> None:
    """Copy MRS concentration maps into ``mrs/`` with BIDS names."""
    ses_label = f"ses-{session:02d}"
    mrs_out   = bids_root / f"sub-{subject}" / ses_label / "mrs"

    for nii in sorted(sub_dir.glob("*.nii.gz")):
        desc = _metabolite_label(nii.name)
        acq  = "OrigRes" if nii.name.startswith("OrigRes_") else None

        entities: dict[str, str | None] = {
            "sub": subject,
            "ses": f"{session:02d}",
            "acq": acq,
            "desc": desc,
        }
        bids_name = _bids_filename(entities, "mrsi", ".nii.gz")
        dst = mrs_out / bids_name
        _transfer(nii, dst, overwrite)
        print(f"  [MRS] sub-{subject}: {nii.name}  →  mrs/{bids_name}")


def _write_dataset_description(bids_root: Path):
    dst = bids_root / "dataset_description.json"
    if dst.exists():
        return
    with open(dst, "w") as fh:
        json.dump({
            "Name": "FCBG Connectome Registration Dataset",
            "BIDSVersion": "1.10.0",
            "DatasetType": "raw",
            "Authors": ["FCBG"],
            "License": "CC0",
        }, fh, indent=2)


def _write_participants_tsv(bids_root: Path, participants: list[str]):
    dst = bids_root / "participants.tsv"
    with open(dst, "w") as fh:
        fh.write("participant_id\n")
        for p in participants:
            fh.write(f"{p}\n")

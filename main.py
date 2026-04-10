import os
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
import bids_structure

from data_utils import (
    save_metabolite_sum,
    downsample_t1w_to_mrs,
    fill_mask_holes,
    rank_metabolites_by_snr,
    skull_strip_t1w,
    segment_t1w,
    segment_t1w_atlas)
from registration_utils import (
    register_mrsi_to_t1w,
    register_t1w_to_mrsi,
    register_t1w_to_mrsi_weighted,
    apply_transform,
    run_total_pipeline)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
BIDS_DIR     = os.path.join(DATA_DIR, "bids")
OUTPUT_DIR   = os.path.join(DATA_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUBJ = "sub-01"
SES  = "ses-01"

ANAT_DIR = os.path.join(BIDS_DIR, SUBJ, SES, "anat")
MRS_DIR  = os.path.join(BIDS_DIR, SUBJ, SES, "mrs")
T1W_PATH = os.path.join(ANAT_DIR, f"{SUBJ}_{SES}_acq-UNIDEN_T1w.nii")

bids_structure.run(
    data_dir    = DATA_DIR,
    output_dir  = BIDS_DIR,
    subject_t1w = "01",
    session     = 1,
    overwrite   = False,
)

mrs_files = sorted(
    f for f in os.listdir(MRS_DIR) if f.endswith(".nii.gz")
)

t1w_img     = nib.load(T1W_PATH)
mrs_example = nib.load(os.path.join(MRS_DIR, mrs_files[0]))

SUM_NAME  = f"{SUBJ}_{SES}_acq-OrigRes_desc-AllMetabSum_mrsi.nii.gz"
SUM_PATH  = os.path.join(OUTPUT_DIR, SUM_NAME)
save_metabolite_sum(BIDS_DIR, ses=SES, overwrite=True, out_dir=OUTPUT_DIR, mask_dir=OUTPUT_DIR, subjects=[SUBJ])
sum_img   = nib.load(SUM_PATH) if os.path.exists(SUM_PATH) else None

# Downsampled T1w resampled to the MRSI voxel grid
T1W_DS_NAME = f"{SUBJ}_{SES}_acq-MRSIres_T1w.nii.gz"
T1W_DS_PATH = os.path.join(OUTPUT_DIR, T1W_DS_NAME)
downsample_t1w_to_mrs(BIDS_DIR, ses=SES, overwrite=False, out_dir=OUTPUT_DIR, subjects=[SUBJ])
t1w_ds_img  = nib.load(T1W_DS_PATH) if os.path.exists(T1W_DS_PATH) else None

# Fill holes in a water-derived mask, then apply that mask to water in main.
WATER_NAME = f"{SUBJ}_{SES}_desc-WaterSignal_mrsi.nii.gz"
WATER_PATH = os.path.join(MRS_DIR, WATER_NAME)

MASK_NAME = f"{SUBJ}_{SES}_desc-WaterMask_mrsi.nii.gz"
WATER_MASK_PATH = os.path.join(OUTPUT_DIR, MASK_NAME)

WATER_MASKED_NAME = f"{SUBJ}_{SES}_desc-WaterSignalMasked_mrsi.nii.gz"
WATER_MASKED_PATH = os.path.join(OUTPUT_DIR, WATER_MASKED_NAME)

water_img = nib.load(WATER_PATH)
water_mask_img = fill_mask_holes(
    water_img_path=WATER_PATH,
    out_mask_path=WATER_MASK_PATH,
    overwrite=True,
)

# Boolean numpy mask ready to reuse 
water_mask = water_mask_img.get_fdata().astype(bool)


snr_records = rank_metabolites_by_snr(MRS_DIR, subj=SUBJ, ses=SES)
best_mrsi_name = snr_records[0]["filename"]
best_mrsi_img  = nib.load(os.path.join(MRS_DIR, best_mrsi_name))
best_mrsi_label = snr_records[0]["metabolite"]

# Registration 1: bestbSNR MRSI  downsampled T1w (rigid, MRSI-res space)
MRSI_REG_DS_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-Registered_mrsi.nii.gz"
MRSI_REG_DS_PATH = os.path.join(OUTPUT_DIR, MRSI_REG_DS_NAME)
mrsi_reg_ds_img, reg1_transforms = register_mrsi_to_t1w(
    mrsi_img=best_mrsi_img,
    t1w_img=t1w_ds_img,
    mask=water_mask,
    out_path=MRSI_REG_DS_PATH,
    overwrite=True,
)

# Registration 2: best-SNR MRSI to full-resolution T1w (rigid, full T1w space)
MRSI_REG_FULLRES_NAME = f"{SUBJ}_{SES}_acq-FullRes_desc-Registered_mrsi.nii.gz"
MRSI_REG_FULLRES_PATH = os.path.join(OUTPUT_DIR, MRSI_REG_FULLRES_NAME)
mrsi_reg_fullres_img, _ = register_mrsi_to_t1w(
    mrsi_img=best_mrsi_img,
    t1w_img=t1w_img,
    mask=water_mask,
    out_path=MRSI_REG_FULLRES_PATH,
    overwrite=True,
)

# Registration 3: independently register the metabolite sum to DS T1w.
MRSI_SUM_REG_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-RegisteredSum_mrsi.nii.gz"
MRSI_SUM_REG_PATH = os.path.join(OUTPUT_DIR, MRSI_SUM_REG_NAME)
if sum_img is not None:
    mrsi_sum_reg_img, reg3_transforms = register_mrsi_to_t1w(
        mrsi_img=sum_img,
        t1w_img=t1w_ds_img,
        mask=water_mask,
        out_path=MRSI_SUM_REG_PATH,
        overwrite=True,
    )
else:
    mrsi_sum_reg_img, reg3_transforms = None, None

# Registration 4: apply Registration 1's best SNR transform to the sum
# (no re-optimisation)  used only for comparison with Registration 3.
MRSI_SUM_XFM_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-SumViaBestSNRXfm_mrsi.nii.gz"
MRSI_SUM_XFM_PATH = os.path.join(OUTPUT_DIR, MRSI_SUM_XFM_NAME)
if sum_img is not None and reg1_transforms is not None:
    mrsi_sum_xfm_img, _ = register_mrsi_to_t1w(
        mrsi_img=sum_img,
        t1w_img=t1w_ds_img,
        mask=water_mask,
        out_path=MRSI_SUM_XFM_PATH,
        overwrite=True,
        init_transforms=reg1_transforms,
    )
else:
    mrsi_sum_xfm_img = None

# Registration 5: apply the sum's transform (Reg 3) to the best-SNR (Gly) image.
MRSI_GLY_VIA_SUM_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-BestSNRViaSumXfm_mrsi.nii.gz"
MRSI_GLY_VIA_SUM_PATH = os.path.join(OUTPUT_DIR, MRSI_GLY_VIA_SUM_NAME)
if reg3_transforms is not None:
    mrsi_gly_via_sum_img, _ = register_mrsi_to_t1w(
        mrsi_img=best_mrsi_img,
        t1w_img=t1w_ds_img,
        mask=water_mask,
        out_path=MRSI_GLY_VIA_SUM_PATH,
        overwrite=True,
        init_transforms=reg3_transforms,
    )
else:
    mrsi_gly_via_sum_img = None

# Best-coverage metabolite: the map with the most voxels containing signal.
coverage_records = sorted(snr_records, key=lambda r: r["n_voxels"], reverse=True)
best_cov_record = coverage_records[0]
best_cov_label  = best_cov_record["metabolite"]
best_cov_img    = nib.load(os.path.join(MRS_DIR, best_cov_record["filename"]))
print(f"Best-coverage metabolite: {best_cov_label}  ({best_cov_record['n_voxels']} voxels)")

# Registration 6: best-coverage metabolite independently registered to DS T1w.
MRSI_COV_REG_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-BestCoverageReg_mrsi.nii.gz"
MRSI_COV_REG_PATH = os.path.join(OUTPUT_DIR, MRSI_COV_REG_NAME)
mrsi_cov_reg_img, reg6_transforms = register_mrsi_to_t1w(
    mrsi_img=best_cov_img,
    t1w_img=t1w_ds_img,
    mask=water_mask,
    out_path=MRSI_COV_REG_PATH,
    overwrite=True,
)

# Inverse registrations: T1w DS to MRSI space (MRSI fixed, T1w moving) 

# Registration 7: T1w DS to  MRSI sum map 
T1W_IN_SUM_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-T1wInSum_T1w.nii.gz"
T1W_IN_SUM_PATH = os.path.join(OUTPUT_DIR, T1W_IN_SUM_NAME)
if sum_img is not None:
    t1w_in_sum_img, inv_sum_transforms = register_t1w_to_mrsi(
        t1w_img=t1w_ds_img,
        mrsi_img=sum_img,
        out_path=T1W_IN_SUM_PATH,
        overwrite=False,
        mask=water_mask,
    )
else:
    t1w_in_sum_img, inv_sum_transforms = None, None

# Registration 8: T1w DS to MRSI best-SNR (Gly) map 
T1W_IN_GLY_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-T1wInGly_T1w.nii.gz"
T1W_IN_GLY_PATH = os.path.join(OUTPUT_DIR, T1W_IN_GLY_NAME)
t1w_in_gly_img, inv_gly_transforms = register_t1w_to_mrsi(
    t1w_img=t1w_ds_img,
    mrsi_img=best_mrsi_img,
    mask=water_mask,
    out_path=T1W_IN_GLY_PATH,
    overwrite=False,
)

# Registration 9: T1w DS to Gly space using the sum transform (reuse Reg 7)
T1W_VIA_SUM_IN_GLY_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-T1wViaSumInGly_T1w.nii.gz"
T1W_VIA_SUM_IN_GLY_PATH = os.path.join(OUTPUT_DIR, T1W_VIA_SUM_IN_GLY_NAME)
if inv_sum_transforms is not None:
    t1w_via_sum_in_gly_img, _ = register_t1w_to_mrsi(
        t1w_img=t1w_ds_img,
        mrsi_img=best_mrsi_img,
        out_path=T1W_VIA_SUM_IN_GLY_PATH,
        overwrite=False,
        init_transforms=inv_sum_transforms,
    )
else:
    t1w_via_sum_in_gly_img = None

#  Weighted inverse registration paths 
BEST_MRSI_PATH = os.path.join(MRS_DIR, best_mrsi_name)

# Registration 10 paths: T1w DS to best SNR MRSI, water-weighted
T1W_IN_GLY_W_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-T1wInGlyWeighted_T1w.nii.gz"
T1W_IN_GLY_W_PATH = os.path.join(OUTPUT_DIR, T1W_IN_GLY_W_NAME)
T1W_IN_GLY_W_XFM  = T1W_IN_GLY_W_PATH.replace(".nii.gz", "_fwdtransform.mat")
t1w_in_gly_w_img, inv_gly_w_transforms = register_t1w_to_mrsi_weighted(
    fixed_path=BEST_MRSI_PATH,
    moving_path=T1W_DS_PATH,
    mask_path=WATER_MASK_PATH,
    out_path=T1W_IN_GLY_W_PATH,
    transform_path=T1W_IN_GLY_W_XFM,
    overwrite=False,
)

# Registration 11 paths: T1w DS to sum MRSI, water-weighted
T1W_IN_SUM_W_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-T1wInSumWeighted_T1w.nii.gz"
T1W_IN_SUM_W_PATH = os.path.join(OUTPUT_DIR, T1W_IN_SUM_W_NAME)
T1W_IN_SUM_W_XFM  = T1W_IN_SUM_W_PATH.replace(".nii.gz", "_fwdtransform.mat")
if sum_img is not None:
    t1w_in_sum_w_img, inv_sum_w_transforms = register_t1w_to_mrsi_weighted(
        fixed_path=SUM_PATH,
        moving_path=T1W_DS_PATH,
        mask_path=WATER_MASK_PATH,
        out_path=T1W_IN_SUM_W_PATH,
        transform_path=T1W_IN_SUM_W_XFM,
        overwrite=False,
    )
else:
    t1w_in_sum_w_img, inv_sum_w_transforms = None, None

# Registration 12: T1w DS to  Gly via sum's water-weighted transform (reuse Reg 11)
T1W_VIA_WSUM_IN_GLY_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-T1wViaWSumInGly_T1w.nii.gz"
T1W_VIA_WSUM_IN_GLY_PATH = os.path.join(OUTPUT_DIR, T1W_VIA_WSUM_IN_GLY_NAME)
T1W_VIA_WSUM_IN_GLY_XFM  = T1W_VIA_WSUM_IN_GLY_PATH.replace(".nii.gz", "_fwdtransform.mat")
if inv_sum_w_transforms is not None:
    t1w_via_wsum_in_gly_img, _ = register_t1w_to_mrsi_weighted(
        fixed_path=BEST_MRSI_PATH,
        moving_path=T1W_DS_PATH,
        mask_path=WATER_MASK_PATH,
        out_path=T1W_VIA_WSUM_IN_GLY_PATH,
        transform_path=T1W_VIA_WSUM_IN_GLY_XFM,
        overwrite=False,
        init_transforms=inv_sum_w_transforms,
    )
else:
    t1w_via_wsum_in_gly_img = None

# Registration 13: water signal to DS T1w (reuse Reg 1 transform, no re-optimisation)
WATER_REG_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-WaterRegistered_T1w.nii.gz"
WATER_REG_PATH = os.path.join(OUTPUT_DIR, WATER_REG_NAME)
water_reg_img, _ = register_mrsi_to_t1w(
    mrsi_img=water_img,
    t1w_img=t1w_ds_img,
    mask=water_mask,
    out_path=WATER_REG_PATH,
    overwrite=False,
    init_transforms=reg1_transforms,
)

# Skull-stripping plus skull-stripped inverse registrations

# Step 1 – Skull-strip the full-resolution T1w 
T1W_BRAIN_NAME = f"{SUBJ}_{SES}_acq-FullRes_desc-Brain_T1w.nii.gz"
T1W_BRAIN_PATH = os.path.join(OUTPUT_DIR, T1W_BRAIN_NAME)
t1w_brain_img = skull_strip_t1w(
    in_path=T1W_PATH,
    out_path=T1W_BRAIN_PATH,
    frac=0.5,
    overwrite=False,
)

# Step 2 – Downsample the stripped full-res brain to the MRSI voxel grid
T1W_DS_BRAIN_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-Brain_T1w.nii.gz"
T1W_DS_BRAIN_PATH = os.path.join(OUTPUT_DIR, T1W_DS_BRAIN_NAME)
if not os.path.exists(T1W_DS_BRAIN_PATH):
    _ds = resample_from_to(t1w_brain_img, mrs_example, order=1)
    _ds = nib.Nifti1Image(np.array(_ds.dataobj, dtype=np.float32), _ds.affine, _ds.header)
    nib.save(_ds, T1W_DS_BRAIN_PATH)
    print(f"  [t1w-ds-brain] saved {T1W_DS_BRAIN_NAME}")
t1w_ds_brain_img = nib.load(T1W_DS_BRAIN_PATH)

# Step 3 – Downsample the BET binary brain mask to the MRSI voxel grid
# (used as fixed-image mask for the water-space registrations in Reg 18)
T1W_BRAIN_MASK_PATH    = T1W_BRAIN_PATH.replace(".nii.gz", "_mask.nii.gz")
T1W_BRAIN_MASK_DS_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-BrainMask_T1w.nii.gz"
T1W_BRAIN_MASK_DS_PATH = os.path.join(OUTPUT_DIR, T1W_BRAIN_MASK_DS_NAME)
if not os.path.exists(T1W_BRAIN_MASK_DS_PATH) and os.path.exists(T1W_BRAIN_MASK_PATH):
    _mask_img = nib.load(T1W_BRAIN_MASK_PATH)
    _mask_ds  = resample_from_to(_mask_img, mrs_example, order=0)  # nearest-neighbour for binary mask
    _mask_ds  = nib.Nifti1Image(np.array(_mask_ds.dataobj, dtype=np.uint8), _mask_ds.affine, _mask_ds.header)
    nib.save(_mask_ds, T1W_BRAIN_MASK_DS_PATH)
    print(f"  [bet-mask-ds] saved {T1W_BRAIN_MASK_DS_NAME}")

#  FSL FAST tissue segmentation on the full-resolution skull-stripped T1w
SEG_PREFIX  = os.path.join(OUTPUT_DIR, f"{SUBJ}_{SES}_acq-FullRes_desc-Brain_T1w_seg")
seg_imgs = segment_t1w(
    brain_path=T1W_BRAIN_PATH,
    out_prefix=SEG_PREFIX,
    n_classes=3,
    overwrite=False,
)
PVE_CSF_PATH = f"{SEG_PREFIX}_pve_0.nii.gz"
PVE_GM_PATH  = f"{SEG_PREFIX}_pve_1.nii.gz"
PVE_WM_PATH  = f"{SEG_PREFIX}_pve_2.nii.gz"

#  # Atlas-based parcellation: FLIRT to MNI152, Harvard-Oxford cortical + subcortical
# ATLAS_PREFIX = os.path.join(OUTPUT_DIR, f"{SUBJ}_{SES}_acq-FullRes_desc-Brain_T1w_atlas")
# atlas_imgs = segment_t1w_atlas(
#      brain_path=T1W_BRAIN_PATH,
#      out_prefix=ATLAS_PREFIX,
#      nonlinear=False,
#      overwrite=False,
#  )

# Registration 14: skull-stripped T1w DS to best SNR MRSI (Gly), water-weighted
T1W_BRAIN_IN_GLY_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-BrainT1wInGly_T1w.nii.gz"
T1W_BRAIN_IN_GLY_PATH = os.path.join(OUTPUT_DIR, T1W_BRAIN_IN_GLY_NAME)
T1W_BRAIN_IN_GLY_XFM  = T1W_BRAIN_IN_GLY_PATH.replace(".nii.gz", "_fwdtransform.mat")
_brain_mask_arg = T1W_BRAIN_MASK_DS_PATH if os.path.exists(T1W_BRAIN_MASK_DS_PATH) else None
t1w_brain_in_gly_img, brain_gly_transforms = register_t1w_to_mrsi_weighted(
    fixed_path=BEST_MRSI_PATH,
    moving_path=T1W_DS_BRAIN_PATH,
    mask_path=WATER_MASK_PATH,
    out_path=T1W_BRAIN_IN_GLY_PATH,
    transform_path=T1W_BRAIN_IN_GLY_XFM,
    overwrite=False,
    moving_mask_path=_brain_mask_arg,
)

# Registration 15: skull-stripped T1w DS to sum MRSI, water-weighted
T1W_BRAIN_IN_SUM_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-BrainT1wInSum_T1w.nii.gz"
T1W_BRAIN_IN_SUM_PATH = os.path.join(OUTPUT_DIR, T1W_BRAIN_IN_SUM_NAME)
T1W_BRAIN_IN_SUM_XFM  = T1W_BRAIN_IN_SUM_PATH.replace(".nii.gz", "_fwdtransform.mat")

t1w_brain_in_sum_img, brain_sum_transforms = register_t1w_to_mrsi_weighted(
        fixed_path=SUM_PATH,
        moving_path=T1W_DS_BRAIN_PATH,
        mask_path=WATER_MASK_PATH,
        out_path=T1W_BRAIN_IN_SUM_PATH,
        transform_path=T1W_BRAIN_IN_SUM_XFM,
        overwrite=False,
        moving_mask_path=_brain_mask_arg)


# Registration 16: skull-stripped T1w in Gly space via sum transform (reuse Reg 15)
T1W_BRAIN_VIA_SUM_IN_GLY_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-BrainT1wViaSumInGly_T1w.nii.gz"
T1W_BRAIN_VIA_SUM_IN_GLY_PATH = os.path.join(OUTPUT_DIR, T1W_BRAIN_VIA_SUM_IN_GLY_NAME)
T1W_BRAIN_VIA_SUM_IN_GLY_XFM  = T1W_BRAIN_VIA_SUM_IN_GLY_PATH.replace(".nii.gz", "_fwdtransform.mat")
if brain_sum_transforms is not None:
    t1w_brain_via_sum_in_gly_img, _ = register_t1w_to_mrsi_weighted(
        fixed_path=BEST_MRSI_PATH,
        moving_path=T1W_DS_BRAIN_PATH,
        mask_path=WATER_MASK_PATH,
        out_path=T1W_BRAIN_VIA_SUM_IN_GLY_PATH,
        transform_path=T1W_BRAIN_VIA_SUM_IN_GLY_XFM,
        overwrite=False,
        init_transforms=brain_sum_transforms,
    )
else:
    t1w_brain_via_sum_in_gly_img = None


#  Total pipeline registration 17 
(t1w_brain_in_sum_ras_img,
    brain_sum_ras_transforms,
    final_reg_imgs,
    metrics) = run_total_pipeline(
    bids_dir=BIDS_DIR,
    mrs_dir=MRS_DIR,
    water_path=WATER_PATH,
    t1w_ds_brain_path=T1W_DS_BRAIN_PATH,
    t1w_ds_brain_img=t1w_ds_brain_img,
    subj=SUBJ,
    ses=SES,
    output_dir=OUTPUT_DIR,
    overwrite=False,
    t1w_brain_mask_ds_path=_brain_mask_arg)

SUM_RAS_NAME              = f"{SUBJ}_{SES}_acq-OrigRes_desc-AllMetabSumRAS_mrsi.nii.gz"
SUM_RAS_PATH              = os.path.join(OUTPUT_DIR, SUM_RAS_NAME)
WATER_RAS_NAME            = f"{SUBJ}_{SES}_desc-WaterSignalRAS_mrsi.nii.gz"
WATER_RAS_PATH            = os.path.join(OUTPUT_DIR, WATER_RAS_NAME)
T1W_BRAIN_IN_SUM_RAS_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-BrainT1wInSumRAS_T1w.nii.gz"
T1W_BRAIN_IN_SUM_RAS_PATH = os.path.join(OUTPUT_DIR, T1W_BRAIN_IN_SUM_RAS_NAME)
FINAL_REG_DIR             = os.path.join(OUTPUT_DIR, "final_reg")

sum_ras_img = nib.load(SUM_RAS_PATH) if os.path.exists(SUM_RAS_PATH) else None

# Registration 18 t1w resampled to water vis ANTS with mask provided

# (a) Reg-11 sum transform reused  apply inv_sum_w_transforms to T1w DS
T1W_IN_WATER_VIA17_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-T1wInWaterViaReg11_T1w.nii.gz"
T1W_IN_WATER_VIA17_PATH = os.path.join(OUTPUT_DIR, T1W_IN_WATER_VIA17_NAME)
T1W_IN_WATER_VIA17_XFM  = T1W_IN_WATER_VIA17_PATH.replace(".nii.gz", "_fwdtransform.mat")
t1w_in_water_via17_img, _ = register_t1w_to_mrsi_weighted(
        fixed_path=WATER_PATH,
        moving_path=T1W_DS_PATH,
        mask_path=WATER_MASK_PATH,
        out_path=T1W_IN_WATER_VIA17_PATH,
        transform_path=T1W_IN_WATER_VIA17_XFM,
        overwrite=False,
        init_transforms=inv_sum_w_transforms,
        moving_mask_path=T1W_BRAIN_MASK_DS_PATH)

# (b) Reg-18: DS T1w to  water signal map (ANTs CLI rigid, brain mask)
T1W_IN_WATER_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-T1wInWaterReg18_T1w.nii.gz"
T1W_IN_WATER_PATH = os.path.join(OUTPUT_DIR, T1W_IN_WATER_NAME)
T1W_IN_WATER_XFM  = T1W_IN_WATER_PATH.replace(".nii.gz", "_fwdtransform.mat")
t1w_in_water_img, t1w_water_transforms = register_t1w_to_mrsi_weighted(
    fixed_path=WATER_PATH,
    moving_path=T1W_DS_PATH,
    mask_path=WATER_MASK_PATH,
    out_path=T1W_IN_WATER_PATH,
    transform_path=T1W_IN_WATER_XFM,
    overwrite=False,
    moving_mask_path=T1W_BRAIN_MASK_DS_PATH 
)


# BET brain mask to  MRSI space (Reg-15 brain to sum forward transform)

BET_MASK_REG_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-BrainMaskReg_T1w.nii.gz"
BET_MASK_REG_PATH = os.path.join(OUTPUT_DIR, BET_MASK_REG_NAME)

# freesurfer brain mask to  MRSI space (Reg-15 brain to sum forward transform)
FREESURFER_MASK_REG_NAME = f"{SUBJ}_{SES}_acq-UNIDEND_T1w_brainatlasmore_mask.nii"
FREESURFER_MASK_REG_PATH = os.path.join(OUTPUT_DIR, FREESURFER_MASK_REG_NAME)

bet_mask_reg_img = apply_transform(
        in_path=T1W_BRAIN_MASK_PATH,
        ref_path=SUM_PATH,
        transform_path=brain_sum_transforms[0],
        out_path=BET_MASK_REG_PATH,
        overwrite=False)

freesurfer_mask_reg_img = apply_transform(
        in_path=T1W_BRAIN_MASK_PATH,
        ref_path=SUM_PATH,
        transform_path=brain_sum_transforms[0],
        out_path=FREESURFER_MASK_REG_PATH,
        overwrite=False)

water_mask_reg_img = apply_transform(
        in_path=WATER_MASK_PATH,
        ref_path=SUM_PATH,
        transform_path=brain_sum_transforms[0],
        out_path=FREESURFER_MASK_REG_PATH,
        overwrite=False)

# # ──────────────────────────────────────────────────────────────────────────
# # Section 25 – Registration comparison dict (T1w images in MRSI space)
# # ──────────────────────────────────────────────────────────────────────────
# REG_COMPARE_IMGS = {}
# for _label, _path in [
#     ("Reg10 T1w→Gly",       T1W_IN_GLY_W_PATH),
#     ("Reg11 T1w→Sum",       T1W_IN_SUM_W_PATH),
#     ("Reg14 Brain→Gly",     T1W_BRAIN_IN_GLY_PATH),
#     ("Reg15 Brain→Sum",     T1W_BRAIN_IN_SUM_PATH),
#     ("Reg18 T1w→Water",     T1W_IN_WATER_PATH),
# ]:
#     if os.path.exists(_path):
#         REG_COMPARE_IMGS[_label] = nib.load(_path)

# # ──────────────────────────────────────────────────────────────────────────
# # Section 26 – Tissue fractions in MRSI space + original metabolite maps
# # ──────────────────────────────────────────────────────────────────────────
# # Warp FAST GM/WM/CSF PVE maps from T1w space → MRSI space using Reg-11
# # forward transform (T1w DS → MRSI sum).  ANTs physical-space transforms
# # are valid for the full-res T1w PVE maps since they share the same
# # physical coordinate system as T1w DS.
# TISSUE_FRACS = {}
# if os.path.exists(T1W_IN_SUM_W_XFM) and os.path.exists(PVE_GM_PATH):
#     TISSUE_FRACS = compute_tissue_fractions_in_mrsi(
#         pve_gm_path=PVE_GM_PATH,
#         pve_wm_path=PVE_WM_PATH,
#         pve_csf_path=PVE_CSF_PATH,
#         mrsi_ref_path=SUM_PATH,
#         xfm_path=T1W_IN_SUM_W_XFM,
#         out_dir=OUTPUT_DIR,
#         subj=SUBJ,
#         ses=SES,
#         overwrite=False,
#     )

# # All original-resolution metabolite concentration maps in MRSI space
# MRSI_CONC_IMGS = {
#     metabolite_name(f): nib.load(os.path.join(MRS_DIR, f))
#     for f in sorted(os.listdir(MRS_DIR))
#     if f.endswith(".nii.gz") and "acq-OrigRes" in f and "AllMetab" not in f
# }


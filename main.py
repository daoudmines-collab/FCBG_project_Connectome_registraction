import os
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
import bids_structure

from data_utils import (
    save_metabolite_sum,
    save_reoriented_metabolite_sum,
    downsample_t1w_to_mrs,
    fill_mask_holes,
    rank_metabolites_by_snr,
    skull_strip_t1w,
    segment_t1w,
    segment_t1w_atlas,
    metabolite_name)
from registration_utils import (
    register_mrsi_to_t1w,
    register_t1w_to_mrsi,
    register_t1w_to_mrsi_weighted,
    apply_transform,
    apply_transforms_multi,
    run_total_pipeline,
    compute_registration_metrics,
    compute_image_quality_metrics,
    prepare_tissue_fraction_maps,
    compute_tissue_concentration_metrics,
    compare_tissue_metric_states,
    print_tissue_metric_comparison)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
BIDS_DIR     = os.path.join(DATA_DIR, "bids")
OUTPUT_DIR   = os.path.join(DATA_DIR, "output")

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



SUM_RAS_NAME              = f"{SUBJ}_{SES}_acq-OrigRes_desc-AllMetabSumRAS_mrsi.nii.gz"
SUM_RAS_PATH              = os.path.join(OUTPUT_DIR, SUM_RAS_NAME)
WATER_RAS_NAME            = f"{SUBJ}_{SES}_desc-WaterSignalRAS_mrsi.nii.gz"
WATER_RAS_PATH            = os.path.join(OUTPUT_DIR, WATER_RAS_NAME)
T1W_BRAIN_IN_SUM_RAS_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-BrainT1wInSumRAS_T1w.nii.gz"
T1W_BRAIN_IN_SUM_RAS_PATH = os.path.join(OUTPUT_DIR, T1W_BRAIN_IN_SUM_RAS_NAME)
FINAL_REG_DIR             = os.path.join(OUTPUT_DIR, "final_reg")


save_reoriented_metabolite_sum(
    bids_dir=BIDS_DIR, ses=SES, out_dir=OUTPUT_DIR, overwrite=False, subjects=[SUBJ])
sum_ras_img = nib.load(SUM_RAS_PATH) 

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


# BET brain mask to  MRSI space (Reg-17 brain to sum forward transform)

BET_MASK_REG_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-BrainMaskReg_T1w.nii.gz"
BET_MASK_REG_PATH = os.path.join(OUTPUT_DIR, BET_MASK_REG_NAME)

# freesurfer brain mask to  MRSI space (Reg-15 brain to sum forward transform)

FREESURFER_BRAIN_SRC_NAME = f"{SUBJ}_{SES}_acq-UNIDEND_T1w_brain_synthstrip.nii"
FREESURFER_BRAIN_SRC_PATH = os.path.join(OUTPUT_DIR, FREESURFER_BRAIN_SRC_NAME)

FREESURFER_MASK_SRC_NAME = f"{SUBJ}_{SES}_acq-UNIDEND_T1w_brainmask_synthstrip.nii"
FREESURFER_MASK_SRC_PATH = os.path.join(OUTPUT_DIR, FREESURFER_MASK_SRC_NAME)

FREESURFER_MASK_REG_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-FreesurferMaskReg_T1w.nii.gz"
FREESURFER_MASK_REG_PATH = os.path.join(OUTPUT_DIR, FREESURFER_MASK_REG_NAME)

WATER_MASK_REG_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-WaterMaskReg_mrsi.nii.gz"
WATER_MASK_REG_PATH = os.path.join(OUTPUT_DIR, WATER_MASK_REG_NAME)

# Use SUM_RAS_PATH as reference: it is now computed above, and is the same space
# that Reg-17 registers into, so the masks are in the correct grid.
bet_mask_reg_img = apply_transform(
        in_path=T1W_BRAIN_MASK_PATH,
        ref_path=SUM_RAS_PATH,
        transform_path=brain_sum_transforms[0],
        out_path=BET_MASK_REG_PATH,
        overwrite=False)

freesurfer_mask_reg_img = apply_transform(
        in_path=FREESURFER_MASK_SRC_PATH,
        ref_path=SUM_RAS_PATH,
        transform_path=brain_sum_transforms[0],
        out_path=FREESURFER_MASK_REG_PATH,
        overwrite=False)
 
water_mask_reg_img = apply_transform(
        in_path=WATER_MASK_PATH,
        ref_path=SUM_RAS_PATH,
        transform_path=brain_sum_transforms[0],
        out_path=WATER_MASK_REG_PATH,
        overwrite=False)



# Reg-17a – Total pipeline with BET brain mask (moving mask)
# sum_ras_path is passed in so the pipeline skips recomputing AllMetabSumRAS.
(t1w_brain_in_sum_ras_img_bet_mask,
    brain_sum_ras_transforms_bet_mask,
    final_reg_imgs_bet_mask,
    metrics) = run_total_pipeline(
    mrs_dir=MRS_DIR,
    water_path=WATER_PATH,
    t1w_ds_brain_path=T1W_DS_BRAIN_PATH,
    t1w_ds_brain_img=t1w_ds_brain_img,
    subj=SUBJ,
    ses=SES,
    output_dir=OUTPUT_DIR,
    sum_ras_path=SUM_RAS_PATH,
    overwrite=False,
    t1w_brain_mask_ds_path=_brain_mask_arg,
    out_suffix="")

# Downsample the FreeSurfer atlas mask to the MRSI-DS voxel grid
# (required as ANTs moving mask: must match the moving image resolution)
FREESURFER_BRAIN_DS_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-BrainSynthstrip_T1w.nii.gz"
FREESURFER_BRAIN_DS_PATH = os.path.join(OUTPUT_DIR, FREESURFER_BRAIN_DS_NAME)
if not os.path.exists(FREESURFER_BRAIN_DS_PATH) and os.path.exists(FREESURFER_BRAIN_SRC_PATH):
    _fs_brain_img = nib.load(FREESURFER_BRAIN_SRC_PATH)
    _fs_brain_ds  = resample_from_to(_fs_brain_img, mrs_example, order=1)
    _fs_brain_ds  = nib.Nifti1Image(np.array(_fs_brain_ds.dataobj, dtype=np.float32),
                                    _fs_brain_ds.affine, _fs_brain_ds.header)
    nib.save(_fs_brain_ds, FREESURFER_BRAIN_DS_PATH)
    print(f"  [fs-brain-ds] saved {FREESURFER_BRAIN_DS_NAME}")
fs_ds_brain_img = nib.load(FREESURFER_BRAIN_DS_PATH) if os.path.exists(FREESURFER_BRAIN_DS_PATH) else t1w_ds_brain_img

FREESURFER_MASK_DS_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-FreesurferMaskDS_T1w.nii.gz"
FREESURFER_MASK_DS_PATH = os.path.join(OUTPUT_DIR, FREESURFER_MASK_DS_NAME)
if not os.path.exists(FREESURFER_MASK_DS_PATH) and os.path.exists(FREESURFER_MASK_SRC_PATH):
    _fs_img = nib.load(FREESURFER_MASK_SRC_PATH)
    _fs_ds  = resample_from_to(_fs_img, mrs_example, order=0)
    _fs_ds  = nib.Nifti1Image(np.array(_fs_ds.dataobj, dtype=np.uint8),
                              _fs_ds.affine, _fs_ds.header)
    nib.save(_fs_ds, FREESURFER_MASK_DS_PATH)
    print(f"  [fs-mask-ds] saved {FREESURFER_MASK_DS_NAME}")
_fs_mask_arg = FREESURFER_MASK_DS_PATH if os.path.exists(FREESURFER_MASK_DS_PATH) else None

# Reg-17b – Total pipeline with FreeSurfer atlas mask (moving mask)
# out_suffix="-FSMask" writes to a distinct filename so it does not
# collide with the BET variant above.
(t1w_brain_in_sum_ras_img_freesurfer_mask,
    brain_sum_ras_transforms_freesurfer_mask,
    final_reg_imgs_freesurfer_mask,
    metrics_freesurfer_mask) = run_total_pipeline(
    mrs_dir=MRS_DIR,
    water_path=WATER_PATH,
    t1w_ds_brain_path=FREESURFER_BRAIN_DS_PATH,
    t1w_ds_brain_img=fs_ds_brain_img,
    subj=SUBJ,
    ses=SES,
    output_dir=OUTPUT_DIR,
    sum_ras_path=SUM_RAS_PATH,
    overwrite=False,
    t1w_brain_mask_ds_path=_fs_mask_arg,
    out_suffix="-FSMask")

# Convenient aliases kept for backward compatibility with notebook imports
t1w_brain_in_sum_ras_img = t1w_brain_in_sum_ras_img_bet_mask
brain_sum_ras_transforms  = brain_sum_ras_transforms_bet_mask
final_reg_imgs            = final_reg_imgs_bet_mask

# Image quality metrics (FBER + EFC) for BET and FS pipelines
iqm_metrics               = compute_image_quality_metrics(t1w_ds_brain_img, final_reg_imgs_bet_mask)
iqm_metrics_freesurfer    = compute_image_quality_metrics(t1w_ds_brain_img, final_reg_imgs_freesurfer_mask)

# ── Before-registration baseline ──────────────────────────────────────────────
# Raw MRSI maps simply resampled (no transform) to the DS T1w grid.
# Used as a "before" baseline when plotting metrics.
_raw_mrsi_maps = sorted(
    f for f in os.listdir(MRS_DIR)
    if f.endswith(".nii.gz") and "acq-OrigRes" in f and "AllMetab" not in f
)
_raw_mrsi_dict = {}
for _fname in _raw_mrsi_maps:
    _label = metabolite_name(_fname)
    _raw_img = nib.load(os.path.join(MRS_DIR, _fname))
    _raw_ds  = resample_from_to(_raw_img, t1w_ds_brain_img, order=1)
    _raw_mrsi_dict[_label] = nib.Nifti1Image(
        np.array(_raw_ds.dataobj, dtype=np.float32),
        t1w_ds_brain_img.affine, t1w_ds_brain_img.header)
# Also include water signal (same label as used in run_total_pipeline)
_water_raw_ds = resample_from_to(nib.load(WATER_PATH), t1w_ds_brain_img, order=1)
_raw_mrsi_dict["WaterSignal"] = nib.Nifti1Image(
    np.array(_water_raw_ds.dataobj, dtype=np.float32),
    t1w_ds_brain_img.affine, t1w_ds_brain_img.header)
metrics_before = compute_registration_metrics(t1w_ds_brain_img, _raw_mrsi_dict)
iqm_metrics_before = compute_image_quality_metrics(t1w_ds_brain_img, _raw_mrsi_dict)

# ── Toolbox (article) pipeline metrics ────────────────────────────────────────
# Apply the toolbox SyN + affine forward transforms (MRSI → T1w) to every
# OrigRes metabolite and compute quality metrics on the same DS T1w grid.
_TOOLBOX_XFM_DIR  = os.path.join(DATA_DIR, "bids", "derivatives", "transforms",
                                  "ants", SUBJ, SES, "mrsi")
_TOOLBOX_SYN_PATH = os.path.join(_TOOLBOX_XFM_DIR,
                                  f"{SUBJ}_{SES}_desc-mrsi_to_t1w.syn.nii.gz")
_TOOLBOX_AFF_PATH = os.path.join(_TOOLBOX_XFM_DIR,
                                  f"{SUBJ}_{SES}_desc-mrsi_to_t1w.affine.mat")
_toolbox_mrsi_dict = {}
if os.path.exists(_TOOLBOX_SYN_PATH) and os.path.exists(_TOOLBOX_AFF_PATH):
    print("[toolbox] applying SyN+affine transforms to all metabolites...")
    for _fname in _raw_mrsi_maps:
        _label    = metabolite_name(_fname)
        _out_name = (f"{SUBJ}_{SES}_acq-MRSIres_desc-{_label}ToolboxSyN_T1w.nii.gz")
        _out_path = os.path.join(OUTPUT_DIR, "final_reg", _out_name)
        _img = apply_transforms_multi(
            in_path=os.path.join(MRS_DIR, _fname),
            ref_path=T1W_DS_BRAIN_PATH,
            transform_paths=[_TOOLBOX_SYN_PATH, _TOOLBOX_AFF_PATH],
            out_path=_out_path,
            overwrite=False,
        )
        if _img is not None:
            _toolbox_mrsi_dict[_label] = _img
    # Also apply toolbox transform to the water signal (using RAS-reoriented version if available)
    _water_ras_path = os.path.join(OUTPUT_DIR, f"{SUBJ}_{SES}_desc-WaterSignalRAS_mrsi.nii.gz")
    _water_src = _water_ras_path if os.path.exists(_water_ras_path) else WATER_PATH
    _water_tb_out = os.path.join(OUTPUT_DIR, "final_reg",
                                  f"{SUBJ}_{SES}_acq-MRSIres_desc-WaterSignalToolboxSyN_T1w.nii.gz")
    _water_tb_img = apply_transforms_multi(
        in_path=_water_src,
        ref_path=T1W_DS_BRAIN_PATH,
        transform_paths=[_TOOLBOX_SYN_PATH, _TOOLBOX_AFF_PATH],
        out_path=_water_tb_out,
        overwrite=False,
    )
    if _water_tb_img is not None:
        _toolbox_mrsi_dict["WaterSignal"] = _water_tb_img
    metrics_toolbox = compute_registration_metrics(t1w_ds_brain_img, _toolbox_mrsi_dict)
    iqm_metrics_toolbox = compute_image_quality_metrics(t1w_ds_brain_img, _toolbox_mrsi_dict)
else:
    print(f"  [toolbox] transforms not found at {_TOOLBOX_XFM_DIR}, skipping metrics_toolbox")
    metrics_toolbox = []
    iqm_metrics_toolbox = []

# ── Tissue-fraction metabolite comparison: before vs our pipeline vs article ──
def _is_tissue_metric_metabolite(label: str):
    upper = label.upper()
    if label == "WaterSignal":
        return True
    if "add" in label or upper.startswith("LIP") or upper.startswith("MM"):
        return False
    if upper.startswith("MINUS") or upper in {"EIB", "VOXELSNR", "VOXELFWHM", "VOXELFREQSHIFT"}:
        return False
    return True


TISSUE_METRIC_LABELS = sorted(
    {label for label in (list(_raw_mrsi_dict.keys()) + ["WaterSignal"])
    if _is_tissue_metric_metabolite(label)
    }
)

if all(os.path.exists(path) for path in [PVE_GM_PATH, PVE_WM_PATH, PVE_CSF_PATH]):
    TISSUE_FRACTION_MAPS_DS = prepare_tissue_fraction_maps(
        gm_img=nib.load(PVE_GM_PATH),
        wm_img=nib.load(PVE_WM_PATH),
        csf_img=nib.load(PVE_CSF_PATH),
        ref_img=t1w_ds_brain_img,
    )

    tissue_metrics_before = compute_tissue_concentration_metrics(
        tissue_maps=TISSUE_FRACTION_MAPS_DS,
        mrsi_imgs=_raw_mrsi_dict,
        labels=TISSUE_METRIC_LABELS,
        min_signal=0.0,
    )
    tissue_metrics_pipeline = compute_tissue_concentration_metrics(
        tissue_maps=TISSUE_FRACTION_MAPS_DS,
        mrsi_imgs=final_reg_imgs,
        labels=TISSUE_METRIC_LABELS,
        min_signal=0.0,
    )
    tissue_metrics_toolbox = compute_tissue_concentration_metrics(
        tissue_maps=TISSUE_FRACTION_MAPS_DS,
        mrsi_imgs=_toolbox_mrsi_dict,
        labels=TISSUE_METRIC_LABELS,
        min_signal=0.0,
    )
    tissue_metrics_comparison = compare_tissue_metric_states(
        metric_states={
            "Before": tissue_metrics_before,
            "Pipeline": tissue_metrics_pipeline,
            "Article": tissue_metrics_toolbox,
        },
        labels=TISSUE_METRIC_LABELS,
    )
    print_tissue_metric_comparison(tissue_metrics_comparison, subject=SUBJ)
else:
    print("  [tissue-metrics] FAST PVE maps not found, skipping tissue comparison")
    TISSUE_FRACTION_MAPS_DS = {}
    tissue_metrics_before = []
    tissue_metrics_pipeline = []
    tissue_metrics_toolbox = []
    tissue_metrics_comparison = []

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


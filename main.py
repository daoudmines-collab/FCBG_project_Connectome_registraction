import os
import numpy as np
import nibabel as nib
import bids_structure
import utils

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

# Pre-computed sum of all metabolite maps (one per subject, native MRS space)
SUM_NAME  = f"{SUBJ}_{SES}_acq-OrigRes_desc-AllMetabSum_mrsi.nii.gz"
SUM_PATH  = os.path.join(OUTPUT_DIR, SUM_NAME)
utils.save_metabolite_sum(BIDS_DIR, ses=SES, overwrite=True, out_dir=OUTPUT_DIR)
sum_img   = nib.load(SUM_PATH) if os.path.exists(SUM_PATH) else None

# Downsampled T1w resampled to the MRSI voxel grid
T1W_DS_NAME = f"{SUBJ}_{SES}_acq-MRSIres_T1w.nii.gz"
T1W_DS_PATH = os.path.join(OUTPUT_DIR, T1W_DS_NAME)
utils.downsample_t1w_to_mrs(BIDS_DIR, ses=SES, overwrite=False, out_dir=OUTPUT_DIR)
t1w_ds_img  = nib.load(T1W_DS_PATH) if os.path.exists(T1W_DS_PATH) else None

# Fill holes in a water-derived mask, then apply that mask to water in main.
WATER_NAME = f"{SUBJ}_{SES}_desc-WaterSignal_mrsi.nii.gz"
WATER_PATH = os.path.join(MRS_DIR, WATER_NAME)

MASK_NAME = f"{SUBJ}_{SES}_desc-WaterMask_mrsi.nii.gz"
MASK_PATH = os.path.join(OUTPUT_DIR, MASK_NAME)
WATER_MASKED_NAME = f"{SUBJ}_{SES}_desc-WaterSignalMasked_mrsi.nii.gz"
WATER_MASKED_PATH = os.path.join(OUTPUT_DIR, WATER_MASKED_NAME)

water_img = nib.load(WATER_PATH)
mask_img = utils.fill_mask_holes(
    water_img_path=WATER_PATH,
    threshold=0.0,
    out_mask_path=MASK_PATH,
    overwrite=False,
)

# Boolean numpy mask ready to reuse in analyses
mask = mask_img.get_fdata().astype(bool)

water_masked_data = np.where(mask, water_img.get_fdata(), 0.0).astype(np.float32)
water_masked_img = nib.Nifti1Image(water_masked_data, water_img.affine, water_img.header)
water_masked_img.set_data_dtype(np.float32)
if not os.path.exists(WATER_MASKED_PATH):
    nib.save(water_masked_img, WATER_MASKED_PATH)


snr_records = utils.rank_metabolites_by_snr(MRS_DIR, subj=SUBJ, ses=SES)
best_mrsi_name = snr_records[0]["filename"]
best_mrsi_img  = nib.load(os.path.join(MRS_DIR, best_mrsi_name))
best_mrsi_label = snr_records[0]["metabolite"]

# Registration 1: best-SNR MRSI  downsampled T1w (rigid, MRSI-res space)
MRSI_REG_DS_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-Registered_mrsi.nii.gz"
MRSI_REG_DS_PATH = os.path.join(OUTPUT_DIR, MRSI_REG_DS_NAME)
mrsi_reg_ds_img, reg1_transforms = utils.register_mrsi_to_t1w(
    mrsi_img=best_mrsi_img,
    t1w_img=t1w_ds_img,
    mask=mask,
    out_path=MRSI_REG_DS_PATH,
    overwrite=True,
)

# Registration 2: best-SNR MRSI →full-resolution T1w (rigid, full T1w space)
MRSI_REG_FULLRES_NAME = f"{SUBJ}_{SES}_acq-FullRes_desc-Registered_mrsi.nii.gz"
MRSI_REG_FULLRES_PATH = os.path.join(OUTPUT_DIR, MRSI_REG_FULLRES_NAME)
mrsi_reg_fullres_img, _ = utils.register_mrsi_to_t1w(
    mrsi_img=best_mrsi_img,
    t1w_img=t1w_img,
    mask=mask,
    out_path=MRSI_REG_FULLRES_PATH,
    overwrite=True,
)

# Registration 3: independently register the metabolite sum to DS T1w.
MRSI_SUM_REG_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-RegisteredSum_mrsi.nii.gz"
MRSI_SUM_REG_PATH = os.path.join(OUTPUT_DIR, MRSI_SUM_REG_NAME)
if sum_img is not None:
    mrsi_sum_reg_img, reg3_transforms = utils.register_mrsi_to_t1w(
        mrsi_img=sum_img,
        t1w_img=t1w_ds_img,
        mask=mask,
        out_path=MRSI_SUM_REG_PATH,
        overwrite=True,
    )
else:
    mrsi_sum_reg_img, reg3_transforms = None, None

# Registration 4: apply Registration 1's best-SNR transform to the sum
# (no re-optimisation)  used only for comparison with Registration 3.
MRSI_SUM_XFM_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-SumViaBestSNRXfm_mrsi.nii.gz"
MRSI_SUM_XFM_PATH = os.path.join(OUTPUT_DIR, MRSI_SUM_XFM_NAME)
if sum_img is not None and reg1_transforms is not None:
    mrsi_sum_xfm_img, _ = utils.register_mrsi_to_t1w(
        mrsi_img=sum_img,
        t1w_img=t1w_ds_img,
        mask=mask,
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
    mrsi_gly_via_sum_img, _ = utils.register_mrsi_to_t1w(
        mrsi_img=best_mrsi_img,
        t1w_img=t1w_ds_img,
        mask=mask,
        out_path=MRSI_GLY_VIA_SUM_PATH,
        overwrite=True,
        init_transforms=reg3_transforms,
    )
else:
    mrsi_gly_via_sum_img = None

# Best-coverage metabolite: the map with the most voxels containing signal.
# This tends to be a metabolite with a strong, spatially broad distribution
# (often Ins) and gives ANTs the most overlap area to optimise against.
coverage_records = sorted(snr_records, key=lambda r: r["n_voxels"], reverse=True)
best_cov_record = coverage_records[0]
best_cov_label  = best_cov_record["metabolite"]
best_cov_img    = nib.load(os.path.join(MRS_DIR, best_cov_record["filename"]))
print(f"Best-coverage metabolite: {best_cov_label}  ({best_cov_record['n_voxels']} voxels)")

# Registration 6: best-coverage metabolite independently registered to DS T1w.
MRSI_COV_REG_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-BestCoverageReg_mrsi.nii.gz"
MRSI_COV_REG_PATH = os.path.join(OUTPUT_DIR, MRSI_COV_REG_NAME)
mrsi_cov_reg_img, reg6_transforms = utils.register_mrsi_to_t1w(
    mrsi_img=best_cov_img,
    t1w_img=t1w_ds_img,
    mask=mask,
    out_path=MRSI_COV_REG_PATH,
    overwrite=True,
)


import os
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
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

# Inverse registrations: T1w DS to MRSI space (MRSI fixed, T1w moving) 

# Registration 7: T1w DS to  MRSI sum map 
T1W_IN_SUM_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-T1wInSum_T1w.nii.gz"
T1W_IN_SUM_PATH = os.path.join(OUTPUT_DIR, T1W_IN_SUM_NAME)
if sum_img is not None:
    t1w_in_sum_img, inv_sum_transforms = utils.register_t1w_to_mrsi(
        t1w_img=t1w_ds_img,
        mrsi_img=sum_img,
        out_path=T1W_IN_SUM_PATH,
        overwrite=False,
    )
else:
    t1w_in_sum_img, inv_sum_transforms = None, None

# Registration 8: T1w DS to MRSI best-SNR (Gly) map 
T1W_IN_GLY_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-T1wInGly_T1w.nii.gz"
T1W_IN_GLY_PATH = os.path.join(OUTPUT_DIR, T1W_IN_GLY_NAME)
t1w_in_gly_img, inv_gly_transforms = utils.register_t1w_to_mrsi(
    t1w_img=t1w_ds_img,
    mrsi_img=best_mrsi_img,
    out_path=T1W_IN_GLY_PATH,
    overwrite=False,
)

# Registration 9: T1w DS to Gly space using the sum transform (reuse Reg 7)
T1W_VIA_SUM_IN_GLY_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-T1wViaSumInGly_T1w.nii.gz"
T1W_VIA_SUM_IN_GLY_PATH = os.path.join(OUTPUT_DIR, T1W_VIA_SUM_IN_GLY_NAME)
if inv_sum_transforms is not None:
    t1w_via_sum_in_gly_img, _ = utils.register_t1w_to_mrsi(
        t1w_img=t1w_ds_img,
        mrsi_img=best_mrsi_img,
        out_path=T1W_VIA_SUM_IN_GLY_PATH,
        overwrite=False,
        init_transforms=inv_sum_transforms,
    )
else:
    t1w_via_sum_in_gly_img = None

#  Weighted inverse registration paths (registrations called in results.ipynb)
BEST_MRSI_PATH = os.path.join(MRS_DIR, best_mrsi_name)

# Registration 10 paths: T1w DS to best-SNR MRSI, water-weighted
T1W_IN_GLY_W_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-T1wInGlyWeighted_T1w.nii.gz"
T1W_IN_GLY_W_PATH = os.path.join(OUTPUT_DIR, T1W_IN_GLY_W_NAME)
T1W_IN_GLY_W_XFM  = T1W_IN_GLY_W_PATH.replace(".nii.gz", "_fwdtransform.mat")
t1w_in_gly_w_img, inv_gly_w_transforms = utils.register_t1w_to_mrsi_weighted(
    fixed_path=BEST_MRSI_PATH,
    moving_path=T1W_DS_PATH,
    mask_path=WATER_PATH,
    out_path=T1W_IN_GLY_W_PATH,
    transform_path=T1W_IN_GLY_W_XFM,
    overwrite=False,
)

# Registration 11 paths: T1w DS to sum MRSI, water-weighted
T1W_IN_SUM_W_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-T1wInSumWeighted_T1w.nii.gz"
T1W_IN_SUM_W_PATH = os.path.join(OUTPUT_DIR, T1W_IN_SUM_W_NAME)
T1W_IN_SUM_W_XFM  = T1W_IN_SUM_W_PATH.replace(".nii.gz", "_fwdtransform.mat")
if sum_img is not None:
    t1w_in_sum_w_img, inv_sum_w_transforms = utils.register_t1w_to_mrsi_weighted(
        fixed_path=SUM_PATH,
        moving_path=T1W_DS_PATH,
        mask_path=WATER_PATH,
        out_path=T1W_IN_SUM_W_PATH,
        transform_path=T1W_IN_SUM_W_XFM,
        overwrite=False,
    )
else:
    t1w_in_sum_w_img, inv_sum_w_transforms = None, None

# Registration 12: T1w DS → Gly via sum's water-weighted transform (reuse Reg 11)
T1W_VIA_WSUM_IN_GLY_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-T1wViaWSumInGly_T1w.nii.gz"
T1W_VIA_WSUM_IN_GLY_PATH = os.path.join(OUTPUT_DIR, T1W_VIA_WSUM_IN_GLY_NAME)
T1W_VIA_WSUM_IN_GLY_XFM  = T1W_VIA_WSUM_IN_GLY_PATH.replace(".nii.gz", "_fwdtransform.mat")
if inv_sum_w_transforms is not None:
    t1w_via_wsum_in_gly_img, _ = utils.register_t1w_to_mrsi_weighted(
        fixed_path=BEST_MRSI_PATH,
        moving_path=T1W_DS_PATH,
        mask_path=WATER_PATH,
        out_path=T1W_VIA_WSUM_IN_GLY_PATH,
        transform_path=T1W_VIA_WSUM_IN_GLY_XFM,
        overwrite=False,
        init_transforms=inv_sum_w_transforms,
    )
else:
    t1w_via_wsum_in_gly_img = None

# ── Skull-stripping + skull-stripped inverse registrations ──────────────────

# Skull-strip the downsampled T1w with FSL BET
T1W_DS_BRAIN_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-Brain_T1w.nii.gz"
T1W_DS_BRAIN_PATH = os.path.join(OUTPUT_DIR, T1W_DS_BRAIN_NAME)
t1w_ds_brain_img = utils.skull_strip_t1w(
    in_path=T1W_DS_PATH,
    out_path=T1W_DS_BRAIN_PATH,
    frac=0.3,
    overwrite=False,
)

# Registration 14: skull-stripped T1w DS → best-SNR MRSI (Gly), water-weighted
T1W_BRAIN_IN_GLY_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-BrainT1wInGly_T1w.nii.gz"
T1W_BRAIN_IN_GLY_PATH = os.path.join(OUTPUT_DIR, T1W_BRAIN_IN_GLY_NAME)
T1W_BRAIN_IN_GLY_XFM  = T1W_BRAIN_IN_GLY_PATH.replace(".nii.gz", "_fwdtransform.mat")
t1w_brain_in_gly_img, brain_gly_transforms = utils.register_t1w_to_mrsi_weighted(
    fixed_path=BEST_MRSI_PATH,
    moving_path=T1W_DS_BRAIN_PATH,
    mask_path=WATER_PATH,
    out_path=T1W_BRAIN_IN_GLY_PATH,
    transform_path=T1W_BRAIN_IN_GLY_XFM,
    overwrite=False,
)

# Registration 15: skull-stripped T1w DS → sum MRSI, water-weighted
T1W_BRAIN_IN_SUM_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-BrainT1wInSum_T1w.nii.gz"
T1W_BRAIN_IN_SUM_PATH = os.path.join(OUTPUT_DIR, T1W_BRAIN_IN_SUM_NAME)
T1W_BRAIN_IN_SUM_XFM  = T1W_BRAIN_IN_SUM_PATH.replace(".nii.gz", "_fwdtransform.mat")
if sum_img is not None:
    t1w_brain_in_sum_img, brain_sum_transforms = utils.register_t1w_to_mrsi_weighted(
        fixed_path=SUM_PATH,
        moving_path=T1W_DS_BRAIN_PATH,
        mask_path=WATER_PATH,
        out_path=T1W_BRAIN_IN_SUM_PATH,
        transform_path=T1W_BRAIN_IN_SUM_XFM,
        overwrite=False,
    )
else:
    t1w_brain_in_sum_img, brain_sum_transforms = None, None

# Registration 16: skull-stripped T1w in Gly space via sum transform (reuse Reg 15)
T1W_BRAIN_VIA_SUM_IN_GLY_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-BrainT1wViaSumInGly_T1w.nii.gz"
T1W_BRAIN_VIA_SUM_IN_GLY_PATH = os.path.join(OUTPUT_DIR, T1W_BRAIN_VIA_SUM_IN_GLY_NAME)
T1W_BRAIN_VIA_SUM_IN_GLY_XFM  = T1W_BRAIN_VIA_SUM_IN_GLY_PATH.replace(".nii.gz", "_fwdtransform.mat")
if brain_sum_transforms is not None:
    t1w_brain_via_sum_in_gly_img, _ = utils.register_t1w_to_mrsi_weighted(
        fixed_path=BEST_MRSI_PATH,
        moving_path=T1W_DS_BRAIN_PATH,
        mask_path=WATER_PATH,
        out_path=T1W_BRAIN_VIA_SUM_IN_GLY_PATH,
        transform_path=T1W_BRAIN_VIA_SUM_IN_GLY_XFM,
        overwrite=False,
        init_transforms=brain_sum_transforms,
    )
else:
    t1w_brain_via_sum_in_gly_img = None

# Registration 13: water signal → DS T1w (reuse Reg 1 transform, no re-optimisation)
WATER_REG_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-WaterRegistered_T1w.nii.gz"
WATER_REG_PATH = os.path.join(OUTPUT_DIR, WATER_REG_NAME)
water_reg_img, _ = utils.register_mrsi_to_t1w(
    mrsi_img=water_img,
    t1w_img=t1w_ds_img,
    mask=mask,
    out_path=WATER_REG_PATH,
    overwrite=False,
    init_transforms=reg1_transforms,
)

# ── Total pipeline ────────────────────────────────────────────────────────────
# Step 1 – Reorient all MRSI maps to RAS canonical and sum them.
SUM_RAS_NAME = f"{SUBJ}_{SES}_acq-OrigRes_desc-AllMetabSumRAS_mrsi.nii.gz"
SUM_RAS_PATH = os.path.join(OUTPUT_DIR, SUM_RAS_NAME)
utils.save_reoriented_metabolite_sum(BIDS_DIR, ses=SES, out_dir=OUTPUT_DIR, overwrite=False)
sum_ras_img = nib.load(SUM_RAS_PATH) if os.path.exists(SUM_RAS_PATH) else None

# Step 2 – Reorient water signal to the same RAS canonical space (used as
#          the fixed-image weight in antsRegistration --masks).
WATER_RAS_NAME = f"{SUBJ}_{SES}_desc-WaterSignalRAS_mrsi.nii.gz"
WATER_RAS_PATH = os.path.join(OUTPUT_DIR, WATER_RAS_NAME)
if not os.path.exists(WATER_RAS_PATH):
    water_ras_img = nib.as_closest_canonical(nib.load(WATER_PATH))
    nib.save(water_ras_img, WATER_RAS_PATH)
    print(f"  [ras] saved reoriented water: {WATER_RAS_NAME}")

# Step 3 – Registration 17: skull-stripped DS T1w → reoriented sum,
#          guided by the reoriented water signal as a fixed-image weight.
T1W_BRAIN_IN_SUM_RAS_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-BrainT1wInSumRAS_T1w.nii.gz"
T1W_BRAIN_IN_SUM_RAS_PATH = os.path.join(OUTPUT_DIR, T1W_BRAIN_IN_SUM_RAS_NAME)
T1W_BRAIN_IN_SUM_RAS_XFM  = T1W_BRAIN_IN_SUM_RAS_PATH.replace(".nii.gz", "_fwdtransform.mat")
if sum_ras_img is not None:
    t1w_brain_in_sum_ras_img, brain_sum_ras_transforms = utils.register_t1w_to_mrsi_weighted(
        fixed_path=SUM_RAS_PATH,
        moving_path=T1W_DS_BRAIN_PATH,
        mask_path=WATER_RAS_PATH,
        out_path=T1W_BRAIN_IN_SUM_RAS_PATH,
        transform_path=T1W_BRAIN_IN_SUM_RAS_XFM,
        overwrite=False,
    )
else:
    t1w_brain_in_sum_ras_img, brain_sum_ras_transforms = None, None

# Step 4 – Apply the Reg-17 transform (inverted) to every individual metabolite,
#          warping each reoriented MRSI map into T1w (brain DS) space.
FINAL_REG_DIR = os.path.join(OUTPUT_DIR, "final_reg")
if brain_sum_ras_transforms is not None:
    final_reg_imgs = utils.apply_transform_all_metabolites(
        mrs_dir=MRS_DIR,
        transform_path=brain_sum_ras_transforms[0],
        t1w_ref_path=T1W_DS_BRAIN_PATH,
        out_dir=FINAL_REG_DIR,
        subj=SUBJ,
        ses=SES,
        overwrite=False,
    )
else:
    final_reg_imgs = {}
    print("  [pipeline] Step 4 skipped – Reg-17 transform not available.")

# Step 5 – Compute per-metabolite registration quality metrics in T1w space.
if final_reg_imgs:
    metrics = utils.compute_registration_metrics(t1w_ds_brain_img, final_reg_imgs)
    print("\nRegistration quality metrics (Reg-17 total pipeline):")
    for r in metrics:
        print(f"  {r['label']:<28}  coverage={r['coverage']:.3f}"
              f"  NCC={r['ncc']:+.4f}  NMI={r['nmi']:.4f}")
else:
    metrics = []

# ── Registration 18 ───────────────────────────────────────────────────────────
# Three-way comparison in water-signal coordinate space:
#   a) Full-res T1w naively resampled to water grid (no registration)
#   b) T1w DS warped to water space by reusing the Reg-11 sum water-weighted transform
#   c) T1w DS directly registered to water signal (ANTs CLI rigid + water mask)

# (a) Naive resample – full-res T1w directly onto water grid (no intermediate DS)
t1w_in_water_naive = resample_from_to(nib.load(T1W_PATH), water_img, order=1)

# (b) Reg-11 sum transform reused – apply inv_sum_w_transforms to T1w DS,
#     resampled onto the water grid (no re-optimisation)
T1W_IN_WATER_VIA17_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-T1wInWaterViaReg11_T1w.nii.gz"
T1W_IN_WATER_VIA17_PATH = os.path.join(OUTPUT_DIR, T1W_IN_WATER_VIA17_NAME)
T1W_IN_WATER_VIA17_XFM  = T1W_IN_WATER_VIA17_PATH.replace(".nii.gz", "_fwdtransform.mat")
if inv_sum_w_transforms is not None:
    t1w_in_water_via17_img, _ = utils.register_t1w_to_mrsi_weighted(
        fixed_path=WATER_PATH,
        moving_path=T1W_DS_PATH,
        mask_path=WATER_MASKED_PATH,
        out_path=T1W_IN_WATER_VIA17_PATH,
        transform_path=T1W_IN_WATER_VIA17_XFM,
        overwrite=False,
        init_transforms=inv_sum_w_transforms,
    )
else:
    t1w_in_water_via17_img = None
    print("  [reg18b] Reg-11 transforms not available – skipping.")

# (c) Reg-18: DS T1w → water signal map (ANTs CLI rigid, water-filled mask)
T1W_IN_WATER_NAME = f"{SUBJ}_{SES}_acq-MRSIres_desc-T1wInWaterReg18_T1w.nii.gz"
T1W_IN_WATER_PATH = os.path.join(OUTPUT_DIR, T1W_IN_WATER_NAME)
T1W_IN_WATER_XFM  = T1W_IN_WATER_PATH.replace(".nii.gz", "_fwdtransform.mat")
t1w_in_water_img, t1w_water_transforms = utils.register_t1w_to_mrsi_weighted(
    fixed_path=WATER_PATH,
    moving_path=T1W_DS_PATH,
    mask_path=WATER_MASKED_PATH,
    out_path=T1W_IN_WATER_PATH,
    transform_path=T1W_IN_WATER_XFM,
    overwrite=False,
)


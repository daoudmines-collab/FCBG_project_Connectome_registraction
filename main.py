import os
import nibabel as nib
import bids_structure
import utils

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
BIDS_DIR     = os.path.join(DATA_DIR, "bids")

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
SUM_PATH  = os.path.join(MRS_DIR, SUM_NAME)
utils.save_metabolite_sum(BIDS_DIR, ses=SES, overwrite=True)
sum_img   = nib.load(SUM_PATH) if os.path.exists(SUM_PATH) else None

# Downsampled T1w resampled to the MRSI voxel grid and saved into anat/
utils.downsample_t1w_to_mrs(BIDS_DIR, ses=SES, overwrite=False)
T1W_DS_NAME = f"{SUBJ}_{SES}_acq-MRSIres_T1w.nii.gz"
T1W_DS_PATH = os.path.join(ANAT_DIR, T1W_DS_NAME)
t1w_ds_img  = nib.load(T1W_DS_PATH) if os.path.exists(T1W_DS_PATH) else None


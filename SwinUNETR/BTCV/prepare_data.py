import datetime
import argparse
import pydicom, os, pdb, json, random
from pathlib import Path
import numpy as np
import nibabel as nib
from rt_utils import RTStructBuilder
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Convert dcm files to NII files")
parser.add_argument( "--pid_path", default="/mnt/disk16T/datasets/houZhuang_Wangqx/", type=str, help="folder contains Patient DCM files")
parser.add_argument( "--save_path", default="/mnt/disk16T/datasets/houZhuang_Wangqx_nii/", type=str, help="folder to save NII files")

# Set organ ID dictionary
organ_id_dict = {'bladder':1, 'colon':2, 'rectum':3, 'small intestine':4, 'hrctv':5, 'hrctc':5, 'hr':5, 'hr-ctv':5, 'hrctv1':5}

def DCM_to_NII(pid_path, save_path):
    # Get all patient IDs
    pids_paths = [pid for pid in pid_path.iterdir() if pid.is_dir()] 
    pids_paths = sorted(pids_paths)

    # Loop through all patients
    for pid_path in tqdm(pids_paths):
        pid = pid_path.stem

        # Skip if nii file already exists
        if (save_path/'imgs'/f"{pid}.nii.gz").exists() and (save_path/'masks'/f"{pid}.nii.gz").exists():
            # print(f"Skipping {pid} as nii file already exists")
            continue

        print(pid)

        # Skip if pid_path does not contain CT or RTSTRUCT files
        if len(list(pid_path.glob('*.dcm'))) == 0:
            print(f"Skipping {pid} as pid_path does not contain CT or RTSTRUCT files")
            continue

        # Read all CT DICOM files and sort by slice location
        ct_files = [pydicom.dcmread(f) for f in pid_path.glob('*.dcm') if 'CT' in str(f)]
        for ds in ct_files:
            assert hasattr(ds, 'SliceLocation'), ds.filename
        ct_files.sort(key=lambda x: float(x.SliceLocation))

        # Stack pixel arrays to form 3D CT array
        ct_array = np.stack([f.pixel_array for f in ct_files], axis=-1)
        ct_array = ct_array.astype(np.int16) # adjust data type
        ct_array = np.transpose(ct_array, axes=[1,0,2]) # adjust orientation

        # Create nifti image object from 3D CT array and metadata
        affine = np.eye(4) # identity matrix for affine transformation
        affine[0,0] = ct_files[0].PixelSpacing[0] # x voxel size
        affine[1,1] = ct_files[0].PixelSpacing[1] # y voxel size
        affine[2,2] = ct_files[0].SliceThickness # z voxel size
        ct_nii = nib.Nifti1Image(ct_array, affine)

        # Load rtstruct.dcm file and extract 3D mask array for each ROI
        rt_fn_list = list(pid_path.glob('RS*.dcm'))
        if len(rt_fn_list) != 1:
            print(f'No or Multiple RS file found for {pid}')
            continue

        # get all ROI names
        rtstruct = RTStructBuilder.create_from(pid_path, rt_fn_list[0])
        roi_names = rtstruct.get_roi_names()
        print(roi_names)

        # Skip if HRCTV, HR or HR-CTV is not in ROI names
        if not ('HRCTV' in roi_names or 'HR' in roi_names or 'HR-CTV' in roi_names or 'HRCTC' in roi_names or 'HRCTV1' in roi_names):
            print(f"Skipping {pid} as HRCTV, HRCTC, HR or HR-CTV is not in ROI names")
            continue
        
        # Skip if CT array shape is not the same as mask array shape
        if ct_array.shape != rtstruct.get_roi_mask_by_name(roi_names[0]).shape:
            print(f"Skipping {pid} as CT array shape is not the same as mask array shape {ct_array.shape} {rtstruct.get_roi_mask_by_name(roi_names[0]).shape}")
            continue

        # Create 3D mask array for all ROIs
        mask_array = np.zeros(ct_array.shape, dtype=np.int8)
        for roi_name in roi_names:
            # Skip if roi_name is type of list
            if isinstance(roi_name, pydicom.multival.MultiValue):
                continue
            try:
                # Skip if roi_name is not in organ_id_dict
                if roi_name.lower() not in organ_id_dict.keys():
                    continue
            except:
                print('error')
                pdb.set_trace()
                print(roi_name)

            # For each ROI, create nifti image object from 3D mask array and metadata
            mask = rtstruct.get_roi_mask_by_name(roi_name) # get 3D mask array for ROI
            mask = np.transpose(mask, axes=[1,0,2]) # adjust orientation
            mask_array[mask] = organ_id_dict[roi_name.lower()]

        # Whether all the values in organ_id_dict is in mask_array
        if not np.all(np.isin(list(organ_id_dict.values()), mask_array.astype(np.int8))):
            print('Not all organ IDs are in mask array')

        mask_nii = nib.Nifti1Image(mask_array.astype(np.int8), affine) # create nifti image object

        # Save nifti object to nii file
        nib.save(ct_nii, save_path/'imgs'/f"{pid}.nii.gz")
        nib.save(mask_nii, save_path/'masks'/f"{pid}.nii.gz")

    # Save organ ID dictionary to json file
    with open(save_path/'masks/organ_id.json', 'w') as f:
        json.dump(organ_id_dict, f, indent=4)

def training_json(save_path):
    # Define the root directory and the subdirectories for images and masks
    root_dir = save_path
    img_dir = root_dir/"imgs"
    mask_dir = root_dir/"masks"

    # Define the template json file
    template_json = {
        "description": "Tianjin Cancer Hospital Brachys Dataset",
        "labels": {
            "0": "background",
            "1": "bladder",
            "2": "colon",
            "3": "rectum",
            "4": "small intestine",
            "5": "hrctv"
        },
        "licence": "yt",
        "modality": {
            "0": "CT"
        },
        "name": "tianjinCancerHospitalBrachys",
        "numTest": 0, # fill this in later. 20% of total data.
        "numTraining": 0, # fill this in later. 80% of total data.
        "reference": "congliu",
        "release": "06/26/2023", # change this in later
        "tensorImageSize": "3D",
        # Initialize empty lists for test and training data
        # We will fill them later with the file names
        "validation": [],
        "training": []
    }

    # set the release date to today and current time
    dt = datetime.datetime.now().strftime("%Y-%m-%d-%H")
    template_json["release"] = dt

    # Get all the image files then shuffle them 
    img_files = list(img_dir.glob('*.nii.gz'))
    random.shuffle(img_files)

    # split them into test and training sets
    num_test = int(len(img_files) * 0.2)
    num_train = len(img_files) - num_test
    template_json["numTest"] = num_test
    template_json["numTraining"] = num_train

    for i in range(len(img_files)):
        img_file = Path('imgs')/img_files[i].name
        mask_file = Path('masks')/img_files[i].name

        img_file = str(img_file)
        mask_file = str(mask_file)
        
        # Check if we have reached the limit for test data or not
        if i < num_test:
            template_json["validation"].append({"image": img_file, "label": mask_file})
        else:
            template_json["training"].append({"image": img_file, "label": mask_file})

    with open(os.path.join(root_dir, f"dataset_{dt}.json"), "w") as f:
        json.dump(template_json, f, indent=4)

    print("Dataset json file generated successfully!")

if __name__ == '__main__':
    args = parser.parse_args()

    # Define directory path
    pid_path = Path(args.pid_path)
    save_path = Path(args.save_path)

    (save_path/'imgs').mkdir(exist_ok=True, parents=True)
    (save_path/'masks').mkdir(exist_ok=True, parents=True)

    DCM_to_NII(pid_path, save_path)
    training_json(save_path)
import os

import numpy as np
import pandas as pd
import nibabel as nib

save_path = '../T2-Weighted-MRI/train_slice'
os.makedirs(save_path, exist_ok=True)
os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
os.makedirs(os.path.join(save_path, 'masks'), exist_ok=True)

data_path = ['t2mri_validation_data.csv']

# idx = 0
for file in data_path:
    data = pd.read_csv(file)
    for i in range(len(data)):
        try:
            img_path = data.iloc[i, 0]
            label_path = data.iloc[i, 1]

            volume = nib.load(img_path)

            volume_data = volume.get_fdata()
            label_data = nib.load(label_path).get_fdata()

            for slice_number in range(volume_data.shape[-1]):
                slice_ = volume_data[:, :, slice_number, np.newaxis]
                label_ = label_data[:, :, slice_number, np.newaxis]

                nib.save(nib.Nifti1Image(slice_, volume.affine, volume.header),
                         os.path.join(save_path, "images",
                                      "case{:04d}".format(i) + "_slice{:02d}".format(slice_number) + ".nii.gz"))

                nib.save(nib.Nifti1Image(label_, volume.affine, volume.header),
                         os.path.join(save_path, "masks",
                                      "case{:04d}".format(i) + "_slice{:02d}_mask".format(slice_number) + ".nii.gz"))

                # idx += 1

        except Exception as error:
            # handle the exception
            print("An exception occurred:", type(error).__name__)  # An exception occurred: ZeroDivisionError

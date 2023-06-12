# [E2MIP Challenge](https://e2mip.github.io/), MICCAI 2023
## [Starting point for E2MIP challenge, task 3:](https://e2mip.org/overview/#task-3-segmentation-of-an-unknown-dataset-dataset-2)
## 2D Segmentation of an unknown dataset (fetal brain MRI)

This repository can be used as a starting point for the E2MIP challenge on the fetal brain MRI dataset with Monai and PyTorch.

This repository contains:
1. information about the submission for the task 3 of the challenge
2. the structure of the training and testing data folders that will be used for your submission
3. few examples of the data

### 1. Challenge submission:
**The following information is preliminary. Final submission information will be available soon.**

* Please provide your complete algorithm for training and predicting in a docker script.
  Further information about how the script should look like will be published here soon.
  * script takes as input the path to the "training_data" and "testing_data" folders
  * script outputs predicted segmentation in a newly created folder "testing_data_prediction". 
The predictions need to be filed in the folder in a certain folder structure (see 2. Data for Challenge)
* Besides the performance metric also the energy consumption during training and evaluation is being measured
  and both determine the Challenge ranking.
### 2. Data for Challenge, Task 3:
The folder structure of the training and testing data used for evaluating your code will look like the following:

***important: The data in "training_data" folder are 2D slices with different resolutions and 
The data in "testing_data" folder are 3D stacks with different resolutions**
***Your saved predictions in "testing_data_prediction" are also corresponding stack prediction of "testing_data" **

```bash
training_data
├── images
│   ├── img_0000.nii.gz
│   ├── img_0001.nii.gz
│   ├── img_0003.nii.gz
    ...
│   
├── masks
│   ├── mask_0000.nii.gz
│   ├── mask_0001.nii.gz
│   ├── mask_0003.nii.gz
    ...
```


```bash
testing_data
├── images
│   ├── img_0000.nii.gz
│   ├── img_0001.nii.gz
│   ├── img_0003.nii.gz
    ...
│   
├── masks
│   ├── mask_0000.nii.gz
│   ├── mask_0001.nii.gz
│   ├── mask_0003.nii.gz
    ...
```

The folder structure of the segmentation predictions that your script should create from  "testing_data" should have the following structure:
```bash
testing_data_prediction
├── masks
│   ├── img_0000_mask.nii.gz
│   ├── img_0001_mask.nii.gz
│   ├── img_0003_mask.nii.gz
    ...
```

#### 3. Sample Data
Please see the "data" folder.


For further questions about this code, please contact razieh.faghihpirayesh@childrens.harvard.edu
with subject "E2MIP Challenge, task 3"

# DBT-Classification-and-2D-Synthesis
## 

The DBT Framework proposed aims to build and classify a synthetic 2D tomosynthesis 
image given the original 3D input data of a DBT tomosynthesis image.

----------------------------------------------------------------------
## Reference

**Paper**: 

    @inproceedings{
    }

----------------------------------------------------------------------
## Launchable scripts and their purpose
**Recommendations**:

This framework requires a proper configuration YAML file for each runnable script (examples given per script for each purpose).
***


**experiment_attnnet.py**: Train and evaluate 3D model, 
this has nothing to do explicitly with the proposed method, 
yet required to build and/or evaluate the source 3D model used for the synthetic image generation. 
(for training or inferation purposes). 
To use whether for training or inference, depends on how setting the number of epochs (Setting anything higher then 0 will prompt training, otherwise evaluation and therefore inference on the defined test set).
[Configuration file example](configs/config.yaml) 

**experiment_2Dsyn.py**: Train and evaluate 2D model, 
this has nothing to do explicitly with the proposed method,
yet required to build and/or evaluate the source 2D model 
(Training and evalutation on the synthtic 2D image produced by the 3D generation model). 
To use whether for training or inference, depends on how setting the number of epochs (Setting anything higher then 0 will prompt training, otherwise evaluation and therefore inference on the defined test set).
[Configuration file example](configs/config_2D.yaml)

**saliency.py**: Build 2D synthetic images given a 3D model. (This file was built specifically for Optimam dataset).
[Configuration file example](configs/config_saliency.yaml) 

**saliency_BCS.py**: Build 2D synthtic images given a 2D model. (This file was built specifically for BCS-DBT dataset).
[Configuration file example](configs/config_saliency_BCS.yaml) 

**saliency_metrics.py**: Compute metrics such as SSIM and LPIPS, this is meant for samples with ground truth bounding boxes (as demonstrated in BCS-DBT case), without proper bounding boxes as ground truth, this script won't work.
[Configuration file example](configs/config_saliency_BCS_metrics.yaml)

Building its own custom dataset may require a custom class.
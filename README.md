# CT Lung Segmentation

This repository contains the codes for lung and airway segmentation from CT scans, used in "Discriminative Localization in CNNs for Weakly-Supervised Segmentation of Pulmonary Nodules", [MICCAI'17](https://arxiv.org/abs/1707.01086).

**Demo code**
```ct_lung_segment.ipynb```
**Demo input**: <br />
``sample_ct.nii.gz`` <br />
**Demo outputs**: <br />
``sample_lungaw.nii.gz`` (segmentation of lung and airway) <br />
``sample_lung.nii.gz``   (segmentation of lung) <br />
``sample_aw.nii.gz``     (segmentation of airway)

***Note***: the sample image is downsampled from a CT scans in the [LIDC-IDRI](http://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX) dataset.

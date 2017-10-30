import numpy as np
import nibabel as nib
from utils import *
from segment_lung import segment_lung
from segment_airway import segment_airway

params = define_parameter()

#####################################################
# Load image 
#####################################################

I         = nib.load('./data/sample_ct.nii.gz')
I_affine  = I.affine
I         = I.get_data()

#####################################################
# Coarse segmentation of lung & airway 
#####################################################

Mlung = segment_lung(params, I, I_affine)

#####################################################
# Romove airway from lung mask 
#####################################################

Mlung, Maw = segment_airway(params, I, I_affine, Mlung)

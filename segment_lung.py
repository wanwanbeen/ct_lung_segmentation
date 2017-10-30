import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage import measure
from scipy import stats

def segment_lung(params, I, I_affine):
	
	#####################################################
	# Intensity thresholding & Morphological operations
	#####################################################

	M = np.zeros(I.shape)
	M [I > params['lungMinValue']] = 1
	M [I > params['lungMaxValue']] = 0

	struct_s = ndimage.generate_binary_structure(3, 1)
	struct_m = ndimage.iterate_structure(struct_s, 2)
	M = ndimage.binary_closing(M, structure = struct_s, iterations = 1)
	M = ndimage.binary_opening(M, structure = struct_m, iterations = 1)

	#####################################################
	# Estimate lung filed of view
	#####################################################

	[m, n, p] = I.shape;
	medx      = int(m/2)
	medy      = int(n/2)
	xrange1   = int(m/2*params['xRangeRatio1'])
	xrange2   = int(m/2*params['xRangeRatio2'])
	zrange1   = int(p*params['zRangeRatio1'])
	zrange2   = int(p*params['zRangeRatio2'])

	#####################################################
	# Select largest connected components & save nii
	#####################################################

	M = measure.label(M)
	label1 = M[medx - xrange2 : medx - xrange1, medy, zrange1 : zrange2]
	label2 = M[medx + xrange1 : medx + xrange2, medy, zrange1 : zrange2]
	label1 = stats.mode(label1[label1 > 0])[0][0]
	label2 = stats.mode(label2[label2 > 0])[0][0]
	M[M == label1] = -1
	M[M == label2] = -1
	M[M > 0] = 0
	M = M*-1

	M     = ndimage.binary_closing(M, structure = struct_m, iterations = 1)
	M     = ndimage.binary_fill_holes(M)
	Mlung = np.int8(M)
	nib.Nifti1Image(Mlung,I_affine).to_filename('./result/sample_lungaw.nii.gz')
	
	return Mlung










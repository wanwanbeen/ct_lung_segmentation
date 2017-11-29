import numpy as np
from scipy.spatial import distance
from scipy import ndimage
from skimage import measure

#####################################################
# Pre-define parameters
#####################################################

def define_parameter():
    
    params = {}
    
    #----------------------------------------------------
    # Parameters for intensity (fixed)
    #----------------------------------------------------
    
    params['lungMinValue']      = -1024
    params['lungMaxValue']      = -400
    params['lungThreshold']     = -900
    
    #----------------------------------------------------
    # Parameters for lung segmentation (fixed)
    #----------------------------------------------------
    
    params['xRangeRatio1']      = 0.4
    params['xRangeRatio2']      = 0.75
    params['zRangeRatio1']      = 0.5
    params['zRangeRatio2']      = 0.75
    
    #----------------------------------------------------
    # Parameters for airway segmentation
    # NEED TO ADAPT for image resolution and orientation
    # [current values work for demo image with resolution = 1mm^3]
    #----------------------------------------------------
    params['airwayRadiusMask']  = 15  # increase the value if you have high resolution image
    params['airwayRadiusX']     = 8   # ditto
    params['airwayRadiusZ']     = 15  # ditto
    params['super2infer']       = 0   # value = 1 if slice no. increases from superior to inferior, else value = 0
    
    return params

#####################################################
# Generate binary structure to mimic trachea
#####################################################

def generate_structure_trachea(Radius, RadiusX, RadiusZ):
    
    struct_trachea = np.zeros([2*Radius+1,2*Radius+1,RadiusZ])
    for i in range(0,2*Radius+1):
        for j in range(0,2*Radius+1):
            if distance.euclidean([Radius+1,Radius+1],[i,j]) < RadiusX:
                struct_trachea[i,j,:] = 1
            else:
                struct_trachea[i,j,:] = 0
    
    return struct_trachea

#####################################################
# Generate bounding box
#####################################################

def bbox2_3D(img,label,margin,limit):
    
    imgtmp = np.zeros(img.shape)
    imgtmp[img == label] = 1
    
    x = np.any(imgtmp, axis=(1, 2))
    y = np.any(imgtmp, axis=(0, 2))
    z = np.any(imgtmp, axis=(0, 1))

    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    xmin = xmin - margin - 1
    xmin = max(0,xmin)
    ymin = ymin - margin - 1
    ymin = max(0,ymin)
    zmin = zmin - margin - 1
    zmin = max(0,zmin)        
    xmax = xmax + margin + 1
    xmax = min(xmax,limit[0])
    ymax = ymax + margin + 1
    ymax = min(ymax,limit[1])
    zmax = zmax + margin + 1
    zmax = min(zmax,limit[2])
        
    return xmin, xmax, ymin, ymax, zmin, zmax

#####################################################
# Generate inital point in trachea
#####################################################
	
def generate_initLoc(params, I, Mlung, Radius, RadiusZ, struct_trachea):
	
	mind           = np.argwhere(Mlung == 1)
	initLoc        = [0,0,0]
	minDiff        = float('inf')
	
	if params['super2infer']:
	    slice_no  = np.min(mind[:,2])
	    Itmp      = I[:,:,slice_no:slice_no+RadiusZ]
	else:
	    slice_no  = np.max(mind[:,2])
	    Itmp      = I[:,:,slice_no-RadiusZ:slice_no]

	Mtmp = np.ones(Itmp.shape);
	Mtmp[Itmp < params['lungMinValue']] = 0
	Mtmp[Itmp > params['lungMaxValue']] = 0
	Itmp = Mtmp;
	Mtmp = np.sum(Mtmp, axis = 2)

	for i in range(Radius, Itmp.shape[0]-Radius):
	    for j in range(Radius, Itmp.shape[1]-Radius):
	        if Mtmp[i,j] > 0:   
	            struct_Itmp = Itmp[i-Radius:i+Radius+1,j-Radius:j+Radius+1,:]
	            currVal     = struct_Itmp - struct_trachea
	            currVal     = np.sum(np.square(currVal))

	            if currVal  < minDiff:
	                initLoc = [i,j,slice_no]
	                minDiff = currVal

	print 'initial location = '+str(initLoc)
	
	return slice_no, initLoc

#####################################################
# Closed space dialation to segment airway
#####################################################

def close_space_dilation(params, I, Mlung, Radius, RadiusX, RadiusZ, struct_s, slice_no, initLoc):
	
	iterNoPerSlice = RadiusX
	maxFactor      = RadiusX/2
	maxChange      = RadiusX*RadiusX*RadiusX*50
	totalChange    = 1
	tempCheck      = 0	
	[m,n,p]        = I.shape
	Mtmp           = np.zeros([m,n,p])
	struct_m       = ndimage.iterate_structure(struct_s, 2)
	
	if params['super2infer']:
	    Mtmp[initLoc[0]-Radius:initLoc[0]+Radius+1,
	         initLoc[1]-Radius:initLoc[1]+Radius+1,
	         0:slice_no+RadiusZ] = 1
	else:
	    Mtmp[initLoc[0]-Radius:initLoc[0]+Radius+1,
	         initLoc[1]-Radius:initLoc[1]+Radius+1,
	         slice_no-RadiusZ:p-1] = 1
	Mtmp  = np.multiply(Mtmp, Mlung)
	Minit = ndimage.binary_closing(Mtmp, structure = struct_s, iterations = 1)
	Minit = np.int8(Minit)
	Minit[Minit > 0] = 2 

	while totalChange > 0:

	    maxSegmentChange = 0;
	    tempCheck        = tempCheck + 1     
	    L                = measure.label(np.floor(Minit/2))
	    Minit[Minit > 1] = 1  

	    for label in np.unique(L[:]):

	        if label != 0 and np.sum(L[:] == label) > 10:

	            # Process each component in local FOV 

	            xmin, xmax, ymin, ymax, zmin, zmax = bbox2_3D(L,label,iterNoPerSlice,[m,n,p])                                       
	            Mtmp                = Minit[xmin:xmax,ymin:ymax,zmin:zmax]
	            Itmp                = I[xmin:xmax,ymin:ymax,zmin:zmax]
	            Ltmp                = L[xmin:xmax,ymin:ymax,zmin:zmax]
	            Ltmp[Ltmp != label] = 0
	            Ltmp[Ltmp > 0]      = 1;

	            for iterCount in range(0, iterNoPerSlice):
	                Ltmp = ndimage.binary_dilation(Ltmp, structure = struct_s, iterations = 1)
	                Ltmp = np.int8(Ltmp)
	                Ltmp[Itmp > params['lungThreshold']] = 0                

	            Ltmp = ndimage.binary_closing(Ltmp, structure = struct_s, iterations = 1)
	            Ltmp = np.int8(Ltmp)
	            Ltmp[Mtmp > 0] = 0
	            Ltmp[Ltmp > 0] = 2
	            Ltmp = Ltmp + Mtmp

	            segmentChange = np.sum(Ltmp[:]>1)        
	            if segmentChange < maxChange or tempCheck < 10:
	                Minit[xmin:xmax,ymin:ymax,zmin:zmax] = Ltmp
	                if segmentChange > maxSegmentChange:
	                    maxSegmentChange = segmentChange

	    if tempCheck < 10:
	        maxChange = max(maxFactor*maxSegmentChange,maxChange)
	    else:        
	        maxChange = min(maxFactor*maxSegmentChange,maxChange)

	    totalChange = np.sum(Minit[:]>1)

	    print 'iter = '+str(tempCheck)+' airway sum = '+str(np.sum(Minit[:]>0))\
	                        +' airway change = '+str(totalChange)
	
	Minit[Minit > 0] = 1
	Minit = ndimage.binary_opening(Minit, structure = struct_s, iterations = 1)
	Minit = ndimage.binary_dilation(Minit, structure = struct_m, iterations = 1)
	Maw = np.int8(Minit)
							
	return Maw

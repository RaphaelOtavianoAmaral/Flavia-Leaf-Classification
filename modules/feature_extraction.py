from numpy import pi
from skimage.util import img_as_float
from skimage.feature import corner_harris, corner_peaks
from skimage.measure import label, regionprops

def count_corners(img,sensitivity=0.04,sigma=1.0):
    img = img_as_float(img)
    harris_response = corner_harris(img,k=sensitivity,sigma=sigma)
    corner_coords = corner_peaks(harris_response,min_distance=60,threshold_rel=0.05)
    return len(corner_coords), corner_coords
    
    

def extract_features(bin_img):
    features = []
    
    img_label = label(bin_img,connectivity=2)
    img_props = regionprops(img_label)

    if not img_props:
        return features
    
    for label_id, img_prop in enumerate(img_props,start=1):
        roi = img_prop.image
        area = img_prop.area
        perimeter = img_prop.perimeter
        axis_major = img_prop.axis_major_length
        axis_minor = img_prop.axis_minor_length
        eccentricity = img_prop.eccentricity
        solidity = img_prop.solidity

        circularity = 0.0
        if  perimeter != 0: circularity = (4*pi*area)/(perimeter**2)

        compacidade = 0.0
        if area != 0: compacidade = (perimeter**2)/area

        alongamento = 0.0
        if axis_minor != 0: alongamento = axis_major/axis_minor 

        redondeza = 0.0
        if axis_major != 0: redondeza = (4*area)/(pi*(axis_major**2))

        corners,corners_coords = count_corners(bin_img,sensitivity=0.2,sigma=1.0)

        features.append(area)
        features.append(perimeter)
        features.append(axis_major)
        features.append(axis_minor)
        features.append(eccentricity)
        features.append(solidity)
        features.append(circularity)
        features.append(compacidade)
        features.append(alongamento)
        features.append(redondeza)
        features.append(corners)



    return features
        


    
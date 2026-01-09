from numpy import ndarray, ones, uint8,float32
from matplotlib import pyplot as plt
from skimage.color import rgb2gray, gray2rgb
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_holes, remove_small_objects, closing
from skimage.util import img_as_float,img_as_ubyte
from sklearn.cluster import KMeans

def thresholding_segmentation(image: ndarray|None=None)->dict|None:
    if image is None: return None

    processed_imgs = {}

    processed_imgs["Original"] = img_as_float(image)
    processed_imgs["Gray"] = rgb2gray(image)
    processed_imgs["Blur"] = gaussian(processed_imgs["Gray"],sigma=3.0)
    threshold = threshold_otsu(processed_imgs["Blur"])
    processed_imgs["Binary"] = processed_imgs["Blur"] < threshold
    processed_imgs["Closed"] = closing(processed_imgs["Binary"],ones((20,20)))
    processed_imgs["Holes Removed"] = remove_small_holes(processed_imgs["Closed"],area_threshold=30000,connectivity=2)
    processed_imgs["Final"] = remove_small_objects(processed_imgs["Holes Removed"],min_size=5000)

    return processed_imgs

def kmeans_segmentation(image: ndarray|None=None,k: int=2)->ndarray|None:
    if image is None: return None
    
    processed_imgs = {"Original":image}

    (h,w,d) = image.shape
    pixel_features =  float32(image.reshape((h*w,d))/255.0)

    kmeans = KMeans(n_clusters=k,random_state=42,n_init="auto",max_iter=300)
    kmeans.fit(pixel_features)

    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    segmented_img_flat = centers[labels]
    segmented_img_flat = (segmented_img_flat * 255.0)
    segmented_img_flat = segmented_img_flat.astype(uint8)

    processed_imgs["Segmented"] = segmented_img_flat.reshape((h,w,d))

    processed_imgs["Gray"] = rgb2gray(processed_imgs["Segmented"])
    processed_imgs["Binary"] = img_as_ubyte(processed_imgs["Gray"]) < 200 
    processed_imgs["Holes Removed"] = remove_small_holes(processed_imgs["Binary"],area_threshold=30000,connectivity=2)
    processed_imgs["Final"] = remove_small_objects(processed_imgs["Holes Removed"],min_size=5000)

    return processed_imgs
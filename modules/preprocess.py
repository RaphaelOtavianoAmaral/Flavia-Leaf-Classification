from pathlib import Path
from random import randint
from numpy import ndarray, ones
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_holes, remove_small_objects, closing
from skimage.util import img_as_float

SPECIES = {
    "Pubescent Bamboo": [1001,1059],
    "Chinese Horse Chestnut": [1060,1122],
    "Chinese Redbud": [1123,1194],
    "True Indigo": [1195,1267],
    "Japanese Maple": [1268,1323],
    "Nanmu": [1324,1385],
    "Castor Aralia": [1386,1437],
    "Goldenrain Tree": [1438,1496],
    "Chinese Cinnamon": [1497,1551],
    "Anhui Barberry": [1552,1616],
    "Big-fruited Holly": [2001,2050],
    "Japanese Cheesewood": [2051,2113],
    "Wintersweet": [2114,2165],
    "Camphortree": [2166,2230],
    "Japan Arrowwood": [2231,2290],
    "Sweet Osmanthus": [2291,2346],
    "Deodar": [2347,2423],
    "Maidenhair Tree": [2424,2485],
    "Crape Myrtle": [2486,2546],
    "Oleander": [2547,2612],
    "Yew Plum Pine": [2616,2675],
    "Japanese Flowering Cherry": [3001,3055],
    "Glossy Privet": [3056,3110],
    "Chinese Toon": [3111,3175],
    "Peach": [3176,3229],
    "Ford Woodlotus": [3230,3281],
    "Trident Maple": [3282,3334],
    "Beales Barberry": [3335,3389],
    "Southern Magnolia": [3390,3446],
    "Canadian Poplar": [3447,3510],
    "Chinese Tulip Tree": [3511,3563],
    "Tangerine": [3566,3621]
}

def preprocess_image(image: ndarray | None)-> ndarray | None:
    if image is None: return None

    image = img_as_float(image)
    gray = rgb2gray(image)
    blur = gaussian(gray,sigma=3.0)
    otsu = threshold_otsu(blur)
    binary = blur < otsu
    closed = closing(binary,ones((20,20)))
    holes = remove_small_holes(closed,area_threshold=30000,connectivity=2)
    preprocessed = remove_small_objects(holes,min_size=5000)

    return preprocessed
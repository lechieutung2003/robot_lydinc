from skimage.feature import local_binary_pattern
from skimage.io import imread
import numpy as np
import cv2
def compute_lbp(img_path, P=8, R=1):
    """
    Compute the Local Binary Patterns (LBP) of an image.
    
    Parameters:
    - image: Input grayscale image (2D array)
    - P: Number of circularly symmetric neighbor set points (default is 8)
    - R: Radius of circle (default is 1)
    
    Returns:
    - lbp_image: LBP image
    - lbp_hist: Histogram of LBP features
    """
    img = imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    print(img.shape)
    lbp_image = local_binary_pattern(img, P, R, method='uniform')
    
    # Compute the histogram of LBP features
    (hist, _) = np.histogram(lbp_image.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    
    # Normalize the histogram
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)
      
    return lbp_image, hist

path = (r"C:\Users\sang1\OneDrive - The University of Technology\Desktop\DATABASE\databaseV2\train\00\HM\01_60_00_N_S0_FDR0_IQ3_F1.jpg")

img, hist = compute_lbp(path)

print(img)
print(len(hist))
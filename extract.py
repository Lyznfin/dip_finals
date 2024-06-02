import numpy as np
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from skimage import io

def extract_glcm_features(image_path):
    props = ['contrast', 'homogeneity', 'energy', 'correlation'] # 4 fitur yang akan diekstrak, sesuai jurnal
    image = io.imread(image_path)
    gray_img = rgb2gray(image) # Ubah citra ke format greyscale
    gray_img = np.uint8(gray_img * 255)
    glcm = graycomatrix(gray_img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True) # Kalkulasi GLCM
    features = {prop: graycoprops(glcm, prop).ravel() for prop in props} # Ekstraksi fitur dari GLCM
    return features
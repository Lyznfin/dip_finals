import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import io
from .togray import to_gray

props = ['contrast', 'homogeneity', 'energy', 'correlation'] # 4 fitur yang akan diekstrak, sesuai jurnal

def extract_glcm_features(image):
    if isinstance(image, str): # Jika input berupa path, baca citra dulu
        image = io.imread(image)
    grey_image = to_gray(image)
    grey_image = np.uint8(grey_image * 255)
    glcm = graycomatrix(grey_image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True) # Kalkulasi GLCM
    features = {prop: graycoprops(glcm, prop).ravel() for prop in props} # Ekstraksi fitur dari GLCM
    return features
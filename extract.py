import numpy as np
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from skimage import io
import os

def extract_glcm_features(image_path):
    props = ['contrast', 'homogeneity', 'energy', 'correlation'] # 4 fitur yang akan diekstrak
    image = io.imread(image_path)
    gray_img = rgb2gray(image) # Ubah citra ke format greyscale
    gray_img = np.uint8(gray_img * 255)
    glcm = graycomatrix(gray_img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True) # Kalkulasi GLCM
    features = {prop: graycoprops(glcm, prop).ravel() for prop in props} # Ekstraksi fitur dari GLCM
    return features

def extract_from_dataset(directory):
    feature_list = []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label) # Buka tiap sub-folder dataset (normal, cataract, etc)
        if os.path.isdir(label_dir):
            for image_file in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_file)
                features = extract_glcm_features(image_path)
                features['label'] = label # Melabeli fitur dengan tipe citra asal (normal, cataract, etc)
                feature_list.append(features)
    return feature_list

features = extract_from_dataset('dataset')
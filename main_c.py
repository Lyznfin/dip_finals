import joblib
from utils.extract import extract_glcm_features
from utils.sharpen import sharpen_image
from utils.histogram import histogram_equalizer
from utils.togray import to_gray
from utils.compare_image import compare_image
from utils.flatten_feature import flatten_features
from skimage import io
import pandas as pd

# load model
knn = joblib.load('model_2/knn_model.pkl')
label_encoder = joblib.load('model_2/label_encoder.pkl')
scaler = joblib.load('model_2/scaler.pkl')

def classify_c(image_path):
    # baca citra
    # image_path = 'image_test/image_cataract_2.png'
    image = io.imread(image_path)

    image_dev = histogram_equalizer(image)
    # image_dev = sharpen_image(image)

    # image_dev = histogram_equalizer(image)
    # image_dev = sharpen_image(image_dev)

    compare_image(image, image_dev)
    image_features = extract_glcm_features(image_dev)
    image_features_df = pd.DataFrame([image_features], columns=['contrast', 'homogeneity', 'energy', 'correlation'])

    new_features_flattened = flatten_features(image_features_df)
    new_features_scaled = scaler.transform(new_features_flattened)
    predicted_class_encoded = knn.predict(new_features_scaled)
    predicted_class = label_encoder.inverse_transform(predicted_class_encoded)

    print(f'Prediksi mata: {predicted_class}')
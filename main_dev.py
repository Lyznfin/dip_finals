import joblib
from utils.extract import extract_glcm_features
from utils.sharpen import sharpen_image
from utils.histogram import histogram_equalizer
from utils.togray import to_gray
from utils.compare_image import compare_image
from skimage import io
import pandas as pd

# load model
knn = joblib.load('model_vanilla_2/knn_model.pkl')
label_encoder = joblib.load('model_vanilla_2/label_encoder.pkl')
scaler = joblib.load('model_vanilla_2/scaler.pkl')

# extract fitur dari citra yang ingin diklasifikasi
image_path = 'image_test/image_cataract_2.png'
image = io.imread(image_path)

# image2 = histogram_equalizer(image)
# image2 = sharpen_image(image)

image2 = histogram_equalizer(image)
image2 = sharpen_image(image2)

# compare_image(image, image2)
image_features = extract_glcm_features(image2)
image_features_df = pd.DataFrame([image_features], columns=['contrast', 'homogeneity', 'energy', 'correlation'])

# flatten dataframe, seperti pada file model
def flatten_features(features_df: pd.DataFrame):
    flattened = pd.DataFrame()
    for col in features_df.columns:
        flattened = pd.concat([flattened, pd.DataFrame(features_df[col].tolist(), columns=[f"{col}_{i}" for i in range(len(features_df[col][0]))])], axis=1)
    return flattened

new_features_flattened = flatten_features(image_features_df)
new_features_scaled = scaler.transform(new_features_flattened)
predicted_class_encoded = knn.predict(new_features_scaled)
predicted_class = label_encoder.inverse_transform(predicted_class_encoded)

print(f'Prediksi mata: {predicted_class}')
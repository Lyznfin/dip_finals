import joblib
from utils.extract import extract_glcm_features
from skimage import io
import pandas as pd

# load model
knn = joblib.load('model_vanilla_2/knn_model.pkl')
label_encoder = joblib.load('model_vanilla_2/label_encoder.pkl')
scaler = joblib.load('model_vanilla_2/scaler.pkl')

# extract fitur dari citra yang ingin diklasifikasi
image_path = 'image_test/image_cataract_2.png'
image_features = extract_glcm_features(image_path)
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
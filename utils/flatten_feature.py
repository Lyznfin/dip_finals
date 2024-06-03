import pandas as pd

# flatten dataframe, seperti pada file model
def flatten_features(features_df: pd.DataFrame):
    flattened = pd.DataFrame()
    for col in features_df.columns:
        flattened = pd.concat([flattened, pd.DataFrame(features_df[col].tolist(), columns=[f"{col}_{i}" for i in range(len(features_df[col][0]))])], axis=1)
    return flattened
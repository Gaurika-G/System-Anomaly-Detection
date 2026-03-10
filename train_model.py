import kagglehub
import pandas as pd
import os

path = kagglehub.dataset_download("alextamboli/unsw-nb15")

file_path = os.path.join(path, "UNSW_NB15_training-set.csv")

data = pd.read_csv(file_path)

# Load dataset
data = pd.read_csv("data/UNSW_NB15_training-set.csv")

print(data.head())
print(data.shape)
print(data.columns)
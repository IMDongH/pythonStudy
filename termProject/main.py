import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import numpy as np
warnings.filterwarnings('ignore')

df = pd.read_csv("data/star_classification.csv")
le = LabelEncoder()
df["class"] = le.fit_transform(df["class"])
df["class"] = df["class"].astype(int)
df = df.drop(["obj_ID"],axis=1)
df = df.drop("rerun_ID",axis=1)
df = df.drop("run_ID",axis=1)
df = df.drop("field_ID",axis=1)
df = df.drop("fiber_ID",axis=1)
df = df.drop("cam_col",axis=1)
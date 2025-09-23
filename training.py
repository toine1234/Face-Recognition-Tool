import cv2, os, numpy as np

EMBEDDING_FL = os.path.join("nn4.small2.v1.t7")
DATASET_PATH = os.path.join("dataset")

def _load_torch(model_path_fl):
    model = cv2.dnn.readNetFromTorch(model_path_fl)
    return model
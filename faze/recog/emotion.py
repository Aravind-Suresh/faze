# Recognizing emotions from frontal-faced images

import dlib, cv2
import numpy as np
from sklearn import svm
from sklearn import cross_validation
from sklearn.externals import joblib

EMOTION_MAP = np.array(map(str, range(0, 9)))

def predict(batch, model_path):
    model = joblib.load(model_path)
    #if model is None:
        # Throw error
    return model.predict(batch)

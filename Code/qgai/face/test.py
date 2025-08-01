# 简单的KNN示例
from xml.sax.handler import all_features

import cv2
from sklearn.neighbors import KNeighborsClassifier
import pickle

from Train import get_feature_and_labels, rec_model
from face_fetcher import face_fetcher
from utils.face_utils import img_to_bin, extract_features

knn = KNeighborsClassifier()

with open("./facedata/face_yingrui_bin_data.pkl", "rb") as f:
    data = pickle.load(f)
    face_id = data['face_id']
    images = data['images']

a_face_images = []
for image in images:
    a_face_images.append(face_fetcher(image))

all_features, all_labels = get_feature_and_labels(a_face_images, face_id)
knn.fit(all_features, all_labels)

test_img = cv2.imread("./facedata/testt.jpg")
img_bin = img_to_bin(test_img)
test_face = face_fetcher(img_bin)
test_f = extract_features(test_face, rec_model).reshape(1, -1)

a = knn.predict_proba(test_f)
print(a)

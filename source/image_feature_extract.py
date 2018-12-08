from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.utils import plot_model
import numpy as np
import os
import pickle
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn import cluster
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

import sklearn
sklearn.__version__

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import KMeans

model = VGG16(weights='imagenet', include_top=False)
model.summary()

vgg16_feature_list = []
filename_list = []
ROOT_DIR = "../"
IMAGE_DIR = ROOT_DIR + "data"
directory = os.fsencode(IMAGE_DIR)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        img_path = IMAGE_DIR + '/' + filename
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        print(img_data.shape)
        print(type(img_data))
        print(img_data)
        vgg16_feature = model.predict(img_data)
        vgg16_feature_np = np.array(vgg16_feature)
        vgg16_feature_list.append(vgg16_feature_np.flatten())
        filename_list.append(filename)
        print(filename + ' extracted')


vgg16_feature_list_pfile = open(ROOT_DIR + 'pickle/vgg16_feature_list.p', 'wb')
pickle.dump(vgg16_feature_list, vgg16_feature_list_pfile)
filename_list_pfile = open(ROOT_DIR + 'pickle/filename_list.p', 'wb')
pickle.dump(filename_list, filename_list_pfile)











#vgg16_feature_list_np = np.array(vgg16_feature_list)
#kmeans = KMean(n_clusters=2, random_state=0).fit(vgg16_feature_list_np)


#print('Predicted:', decode_predictions(vgg16_feature, top=3)[0])
#plot_model(model, to_file='model.png')

# # 從頂部移出一層
# model.layers.pop()
# model.outputs = [model.layers[-1].output]
# model.layers[-1].outbound_nodes = []
# # 加一層，只辨識10類
# from keras.layers import Dense
# num_classes=10
# x=Dense(num_classes, activation='softmax')(model.output)
# # 重新建立模型結構
# model=Model(model.input,x)



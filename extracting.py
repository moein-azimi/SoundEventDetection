import numpy as np
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#from keras.applications.vgg16 import preprocess_input
#from keras.applications.xception import preprocess_input
from keras.applications.resnet import preprocess_input
#from keras.applications.vgg16 import decode_predictions
#from keras.applications.xception import decode_predictions
from keras.applications.resnet import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.applications.resnet import ResNet50
from keras.models import Model
import pickle 

#model = VGG16(weights='imagenet', input_shape=(224, 224, 3))
#model = Xception(weights='imagenet', input_shape=(299, 299, 3))
model = ResNet50()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
model.summary()
path = 'C:/UCC/Final/DoorLocalization/resnet50-spect'
A = os.listdir(path)
feature_data = list()
for i in range(0,len(A)):
    image = load_img(path+'/'+A[i], target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    features = model.predict(image)
    #print(A[i],features[0].shape) vgg16
    print(A[i],features[0].shape)
    single_data = list()
    #single_data.append(features[0].T) vgg16
    single_data.append(features[0])
    single_data.append(A[i])
    feature_data.append(single_data)
print('Saving as a pkl file')
#with open('feature_data_vgg16.pkl','wb') as f:
with open('feature_data_resnet50.pkl','wb') as f:
    pickle.dump(feature_data, f)
    print('Finished')
# example of horizontal shift image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os

datagen = ImageDataGenerator(rotation_range=10, width_shift_range=[-50.0,50.0])
path = 'C:/UCC/codes/eventdetection/spectrograms/0train/'
file = os.listdir(path)
for i in range(0,len(file)):
    if file[i].endswith('png'):
        img = load_img(path+file[i])  
        x = img_to_array(img)  
        x = x.reshape((1,) + x.shape)  
        a = file[i].split('-')[0]
        print(a)
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir='aug', save_prefix=a, save_format='png'):
            i += 1
            if i > 13:
                break  # otherwise the generator would loop indefinitely
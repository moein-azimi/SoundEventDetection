import librosa
import matplotlib.pyplot as plt 
import numpy as np 
import librosa.display
import os

path = 'C:/UCC/Final/DoorLocalization/database/'
path2 = 'C:/UCC/Final/DoorLocalization/VGG16-spect/'
path3 = 'C:/UCC/Final/DoorLocalization/Xception-spect/'
path4 = 'C:/UCC/Final/DoorLocalization/resnet50-spect/'
x = os.listdir(path)
#print(x)

for i in range(0,len(x)):
    if x[i].endswith('wav'):
        print(i, x[i])
        try:
            y,sr = librosa.load(path+'/'+x[i])
            #fig =plt.figure(figsize=[0.725,0.73]) #VGG16
            fig =plt.figure(figsize=[0.965,0.974]) #Xception
            #fig =plt.figure(figsize=[1.654,1.665]) #resnet50
            ax = fig.add_subplot(1, 1, 1)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            librosa.display.specshow(librosa.power_to_db(librosa.feature.melspectrogram(y=y,sr=sr),ref=np.max))
            plt.savefig(path3+'/'+x[i]+'.png',dpi=400,bbox_inches='tight',pad_inches=0)
            plt.close('all')
        except:
            print('failed',x[i])



'''
y,sr = librosa.load('1.wav')
fig =plt.figure(figsize=[1.654,1.665])
ax = fig.add_subplot(1, 1, 1)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.set_frame_on(False)
librosa.display.specshow(librosa.power_to_db(librosa.feature.melspectrogram(y=y,sr=sr),ref=np.max))
plt.savefig('spec.png',dpi=400,bbox_inches='tight',pad_inches=0)
plt.close('all')
'''

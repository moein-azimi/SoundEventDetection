import pickle
import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Model
from keras.layers import Dense, Conv1D,SimpleRNN, Flatten, MaxPooling1D
from keras.layers import Activation, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import optimizers
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt

with open('feature_data_Xception.pkl', 'rb') as f:
    dataset_Xception = pickle.load(f)
    
embedding = []
label1 = []

for emb, label in dataset_Xception[:]:
    print(label.split('-')[0])
    embedding.append(emb)
    label1.append(label.split('-')[0])

X = np.array(embedding)
label1 = pd.DataFrame(label1,columns=['class'])
y = np.array(label1['class'])
y = np.array(pd.get_dummies(y))
print(X.shape)
print(y.shape)
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=8)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)    
print(y_test.shape)    


model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(2048,)))
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="sigmoid"))
model.add(Dense(23))
model.add(Activation('softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=5e-5), metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='saved_models/1.hdf5',verbose=1,save_best_only=True)

history = model.fit(X_train, y_train, batch_size=20, epochs=300, validation_data=(X_test,y_test), callbacks=[checkpointer], verbose=2)
# plot metrics
y_pred = model.predict(X_test)
matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
f1score = metrics.f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1),average='macro')
print(matrix)
print(f1score)

class_names = ['BabyCry', 'Breathing', 'Cough', 'Dishes', 'DoorClapping', 'DoorOpening', 
               'ElectricalShaver', 'FemaleCry', 'FemaleScream', 'GlassBreaking', 
               'HairDryer', 'HandsClapping', 'Keyboard', 'Keys', 'Laugh', 'MaleScream', 
               'Paper', 'Sneeze', 'Speech', 'Switch', 'Water', 'Window', 'Yawn']
print(metrics.classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))
df = pd.DataFrame(metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)), index=class_names, columns=class_names)
print(df)
figsize = (10,7)
fontsize=10
fig = plt.figure(figsize=figsize)
heatmap = sns.heatmap(df, annot=True, fmt="d")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=25, ha='right', fontsize=fontsize)
#plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix: Multi-Class Classification')
fig.savefig('save_as_a_png.png')

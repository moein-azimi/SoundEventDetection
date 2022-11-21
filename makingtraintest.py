import os
import shutil 
from sklearn.model_selection import train_test_split
path = 'C:/UCC/codes/eventdetection/spectrograms/'
name = os.listdir(path)
X = []
y = []
for i in range(0,len(name)):
    if name[i].endswith('png'):
        X.append(name[i])
        y.append(i)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=8)


path_train = path+'/0train/'
path_test = path+'/0test/'

for i in range(0,len(X_train)): 
    func = shutil.copy(path+X_train[i], path_train+X_train[i]) 
    
for i in range(0,len(X_test)): 
    func = shutil.copy(path+X_test[i], path_test+X_test[i]) 
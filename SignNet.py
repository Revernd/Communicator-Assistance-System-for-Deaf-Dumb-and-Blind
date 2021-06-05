import numpy as np
import pandas as pd
import cv2, os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

path = "Dataset/Processed_ISL_data"
labelFile = 'Dataset/labels_ISL.csv'
images = []
classNo = []
count = 0
folders = os.listdir(path)
print("Total Classes Detected:",len(folders))
noOfClasses=len(folders)
print("Importing Classes.....")
for folderName in folders:
    print(count, end =" ")
    count+=1
    if not folderName.startswith('.'):
        if folderName in ['1']:
            label = 0
        elif folderName in ['2']:
            label = 1
        elif folderName in ['3']:
            label = 2
        elif folderName in ['4']:
            label = 3
        elif folderName in ['5']:
            label = 4
        elif folderName in ['6']:
            label = 5
        elif folderName in ['7']:
            label = 6
        elif folderName in ['8']:
            label = 7
        elif folderName in ['9']:
            label = 8
        elif folderName in ['A']:
            label = 9
        elif folderName in ['B']:
            label = 10
        elif folderName in ['C']:
            label = 11
        elif folderName in ['D']:
            label = 12
        elif folderName in ['E']:
            label = 13
        elif folderName in ['F']:
            label = 14
        elif folderName in ['G']:
            label = 15
        elif folderName in ['H']:
            label = 16
        elif folderName in ['I']:
            label = 17
        elif folderName in ['J']:
            label = 18
        elif folderName in ['K']:
            label = 19
        elif folderName in ['L']:
            label = 20
        elif folderName in ['M']:
            label = 21
        elif folderName in ['N']:
            label = 22
        elif folderName in ['O']:
            label = 23
        elif folderName in ['P']:
            label = 24
        elif folderName in ['Q']:
            label = 25
        elif folderName in ['R']:
            label = 26
        elif folderName in ['S']:
            label = 27
        elif folderName in ['T']:
            label = 28
        elif folderName in ['U']:
            label = 29
        elif folderName in ['V']:
            label = 30
        elif folderName in ['W']:
            label = 31
        elif folderName in ['X']:
            label = 32
        elif folderName in ['Y']:
            label = 33
        elif folderName in ['Z']:
            label = 34
        elif folderName in ['Nothing']:
            label = 35
        else:
            label = 36
        image_folders = os.listdir(path+'/'+folderName)
        for image_filename in image_folders:
            img = cv2.imread(path + '/'+folderName + '/' + image_filename, 0)
            images.append(img)
            classNo.append(label)

images = np.array(images)
classNo = np.array(classNo)

 
X_train, X_validation, y_train, y_validation = train_test_split(images, classNo, test_size=0.2)
 
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_validation=X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)

dataGen= ImageDataGenerator(rescale = 1./255, width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2,shear_range=0.1,rotation_range=10)
dataGen.fit(X_train)
batches= dataGen.flow(X_train,y_train,batch_size=20)  
X_batch,y_batch = next(batches)
 
y_train = to_categorical(y_train,noOfClasses)
y_validation = to_categorical(y_validation,noOfClasses)

model= Sequential()
model.add((Conv2D(64, (5,5),input_shape=(128,128,1),activation='relu')))
model.add((Conv2D(64, (5,5), activation='relu')))
model.add(MaxPooling2D(2,2))
model.add((Conv2D(32, (5,5) ,activation='relu')))
model.add((Conv2D(32, (5,5), activation='relu')))
model.add(MaxPooling2D(2,2))
model.add((Conv2D(32, (3,3),activation='relu')))
model.add((Conv2D(32, (3,3), activation='relu')))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.5)) 
model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))    
model.add(Dropout(0.5))
model.add(Dense(noOfClasses,activation='softmax'))
model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

print(model.summary())
history=model.fit_generator(dataGen.flow(X_train,y_train,batch_size=20),steps_per_epoch=X_train.shape[0]//20,epochs=10,validation_data=(X_validation,y_validation),shuffle=1)
model.save("Dataset/SignNet_ISL.h5")
pd.DataFrame(history.history).reset_index().to_csv('Dataset/history_ISL.csv', header=True, index=False)

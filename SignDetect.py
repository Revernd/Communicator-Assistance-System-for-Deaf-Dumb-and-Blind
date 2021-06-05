#Importing Libraries
import cv2
import numpy as np
import pandas as pd
# from time import sleep
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from preprocess import video_preprocessing

labels = pd.read_csv("D:/BOOK/Datasets/Dataset/labels_ISL.csv")
label = labels['Name']
label.index = labels['ClassId']
label = dict(label.to_dict())
label[35] = 'Nothing'

model = load_model("Dataset/SignNet_ISL.h5")
print("Model Loaded!!")
history = pd.read_csv("Dataset/history_ISL.csv")
 
plt.figure(1)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()

def getClassName(classNo):
    return label[int(classNo)]

# img = cv2.imread("Dataset/data_high Quality/ASL/A/IMG_20181119_155215403 - Copy - Copy - Copy.jpg")
# img = cv2.resize(img, (400, 400))
# classIndex = 11
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img, "CLASS: " , (20, 35), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
# cv2.putText(img, "PROBABILITY: ", (20, 75), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
# cv2.putText(img,str(classIndex)+" "+str(getClassName(classIndex)), (120, 35), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
# cv2.putText(img, str(round(0.956*100,2) )+"%", (180, 75), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
# cv2.imshow("image", img)
def sign_Detection():
    letter = ""
    cv2.waitKey(0) 
    font = cv2.FONT_HERSHEY_SIMPLEX
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(10, 180)
    cap.set(cv2.CAP_PROP_FPS, 1)
    word = ''
    while True:
        success, imgOrignal = cap.read()
        imgOrignal = cv2.flip(imgOrignal, 1)
        img = np.asarray(imgOrignal)
        cv2.rectangle(img, (300, 400), (100, 200),(255, 0 ,0), 2)
        img = img[200:400, 100:300]
        img = cv2.resize(img, (128, 128))
        img = video_preprocessing(img)
        cv2.imshow("Processed Image", img)
        img = img.reshape(1, 128, 128, 1)
        cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        predictions = model.predict(img)
        classIndex = np.argmax(predictions, axis=-1)
        probabilityValue =np.amax(predictions)
        
        print(letter)
        if probabilityValue > 0.96:
            cv2.putText(imgOrignal,str(classIndex)+" "+str(getClassName(classIndex)), (120, 35), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Result", imgOrignal)
            if str(getClassName(classIndex))!='Nothing':
                word += str(getClassName(classIndex))
            else:
                word  += ""
        
        if cv2.waitKey(1)&0xFF == ord('q') or cv2.waitKey(1)&0xFF==27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return word
        
import cv2
import pandas as pd

def sign_slides(text):
    text_new = ""
    path = "Dataset/ISL_data/"
    labelFile = 'Dataset/labels_ISL.csv'
    labels = pd.read_csv(labelFile)
    labels = labels.drop('ClassId', axis=1)
    dict_new = labels.to_dict()['Name']
    
    for x in text.upper():
        print(x)
        for key, value in dict_new.items():    
            if value==x:
                img = cv2.imread(path+'/'+x+"/1.jpg", cv2.IMREAD_UNCHANGED)
                text_new.join(x)
                resized = cv2.resize(img, (372, 372))
                cv2.imshow("Image", resized)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()

sign_slides('rutvik')
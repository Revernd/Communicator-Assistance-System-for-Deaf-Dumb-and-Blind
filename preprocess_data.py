#Importing Libraries
import numpy as np
import pandas as pd
import cv2, os, random
import matplotlib.pyplot as plt
from preprocess import image_preprocessing1

#gathering data
path = "Dataset/ISL_data/"
labelFile = 'Dataset/labels_ISL.csv'
images = []
classNo = []
count = 0
folders = os.listdir(path)
print("Total Classes Detected:",len(folders))
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
            img = cv2.imread(path + '/'+folderName + '/' + image_filename)
            images.append(img)
            classNo.append(label)

images = np.array(images)
classNo = np.array(classNo)

print("Data Shapes")
print("Images",end = "");print(images.shape,classNo.shape)

data=pd.read_csv(labelFile)
print("data shape ",data.shape,type(data))
data_to_plot = data[0:10]
fig, axs = plt.subplots(1, 10, figsize=(20, 20))
fig.tight_layout()
for j,row in data_to_plot.iterrows():
    x_selected = images[classNo == j]
    image = x_selected[random.randint(0, len(x_selected)- 1), :, :]
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axs[j].imshow(img)
    axs[j].axis("off")
    axs[j].set_title(str(j)+ "-"+row["Name"])
 
images = np.array(list(map(image_preprocessing1,images))) 
images=images.reshape(images.shape[0],images.shape[1],images.shape[2],1)


fig, axs = plt.subplots(1, 10, figsize=(20, 20))
fig.tight_layout()
for j,row in data_to_plot.iterrows():
    x_selected = images[classNo == j]
    image = x_selected[random.randint(0, len(x_selected)- 1), :, :]
    axs[j].imshow(image)
    axs[j].axis("off")
    axs[j].set_title(str(j)+ "-"+row["Name"])


folders.remove('Nothing')
folders.append('Nothing')

os.mkdir("Dataset/Processed_ISL_data")    
save_path = "Dataset/Processed_ISL_data"
for i,folder in enumerate(folders):
    label_folder = save_path+'/'+folder    
    os.mkdir(label_folder)
    images_selected = images[classNo==i]
    for j in range(0,1200):
        image_selected = images_selected[j, :, :]
        cv2.imwrite(label_folder+"/"+str(j)+".png", image_selected,[cv2.IMWRITE_PNG_COMPRESSION, 0])
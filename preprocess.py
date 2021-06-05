import cv2
import numpy as np
# import os

def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	return cv2.LUT(image, table)

def BackgroundSubstract(frame):
    fgbg = cv2.createBackgroundSubtractorMOG2(history=5, varThreshold=10,detectShadows=False)
    substracted = fgbg.apply(frame)
    return substracted

def image_preprocessing1(frame):
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)    	
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 255)
    canny_dilate = cv2.dilate(canny, kernel, iterations=1)
    canny_erode = cv2.erode(canny_dilate, kernel, iterations=1)
    return canny_erode

def image_preprocessing2(frame):
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)    	
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 255)
    canny_dilate = cv2.dilate(canny, kernel, iterations=1)
    canny_erode = cv2.erode(canny_dilate, kernel, iterations=1)
    return skin, gray, canny, canny_dilate, canny_erode

def video_preprocessing(frame):
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)    	
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skinMask = cv2.dilate(skinMask, kernel1, iterations = 1)
    skinMask = cv2.erode(skinMask, kernel1, iterations = 1)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 50, 150)
    canny_dilate = cv2.dilate(canny, kernel1, iterations=1)
    canny_erode = cv2.erode(canny_dilate, kernel1, iterations=1)
    return canny_erode

# cv2.waitKey(0) 
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
# cap.set(10, 180)
# # word = ""
# while True:
#     success, imgOrignal = cap.read()
#     imgOrignal = cv2.flip(imgOrignal, 1)
#     img = np.asarray(imgOrignal)
#     cv2.rectangle(img, (300, 300), (100, 100),(255, 0 ,0), 2)
#     img = img[100:300, 100:300]
#     img = cv2.resize(img, (128, 128))
#     # img_gamma = adjust_gamma(img, 1.75)
#     # lower = np.array([0, 48, 80], dtype = "uint8")
#     # upper = np.array([20, 255, 255], dtype = "uint8")
#     # converted = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2HSV)
#     # skinMask = cv2.inRange(converted, lower, upper)    	
#     # kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
#     # skinMask = cv2.dilate(skinMask, kernel1, iterations = 2)
#     # skinMask = cv2.erode(skinMask, kernel1, iterations = 2)
#     # skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
#     # skin = cv2.bitwise_and(img_gamma, img_gamma, mask = skinMask)
#     # gray = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
#     # canny = cv2.Canny(gray, 10, 255)
#     # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     # canny_dilate = cv2.dilate(canny, kernel2, iterations=2)
#     # canny_erode = cv2.erode(canny_dilate, kernel2, iterations=2)
#     cv2.imshow("Result", imgOrignal)
#     # cv2.imshow("Skin", np.hstack([skin,img_gamma]))
#     # cv2.imshow("Processed Image", np.hstack((gray, canny, canny_dilate, canny_erode)))    
#     mask = BackgroundSubstract(img)
#     cv2.imshow("Original", img)
#     cv2.imshow("Substracted", mask)
#     if cv2.waitKey(1)&0xFF == ord('q') or cv2.waitKey(1)&0xFF==27:
#         break
# cap.release()

image = cv2.imread("Dataset/ISL_data/A/1.jpg")
skin, gray, canny, canny_dilate,canny_erode  = image_preprocessing2(image)
cv2.imshow("image",np.hstack([image, skin]))
cv2.imshow("Preprocessed image", np.hstack([gray, canny, canny_dilate, canny_erode]))
cv2.waitKey(0)

# # path = "Dataset/data_high Quality/ASL/"
# # folders = os.listdir(path)
# # os.mkdir("Dataset/Processed_data_high_Quality")
# # save_path = "Dataset/Processed_data_high_Quality/"
# # for folder in folders:
# #     files = os.listdir(path+folder+'/')
# #     os.mkdir(save_path+folder)
# #     for j, file in enumerate(files):
# #         img_file = path+folder+'/'+file
# #         print(img_file)
# #         image = cv2.imread(img_file)
# #         image = cv2.resize(image, (400, 400))
# #         processed_image = image_preprocessing(image)
# #         cv2.imwrite(save_path+folder+'/'+str(j)+'.png', processed_image, [cv2.IMWRITE_PNG_COMPRESSION,0])

# # files = os.listdir(path+folders[0]+'/')
# # os.mkdir(save_path+folders[0])
# # for j, file in enumerate(files):
# #     img_file = path+folders[0]+'/'+file
# #     print(img_file)
# #     image = cv2.imread(img_file)
# #     image = cv2.resize(image, (400, 400))
# #     processed_image = image_preprocessing(image)
# #     cv2.imwrite(save_path+folders[0]+'/'+str(j)+'.png', processed_image, [cv2.IMWRITE_PNG_COMPRESSION,0])

# # palm_cascade = cv2.CascadeClassifier('Haar Cascades/hand_haar_cascade.xml')
# # palms = palm_cascade.detectMultiScale(image)
# # for (x,y,w,h) in palms:
# #     cv2.rectangle(image,(x,y),(x+w,y+h),(255,155,0),2)
# # cv2.imshow("Cascaded image",image)
# # cv2.waitKey(0)
# cv2.destroyAllWindows()
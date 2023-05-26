import cv2
import numpy as np
from PIL import Image
import os

path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
derector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def getImagesAndLables(path):

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = derector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples, ids
print("\n [INFO] DANG TRAINING du lieu ...")
faces,ids = getImagesAndLables(path)
recognizer.train(faces, np.array(ids))

recognizer.write('trainer/trainer.h5')

print("\n [INFO] {0} khuon mat duoc train. Thoat".format(len(np.unique(ids))))



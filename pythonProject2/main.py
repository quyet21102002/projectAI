import cv2
import face_recognition
import os

path = "pic2"
images = []
classNames = []
mylist = os.listdir(path)
print(mylist)
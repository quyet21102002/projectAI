import cv2
import os

cam = cv2.VideoCapture(0) # bật và khai báo , set camera
cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)


face_derector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #khai báo thư viện

face_id = input('\n Xin moi nhap ID khuon mat = ')

print("\n [INFO] khoi tao camera ")
count = 0

while(True):
    ret, img = cam.read() #dùng để bật cam
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_derector.detectMultiScale(gray, 1.3, 5) # được dùng để phát hiện&nhận dạng đối tượng trong h/a

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h),(255,0,0), 2)  #khoanh vùng khuôn mặt
        count +=1

        cv2.imwrite("dataset/User." + str(face_id) + '.' +str(count) +".jpg", gray[y:y+h,x:x+w]) # lưu các ảnh ở định dạng jpg vào thư mục dataSet

        cv2.imshow('show', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 20: #sau khi đổi ảnh sẽ break thoát ra khỏi vòng lập while
        break


print("\n [INFO THOAT]")
cam.release()
cv2.destroyAllWindows()



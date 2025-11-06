import cv2 as cv

#-----------------------IMage Detection--------------------

# face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml') # Pre loaded data for face detection.
# img = cv.imread('lena.jpg')
# imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(imgGray,1.1,4)

# for (x,y,w,h) in faces:
#     cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

# cv.imshow('img',img)
# cv.waitKey()
# cv.destroyAllWindows()


#---------------------------------------------Face and Video detection-------------------------------------------
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml') #A CascadeClassifier in OpenCV is a machine learning–based object detection algorithm that can identify objects in images or video streams
eye_cascade = cv.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
cap = cv.VideoCapture(0)
while cap.isOpened():
    _,img = cap.read()
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(imgGray,1.1,4) #detectMultiScale() is the method that actually detects objects (like faces, eyes, or cars) in an image.

    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        roi_gray = imgGray[y:y+h,x:x+w] # To extract the face area only.
        roi_Color = img[y:y+h,x:x+w] #Keep a color copy of the face area so we can draw eye rectangles in color later.
        # ROI means simply the region you want to focus on
        eyes = eye_cascade.detectMultiScale(roi_gray) # This scans the area occupied by the face which makes it faster to scan for eyes.+++
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_Color,(ex,ey),(ex+ew,ey+eh),(0,255,255),5)

    cv.imshow('img',img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()

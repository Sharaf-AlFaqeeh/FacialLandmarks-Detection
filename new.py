import cv2
import numpy as np
import dlib

webcam = True 
camera=1
cap = cv2.VideoCapture(camera)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("FacialLandmarks/shape_predictor_68_face_landmarks.dat")

def empty(a):
    pass

cv2.namedWindow("BGR")
cv2.resizeWindow("BGR",640,240)
cv2.createTrackbar("Blue","BGR",153,255,empty)
cv2.createTrackbar("Green","BGR",0,255,empty)
cv2.createTrackbar("Red","BGR",137,255,empty)

def createBox(img, points, target_size=(100, 100), masked=False, cropped=True):
    if masked:
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask, [points], (255,255,255))
        img = cv2.bitwise_and(img, mask)
    
    if cropped:
        bbox = cv2.boundingRect(points)
        x, y, w, h = bbox
        imgCrop = img[y:y+h, x:x+w]
        imgCrop = cv2.resize(imgCrop, target_size)
        return imgCrop
    else:
        return mask

while True:
    if webcam: 
        success, img = cap.read()
    else: 
        img = cv2.imread('FacialLandmarks/1.jpg')
    
    img = cv2.resize(img, (0,0), None, 0.6, 0.6)
    imgOriginal = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(imgOriginal)

    for face in faces:
        landmarks = predictor(imgGray, face)
        myPoints = []
        
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x,y])
        
        if len(myPoints) != 0:
            try:
                myPoints = np.array(myPoints)
                
                # استخراج الأجزاء بحجم موحد
                imgEyeBrowLeft = createBox(img, myPoints[17:22])
                imgEyeBrowRight = createBox(img, myPoints[22:27])
                imgNose = createBox(img, myPoints[27:36])
                imgLeftEye = createBox(img, myPoints[36:42])
                imgRightEye = createBox(img, myPoints[42:48])
                # create box for lips
                imgLips = createBox(img, myPoints[48:61])

                # إنشاء الشبكة
                row1 = np.hstack([imgEyeBrowLeft, imgEyeBrowRight])
                row2 = np.hstack([imgLeftEye, imgRightEye])
                row3 = np.hstack([imgNose, imgLips])
                grid = np.vstack([row1, row2, row3])

                # عرض الشبكة
                cv2.imshow('Facial Parts Grid', grid)

                # إعداد الشفاه الملونة
                maskLips = createBox(img, myPoints[48:61], masked=True, cropped=False)
                imgColorLips = np.zeros_like(maskLips)
                
                b = cv2.getTrackbarPos("Blue", "BGR")
                g = cv2.getTrackbarPos("Green", "BGR")
                r = cv2.getTrackbarPos("Red", "BGR")
                
                imgColorLips[:] = b, g, r
                imgColorLips = cv2.bitwise_and(maskLips, imgColorLips)
                imgColorLips = cv2.GaussianBlur(imgColorLips, (7,7), 10)
                
                imgOriginalGray = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
                imgOriginalGray = cv2.cvtColor(imgOriginalGray, cv2.COLOR_GRAY2BGR)
                imgColorLips = cv2.addWeighted(imgOriginalGray, 1, imgColorLips, 0.4, 0)
                cv2.imshow('BGR', imgColorLips)

            except Exception as e:
                print(e)

    cv2.imshow("Originial", imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2

faceCascade = cv2.CascadeClassifier('frontalface.xml')


smileCascade = cv2.CascadeClassifier('smile.xml')

webcam = cv2.VideoCapture(0)

while True:

    true, frame = webcam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Sec 1

    face = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5) #Sec 2

    for (x,y,w,h) in face:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,204),2)

        gray_temp = gray[y:y+h, x:x+w] #Sec 3

        smile = smileCascade.detectMultiScale(gray_temp, scaleFactor= 1.3, minNeighbors=5) #Sec 4

        for i in smile:

            if len(smile)>1:

                cv2.putText(frame,"Smiling",(x,y-50),cv2.FONT_HERSHEY_PLAIN, 2,(255,0,0),3,cv2.LINE_AA) #Sec 5

    cv2.imshow('RESULT', frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):

        break #Sec 6
        
webcam.release()

cv2.destroyAllWindows()
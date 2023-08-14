import cv2                      #Opencv inbuilt object
from functools import wraps
import time
import multiprocessing
from playsound import playsound
lastsave = 0

def counter(func):
    @wraps(func)
    def tmp(*args, **kwargs):
        tmp.count += 1
        global lastsave
        if time.time() - lastsave > 3:
                 # this is in seconds, so 5 minutes = 300 seconds
            lastsave = time.time()
            tmp.count = 0
        return func(*args, **kwargs)
    tmp.count = 0
    return tmp

#loading the xml files
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')             #for the face

eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')                              #for the eye

cap = cv2.VideoCapture(0) #0 for default cam and 1 for external cam
@counter
def closed():       
  print ("Eye Closed")
def openeye():
  print ("Eye is Open")

#function for alarm sound
def sound():
   p = multiprocessing.Process(target=playsound,args=("hello.mp3",))
   p.start()
   time.sleep(30)
   p.terminate()
  while 1:
    ret, img = cap.read()   #reading the videofeed
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #conversion to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.3,minNeighbors = 3) #harcasscade
   
    #drawing rectangles around the face
    for (x, y, w, h) in faces:
    #        print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        #detecting eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
 #if eyes detected draw rectangle and declare eye open else closed
        if eyes is not ():
            for (ex, ey, ew, eh) in eyes:
                #print(ex,ey,ew,eh)
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                openeye()
        else:
           closed()
           if closed.count == 30:             #checking condition
               #print ("Control room operator is drowsy")
               font = cv2.FONT_HERSHEY_SIMPLEX
               color = (255,0,0)
               cv2.putText(img,"You are drowsy",(x,y),font,1,color,2,cv2.LINE_AA)
               sound()                      #sound the alarm
  #show the frame window
    cv2.imshow('Control room sleep alert', img)
    if cv2.waitKey(20) & 0xff == ord('k'):
          break

#free the camera and destroy the window
cap.release()
cv2.destroyAllWindows()
     

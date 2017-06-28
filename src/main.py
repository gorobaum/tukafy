import sys
import time
import telepot
import cv2
import numpy as np
from telepot.loop import MessageLoop

def handle(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
    print(content_type, chat_type, chat_id)

    if content_type == 'photo':
        file_id = msg['photo'][2]['file_id']
        bot.download_file(file_id, '../imgs/img.jpg')
        img = cv2.imread('../imgs/img.jpg')
        img = detectFaces(img, tuka)
        cv2.imwrite('pagode.png', img)
        bot.sendPhoto(chat_id, open('pagode.png', 'rb'))

def detectFaces(img, tuka):
    result = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade_path = 'face.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    for (x,y,w,h) in faces:
        pts1 = np.float32([[x,y],[x+w, y], [x,y+h]])
        tukaShape = tuka.shape
        pts2 = np.float32([[0, 0], [tukaShape[1], 0], [0, tukaShape[0]]])

        M = cv2.getAffineTransform(pts2,pts1)
        rows, cols, ch = img.shape
        dst = cv2.warpAffine(tuka,M,(cols,rows))
        result[y:y+h,x:x+w] = 0
        result = result + dst
    return result


TOKEN = sys.argv[1]  # get token from command-line
tuka = cv2.imread(sys.argv[2])

bot = telepot.Bot(TOKEN)
MessageLoop(bot, handle).run_as_thread()
print ('Listening ...')

# Keep the program running.
while 1:
    time.sleep(10)

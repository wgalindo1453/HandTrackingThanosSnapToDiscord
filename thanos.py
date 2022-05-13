import discord
import mediapipe as mp
import time  # to check frame rate
import requests
import cv2
import json
from discord import Webhook, RequestsWebhookAdapter

cap = cv2.VideoCapture(0)  # webcam number 1 or 0
mpHands = mp.solutions.hands  # need this before
hands = mpHands.Hands()  # default params already given dont have to write anything inside
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0





def takeSnapshot():# function to take snapshot
    global pTime
    global cTime
    pTime = cTime
    cTime = time.time()
    print(cTime - pTime)
    if cTime - pTime > 0.05:
        ret, frame = cap.read()
        sendData(frame)
        print("snapshot sent")


# create a function to send data to webhook url
def sendData(data=None):
    thanosQuote = "half of the universe's population is turned into dust"
    ret, frame = cap.read()
    cv2.imwrite("test.png", data)
    # data = "William Just snapped his fingers, half of the universe's population is turned into dust..."
    url = "<DISCORD WEBHOOK>/github"
    webhook = Webhook.from_url(url, adapter=RequestsWebhookAdapter())
    webhook.send(file=discord.File("test.png"))
    webhook.send(thanosQuote)


while True:
    success, img = cap.read()  # this will give us our frame
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)  # processes and give us results
    # print(results.multi_hand_landmarks)#make sure something in results
    # opening up and extracting information multiple hands
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(
                    handLms.landmark):  # land mark we are getting and id that relates to index of landmark
                # print(id, lm)
                h, w, c = img.shape # get the height and width of the image
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy) # prints the x and y coordinates of the landmark
                # draw ids on image
                cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # check if 8(pointing finger) and 4(thumb) are near each other
                if id == 8:
                    if abs(cx - handLms.landmark[4].x * w) < 20 and abs(cy - handLms.landmark[4].y * h) < 20:
                        print("fingers 4 and 8 are close")
                        # create a circle
                        cv2.circle(img, (cx, cy), 40, (255, 0, 64), 10)
                        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)#draws the lines
                        sendData(img)#sends the image to discord
                       

            
    #fps counter display
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

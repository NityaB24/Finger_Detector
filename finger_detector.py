"""
This code snippet is a part of a larger program that uses the OpenCV library to detect and track hand gestures in real-time. It captures video from a webcam, overlays images on the hand based on the number of fingers raised, and displays the processed video with the hand gestures.

Inputs:
- The code snippet requires the OpenCV library to be installed.
- It assumes that there is a webcam connected to the computer.
- It assumes that there is a folder named "Finger_images" containing images of hand gestures.
- It assumes that there are three images in the "emoji" folder named "rockstar.png", "hand_pinky.png", and "ok.png".

Outputs:
- The code snippet displays the processed video with the hand gestures.

"""
import cv2 as cv
import os
import handtrackingmodule as htm

wcam, hcam = 640, 480

capture = cv.VideoCapture(0)
capture.set(3, wcam)
capture.set(4, hcam)
pt = 0
ct = 0

folderpath = "Finger_images"
myList = os.listdir(folderpath)
print(myList)

imgback = cv.imread('NB (1).png')

overlaylist = []
for impath in myList:
    image = cv.imread(f'{folderpath}/{impath}')
    overlaylist.append(image)
print(len(overlaylist))

list2 = []
list2.append(cv.imread('emoji/rockstar.png'))
list2.append(cv.imread('emoji/hand_pinky.png'))
list2.append(cv.imread('emoji/ok.png'))

detector = htm.hand()
tipids = [4, 8, 12, 16, 20]
while True:
    success, img = capture.read()
    imgback[155:155+480,599:599+640] = img
    img = detector.findhands(imgback, draw=False)
    lmlist = detector.findposition(img, draw=False)
    if len(lmlist) != 0:
        fingers = [] * 10
        # thumb
        if lmlist[tipids[0]][1] > lmlist[tipids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

                    # 4 fingers
        for id in range(1, 5):
            if lmlist[tipids[id]][2] < lmlist[tipids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

            totalfingers = fingers.count(1)
            print(totalfingers)

            h, w, c = overlaylist[totalfingers - 1].shape
            img[38:38+h, 70:70+w] = overlaylist[totalfingers - 1]

            if lmlist[tipids[0]][1] < lmlist[tipids[0] - 1][1]:
                if lmlist[tipids[4]][2] < lmlist[tipids[4] - 2][2]:
                    if lmlist[tipids[1]][2] < lmlist[tipids[1] - 2][2]:
                        if lmlist[tipids[2]][2] > lmlist[tipids[2] - 2][2]:
                            if lmlist[tipids[3]][2] > lmlist[tipids[3] - 2][2]:
                                h, w, c = list2[0].shape
                                img[38:38 + h, 70:70 + w] = list2[0]
                    elif lmlist[tipids[2]][2] < lmlist[tipids[2] - 2][2]:
                        if lmlist[tipids[3]][2] < lmlist[tipids[3] - 2][2]:
                            h, w, c = list2[2].shape
                            img[38:38 + h, 70:70 + w] = list2[2]
                    else:
                        h, w, c = list2[1].shape
                        img[38:38 + h, 70:70 + w] = list2[1]

    cv.imshow("Image", imgback)
    cv.waitKey(1)
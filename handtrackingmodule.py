import cv2 as cv
import mediapipe as mp
# import time
import math
class hand():
    def __init__(self, mode=False, maxhands=2):
        """
        Initializes the hand class object with the specified mode and maximum number of hands to detect.
    
        Args:
            mode (bool): Specifies whether to run the hand detection in static image mode or video mode. Default is False.
            maxhands (int): Specifies the maximum number of hands to detect. Default is 2.
        """
        self.mode = mode
        self.maxhands = maxhands

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxhands)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipids = [4, 8, 12, 16, 20]

    def findhands(self, img, draw=True):
        """
        Detects and draws landmarks on hands in an image.

        Args:
            img (numpy array): The input image on which the hand landmarks are to be detected.
            draw (bool, optional): Flag to specify whether to draw the landmarks on the image. Defaults to True.

        Returns:
            numpy array: The input image with the hand landmarks drawn on it.
        """
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)

        return img

    def findposition(self, img, handno=0, draw=True):
        """
        Detects and draws landmarks on hands in an image.

        Args:
            img (numpy array): The input image on which the hand landmarks are to be detected.
            handno (int, optional): The index of the hand to be detected if multiple hands are present in the image. Defaults to 0.
            draw (bool, optional): Flag to specify whether to draw the landmarks on the image. Defaults to True.

        Returns:
            list: A list of landmarks with their corresponding coordinates.
        """

        self.lmlist=[]
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handno]
            for id, lm in enumerate(myhand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                self.lmlist.append([id, cx, cy])
                if draw:
                    if id == 8 or id == 4 :
                        cv.circle(img, (cx, cy), 12, (0, 255, 255), -1)
        return self.lmlist

    def fingersup(self):
        """
        Determines which fingers are raised based on the landmarks detected on the hand.

        Returns:
        - A list of binary values representing the status of each finger. A value of 1 indicates that the finger is raised, while a value of 0 indicates that the finger is not raised.
        """

        fingers = []
        # thumb
        if self.lmlist[self.tipids[0]][1] < self.lmlist[self.tipids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers
        for id in range(1, 5):
            if self.lmlist[self.tipids[id]][2] < self.lmlist[self.tipids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def finddistaance(self, p1, p2, img, draw=True, r=15, t=3):
        """
        Calculates the distance between two landmarks on a hand and visualizes it on an image.

        Args:
            p1 (int): The index of the first landmark.
            p2 (int): The index of the second landmark.
            img (numpy array): The input image on which the landmarks and distance will be visualized.
            draw (bool, optional): Flag to specify whether to draw the landmarks and distance on the image. Defaults to True.
            r (int, optional): The radius of the circles representing the landmarks. Defaults to 15.
            t (int, optional): The thickness of the line representing the distance. Defaults to 3.

        Returns:
            tuple: A tuple containing the calculated distance, the modified image with the landmarks and distance visualized, and a list of the coordinates of the two landmarks and the center.

        """
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv.circle(img, (x1, y1), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (cx, cy), r, (0, 0, 255), cv.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]
def main():
    """
    This function is the entry point of the program. It initializes a video capture object, creates an instance of the `hand` class, and continuously reads frames from the video capture. It then uses the `hand` object to detect and draw landmarks on the hands in each frame. The landmarks are then printed to the console. Finally, the processed frame is displayed in a window.
    """
    capture = cv.VideoCapture(0)
    pt = 0
    ct = 0
    detector = hand()

    while True:
        success, img = capture.read()
        img = detector.findhands(img)
        lmlist = detector.findposition(img)
        if len(lmlist) != 0:
            print(lmlist[4])
            print(lmlist[8])

        # ct = time.time()
        # fps = 1 / (ct - pt)
        # pt = ct
        # cv.putText(img, str(int(fps)), (10, 70), cv.FONT_ITALIC, 3, (0, 255, 0), 3)

        cv.imshow("Image", img)
        cv.waitKey(1)


if __name__ == "__main__":
    main()
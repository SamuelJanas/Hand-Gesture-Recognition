import cv2
import mediapipe as mp
import imutils 

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

def process_image(img):
    # Converting the input to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(gray_image)

    # Returning the detected hands to calling function
    return results

def draw_hand_connections(img, results):
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = 1000
            y_min = 1000
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape

                # Finding the coordinates of each landmark
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_max = max(cx, x_max)
                y_max = max(cy, y_max)
                x_min = min(cx, x_min)
                y_min = min(cy, y_min)
                

                # Printing each landmark ID and coordinates
                # on the terminal
                print(id, cx, cy)

                # Creating a circle around each landmark
                cv2.circle(img, (cx, cy), 10, (0, 255, 0),
                           cv2.FILLED)
                # Drawing the landmark connections
                mpDraw.draw_landmarks(img, handLms,
                                      mpHands.HAND_CONNECTIONS)
            cv2.boundingRect(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        return img


cap = cv2.VideoCapture(0)

while True:
    # Taking the input
    success, image = cap.read()
    image = imutils.resize(image, width=1000, height=1000)
    results = process_image(image)
    image = draw_hand_connections(image, results)

    # Displaying the output
    cv2.imshow("Hand tracker", image)

    # Program terminates when q key is pressed
    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
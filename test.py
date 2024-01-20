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
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape

                # Finding the coordinates of each landmark
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Printing each landmark ID and coordinates
                # on the terminal
                print(id, cx, cy)

                # Creating a circle around each landmark
                cv2.circle(img, (cx, cy), 10, (0, 255, 0),
                           cv2.FILLED)
                # Drawing the landmark connections
                mpDraw.draw_landmarks(img, handLms,
                                      mpHands.HAND_CONNECTIONS)

        return img


# draw bounding box around hand
def draw_bounding_box(img, results, padding=20):
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # bounding box
            x_max = 0
            x_min = 100000
            y_max = 0
            y_min = 100000
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape

                # Finding the coordinates of each landmark
                cx, cy = int(lm.x * w), int(lm.y * h)
                if cx > x_max:
                    x_max = cx
                if cx < x_min:
                    x_min = cx
                if cy > y_max:
                    y_max = cy
                if cy < y_min:
                    y_min = cy

            # Drawing the bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # keep the bounded area with padding = 20 pixels in seperata variable
            # check if the bounded area with padding is within the image
            # if not, then set the padding to the maximum possible value

            if x_min - padding < 0:
                x_min = 0
            else:
                x_min = x_min - padding

            if x_max + padding > w:
                x_max = w
            else:
                x_max = x_max + padding

            if y_min - padding < 0:
                y_min = 0
            else:
                y_min = y_min - padding

            if y_max + padding > h:
                y_max = h
            else:
                y_max = y_max + padding

            bounded_area = img[y_min:y_max, x_min:x_max]


        return img, bounded_area

cap = cv2.VideoCapture(0)

while True:
    # Taking the input
    success, image = cap.read()
    image = imutils.resize(image, width=500, height=500)
    results = process_image(image)
    # draw_hand_connections(image, results)
    img, bounded_area = draw_bounding_box(image, results)

    # Displaying the output
    cv2.imshow("Hand tracker", image)
    cv2.imshow("Bounded area", bounded_area)

    # Program terminates when q key is pressed
    if cv2.waitKey(1) == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import imutils
import torch 
from model.SelfMadeCNN import SelfMadeResNet

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

def process_image(img):
    # Converting the input to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(gray_image)

    # Returning the detected hands to calling function
    return results

def draw_hand_connections(img, results, return_image=False, size=200, draw=True):
    h, w, c = img.shape
    hand_images = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            x_values = [int(lm.x * img.shape[1]) for lm in handLms.landmark]
            y_values = [int(lm.y * img.shape[0]) for lm in handLms.landmark]
            x_min, x_max = min(x_values), max(x_values)
            y_min, y_max = min(y_values), max(y_values)
            x_center, y_center = (x_min + x_max) // 2, (y_min + y_max) // 2

            
            for id, lm in enumerate(handLms.landmark):
                if draw:
                    cx, cy = x_values[id], y_values[id]
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
                    mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            if draw:
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            if return_image:
                # take size x size image around the center of the hand
                # beware of the boundaries
                # returned image should be of size (2*size, 2*size, 3)
                # if the hand is too close to the boundary return 2*size from the boundary
                
                if x_center - size < 0:
                    x_center = size
                if x_center + size > w:
                    x_center = w - size
                if y_center - size < 0:
                    y_center = size
                if y_center + size > h:
                    y_center = h - size
                hand_images.append(img[y_center-size:y_center+size, x_center-size:x_center+size])

    return img, hand_images

checkpoint = torch.load('checkpoints/default_best_acc.pth')
model = SelfMadeResNet(num_blocks=[3, 4], num_classes=18)
model.load_state_dict(checkpoint['state_dict'])
model.to('cuda')
model.eval()
label_names = ['call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three', 'three2', 'two_up', 'two_up_inverted']

cap = cv2.VideoCapture(0)
draw_boxes = False
while True:

    success, image = cap.read()
    image = imutils.resize(image, width=1000, height=800)
    results = process_image(image)
    img, hand_fragments = draw_hand_connections(image, results, return_image=True, draw=draw_boxes)

    # Displaying the output
    cv2.imshow("Hand tracker", image)

    if len(hand_fragments) > 0:
        for i, hand in enumerate(hand_fragments):
            hand_tensor = torch.from_numpy(hand).permute(2, 0, 1).unsqueeze(0).float().to('cuda')
            output = model(hand_tensor)
            # display 3 most probable classes
            _, indices = torch.sort(output, descending=True)
            for j in range(3):
                cv2.putText(hand, label_names[indices[0][j]], (10, 30 + j*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(f"Hand {i}", hand)

    if cv2.waitKey(1) == ord('d'):
        draw_boxes = not draw_boxes

    # Exit condition
    # FIXME: This is not working
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break
    
        
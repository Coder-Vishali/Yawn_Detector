import cv2
import dlib
import numpy as np
import os
import csv
from scipy.spatial import distance as dist

'''
It is a model that can identify various features of our face and provide information.

Dlib will extract all the information from the file and
opencv is used to find out different features.

Shape predictor(dat file) is used to extract information from the image like 
corner of eyes, areas around noes etc. The image is converted into matrix 
and the annotate_landmarks() function marks those features/information which
is used by other functions to extract various features like upper lips, upper mouth etc.
'''

PREDICTOR_PATH = os.path.join(os.getcwd(), 'Lib/shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

def getting_landmarks(img):
    rects = detector(img, 1)
    return np.matrix([[p.x,p.y] for p in predictor(img, rects[0]).parts()])

def annotate_landmarks(img, landmarks):
    img = img.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(img,
                    str(idx),
                    pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(1, 2, 255))
        cv2.circle(img, pos, 3, color=(0, 2, 2))
    return img

######################################################################
# Extract information about various features pivotal to detect Yawning
######################################################################

def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61, 64):
        top_lip_pts.append(landmarks[i])
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56, 59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])

def mouth_open(image):
    landmarks = getting_landmarks(image)
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance

def closing_of_eyes(image):
    landmarks = getting_landmarks(image)
    eyelid_dist = dist.euclidean(landmarks[44], landmarks[46])
    eye_corner_dist = dist.euclidean(landmarks[42], landmarks[45])
    eye_distance = eyelid_dist/eye_corner_dist
    return eye_distance

######################################################################
# Open the video camera and define yawning criteria while yawn_status
# is True and assign action accordingly.
######################################################################
'''
If lip_distance is greater than 25 than it(yawn_status) is defined as yawning (True case).
In that case, “Person is Yawning” text appears on the screen.
If Yawn_status is true, the Yawn Count will go on and show how many times the subject has yawned.
'''
cap = cv2.VideoCapture(0)
yawns = 0
yawn_status = False
yawn_info = ['lip_distance', 'yawn_status']
yawn_dict = []

while True:
    ret, frame = cap.read()
    image_landmarks, lip_distance = mouth_open(frame)
    eye_distance = closing_of_eyes(frame)

    prev_yawn_status = yawn_status

    if lip_distance > 25 and eye_distance < 0.16:
        yawn_status = True

        yawn_dict.append({'lip_distance': lip_distance, 'yawn_status': yawn_status})

        cv2.putText(frame, "Person is Yawning", (50,450),
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0 , 255), 2)

        output_text = "Yawn Count: " + str(yawns + 1)

        cv2.putText(frame, output_text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,127), 2)

    else:
        yawn_status = False

        yawn_dict.append({'lip_distance': lip_distance, 'yawn_status': yawn_status})

    if prev_yawn_status == True and yawn_status == False:
        yawns += 1

    cv2.imshow("Live Landmarks", image_landmarks)
    cv2.imshow("Yawn Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # wait for 'q' to be pressed to exit the code
        break

with open('yawn_info.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=yawn_info)
    writer.writeheader()
    writer.writerows(yawn_dict)

cap.release()
cv2.destroyAllWindows()

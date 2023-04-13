import cv2
import mediapipe as mp
import pyautogui
import numpy as np
# import pyautogui
# import speech_recognition as sr # debug:https://github.com/Uberi/speech_recognition/issues/294 (same for pyaudio)
from transcribe import transcribe

# def transcribe():
#     r = sr.Recognizer()
#     mic = sr.Microphone()
#     with mic as source:
#         r.adjust_for_ambient_noise(source)
#         audio = r.listen(source)
#         transcription = r.recognize_google(audio)
#         print(transcription)
#     return transcription


cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True,
                                            min_detection_confidence=0.5, 
                                            min_tracking_confidence=0.5)

while cam.isOpened():

    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks

    frame_h, frame_w, _ = frame.shape

    # eye blink for mouse click
    if landmark_points:
        landmarks = landmark_points[0].landmark
        left = [landmarks[145], landmarks[159]]
        right = [landmarks[374], landmarks[386]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))
        for landmark in right:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (255, 0, 255))
        if (left[0].y - left[1].y) < 0.01:
            pyautogui.click(button='left')
            pyautogui.sleep(0.5)
        if (right[0].y - right[1].y) < 0.011:
            pyautogui.click(button='right')
            pyautogui.sleep(0.5)

    # head angle for scrolling
    img_h, img_w, img_c = frame.shape
    face_3d = []
    face_2d = []
    if landmark_points:
        
        landmarks = landmark_points[0].landmark
        mouth = [landmarks[13], landmarks[14]]
        
        for idx, lm in enumerate(landmarks):
            if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                if idx == 1:
                    nose_2d = (lm.x * img_w, lm.y * img_h)
                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                x, y = int(lm.x * img_w), int(lm.y * img_h)

                # Get the 2D Coordinates
                face_2d.append([x, y])

                # Get the 3D Coordinates
                face_3d.append([x, y, lm.z])       
        
        for landmark in mouth:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (255, 255, 0))

        # Convert it to the NumPy array
        face_2d = np.array(face_2d, dtype=np.float64)

        # Convert it to the NumPy array
        face_3d = np.array(face_3d, dtype=np.float64)

        # The camera matrix
        focal_length = 1 * img_w

        cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                [0, focal_length, img_w / 2],
                                [0, 0, 1]])

        # The distortion parameters
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # Get the y rotation degree
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360
        
        # See where the user's head tilting
        if (mouth[0].y - mouth[1].y) < -0.01:
            if y < -10:
                text = "Move Cursor Left"
                pyautogui.move(-20, 0)
            elif y > 10:
                text = "Move Cursor Right"
                pyautogui.move(20, 0)
            elif x < -10:
                text = "Move Cursor Down"
                pyautogui.move(0, 20)
            elif x > 10:
                text = "Move Cursor Up"
                pyautogui.move(0, -20)
            else:
                text = "Neutral"
        else:
            if y < -10: # face left
                text = "Nil"
            elif y > 10: # face right
                text = "Transcribing"
                transcription = transcribe()
                pyautogui.write(transcription)
            elif x < -10:
                text = "Scrolling Down"
                pyautogui.scroll(-20)
            elif x > 10:
                text = "Scrolling Up"
                pyautogui.scroll(20)
            else:
                text = "Neutral"

        # Display the nose direction
        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

        p1 = (int(nose_2d[0]), int(nose_2d[1]))
        p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
        
        cv2.line(frame, p1, p2, (255, 0, 0), 3)

        # Add the text on the image
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        # cv2.putText(frame, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(frame, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(frame, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Test Screen', frame)
    cv2.waitKey(1)

cam.release()
cv2.destroyAllWindows()

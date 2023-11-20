import mediapipe as mp
import cv2
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, img = cap.read()
    
        
    cv2.imshow('Webcam', img)
    
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()
cap = cv2.VideoCapture(0)
# intiaizing holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:


    while cap.isOpened():
        ret, frame = cap.read()
        
       
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = holistic.process(image)
        #print(results.face_landmarks)
        
        # face_landmarks , pose_Landmarks , left_hand_landmarks , Right_hand_landmarks
        
        
        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                 mp_drawing.DrawingSpec(color=(240,197,59),thickness=1,circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(0,0,0),thickness=1,circle_radius=1))
        
        # Right Hands
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(240,197,59),thickness=2,circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(255,255,255),thickness=2,circle_radius=2))
        
        # Left Hands 
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(240,197,59),thickness=2,circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(255,255,255),thickness=2,circle_radius=2))
        
        # Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(240,197,59),thickness=2,circle_radius=2),
                                 mp_drawing.DrawingSpec(color=(255,255,255),thickness=2,circle_radius=2))
    
        cv2.imshow('Raw Webcam Feed', image)
    
        if cv2.waitKey(10) & 0xff == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()
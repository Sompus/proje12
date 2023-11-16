import cv2

# Load the pre-trained face, hand, and body classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
hand_cascade = cv2.CascadeClassifier('path/to/your/hand_cascade.xml')  # Replace with the path to your hand cascade file
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Function to detect and draw rectangles around faces, hands, and bodies
def detect_objects(image, scaleFactor, minNeighbors, minSize, color, classifier):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    objects = classifier.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)
    for (x, y, w, h) in objects:
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    return image

# Open a video capture object (you can replace '0' with the path to a video file)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Detect and draw rectangles around faces, hands, and bodies
    frame = detect_objects(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), color=(255, 0, 0), classifier=face_cascade)
    frame = detect_objects(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), color=(0, 255, 0), classifier=hand_cascade)
    frame = detect_objects(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), color=(0, 0, 255), classifier=body_cascade)

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

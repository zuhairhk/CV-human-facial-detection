import cv2
import dlib
import numpy as np
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to input image")
args = parser.parse_args()

# Initialize detectors
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load image
image = cv2.imread(args.image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_detector(gray)

print(f"[INFO] Detected {len(faces)} face(s).")

for face in faces:
    # Draw rectangle around face
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Get landmarks
    landmarks = landmark_predictor(gray, face)
    landmarks_points = np.array([[p.x, p.y] for p in landmarks.parts()])

    # Draw landmarks
    for (x, y) in landmarks_points:
        cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

# Show output
cv2.imshow("Facial Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
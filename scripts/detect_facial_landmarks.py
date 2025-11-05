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

# Combine color and grayscale in a single window
# Convert grayscale to 3-channel so it can be concatenated with color image
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# Ensure both images have same dimensions
if image.shape != gray_bgr.shape:
    gray_bgr = cv2.resize(gray_bgr, (image.shape[1], image.shape[0]))

# Stack them side by side
combined = np.hstack((image, gray_bgr))

# Display
window_name = "Facial Landmarks (left) | Grayscale (right)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1200, 600)
cv2.imshow(window_name, combined)
print("[INFO] Press 'q' to close the window.")
# Keep window open until 'q' key is pressed
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()

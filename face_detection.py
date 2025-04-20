import cv2

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load an image
image = cv2.imread('myImg.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("image gray scale done")
# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)


cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


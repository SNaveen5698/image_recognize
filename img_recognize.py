import cv2
import matplotlib.pyplot as plt

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('/content/xml1.xml')

# Load an image
img = cv2.imread('/content/2024-03-23.png')
image = img.copy()

# Detect faces in the image
faces = face_cascade.detectMultiScale(image)

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Convert the image to RGB for Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image with detected faces using Matplotlib
plt.imshow(image_rgb)
plt.title('Detected Faces')
plt.axis('off')  # Turn off axis numbers and ticks

# Add nearest neighbor interpolation for better understanding
plt.interpolation = 'nearest'

plt.show()

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np

# Load the trained model
model = load_model("cat_dog_classifier_model.h5")

# Set the input image dimensions
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Set image path
img_path = 'archive/train/cat - bombay/Bombay_1_jpg.rf.196d4f5a90867a18ac417d2da342e167.jpg'

# Load and preprocess the image
img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict using the model
prediction = model.predict(img_array)

# Display image and result
plt.imshow(img)
plt.axis('off')

if prediction[0][0] > 0.5:
    plt.title("🐶 It's a Dog!")
    print("🐶 It's a Dog!")
else:
    plt.title("🐱 It's a Cat!")
    print("🐱 It's a Cat!")

plt.show()


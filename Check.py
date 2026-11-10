import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from google.colab import files

print("Please upload your scanned digit image:")
uploaded = files.upload()

for fn in uploaded.keys():
    img_path = fn

# SET THIS TO TRUE IF YOUR IMAGE IS BLACK INK ON A WHITE BACKGROUND
invert_colors = True

try:
    # 1. Load image in grayscale and resize to 28x28
    img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img)

    # 2. Handle Color Inversion if necessary
    if invert_colors:
        img_array = 255.0 - img_array

    img_processed = img_array / 255.0
    img_processed = np.expand_dims(img_processed, axis=0)

    # 3. Predict
    predictions = base_model.predict(img_processed)
    predicted_digit = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    # 4. Show what the model actually saw
    plt.figure(figsize=(4,4))
    plt.imshow(img_processed[0, :, :, 0], cmap='gray')
    plt.title(f"Model Saw This | Pred: {predicted_digit} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

    print(f"\nModel predicts this digit is a: {predicted_digit}")

except Exception as e:
    print(f"Error: {e}")

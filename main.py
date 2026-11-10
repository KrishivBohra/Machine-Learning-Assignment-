import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# ==========================================
# PART A: Dataset Analysis & Preprocessing
# ==========================================
print("--- Part A: Preprocessing ---")
# Load MNIST dataset (acting as our scanned cheque digits)
(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()

# 1. Resizing & Channel Definition: Reshape to 28x28x1 (grayscale)
X_train = X_train_raw.reshape(-1, 28, 28, 1).astype('float32')
X_test = X_test_raw.reshape(-1, 28, 28, 1).astype('float32')

# 2. Normalization: Scale pixel values from 0-255 down to 0.0-1.0
X_train /= 255.0
X_test /= 255.0

# 3. One-Hot Encoding: Convert integer labels (0-9) to categorical vectors
y_train = tf.keras.utils.to_categorical(y_train_raw, 10)
y_test = tf.keras.utils.to_categorical(y_test_raw, 10)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# ==========================================
# PART B: Constructing the CNN Model
# ==========================================
print("\n--- Part B: Building the CNN Model ---")
def build_model(filters_1=32, filters_2=64, kernel_size=(3, 3)):
    model = Sequential([
        # First Convolutional & Pooling Layer
        Conv2D(filters_1, kernel_size=kernel_size, activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),

        # Second Convolutional & Pooling Layer
        Conv2D(filters_2, kernel_size=kernel_size, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        # Flattening and Dense Layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), # Regularization to prevent overfitting
        Dense(10, activation='softmax') # 10 output classes for digits 0-9
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

base_model = build_model()
base_model.summary()

# ==========================================
# PART C: Model Performance Analysis
# ==========================================
print("\n--- Part C: Training & Analysis ---")
# Train the model and store the metrics history
history = base_model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.1, # 10% of training data used for validation
    verbose=1
)

# Evaluate final test accuracy
test_loss, test_acc = base_model.evaluate(X_test, y_test, verbose=0)
print(f"\nBase Model Test Accuracy: {test_acc * 100:.2f}%")

# Plotting Training vs Validation Accuracy and Loss
plt.figure(figsize=(14, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange', linestyle='--')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', color='red')
plt.plot(history.history['val_loss'], label='Val Loss', color='green', linestyle='--')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ==========================================
# PART D: Examining Filter Sizes and Layers
# ==========================================
print("\n--- Part D: Changing Filters and Layers ---")
# Experiment 1: Larger filter size (5x5 instead of 3x3)
print("Training Model with 5x5 Filters...")
model_large_filters = build_model(filters_1=32, filters_2=64, kernel_size=(5, 5))
history_large = model_large_filters.fit(X_train, y_train, batch_size=128, epochs=5, validation_split=0.1, verbose=0)
_, acc_large = model_large_filters.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy with 5x5 filters: {acc_large * 100:.2f}%")

# Experiment 2: Deeper Network (Adding a 3rd Conv Layer)
print("\nTraining Deeper Model (3 Conv Layers)...")
deeper_model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'), # Added layer
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
deeper_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_deeper = deeper_model.fit(X_train, y_train, batch_size=128, epochs=5, validation_split=0.1, verbose=0)
_, acc_deeper = deeper_model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy with 3 Convolutional Layers: {acc_deeper * 100:.2f}%")

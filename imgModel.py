import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Paths
train_dir = "screenshots"  # Main folder containing 'phishing' and 'genuine'

# Parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30

# Data Augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,  # 80% train, 20% validation
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# No augmentation for validation, only rescaling
val_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

# Load training data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

# Load validation data
val_data = val_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# Print counts
print(f"\n[INFO] Training images: {train_data.samples}")
print(f"[INFO] Validation images: {val_data.samples}")
print(f"[INFO] Classes found: {train_data.class_indices}\n")

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2, 2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")  # Binary classification
])

# Compile Model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train Model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# Save Model
model.save("imgModel.h5")
print("\n[INFO] Model saved as imgModel.h5 ✅")

# Plot Accuracy & Loss
plt.figure(figsize=(10, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Accuracy")

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")

plt.show()

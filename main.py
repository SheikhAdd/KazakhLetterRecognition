import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image
import os

# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    try:
        img = Image.open(image_path)
        img = img.resize((64, 64)).convert('L')  # Resize and convert to grayscale
        img_array = np.array(img) / 255.0  # Normalize pixel values
        return img_array
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return None

# Load the dataset
dataset_path = 'D:/CapstoneProject/Capstone_Code/alphabet.xlsx'
data = pd.read_excel(dataset_path)

# Set the correct path to your images
image_folder_path = 'D:/CapstoneProject/Capstone_Code/datasetsaver/dataset_new/new_dataset'

# Check and print column names
print("Column names in the dataset:", data.columns)

# Assuming the correct column name is "File Name"
data['full_path'] = data['File_Name'].apply(lambda x: os.path.join(image_folder_path, x))

# Print some full paths to verify they are correct
print(data['full_path'].head())

# Load images
data['image_array'] = data['full_path'].apply(load_and_preprocess_image)

# Drop rows where image loading failed
data = data.dropna(subset=['image_array'])

# Prepare data for training
X = np.stack(data['image_array'].values)  # Features
X = X.reshape(-1, 64, 64, 1)  # Reshape for CNN (batch, height, width, channels)
y = pd.get_dummies(data['Letter']).values  # One-hot encode labels

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")


# Assuming 'model' is your trained model instance
model.save('test.h5')  # This saves the model in HDF5 format

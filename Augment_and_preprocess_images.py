import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directory paths
input_dir = 'captured_images'
output_dir = 'augmented_images'
os.makedirs(output_dir, exist_ok=True)

# Initialize the ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=30,  # Random rotation from -30 to 30 degrees
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2,  # Random vertical shift
    shear_range=0.2,  # Shear transformation
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    brightness_range=[0.8, 1.2],  # Random brightness adjustment
    fill_mode='nearest'  # Fill missing pixels after transformations
)

# Function for preprocessing image
def preprocess_image(image):
    """
    Preprocess the image (grayscale, normalization, etc.)
    """
    # Optional: Resize image (e.g., 224x224 for model consistency)
    image_resized = cv2.resize(image, (224, 224))

    # Optional: Normalize pixel values
    # image_normalized = image_resized.astype('float32') / 255.0

    # Convert to grayscale (if needed)
    # image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    return image_resized  # Return the preprocessed image

# Loop through all the images in the captured_images directory
for img_file in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_file)
    
    if img_file.endswith('.jpg') or img_file.endswith('.jpeg'):
        # Read the image
        image = cv2.imread(img_path)

        # Apply preprocessing steps
        preprocessed_image = preprocess_image(image)

        # Reshape image for augmentation (required by ImageDataGenerator)
        preprocessed_image = preprocessed_image.reshape((1, ) + preprocessed_image.shape)

        # Extract original file name parts
        file_parts = img_file.split('.')
        name, unique_id, serial_num = file_parts[0], file_parts[1], file_parts[2]

        # Generate augmented images and save with updated names
        i = 1
        for batch in datagen.flow(
            preprocessed_image,
            batch_size=1,
            save_to_dir=output_dir,
            save_prefix=f"{name}.{unique_id}.{serial_num}_aug",
            save_format='jpg'
        ):
            i += 1
            if i > 5:  # Generate 5 augmentations per image
                break

print("Augmentation and preprocessing complete.")

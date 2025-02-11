import os
import cv2
import numpy as np
from PIL import Image

# Function to extract images and labels
def getImagesAndLabels(path):
    """
    Reads images from the specified path, converts them to grayscale,
    and extracts IDs and face data.
    """
    # Get the path of all files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('jpg', 'png', 'jpeg'))]

    # Initialize lists for faces and IDs
    faces = []
    Ids = []

    # Loop through all image paths
    for imagePath in imagePaths:
        try:
            # Load the image and convert it to grayscale
            image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Skipping file {imagePath}: Unable to read image.")
                continue

            # Extract the ID from the image filename (Shahin.101.1.jpg → ID = 101)
            filename = os.path.basename(imagePath)  # Extract filename from path
            filename_parts = filename.split(".")  # Splitting at '.'
            
            if len(filename_parts) < 3:
                print(f"Skipping file {filename}: Incorrect naming format.")
                continue

            try:
                Id = int(filename_parts[1])  # Extracting ID
            except ValueError:
                print(f"Skipping file {filename}: ID should be a number.")
                continue

            # Append the face and ID to their respective lists
            faces.append(image)
            Ids.append(Id)
            print(f"Processed: {filename} | ID: {Id}")  # Debugging output

        except Exception as e:
            print(f"Error processing file {imagePath}: {e}")

    return faces, Ids

# Function to train images
def TrainImages():
    """
    Trains the face recognition model using images in the specified folder
    and saves the trained model.
    """
    # Initialize the LBPH face recognition model
    model = cv2.face.LBPHFaceRecognizer_create()

    # Directory containing training images
    training_images_path = "captured_images"

    # Check if the 'lbph_model.yml' exists, and delete it if it does
    trainer_file = "lbph_model.yml"
    if os.path.exists(trainer_file):
        os.remove(trainer_file)
        print(f"Existing model '{trainer_file}' deleted.")

    # Load images and corresponding IDs
    faces, Ids = getImagesAndLabels(training_images_path)

    # Check if faces and IDs were successfully loaded
    if not faces or not Ids:
        print("No captured images found or error in processing images.")
        return

    # Train the model with faces and IDs
    model.train(faces, np.array(Ids))

    # Save the trained model to a file
    model.save(trainer_file)
    print(f"Images trained successfully and model saved as '{trainer_file}'")

# Run the training function
TrainImages()










































# import os
# import cv2
# import numpy as np
# from PIL import Image

# # Function to extract images and labels
# def getImagesAndLabels(path):
#     """
#     Reads images from the specified path, converts them to grayscale,
#     and extracts IDs and face data.
#     """
#     # Get the path of all files in the folder
#     imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('jpg', 'png', 'jpeg'))]

#     # Initialize lists for faces and IDs
#     faces = []
#     Ids = []

#     # Loop through all image paths
#     for imagePath in imagePaths:
#         try:
#             # Load the image and convert it to grayscale
#             pilImage = Image.open(imagePath).convert('L')

#             # Convert the PIL image into a numpy array
#             imageNp = np.array(pilImage, 'uint8')

#             # Extract the ID from the image filename (assumes format "101--Rayhan_1.jpg")
#             filename = os.path.split(imagePath)[-1]  # Extracts "101--Rayhan_1.jpg"
#             filename_parts = filename.split("--")  # Splits into ['101', 'Rayhan_1.jpg']
            
#             if len(filename_parts) >= 2:
#                 Id = int(filename_parts[0])  # Extract the numeric ID
#             else:
#                 print(f"Skipping file {filename}: Incorrect naming format.")
#                 continue

#             # Append the face and ID to their respective lists
#             faces.append(imageNp)
#             Ids.append(Id)
#             print(f"Processed: {filename} | ID: {Id}")  # Debugging output

#         except Exception as e:
#             print(f"Error processing file {imagePath}: {e}")

#     return faces, Ids

# # Function to train images
# def TrainImages():
#     """
#     Trains the face recognition model using images in the specified folder
#     and saves the trained model.
#     """
#     # Initialize the LBPH face recognition model
#     model = cv2.face.LBPHFaceRecognizer_create()

#     # Directory containing training images
#     training_images_path = "captured_images"

#     # Check if the 'lbph_model.yml' exists, and delete it if it does
#     trainer_file = "lbph_model.yml"
#     if os.path.exists(trainer_file):
#         os.remove(trainer_file)
#         print(f"Existing model '{trainer_file}' deleted.")

#     # Load images and corresponding IDs
#     faces, Ids = getImagesAndLabels(training_images_path)

#     # Check if faces and IDs were successfully loaded
#     if not faces or not Ids:
#         print("No captured images found or error in processing images.")
#         return

#     # Train the model with faces and IDs
#     model.train(faces, np.array(Ids))

#     # Save the trained model to a file
#     model.save(trainer_file)
#     print(f"Images trained successfully and model saved as '{trainer_file}'")

# # Run the training function
# TrainImages()

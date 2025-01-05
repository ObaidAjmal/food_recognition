import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import re
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# Set the paths to your dataset and model
dataset_path = "F:\\food_recognition\\fruits-360\\Training"
model_path = "F:\\food_recognition\\modal\\trained_model.h5"

# Set the parameters for training
image_size = (224, 224)
batch_size = 32
num_epochs = 10

# Get the list of food categories from the dataset folder names
food_categories = sorted(os.listdir(dataset_path))

# Initialize the label encoder
label_encoder = LabelEncoder()

# Create an image data generator
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

# Check if the model already exists
if os.path.exists(model_path):
    print("Loading the pre-trained model...")
    model = keras.models.load_model(model_path)
else:
    print("Training the model...")

    # Create a generator for loading the training data
    train_generator = data_generator.flow_from_directory(
        dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        subset='training'
    )

    # Get the label mappings from the generator
    label_mappings = train_generator.class_indices

    # Convert the label mappings to integer labels
    integer_labels = label_encoder.fit_transform(list(label_mappings.keys()))

    # Create a new generator for validation data
    validation_generator = data_generator.flow_from_directory(
        dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        subset='validation'
    )

    # Load the MobileNetV2 pre-trained model without the top layer
    base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

    # Build the model architecture
    model = keras.models.Sequential()
    model.add(base_model)
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(len(food_categories), activation="softmax"))

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train the model using the generator
    model.fit(
        train_generator,
        epochs=num_epochs,
        validation_data=validation_generator
    )

    # Save the trained model
    model.save(model_path)

print("Model training completed!")


# Perform food recognition on a new image
image_path = "F:\\food_recognition\\fruits-360\\Test\\Cauliflower\\37_100.jpg"
new_image = Image.open(image_path)
new_image = new_image.resize(image_size)
new_image = img_to_array(new_image)
new_image = preprocess_input(new_image)
new_image = np.expand_dims(new_image, axis=0)
predictions = model.predict(new_image)
predicted_category = food_categories[np.argmax(predictions)]

# Print the predicted category
print("Predicted category:", predicted_category)

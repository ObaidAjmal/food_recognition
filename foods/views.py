

from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from PIL import Image
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set the paths to your dataset and model
dataset_path = os.path.join(settings.BASE_DIR, "fruits-360/Training")
model_path = os.path.join(settings.BASE_DIR, "modal/trained_model.h5")

# Set the parameters for training
image_size = (224, 224)
batch_size = 32
num_epochs = 25

# Get the list of food categories from the dataset folder names
food_categories = sorted(os.listdir(dataset_path))

# Initialize the label encoder
label_encoder = LabelEncoder()

# Create an image data generator with data augmentation
train_data_generator = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create a generator for loading the training data with data augmentation
train_generator = train_data_generator.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

# Check if the model already exists
if os.path.exists(model_path):
    print("Loading the pre-trained model...")
    model = keras.models.load_model(model_path)
else:
    print("Training the model...")

    # Create a generator for loading the training data with data augmentation
    train_generator = train_data_generator.flow_from_directory(
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

    # Create a generator for validation data without data augmentation
    validation_generator = train_data_generator.flow_from_directory(
        dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        subset='validation'
    )

    # Build a more complex CNN model with additional hidden layers
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(len(food_categories), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Set up callbacks for early stopping and learning rate reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    # Train the model using the generator
    model.fit(
        train_generator,
        epochs=num_epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr]
    )

    # Save the trained model
    model.save(model_path)

def predict_food_category(image_path):
    new_image = Image.open(image_path)
    new_image = new_image.resize(image_size)
    new_image = img_to_array(new_image) / 255.0  # Normalize the pixel values
    new_image = np.expand_dims(new_image, axis=0)
    predictions = model.predict(new_image)
    predicted_category = food_categories[np.argmax(predictions)]
    return predicted_category

@csrf_exempt
def home(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        upload_path = os.path.join(settings.MEDIA_ROOT, image.name)
        with open(upload_path, 'wb') as file:
            for chunk in image.chunks():
                file.write(chunk)
        predicted_category = predict_food_category(upload_path)
        return render(request, 'result.html', {'image_url': image.name,
                                               'predicted_category': predicted_category})
    return render(request, 'home.html')

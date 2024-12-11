#Initial network from https://www.tensorflow.org/tutorials/images/classification
#Transfer learning from https://www.tensorflow.org/tutorials/images/transfer_learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import os

# Paths to training and validation directories
dataset_dir = r'ASL_Alphabet_Dataset/asl_alphabet_train'
model_path = 'asl_image_classifier_mobilenetv2_480x480.h5'

# Load dataset and split it into training and validation
full_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    image_size=(224, 224),
    batch_size=32
)

# Extract class names before dataset transformations
class_names = full_dataset.class_names

# Define split sizes
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size

# Split dataset into training and validation
train_dataset = full_dataset.take(train_size)
val_dataset = full_dataset.skip(train_size)

# Optimize dataset performance (shuffle, prefetch, etc.)
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.shuffle(buffer_size=256).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# Print dataset summary
print(f"Training batches: {len(train_dataset)}")
print(f"Validation batches: {len(val_dataset)}")

# Normalize pixel values to [-1, 1] for MobileNetV2
normalization_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# Load existing model if available, otherwise create a new one
if os.path.exists(model_path):
    print("Loading existing model...")
    model = load_model(model_path)
else:
    print("Creating a new model...")
    # Initialize MobileNetV2 as the base model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze the base model

    # Create the new model
    model = Sequential([
        base_model,  # MobileNetV2 feature extractor
        GlobalAveragePooling2D(),  # Pool the features into a single vector
        Dense(128, activation='relu'),  # Dense layer for classification
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(class_names), activation='softmax')  # Output layer for number of classes
    ])

# Compile the model (even for saved model)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(train_dataset, epochs=20, validation_data=val_dataset)

# Evaluate the model
loss, accuracy = model.evaluate(val_dataset)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
model.save(model_path)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

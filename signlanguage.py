#Initial network from https://www.tensorflow.org/tutorials/images/classification
#Transfer learning from https://www.tensorflow.org/tutorials/images/transfer_learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
# Path to the folder containing subfolders of images (e.g., 'dataset/train' and 'dataset/val')
train_dir = r'ASL_Alphabet_Dataset\asl_alphabet_train'
val_dir = r'ASL_Alphabet_Dataset\asl_alphabet_test'

# Load the dataset from the directory Dataset from https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset/code
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),  # Resize images to 224x224 for MobileNetV2
    batch_size=32
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(224, 224),
    batch_size=32
)
class_names = train_dataset.class_names
# Normalize pixel values to [-1, 1] for MobileNetV2
normalization_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

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

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Evaluate the model
loss, accuracy = model.evaluate(val_dataset)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
model.save('asl_image_classifier_mobilenetv2_480x480.h5')

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

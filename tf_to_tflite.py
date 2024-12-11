import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
import numpy as np

# Load your trained model
model = tf.keras.models.load_model('asl_image_classifier_mobilenetv2_480x480.h5')

# Paths to training and validation directories
dataset_dir = r'ASL_Alphabet_Dataset/asl_alphabet_train'
model_path = 'asl_image_classifier_mobilenetv2_480x480.h5'

# Load dataset and split it into training and validation
full_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    image_size=(224, 224),
    batch_size=32
)

# Define size of training dataset (from signlanguage.py)
train_size = int(0.9 * len(full_dataset))

# Take data from dataset for quantization
train_dataset = full_dataset.take(train_size)

# Define a representative dataset function with parallel data processing
def representative_data_gen():
    for image_batch, _ in train_dataset.take(100):  # Adjust the number of batches if necessary
        # The model expects inputs of shape (batch_size, 224, 224, 3), so ensure they are normalized.
        # Normalize the images the same way you did during training:
        image_batch = tf.keras.applications.mobilenet_v2.preprocess_input(image_batch)
        
        # Parallelize data loading using tf.data API's `map` function
        # The `map` function applies a transformation in parallel to the dataset.
        # This will preprocess the images in parallel for efficiency
        yield [image_batch]

# Convert the model to TensorFlow Lite format with UINT8 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set the optimization strategy and target format to UINT8
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.representative_dataset = representative_data_gen

# Perform the conversion
tflite_model_quantized = converter.convert()

# Save the quantized model
with open('my_sign_language_model_quantized_uint8.tflite', 'wb') as f:
    f.write(tflite_model_quantized)

print("Model successfully converted and saved as UINT8 TFLite model.")


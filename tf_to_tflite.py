#https://github.com/oleksandr-g-rock/how_to_convert_h5_model_to_tflite/blob/main/How_to_convert_h5_model_keras_tflite.ipynb

import tensorflow as tf
model = tf.keras.models.load_model('asl_image_classifier.h5')
# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_quantized = converter.convert()

# Save the quantized model
open('my_sign_language_model_quantized.tflite', 'wb').write(tflite_model_quantized)
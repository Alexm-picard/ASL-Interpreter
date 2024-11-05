import tensorflow as tf
model = tf.keras.models.load_model('asl_image_classifier.h5')
# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model_quantized = converter.convert()

# Save the quantized model
with open('my_sign_language_model_quantized.tflite', 'wb') as f:
    f.write(tflite_model_quantized)
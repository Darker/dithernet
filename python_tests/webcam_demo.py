from build_ditherer_network import build_ditherer_network
import color_choices
import argparse
import cv2
import numpy as np
import tensorflow as tf
import output_convert
import input_convert
import filename_utils
import math
import os


input_size = (64, 64)  # Size of the input image to the network
# height, width
output_size = (64, 64)  # Size of the output dithering pattern

network_model = build_ditherer_network((64, 64), (64, 64))

network_model = build_ditherer_network((64, 64), (64, 64))

weights_name = filename_utils.format_in_dimensions(input_size, output_size)
weights_path = f"checkpoints/color_decider_latest.{weights_name}.weights.h5"
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Weight file {weights_path} does not exist. Please train the model first or provide a valid path.")
# load weights
network_model.load_weights(weights_path)
color_choices_matrix = color_choices.get_color_choices(output_size)
color_choices_batch = np.array([color_choices_matrix])  

cap = cv2.VideoCapture(0)  # 0 = Default webcam, change for external cameras

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if frame couldn't be read

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Resize the frame to match the input shape of your model
    input_frame = input_convert.scale_input(frame, input_size, zoom=1.0)
    input_frame = input_convert.center_crop(input_frame, input_size) 
    input_frame = np.expand_dims(input_frame/255.0, axis=0)

    # Run inference through TensorFlow model
    network_output = network_model.predict([color_choices_batch, input_frame], batch_size=1)
    dither_output = output_convert.to_image_tf_no_learn(network_output, color_choices_batch, output_size)
    dither_output = tf.squeeze(dither_output, axis=0) 
    image_tensor = tf.image.convert_image_dtype(dither_output, dtype=tf.uint8) 
    upscaled_tensor = tf.image.resize(image_tensor, 
                                    size=(image_tensor.shape[0] * 6, image_tensor.shape[1] * 6), 
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    numpy_img = upscaled_tensor.numpy().astype(np.uint8)
    opencv_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)

    # Display the processed output
    cv2.imshow("Processed Output", opencv_img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
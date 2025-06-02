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

# Create argument parser
parser = argparse.ArgumentParser(description="Parse command-line arguments.")

# Add optional flags with values
parser.add_argument("--zoom", type=float, help="How much to zoom in on the image", default=1.0)
parser.add_argument("--left", type=float, help="Left offset for cropping as fraction", default=0.5)
parser.add_argument("--top", type=float, help="Top offset for cropping as fraction", default=0.5)
parser.add_argument("--upscale", type=int, help="Upscale result that many times (nearest neighbor)", default=1)

# Positional argument (last argument)
parser.add_argument("path", type=str, help="The last positional argument (file path)")

# Parse the arguments
args = parser.parse_args()

input_size = (64, 64)  # Size of the input image to the network
# height, width
output_size = (64, 64)  # Size of the output dithering pattern

network_model = build_ditherer_network((64, 64), (64, 64))

postprocess_upscale:int = args.upscale

weights_name = filename_utils.format_in_dimensions(input_size, output_size)
weights_path = f"checkpoints/color_decider_latest.{weights_name}.weights.h5"
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Weight file {weights_path} does not exist. Please train the model first or provide a valid path.")
# load weights
network_model.load_weights(weights_path)

img = cv2.imread(args.path)  # Read image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

img = input_convert.scale_input(img, input_size, zoom=args.zoom)  # Scale image to fit the input size

croppable_height, croppable_width = img.shape[:2]
croppable_height = croppable_height - output_size[0]
croppable_width = croppable_width - output_size[1]
left_px = int(croppable_width * args.left)
top_px = int(croppable_height * args.top)

cropped_img = img[top_px:top_px + output_size[0], left_px:left_px + output_size[1]]
assert cropped_img.shape[:2] == output_size, f"Cropped image size {cropped_img.shape[:2]} does not match expected output size {output_size}"
print(f"Cropped image shape: {cropped_img.shape}")
color_choices_matrix = color_choices.get_color_choices(output_size)
print(f"Color choices matrix shape: {color_choices_matrix.shape}")

img_batch = np.expand_dims(cropped_img/255.0, axis=0)  # Add batch dimension

color_choices_batch = np.array([color_choices_matrix])  
print(f"Color choices batch shape: {color_choices_batch.shape}")
# Get final output from the second network
network_output = network_model.predict([color_choices_batch, img_batch], batch_size=1)
dither_output = output_convert.to_image_tf_no_learn(network_output, color_choices_batch, output_size)
dither_output = tf.squeeze(dither_output, axis=0)  # Remove batch dimension
input_img_to_save = cropped_img
if postprocess_upscale > 1:
    dither_output = tf.image.resize(dither_output, 
                                    size=(dither_output.shape[0] * postprocess_upscale, 
                                          dither_output.shape[1] * postprocess_upscale), 
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_img_to_save = tf.image.resize(input_img_to_save, 
                                    size=(input_img_to_save.shape[0] * postprocess_upscale, 
                                          input_img_to_save.shape[1] * postprocess_upscale), 
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_img_to_save = tf.image.convert_image_dtype(input_img_to_save, dtype=tf.uint8)

image_tensor = tf.image.convert_image_dtype(dither_output, dtype=tf.uint8)  # Convert to uint8

encoded_png = tf.io.encode_png(image_tensor)
output_name = os.path.basename(args.path)
output_name = os.path.splitext(output_name)[0]  # Remove file extension
output_filename = f"{output_name}.png"
tf.io.write_file(output_filename, encoded_png)

# Save the input as well
input_filename = f"{output_name}_input.png"
encoded_png_input = tf.io.encode_png(input_img_to_save)
tf.io.write_file(input_filename, encoded_png_input)
# cv2.imwrite(input_filename, cv2.cvtColor(input_img_to_save, cv2.COLOR_RGB2BGR))
print(f"Output saved to {output_filename}")
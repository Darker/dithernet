import math
import tensorflow as tf
from datetime import datetime, UTC
from typing import TYPE_CHECKING, TypedDict

import PicturesLRU
from build_ditherer_network import build_ditherer_network
if TYPE_CHECKING:
    import tensorflow.python.keras as keras
    import tensorflow.python.keras.layers as layers
    keras.layers = layers
else:
    from tensorflow import keras
# import tensorflow.python.keras as keras
# import tensorflow.python.keras.layers as layers
# import tensorflow.python.keras.losses as losses
# import tensorflow.python.keras.optimizers as optimizers
import numpy as np
import os
import random
import cv2
import json

# tf.test.is_gpu_available()
# tf.config.set_soft_device_placement(False)
# with tf.device('/device:GPU:0'): # Or just /GPU:0
#     a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#     b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#     c = tf.matmul(a, b)
# print("Matrix multiplication result on GPU:\n", c.numpy())
# print("\nGPU seems to be working!")

# Define input shape
INPUT_SHAPE = (64, 64, 3)  # Example: 32x32 RGB images
OUTPUT_DIMMS = (64, 64)
NUM_COLORS = 4  # The four color choices per pixel


# Future plans:
# This should get some image classes describing roughly the nature of the image
# It should then generate a matrix of 4 colour chouces per pixel that are good for displaying such images
def build_pallete_generator():
    raise NotImplementedError("This function is not implemented yet.")
    model = keras.Sequential([
        keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=INPUT_SHAPE),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(NUM_COLORS * INPUT_SHAPE[0] * INPUT_SHAPE[1], activation='softmax')  # Probability for 4 colors per pixel
    ])
    return model

if not os.path.exists("./input_folders.json"):
    print("No input folders found. Please create a file named 'input_folders.json' with the list of image directories.")
    exit(1)
if not os.path.exists("./config.json"):
    print("Config not found. Should contain \{\"demo_image\":path to\}")
    exit(1)
image_dirs = json.load(open("./input_folders.json", "r"))  # Load image directories from JSON

class DebugConfig(TypedDict):
    demo_image: str  # Path to a demo image for testing

debug_config:DebugConfig = json.load(open("./config.json", "r"))  # Load debug config

def get_random_image_path():
    """Loads a random image from one of the folders."""
    chosen_dir = random.choice(image_dirs)  # Pick a random folder
    images = [x for x in os.listdir(chosen_dir) if x.lower().endswith(".jpg")]  # List images
    img_path = os.path.join(chosen_dir, random.choice(images))  # Pick a random image
    return img_path

def load_image(img_path: str, max_size=1024):
    """Loads an image and resizes it to a maximum size."""
    img = cv2.imread(img_path)  # Read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    h, w, _ = img.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))  # Resize image
    return img

raw_images_lru = PicturesLRU.PicturesLRU(capacity=100, picture_loader=lambda img_path: load_image(img_path, max_size=1024))

# def resample_image(image, target_size=(32, 32)):

def preprocess_image(img_path, downsample_factor=2, target_size=(32, 32)):
    """Loads, downsamples, and crops the image.
        TODO: caching? Especially since photos are on HDD?
    """
    img = cv2.imread(img_path)  # Read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    # Calculate new size so that smallest dimension is same as target size
    h, w, _ = img.shape
    if h < w:
        new_h = target_size[0]
        new_w = int(w * (new_h / h))
    else:
        new_w = target_size[0]
        new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_w, new_h))  # Resize to maintain aspect ratio
    #img = cv2.resize(img, (64, 64))  # Downsample (adjust size)
    
    # Random Crop (Center Crop for simplicity)
    crop_size = target_size[0]  # Assume square crop
    h, w, _ = img.shape
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    img = img[start_y:start_y+crop_size, start_x:start_x+crop_size]  # Crop
    
    img = img / 255.0  # Normalize
    return img

def preprocess_image_w_params(img_path):
    """Preprocesses an image with given parameters."""
    return preprocess_image(img_path, downsample_factor=2, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]))
images_lru = PicturesLRU.PicturesLRU(capacity=100, picture_loader=preprocess_image_w_params)


def data_generator(batch_size=32):
    """Yields batches of processed images."""

    while True:
        batch_images = []
        for _ in range(batch_size):
            img_path = get_random_image_path()
            img = images_lru.get(img_path)
            batch_images.append(img)
        
        yield np.array(batch_images)

# Create the models
# color_selector = build_color_selector()
color_decider = build_ditherer_network(INPUT_SHAPE[0:2], OUTPUT_DIMMS)
if os.path.exists("./checkpoints/color_decider_latest.weights.h5"):
    print("Loading existing weights for color_decider...")
    color_decider.load_weights("./checkpoints/color_decider_latest.weights.h5")
    print("Weights loaded successfully.")
else:
    print("No existing weights found for color_decider. Starting from scratch.")

# To make the next part readable
color_map = {
    "cyan": [0, 255, 255],  
    "magenta": [255, 0, 255],
    "yellow": [255, 255, 0],
    "black": [0, 0, 0],     
    "white": [255, 255, 255],
    "green": [0, 255, 0],   
    "red": [255, 0, 0],     
    "blue": [0, 0, 255],    
    "orange": [255, 165, 0],
    "purple": [128, 0, 128],
    "lightblue": [0, 0x85, 0xff],
    "gray": [128, 128, 128],
    "darkblue": [0, 0, 80],
    "darkgreen": [0, 100, 0],
    "lightgreen": [144, 238, 144],
    "brown": [165, 42, 42],
    "lightred": [255, 182, 193],
    "pink": [255, 192, 203],
    "darkred": [139, 0, 0],
}

# normalize color map values to [0, 1] range
color_map = {k: np.array(v) / 255.0 for k, v in color_map.items()}

# Todo: JSON?
pixel_option_1 = np.array([
    color_map["cyan"],
    color_map["magenta"],
    color_map["yellow"],
    color_map["black"]
])

pixel_option_2 = np.array([
    color_map["blue"],
    color_map["darkgreen"],
    color_map["red"],
    color_map["white"]
])

pixel_option_3 = np.array([
    color_map["green"],
    color_map["orange"],
    color_map["lightred"],
    color_map["lightblue"]
])

pixel_option_4 = np.array([
    color_map["green"],
    color_map["yellow"],
    color_map["blue"],
    color_map["brown"]
])

pixel_option_5 = np.array([
    color_map["red"],
    color_map["darkblue"],
    color_map["cyan"],
    color_map["purple"]
])
pixel_option_6 = np.array([
    color_map["lightred"],
    color_map["darkgreen"],
    color_map["pink"],
    color_map["gray"]
])
pixel_option_7 = np.array([
    color_map["yellow"],
    color_map["darkred"],
    color_map["lightblue"],
    color_map["green"]
])

pixel_option_8 = np.array([
    color_map["darkblue"],
    color_map["lightgreen"],
    color_map["orange"],
    color_map["red"]
])

pixel_option_9 = np.array([
    color_map["darkgreen"],
    color_map["yellow"],
    color_map["purple"],
    color_map["cyan"]
])

all_pixel_options = [
    pixel_option_1,
    pixel_option_2,
    pixel_option_3,
    pixel_option_4,
    pixel_option_5,
    pixel_option_6,
    pixel_option_7,
    pixel_option_8,
    pixel_option_9
]
options_count = len(all_pixel_options)

# first run with preset color choices -  CMYK
color_choices = np.array([
    all_pixel_options[xcolor%options_count] for xcolor in range(OUTPUT_DIMMS[0]*OUTPUT_DIMMS[1])
])

def data_generator_with_color_choices(batch_size=32):
    """Yields batches of processed images with color choices."""

    images_generator = data_generator(batch_size)
    while True:
        batch_color_choices = [color_choices for _ in range(batch_size)]
        
        yield np.array(batch_color_choices), np.array(next(images_generator))

def softargmax(x, beta=1e10):
  x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
  return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)

def gumbel_softmax(logits, temperature=0.2):
    """
    (generated by AI)
    Computes differentiable discrete selections using Gumbel-Softmax.

    Args:
        logits: Tensor of shape (..., num_classes) representing unnormalized predictions.
        temperature: Controls randomness (lower = sharper selections).

    Returns:
        Differentiable one-hot encoded selections.
    """
    # Sample Gumbel noise
    gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), minval=0.0, maxval=1.0)))

    # Add Gumbel noise and apply softmax
    sampled_probs = tf.nn.softmax((logits + gumbel_noise) / temperature)

    return sampled_probs

def convert_to_image_tf(output_matrices, color_choices_batch):
    """Maps network outputs to RGB colors using predefined choices in TensorFlow.
    Args:
        output_matrices: Tensor of shape (batch_size, height*width, 4) with network outputs.
        color_choices_batch: Tensor of shape (batch_size, height*width, 4, 3) with color choices.

    The outputs should be mapped to a flat array of indices (0-3) for each pixel,
    which will then be used to select the corresponding RGB color from color_choices_batch.

    Final output shape will be (batch_size, height, width, 3) - last dimension being RGB colors.
    """
    batch_size, num_pixels, num_choices_per_px, num_colors = color_choices_batch.shape  # Extract dimensions
    print("output_matrices dimensions: " + str(output_matrices.shape))
    print("color_choices_batch dimensions: " + str(color_choices_batch.shape))

    # This solution was too blurry and taught the network to blend the colors instead of choosing one
    #  softmax_output = tf.nn.softmax(output_matrices, axis=-1)  # Shape: (batch_size, height*width, 4)
    # This just plain didn't work
    #  softmax_output = softargmax(output_matrices, beta=1e7)  # Shape: (batch_size, height*width, 4)
    softmax_output = gumbel_softmax(output_matrices, temperature=0.1)  # Shape: (batch_size, height*width, 4)
    print("softmax_output shape: " + str(softmax_output.shape))
    chosen_colors_soft = tf.reduce_sum(softmax_output[..., tf.newaxis] * color_choices_batch, axis=2)  # Shape: (batch_size, height*width, 3)
    print("chosen_colors_soft shape: " + str(chosen_colors_soft.shape))

    mapped_images = tf.reshape(chosen_colors_soft, (batch_size, OUTPUT_DIMMS[0], OUTPUT_DIMMS[1], 3), name="mapped_image_reshaped")  # Reshape to (height, width, 3)

    return mapped_images


def convert_to_image_tf_no_learn(output_matrices, color_choices_batch):
    '''
    This function does the same as convert_to_image_tf, but the operation is
    truly discrete and does not allow gradients to flow through it.
    '''
    batch_size, num_pixels, num_choices_per_px, num_colors = color_choices_batch.shape  # Extract dimensions
    chosen_colors = tf.argmax(output_matrices, axis=2, output_type=tf.int32)
    mapped_images = tf.gather(color_choices_batch, chosen_colors, batch_dims=2, axis=2, name="mapped_image_before_reshape")  # Shape: (batch_size, height*width, 3)
    mapped_images = tf.reshape(mapped_images, (batch_size, OUTPUT_DIMMS[0], OUTPUT_DIMMS[1], 3))  # Reshape to (height, width, 3)
    return mapped_images

train_data = tf.data.Dataset.from_generator(
    lambda: data_generator_with_color_choices(32), 
    output_types=(tf.float32, tf.float32),  # Two inputs: color choices & image data
    output_shapes=((None, OUTPUT_DIMMS[0]*OUTPUT_DIMMS[1], 4, 3), (None, INPUT_SHAPE[0], INPUT_SHAPE[1], 3))  # First is color choices, second is image data
)

loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.0001, 
                                  weight_decay=0.00001)

# Custom training loop
@tf.function
def train_step(model, image_batch, color_choices_batch):
    for var in model.trainable_variables:
        print(f"Variable: {var.name}, Shape: {var.shape}, Trainable: {var.trainable}")

    with tf.GradientTape() as tape:

        raw_outputs = model((color_choices_batch, image_batch))  # Get predictions

        generated_images = convert_to_image_tf(raw_outputs, color_choices_batch)
        # print("Generated images tensor:", generated_images)
        blurred_generated_images = tf.nn.avg_pool2d(generated_images, ksize=2, strides=1, padding="SAME")
        # This is not available but would be cool. Probably can be done via generic convolution function
        #  blurred_generated_images = tf.vision.augment.gaussian_filter2d(generated_images, ksize=4, strides=1, padding="SAME")
        # Stopped blurring originals because they are already kinda blurry considering how downscaled they are
        #   blurred_original_images = tf.nn.avg_pool2d(image_batch, ksize=5, strides=1, padding="SAME")

        # # Compute loss
        loss = loss_fn(tf.cast(image_batch, tf.float32), tf.cast(blurred_generated_images, tf.float32))

        # watched_vars = tape.watched_variables()
        # for var in watched_vars:
        #     print(f"Watched Variable: {var.name}, Shape: {var.shape}, Trainable: {var.trainable}")



    # Compute and apply gradients
    grads = tape.gradient(loss, model.trainable_variables)
    if grads is None or all(g is None for g in grads):
        raise ValueError("No gradients found! Model outputs may not depend on trainable variables.")
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def store_output_samples(img_path):
    img = images_lru.get(img_path) 
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Get color choices from the first network
    # color_choices = color_selector.predict(img)
    # color_choices = np.reshape(color_choices, (1, INPUT_SHAPE[0], INPUT_SHAPE[1], NUM_COLORS, 3))
    color_choices_batch = np.array([color_choices])  # Use the predefined color choices
    # Get final output from the second network
    network_output = color_decider.predict([color_choices_batch, img], batch_size=1)
    dither_output = convert_to_image_tf_no_learn(network_output, color_choices_batch)
    dither_output = tf.squeeze(dither_output, axis=0)  # Remove batch dimension

    

    # Ensure pixel values are between [0, 255]
    image_tensor = tf.image.convert_image_dtype(dither_output, dtype=tf.uint8)  # Convert to uint8
    # Encode as PNG
    encoded_png = tf.io.encode_png(image_tensor)

    what_network_saw_btach = convert_to_image_tf(network_output, color_choices_batch)
    what_network_saw = tf.squeeze(what_network_saw_btach, axis=0)  # Remove batch dimension

    encoded_what_network_saw = tf.io.encode_png(tf.image.convert_image_dtype(what_network_saw, dtype=tf.uint8))

    blurred_dither_output = tf.nn.avg_pool2d(what_network_saw_btach, ksize=4, strides=1, padding="SAME")
    blurred_dither_output = tf.squeeze(blurred_dither_output, axis=0) 
    image_tensor_blurred = tf.image.convert_image_dtype(blurred_dither_output, dtype=tf.uint8)  # Convert to uint8
    encoded_blurred_dither_outp_png = tf.io.encode_png(image_tensor_blurred)
    tf.io.write_file("test_output_blurred.png", encoded_blurred_dither_outp_png)


    # Save to file
    tf.io.write_file("test_output.png", encoded_png)

    os.makedirs("./output_history", exist_ok=True)
    output_timestamp = datetime.now(UTC).replace(microsecond=0).isoformat(timespec='seconds').replace(":","-").replace("+00-00", "")

    tf.io.write_file(os.path.join("./output_history", "test_output_"+output_timestamp+".png"), encoded_png)
    encoded_input = tf.io.encode_png(tf.cast(img[0] * 255, tf.uint8))  # Convert input image to uint8
    tf.io.write_file("test_input.png", encoded_input)
    tf.io.write_file("test_what_network_saw.png", encoded_what_network_saw)
    return dither_output

try:
    # Training loop
    for epoch in range(300):
        store_output_samples(debug_config["demo_image"])

        for step, (color_choices_batch, image_batch) in enumerate(train_data.take(200)):  
            loss = train_step(color_decider, image_batch, color_choices_batch)
            print(f"Epoch {epoch}, Step {step}, Loss: {loss}")

        if epoch % 10 == 0:  # Save every 10 epochs
            color_decider.save_weights("./checkpoints/color_decider_latest.weights.h5".format(epoch))
            print(f"Checkpoint saved for epoch {epoch}")
except KeyboardInterrupt:
    print("Training interrupted. Saving final weights...")
    color_decider.save_weights("./checkpoints/color_decider_latest.weights.h5")
    print("Final weights saved.")
    store_output_samples(debug_config["demo_image"])
    #predict_image("/mnt/d/JAKUB/images/photos/eliska-portrety/results/DSC_6919.jpg")


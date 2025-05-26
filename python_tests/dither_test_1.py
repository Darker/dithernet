import tensorflow as tf
from typing import TYPE_CHECKING
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

# Define input shape
INPUT_SHAPE = (32, 32, 3)  # Example: 32x32 RGB images
NUM_COLORS = 4  # The four color choices per pixel


# First network: Color selector
def build_color_selector():
    model = keras.Sequential([
        keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=INPUT_SHAPE),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(NUM_COLORS * INPUT_SHAPE[0] * INPUT_SHAPE[1], activation='softmax')  # Probability for 4 colors per pixel
    ])
    return model

# Second network: Color decision maker
def build_color_decider():
    # First input: Matrix with 4 values per pixel
    color_choices = keras.Input(shape=(INPUT_SHAPE[0] * INPUT_SHAPE[1], 4, 3), name="ColorChoices")

    # Second input: Normal RGB image
    image_data = keras.Input(shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], 3), name="ImageData")

    # Process color choices matrix
    x1 = keras.layers.Conv2D(32, (3,3), activation="relu")(color_choices)
    x1 = keras.layers.Flatten()(x1)

    # Process image data matrix
    x2 = keras.layers.Conv2D(32, (3,3), activation="relu")(image_data)
    x2 = keras.layers.Flatten()(x2)

    merged = keras.layers.Concatenate()([x1, x2])
    merged = keras.layers.Dense(128, activation="relu")(merged)
    # One number per pixel, representing the chosen color
    output = keras.layers.Dense(INPUT_SHAPE[0] * INPUT_SHAPE[1]*4, activation='linear')(merged)  # Example output layer
    output = keras.layers.Reshape((INPUT_SHAPE[0] * INPUT_SHAPE[1], 4))(output)
    model = keras.Model(inputs=[color_choices, image_data], outputs=output)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.summary()
    # model = keras.Sequential([
    #     layers.Input(shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], 4)),
    #     layers.Conv2D(INPUT_SHAPE[0], (3,3), activation='relu', input_shape=INPUT_SHAPE),
    #     layers.Conv2D(64, (3,3), activation='relu'),
    #     layers.Flatten(),
    #     layers.Dense(1024, activation='relu'),
    #     layers.Dense(INPUT_SHAPE[0] * INPUT_SHAPE[1] * 3, activation='linear')  # Final RGB values
    # ])
    return model

image_dirs = ["D:\\JAKUB\\images\\photos\\eliska-portrety\\results"]  # Example folders

def load_random_image():
    """Loads a random image from one of the folders."""
    chosen_dir = random.choice(image_dirs)  # Pick a random folder
    images = os.listdir(chosen_dir)  # List images
    img_path = os.path.join(chosen_dir, random.choice(images))  # Pick a random image
    return img_path

def preprocess_image(img_path, downsample_factor=2, target_size=(32, 32)):
    """Loads, downsamples, and crops the image."""
    img = cv2.imread(img_path)  # Read image
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

def data_generator(batch_size=32):
    """Yields batches of processed images."""
    while True:
        batch_images = []
        for _ in range(batch_size):
            img_path = load_random_image()
            img = preprocess_image(img_path)
            batch_images.append(img)
        
        yield np.array(batch_images)

# Create the models
# color_selector = build_color_selector()
color_decider = build_color_decider()
if os.path.exists("./checkpoints/color_decider_latest.weights.h5"):
    print("Loading existing weights for color_decider...")
    color_decider.load_weights("./checkpoints/color_decider_latest.weights.h5")
    print("Weights loaded successfully.")

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
}

# normalize color map values to [0, 1] range
color_map = {k: np.array(v) / 255.0 for k, v in color_map.items()}

pixel_option_1 = np.array([
    color_map["cyan"],
    color_map["magenta"],
    color_map["yellow"],
    color_map["black"]
])

pixel_option_2 = np.array([
    color_map["blue"],
    color_map["black"],
    color_map["red"],
    color_map["white"]
])

pixel_option_3 = np.array([
    color_map["green"],
    color_map["orange"],
    color_map["magenta"],
    color_map["lightblue"]
])

# first run with preset color choices -  CMYK
color_choices = np.array([
    pixel_option_1 if xcolor%2==0 else pixel_option_2 for xcolor in range(INPUT_SHAPE[0]*INPUT_SHAPE[1])
])

def data_generator_with_color_choices(batch_size=32):
    """Yields batches of processed images with color choices."""

    images_generator = data_generator(batch_size)
    while True:
        batch_color_choices = [color_choices for _ in range(batch_size)]
        
        yield np.array(batch_color_choices), np.array(next(images_generator))

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
    # chosen_colors = tf.argmax(output_matrices, axis=2, output_type=tf.int32)
    # print("chosen_colors shape: " + str(chosen_colors.shape))
    # print("chosen_colors gradient: " + str(tape.gradient(output_matrices, chosen_colors)))
    # #chosen_colors = tf.expand_dims(chosen_colors, axis=-1)  # Expand to shape (batch_size, height*width, 1)
    # #print("chosen_colors expanded shape: " + str(chosen_colors.shape))
    # batch_size, num_pixels, num_choices_per_px, num_colors = color_choices_batch.shape  # Extract dimensions
    # # Gather correct colors for each pixel while ensuring correct batch-wise indexing
    # # batch_indices = tf.range(batch_size)[:, tf.newaxis]  # Shape: (batch_size, 1)
    # # pixel_indices = tf.range(num_pixels)[tf.newaxis, :]  # Shape: (1, height*width)

    # # Stack indices to correctly select from the second dimension (which has only 4 entries per pixel)
    # # stacked_indices = tf.stack([batch_indices, pixel_indices, chosen_colors], axis=-1)  # Shape: (batch_size, height*width, 3)

    # mapped_images = tf.gather(color_choices_batch, chosen_colors, batch_dims=2, axis=2, name="mapped_image_before_reshape")  # Shape: (batch_size, height*width, 3)

    # print("mapped_image shape: " + str(mapped_images.shape))
    # print("mapped_image gradient: " + str(tape.gradient(output_matrices, mapped_images)))

    softmax_output = tf.nn.softmax(output_matrices, axis=-1)  # Shape: (batch_size, height*width, 4)
    print("softmax_output shape: " + str(softmax_output.shape))
    chosen_colors_soft = tf.reduce_sum(softmax_output[..., tf.newaxis] * color_choices_batch, axis=2)  # Shape: (batch_size, height*width, 3)
    print("chosen_colors_soft shape: " + str(chosen_colors_soft.shape))
    # print("chosen_colors_soft gradient: " + str(tape.gradient(output_matrices, chosen_colors_soft)))
    mapped_images = tf.reshape(chosen_colors_soft, (batch_size, INPUT_SHAPE[0], INPUT_SHAPE[1], 3), name="mapped_image_reshaped")  # Reshape to (height, width, 3)

    return mapped_images

def convert_to_image_tf_no_learn(output_matrices, color_choices_batch):
    batch_size, num_pixels, num_choices_per_px, num_colors = color_choices_batch.shape  # Extract dimensions
    chosen_colors = tf.argmax(output_matrices, axis=2, output_type=tf.int32)
    mapped_images = tf.gather(color_choices_batch, chosen_colors, batch_dims=2, axis=2, name="mapped_image_before_reshape")  # Shape: (batch_size, height*width, 3)
    mapped_images = tf.reshape(mapped_images, (batch_size, INPUT_SHAPE[0], INPUT_SHAPE[1], 3))  # Reshape to (height, width, 3)
    return mapped_images

train_data = tf.data.Dataset.from_generator(
    lambda: data_generator_with_color_choices(32), 
    output_types=(tf.float32, tf.float32),  # Two inputs: color choices & image data
    output_shapes=((None, INPUT_SHAPE[0]*INPUT_SHAPE[1], 4, 3), (None, INPUT_SHAPE[0], INPUT_SHAPE[1], 3))  # First is color choices, second is image data
)

loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Custom training loop
@tf.function
def train_step(model, image_batch, color_choices_batch):
    for var in model.trainable_variables:
        print(f"Variable: {var.name}, Shape: {var.shape}, Trainable: {var.trainable}")

    with tf.GradientTape() as tape:
        tape.watch(image_batch)
        tape.watch(color_choices_batch)
        batch_size = image_batch.shape[0]
        raw_outputs = model((color_choices_batch, image_batch))  # Get predictions
        tape.watch(raw_outputs)
        # loss = loss_fn(tf.cast(tf.random.uniform((32,1024,4)), tf.float32), tf.cast(raw_outputs, tf.float32))
        generated_images = convert_to_image_tf(raw_outputs, color_choices_batch)
        print("Generated images tensor:", generated_images)
        blurred_generated_images = tf.nn.avg_pool2d(generated_images, ksize=5, strides=1, padding="SAME")
        # blurred_original_images = tf.nn.avg_pool2d(image_batch, ksize=5, strides=1, padding="SAME")
        # # # Apply blur to both generated and original images

        # gradient test
        # image_gen_gradient = tape.gradient(raw_outputs, generated_images)
        # print("Image generation gradient shape:", image_gen_gradient.shape)
        # image_gen_blur_gradient = tape.gradient(raw_outputs, blurred_generated_images)
        # print("Image blur gradient shape:", image_gen_blur_gradient.shape)

        # # Compute loss
        loss = loss_fn(tf.cast(image_batch, tf.float32), tf.cast(blurred_generated_images, tf.float32))

        watched_vars = tape.watched_variables()
        for var in watched_vars:
            print(f"Watched Variable: {var.name}, Shape: {var.shape}, Trainable: {var.trainable}")



    # Compute and apply gradients
    grads = tape.gradient(loss, model.trainable_variables)
    if grads is None or all(g is None for g in grads):
        raise ValueError("No gradients found! Model outputs may not depend on trainable variables.")
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def predict_image(img_path):
    img = preprocess_image(img_path)
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Get color choices from the first network
    # color_choices = color_selector.predict(img)
    # color_choices = np.reshape(color_choices, (1, INPUT_SHAPE[0], INPUT_SHAPE[1], NUM_COLORS, 3))
    color_choices_batch = np.array([color_choices])  # Use the predefined color choices
    # Get final output from the second network
    final_output = color_decider.predict([color_choices_batch, img], batch_size=1)
    final_output = convert_to_image_tf_no_learn(final_output, color_choices_batch)
    final_output = tf.squeeze(final_output, axis=0)  # Remove batch dimension
    # Ensure pixel values are between [0, 255]
    image_tensor = tf.image.convert_image_dtype(final_output, dtype=tf.uint8)  # Convert to uint8
    
    # Encode as PNG
    encoded_png = tf.io.encode_png(image_tensor)

    # Save to file
    tf.io.write_file("test_output.png", encoded_png)
    encoded_input = tf.io.encode_png(tf.cast(img[0] * 255, tf.uint8))  # Convert input image to uint8
    tf.io.write_file("test_input.png", encoded_input)
    return final_output

try:
    # Training loop
    for epoch in range(300):
        for step, (color_choices_batch, image_batch) in enumerate(train_data.take(50)):  
            loss = train_step(color_decider, image_batch, color_choices_batch)
            print(f"Epoch {epoch}, Step {step}, Loss: {loss}")

        if epoch % 2 == 0:  # Save every 10 epochs
            color_decider.save_weights("./checkpoints/color_decider_latest.weights.h5".format(epoch))
            print(f"Checkpoint saved for epoch {epoch}")
except KeyboardInterrupt:
    print("Training interrupted. Saving final weights...")
    color_decider.save_weights("./checkpoints/color_decider_latest.weights.h5")
    print("Final weights saved.")
    predict_image("D:\\JAKUB\\images\\photos\\eliska-portrety\\results\\DSC_6919.jpg")


import tensorflow as tf

def to_image_tf_no_learn(output_matrices, color_choices_batch, dimensions_hw:tuple[int, int]) -> tf.Tensor:
    '''
    This function does the same as convert_to_image_tf, but the operation is
    truly discrete and does not allow gradients to flow through it.
    '''
    batch_size, num_pixels, num_choices_per_px, num_colors = color_choices_batch.shape  # Extract dimensions
    # output_dimensions = output_matrices.shape[1:3]  # Get output dimensions (height, width)
    # print(f"Output dimensions: {output_dimensions}, Batch size: {batch_size}, Num pixels: {num_pixels}, Num choices per pixel: {num_choices_per_px}, Num colors: {num_colors}")
    
    chosen_colors = tf.argmax(output_matrices, axis=2, output_type=tf.int32)
    mapped_images = tf.gather(color_choices_batch, chosen_colors, batch_dims=2, axis=2, name="mapped_image_before_reshape")  # Shape: (batch_size, height*width, 3)
    mapped_images = tf.reshape(mapped_images, (batch_size, dimensions_hw[0], dimensions_hw[1], 3))  # Reshape to (height, width, 3)
    return mapped_images
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tensorflow.python.keras as keras
    import tensorflow.python.keras.layers as layers
    keras.layers = layers
else:
    from tensorflow import keras

# This network gets a normal RGB image as well as per-pixel color choices and should output a suutable dithering pattern
def build_ditherer_network(input_wh: tuple[int, int], 
                           output_wh: tuple[int, int]):

    # First input: Matrix with 4 values per pixel
    color_choices = keras.Input(shape=(output_wh[0] * output_wh[1], 4, 3), name="ColorChoices")

    # Second input: Normal RGB image
    image_data = keras.Input(shape=(input_wh[0], input_wh[1], 3), name="ImageData")

    # Process color choices matrix
    x1 = keras.layers.Conv2D(32, (4,4), activation="relu", padding="same")(keras.layers.Reshape((input_wh[0], input_wh[1], 4*3))(color_choices))
    #x1 = keras.layers.Flatten()(x1)

    # Process image data matrix
    x2 = keras.layers.Conv2D(32, (4,4), activation="relu", padding="same")(image_data)
    #x2 = keras.layers.Flatten()(x2)

    merged = keras.layers.Concatenate()([x1, x2])
    # merged = keras.layers.Dense(256, activation="relu")(merged)

    # merged = keras.layers.Reshape((16, 16, 1))(merged) 
    merged = keras.layers.Conv2D(32, (2,2), activation="tanh", padding="same")(merged)
    merged = keras.layers.Conv2D(64, (8,8), activation="relu", padding="same")(merged)
    # merged = keras.layers.Conv2D(8, (2,2), activation="tanh")(merged)
    print("Merged shape: " + str(merged.shape))

    output = keras.layers.Conv2DTranspose(4, (4,4), strides=(1,1), activation="linear", padding="same")(merged)
    print("Output shape: " + str(output.shape))
    output = keras.layers.Reshape((output_wh[0] * output_wh[1], 4))(output)
    model = keras.Model(inputs=[color_choices, image_data], outputs=output)
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

    model.summary()

    return model
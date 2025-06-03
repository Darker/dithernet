import numpy as np
from numpy.typing import NDArray

# To make the next part readable
color_map = {
    "cyan": [0, 255, 255],  
    "magenta": [255, 0, 255],
    "yellow": [255, 255, 0],
    "black": [0, 0, 0],     
    "white": [255, 255, 255],
    "limegreen": [0, 255, 0],   
    "green": [0, 128, 0],   
    "red": [255, 0, 0],     
    "blue": [0, 0, 255],    
    "orange": [255, 165, 0],
    "purple": [128, 0, 128],
    "lightblue": [0, 0x85, 0xff],
    "gray": [128, 128, 128],
    "darkblue": [0, 0, 60],
    "darkgreen": [0, 30, 0],
    "lightgreen": [144, 255, 144],
    "brown": [165, 42, 42],
    "lightred": [255, 182, 193],
    "pink": [255, 192, 203],
    "darkred": [50, 0, 0],
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
    color_map["darkblue"]
])

pixel_option_4 = np.array([
    color_map["limegreen"],
    color_map["gray"],
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
    color_map["magenta"],
    color_map["red"]
])

pixel_option_9 = np.array([
    color_map["darkgreen"],
    color_map["yellow"],
    color_map["purple"],
    color_map["darkblue"]
])

pixel_option_10 = np.array([
    color_map["brown"],
    color_map["limegreen"],
    color_map["lightblue"],
    color_map["gray"]
])

pixel_option_blue_cmyk = np.array([
    color_map["lightblue"],
    color_map["magenta"],
    color_map["yellow"],
    color_map["darkblue"]
])

pixel_option_red_cmyk = np.array([
    color_map["limegreen"],
    color_map["pink"],
    color_map["orange"],
    color_map["darkred"]
])

all_pixel_options = [
    pixel_option_1,
    pixel_option_2,
    pixel_option_3,
    pixel_option_blue_cmyk,
    pixel_option_4,
    pixel_option_5,
    pixel_option_6,
    pixel_option_7,
    pixel_option_red_cmyk,
    pixel_option_8,
    pixel_option_9,
    pixel_option_10
]
options_count = len(all_pixel_options)

# first run with preset color choices -  CMYK
# color_choices = np.array([
#     all_pixel_options[xcolor%options_count] for xcolor in range(OUTPUT_DIMMS[0]*OUTPUT_DIMMS[1])
# ])

choices_cache: 'dict[tuple[int, int], NDArray[np.floating]]' = dict()

def get_color_choices(dimensions_hw: tuple[int, int]) -> NDArray[np.floating]:
    """
    Get color choices for a given width and height.
    
    Args:
        width (int): Width of the output.
        height (int): Height of the output.
    
    Returns:
        np.ndarray: Color choices for the specified dimensions.
    """
    if dimensions_hw in choices_cache:
        return choices_cache[dimensions_hw]
    
    # Create a new array with the specified dimensions
    color_choices = np.array([
        all_pixel_options[xcolor % options_count] for xcolor in range(dimensions_hw[0] * dimensions_hw[1])
    ])
    
    choices_cache[dimensions_hw] = color_choices
    return color_choices
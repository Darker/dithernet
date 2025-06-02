

def format_in_dimensions(input_image_hw: tuple[int, int], output_image_hw: tuple[int, int]) -> str:
    """
    Formats the input and output image sizes into a string representation.

    Args:
        input_image_hw (tuple[int, int]): The size of the input image (height, width).
        output_image_size (tuple[int, int]): The size of the output image (height, width).

    Returns:
        str: A formatted string representing the dimensions.
    """
    return f"{input_image_hw[0]}x{input_image_hw[1]}to{output_image_hw[0]}x{output_image_hw[1]}"
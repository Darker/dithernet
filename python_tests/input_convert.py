import cv2

def scale_input(image, network_input_hw, zoom=1.0):
    if zoom < 1.0:
        raise ValueError("Zoom factor must be >= 1.0")
    height, width = image.shape[:2]
    scale_dimension = min(height, width)
    scale_refference = max(network_input_hw[0], network_input_hw[1])
    if scale_dimension == 0:
        raise ValueError("Image has zero height or width, cannot scale")
    # Zoom of 1.0 means output size is to fit the input size
    # Zoom of 2.0 means image size twice the input size
    scale_factor = scale_refference*zoom / scale_dimension
    # hratio = height / scale_dimension
    # wratio = width / scale_dimension
    # scale_factor_w = input_size[1]*args.zoom / width
    # scale_factor_h = input_size[0]*args.zoom / height

    new_height = max(int(height * scale_factor), network_input_hw[0])
    new_width = max(int(width * scale_factor), network_input_hw[1])
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return image

def center_crop(image, network_input_hw):
    height, width = image.shape[:2]
    if height < network_input_hw[0] or width < network_input_hw[1]:
        raise ValueError("Image is smaller than the network input size")
    
    top = (height - network_input_hw[0]) // 2
    left = (width - network_input_hw[1]) // 2
    cropped_image = image[top:top + network_input_hw[0], left:left + network_input_hw[1]]
    return cropped_image
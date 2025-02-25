import numpy as np
import cv2

def dct2(a):
    """
    Perform a 2D Discrete Cosine Transform (DCT) on the input array.

    Parameters:
    - a: numpy.ndarray
        The input 2D array (image) to be transformed.

    Returns:
    - numpy.ndarray:
        The 2D DCT of the input array.
    """
    return cv2.dct(np.float32(a))

def idct2(a):
    """
    Perform a 2D Inverse Discrete Cosine Transform (IDCT) on the input array.

    Parameters:
    - a: numpy.ndarray
        The input 2D array (DCT coefficients) to be transformed.

    Returns:
    - numpy.ndarray:
        The 2D IDCT of the input array.
    """
    return cv2.idct(a)

def encode_image(carrier_image: np.ndarray, secret_image: np.ndarray) -> np.ndarray:
    """
    Encode a secret image into a carrier image using DCT-based steganography.

    Parameters:
    - carrier_image: numpy.ndarray
        The image that will carry the secret information.
    - secret_image: numpy.ndarray
        The image that will be hidden within the carrier image.

    Returns:
    - numpy.ndarray:
        The carrier image with the secret image encoded into it.

    Raises:
    - ValueError:
        If the secret image is larger than the carrier image.
    """

    # Check if the secret image fits into the carrier image
    if secret_image.shape[0] > carrier_image.shape[0] or secret_image.shape[1] > carrier_image.shape[1]:
        raise ValueError("Secret image must be smaller than the carrier image.")

    # Perform DCT on the carrier image
    dct_carrier = dct2(carrier_image)

    # Encode the secret image into the DCT coefficients of the carrier image
    for i in range(secret_image.shape[0]):
        for j in range(secret_image.shape[1]):
            # Modify the DCT coefficient based on the secret image pixel value
            dct_carrier[i, j] += (secret_image[i, j] / 255.0)  # Normalize secret image pixel value

    # Return the modified carrier image
    return np.uint8(idct2(dct_carrier))

def decode_image(encoded_image: np.ndarray, secret_image_shape: tuple) -> np.ndarray:
    """
    Decode the secret image from the encoded image using DCT-based steganography.

    Parameters:
    - encoded_image: numpy.ndarray
        The image that contains the hidden secret image.
    - secret_image_shape: tuple
        The shape (height, width) of the secret image to be extracted.

    Returns:
    - numpy.ndarray:
        The extracted secret image.

    Raises:
    - ValueError:
        If the shape of the secret image does not match the provided shape.
    """

    # Perform DCT on the encoded image
    dct_encoded = dct2(encoded_image)

    # Initialize an empty array for the secret image
    secret_image = np.zeros(secret_image_shape, dtype=np.float32)

    # Extract the secret image from the DCT coefficients
    for i in range(secret_image_shape[0]):
        for j in range(secret_image_shape[1]):
            # Retrieve the secret image pixel value from the DCT coefficient
            secret_image[i, j] = (dct_encoded[i, j] * 255.0)  # Denormalize to get the pixel value

    # Return the extracted secret image
    return np.clip(secret_image, 0, 255).astype(np.uint8)

# Example usage:
if __name__ == "__main__":
    # Load the carrier and secret images
    carrier = cv2.imread('images/lena.png', cv2.IMREAD_GRAYSCALE)
    secret = cv2.imread('images/133.png', cv2.IMREAD_GRAYSCALE)

    # Encode the secret image into the carrier image
    encoded = encode_image(carrier, secret)
    cv2.imwrite('encoded_image.png', encoded)

    # Decode the secret image from the encoded image
    decoded_secret = decode_image(encoded, secret.shape)
    cv2.imwrite('decoded_secret_image.png', decoded_secret)
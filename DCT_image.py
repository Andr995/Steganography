import cv2
import numpy as np
from scipy.fftpack import dct, idct




class DCT_image:
    def __init__(self, key=None):
        self.key = key
        self.block_size = 8
        self.alpha = 0.01 # default è 0.05

    def _generate_blocks(self, image):
        blocks = []
        rows, cols = image.shape
        for j in range(0, rows, self.block_size):
            for i in range(0, cols, self.block_size):
                blocks.append(image[j:j+self.block_size, i:i+self.block_size])
        return blocks

    def _reconstruct_from_blocks(self, blocks, image_shape):
        reconstructed = np.zeros(image_shape, dtype=np.float32)
        block_idx = 0
        rows, cols = image_shape
        for j in range(0, rows, self.block_size):
            for i in range(0, cols, self.block_size):
                reconstructed[j:j+self.block_size, i:i+self.block_size] = blocks[block_idx]
                block_idx += 1
        return reconstructed

    def _encode_channel(self, host_channel, target_channel):
        # Assicurarsi che i canali siano in float32
        host_channel = np.float32(host_channel)
        target_channel = np.float32(target_channel)
        host_blocks = self._generate_blocks(host_channel)
        target_blocks = self._generate_blocks(target_channel)
        target_blocks = target_blocks[:len(host_blocks)]
        encoded_blocks = []
        for h_block, t_block in zip(host_blocks, target_blocks):
            # Applica la DCT ai blocchi host e target
            dct_host = cv2.dct(h_block)
            dct_target = cv2.dct(t_block)
            # Embedding: aggiungi un piccolo contributo del blocco target al blocco host
            dct_encoded = dct_host + self.alpha * dct_target
            # Applica l'inverse DCT per ottenere il blocco modificato
            encoded_block = cv2.idct(dct_encoded)
            encoded_blocks.append(encoded_block)
        return self._reconstruct_from_blocks(encoded_blocks, host_channel.shape)

    def _decode_channel(self, encoded_channel, host_channel):
        # Converti i canali in float32
        encoded_channel = np.float32(encoded_channel)
        host_channel = np.float32(host_channel)
        encoded_blocks = self._generate_blocks(encoded_channel)
        host_blocks = self._generate_blocks(host_channel)
        encoded_blocks = encoded_blocks[:len(host_blocks)]
        decoded_blocks = []
        for e_block, h_block in zip(encoded_blocks, host_blocks):
            # Applica la DCT ai blocchi
            dct_encoded = cv2.dct(e_block)
            dct_host = cv2.dct(h_block)
            # Estrai il contributo del target calcolando la differenza pesata
            dct_decoded = (dct_encoded - dct_host) / self.alpha
            # Applica l'inverse DCT per ottenere il blocco decodificato
            decoded_block = cv2.idct(dct_decoded)
            decoded_blocks.append(decoded_block)
        return self._reconstruct_from_blocks(decoded_blocks, host_channel.shape)

    def encode_image_with_image(self, host_img, target_img):
        """
        Incorpora l'immagine target nell'immagine host utilizzando i coefficienti DCT.
        Si ridimensiona target_img per avere le stesse dimensioni di host_img.
        """
        # Ridimensiona la target per adattarsi a host
        target_img = cv2.resize(target_img, (host_img.shape[1], host_img.shape[0]))
        # Poiché le immagini lette con cv2 sono in BGR, processiamo ciascun canale mantenendo l'ordine
        encoded_b = self._encode_channel(host_img[:, :, 0], target_img[:, :, 0])
        encoded_g = self._encode_channel(host_img[:, :, 1], target_img[:, :, 1])
        encoded_r = self._encode_channel(host_img[:, :, 2], target_img[:, :, 2])
        # Ricompone l'immagine encoded in ordine BGR
        encoded_img = cv2.merge([encoded_b, encoded_g, encoded_r])
        # Assicura che i valori siano nel range corretto
        return np.uint8(np.clip(encoded_img, 0, 255))

    def decode_image_with_image(self, encoded_img, host_img):
        """
        Estrae l'immagine target nascosta dall'immagine encoded, utilizzando l'immagine host originale come riferimento.
        """
        decoded_b = self._decode_channel(encoded_img[:, :, 0], host_img[:, :, 0])
        decoded_g = self._decode_channel(encoded_img[:, :, 1], host_img[:, :, 1])
        decoded_r = self._decode_channel(encoded_img[:, :, 2], host_img[:, :, 2])
        decoded_img = cv2.merge([decoded_b, decoded_g, decoded_r])
        return np.uint8(np.clip(decoded_img, 0, 255))


def create_sync_marker(block_size=8, value=255):
    """Crea un marker di dimensione block_size x block_size con valore costante per tutti i canali."""
    marker = np.full((block_size, block_size, 3), value, dtype=np.uint8)
    return marker

def insert_sync_marker(img, marker, position=(0, 0)):
    """Inserisce il marker nell'immagine alla posizione specificata."""
    m, n = marker.shape[:2]
    img[position[1]:position[1]+m, position[0]:position[0]+n] = marker
    return img

def detect_sync_marker(img, marker):
    """
    Rileva il marker nell'immagine usando matchTemplate.
    Convertiamo sia l'immagine che il marker in scala di grigi forzando il tipo a uint8.
    """
    img_gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    marker_gray = cv2.cvtColor(marker.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, marker_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return max_loc  # posizione top-left del marker

def estimate_inverse_affine(marker_orig_pos, marker_detected_pos):
    """
    Stima una matrice affine inversa basata sulla differenza di posizione (traslazione).
    Questo esempio gestisce solo traslazioni.
    """
    dx = marker_orig_pos[0] - marker_detected_pos[0]
    dy = marker_orig_pos[1] - marker_detected_pos[1]
    M_inv = np.float32([[1, 0, dx], [0, 1, dy]])
    return M_inv
def compress_image(img, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_img = cv2.imencode('.jpg', img, encode_param)
    decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    return decoded_img

def create_translation(img):
    # Inserisci un marker di sincronizzazione (per esempio, nel blocco in alto a sinistra)
    marker = create_sync_marker(block_size=8, value=255)
    encoded_with_marker = insert_sync_marker(img.copy(), marker, position=(0, 0))
    cv2.imwrite("encoded_with_marker.png", encoded_with_marker)

    # applichiamo col maker la trasformazione affine
    rows, cols, ch = encoded_with_marker.shape
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    affine_transformed = cv2.warpAffine(encoded_with_marker, M, (cols, rows))
    cv2.imwrite("affine.jpeg", affine_transformed)

    # In fase di decodifica, rileva il marker per stimare la trasformazione di sincronizzazione
    detected_marker_pos = detect_sync_marker(affine_transformed, marker)
    # La posizione originale del marker è (0,0)
    M_inv = estimate_inverse_affine ((0, 0), detected_marker_pos)

    # Correggi l'immagine affine applicando M_inv (solo traslazione in questo esempio)
    return cv2.warpAffine(affine_transformed, M_inv, (cols, rows))

## Versione manuale:

def manual_translation(img, dx, dy):
    """Applica una traslazione manuale creando direttamente la matrice di trasformazione affine."""
    rows, cols = img.shape[:2]

    # Creiamo manualmente la matrice di traslazione
    M = np.float32([[1, 0, dx], 
                    [0, 1, dy]])

    # Applichiamo la trasformazione
    translated_img = cv2.warpAffine(img, M, (cols, rows))

    return translated_img, M  # Ritorniamo anche M per l'inversa

def inverse_translation(img, M):
    """Applica l'inversa della traslazione per riportare l'immagine all'originale."""
    rows, cols = img.shape[:2]

    # Invertiamo la traslazione
    M_inv = np.float32([[1, 0, -M[0, 2]], 
                         [0, 1, -M[1, 2]]])

    # Applichiamo la trasformazione inversa
    corrected_img = cv2.warpAffine(img, M_inv, (cols, rows))

    return corrected_img

def manual_rotation(img, angle):
    """Applica una rotazione manuale creando direttamente la matrice di trasformazione affine."""
    rows, cols = img.shape[:2]

    # Convertiamo l'angolo in radianti
    theta = np.radians(angle)

    # Calcoliamo la matrice di rotazione
    M = np.float32([[np.cos(theta), -np.sin(theta), 0], 
                    [np.sin(theta), np.cos(theta), 0]])

    # Per evitare il ritaglio, trasliamo il centro dell'immagine
    center_x, center_y = cols // 2, rows // 2
    M[0, 2] = center_x - (center_x * M[0, 0] + center_y * M[0, 1])
    M[1, 2] = center_y - (center_x * M[1, 0] + center_y * M[1, 1])

    # Applichiamo la trasformazione
    rotated_img = cv2.warpAffine(img, M, (cols, rows))

    return rotated_img, M  # Ritorniamo anche M per l'inversa

def inverse_rotation(img, M):
    """Applica l'inversa della rotazione per riportare l'immagine all'originale."""
    rows, cols = img.shape[:2]

    # Invertiamo la matrice di rotazione
    M_inv = cv2.invertAffineTransform(M)

    # Applichiamo la trasformazione inversa
    corrected_img = cv2.warpAffine(img, M_inv, (cols, rows))

    return corrected_img
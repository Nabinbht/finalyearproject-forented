import heapq
from collections import defaultdict
import numpy as np
import cv2

# Helper functions for Huffman coding
class HuffmanNode:
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_frequency_table(data):
    frequency = defaultdict(int)
    for value in data:
        frequency[value] += 1
    return frequency

def build_huffman_tree(frequency):
    heap = [HuffmanNode(value, freq) for value, freq in frequency.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    return heap[0]

def build_huffman_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    if node:
        if node.value is not None:
            codebook[node.value] = prefix
        build_huffman_codes(node.left, prefix + "0", codebook)
        build_huffman_codes(node.right, prefix + "1", codebook)
    return codebook

def encode_data(data, huffman_codes):
    return ''.join(huffman_codes[value] for value in data)

def decode_data(encoded_data, reverse_codes):
    decoded_data = []
    current_code = ""
    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_codes:
            decoded_data.append(reverse_codes[current_code])
            current_code = ""
    return decoded_data

# Standard JPEG Quantization Tables
Q_Y = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

Q_C = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

# DCT Matrix
def dct_matrix(N):
    c = np.sqrt(2 / N)
    D = np.zeros((N, N))
    for u in range(N):
        for x in range(N):
            if u == 0:
                D[u, x] = np.sqrt(1 / N)
            else:
                D[u, x] = c * np.cos((np.pi / N) * (x + 0.5) * u)
    return D

# 2D DCT and IDCT
def dct_2d(block):
    N, M = block.shape
    D_row = dct_matrix(N)
    D_col = dct_matrix(M)
    return np.dot(np.dot(D_row, block), D_col.T)

def idct_2d(block):
    N, M = block.shape
    D_row = dct_matrix(N)
    D_col = dct_matrix(M)
    return np.dot(np.dot(D_row.T, block), D_col)

# Quantization
def quantize(block, table, quality):
    if quality < 1:
        quality = 1
    elif quality > 100:
        quality = 100
    scale = 1.0 if quality >= 50 else 50 / quality
    Q = np.floor((table * scale) + 0.5)
    Q[Q > 255] = 255  # Clamp to 8-bit range
    return np.round(block / Q) * Q

# Downsampling
def downsample(channel, factor=2):
    height, width = channel.shape
    new_height, new_width = height // factor, width // factor
    downsampled = channel[:new_height * factor, :new_width * factor]
    downsampled = downsampled.reshape(new_height, factor, new_width, factor).mean(axis=(1, 3))
    return downsampled

# RGB to YCbCr Conversion
def rgb_to_ycbcr_manual(image):
    R, G, B = image[:, :, 0].astype(float), image[:, :, 1].astype(float), image[:, :, 2].astype(float)
    Y = (0.299 * R + 0.587 * G + 0.114 * B)
    Cb = 128 + (-0.168736 * R - 0.331264 * G + 0.5 * B)
    Cr = 128 + (0.5 * R - 0.418688 * G - 0.081312 * B)
    return np.clip(Y, 0, 255), np.clip(Cb, 0, 255), np.clip(Cr, 0, 255)

# Resize Channels
def resize_channel_to_match(y_channel, channel_to_resize):
    new_height, new_width = y_channel.shape
    return cv2.resize(channel_to_resize, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

# Merge YCbCr Channels
def merge_ycbcr(Y, Cb, Cr):
    return np.stack((Y, Cb, Cr), axis=-1)

# YCbCr to RGB Conversion
def ycbcr2rgb(ycbcr_image):
    Y, Cb, Cr = ycbcr_image[:, :, 0], ycbcr_image[:, :, 1], ycbcr_image[:, :, 2]
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)
    rgb_image = np.stack((R, G, B), axis=-1)
    return np.clip(rgb_image, 0, 255).astype(np.uint8)

# Split into 8x8 Blocks
def split_into_blocks(image, block_size=8):
    h, w = image.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            if block.shape == (block_size, block_size):  # Ensure full blocks
                blocks.append(block)
    return blocks

# Merge 8x8 Blocks
def merge_blocks(blocks, image_shape):
    h, w = image_shape
    block_size = blocks[0].shape[0]
    image = np.zeros((h, w))
    idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if i + block_size <= h and j + block_size <= w:
                image[i:i+block_size, j:j+block_size] = blocks[idx]
                idx += 1
    return image

# Zigzag Scanning
def zigzag(block):
    return np.concatenate([np.diagonal(block[::-1, :], i)[::(2 * (i % 2) - 1)] for i in range(1 - block.shape[0], block.shape[0])])

# Run-Length Encoding (RLE)
def run_length_encode(data):
    encoded = []
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            encoded.append((data[i - 1], count))
            count = 1
    encoded.append((data[-1], count))
    return encoded

# Compression with Huffman Coding
def compress_image(input_image, quality=1):
    Y, Cb, Cr = rgb_to_ycbcr_manual(input_image)
    Cb_downsampled = downsample(Cb)
    Cr_downsampled = downsample(Cr)

    # Split into 8x8 blocks
    Y_blocks = split_into_blocks(Y)
    Cb_blocks = split_into_blocks(Cb_downsampled)
    Cr_blocks = split_into_blocks(Cr_downsampled)

    # Apply DCT and quantization
    Y_quantized_blocks = [quantize(dct_2d(block), Q_Y, quality) for block in Y_blocks]
    Cb_quantized_blocks = [quantize(dct_2d(block), Q_C, quality) for block in Cb_blocks]
    Cr_quantized_blocks = [quantize(dct_2d(block), Q_C, quality) for block in Cr_blocks]

    # Zigzag scanning
    Y_zigzag = [zigzag(block) for block in Y_quantized_blocks]
    Cb_zigzag = [zigzag(block) for block in Cb_quantized_blocks]
    Cr_zigzag = [zigzag(block) for block in Cr_quantized_blocks]

    # Run-Length Encoding (RLE)
    Y_rle = [run_length_encode(block) for block in Y_zigzag]
    Cb_rle = [run_length_encode(block) for block in Cb_zigzag]
    Cr_rle = [run_length_encode(block) for block in Cr_zigzag]

    # Flatten the RLE data
    Y_flat = [item for sublist in Y_rle for item in sublist]
    Cb_flat = [item for sublist in Cb_rle for item in sublist]
    Cr_flat = [item for sublist in Cr_rle for item in sublist]

    # Build frequency tables and Huffman trees
    Y_freq = build_frequency_table(Y_flat)
    Cb_freq = build_frequency_table(Cb_flat)
    Cr_freq = build_frequency_table(Cr_flat)

    Y_tree = build_huffman_tree(Y_freq)
    Cb_tree = build_huffman_tree(Cb_freq)
    Cr_tree = build_huffman_tree(Cr_freq)

    # Generate Huffman codes
    Y_codes = build_huffman_codes(Y_tree)
    Cb_codes = build_huffman_codes(Cb_tree)
    Cr_codes = build_huffman_codes(Cr_tree)

    # Encode the data
    Y_encoded = encode_data(Y_flat, Y_codes)
    Cb_encoded = encode_data(Cb_flat, Cb_codes)
    Cr_encoded = encode_data(Cr_flat, Cr_codes)

    # Save the encoded data and Huffman codes
    np.savez("compressed_data.npz",
             Y=Y_encoded, Cb=Cb_encoded, Cr=Cr_encoded,
             Y_codes=Y_codes, Cb_codes=Cb_codes, Cr_codes=Cr_codes,
             Y_shape=Y.shape, Cb_shape=Cb_downsampled.shape, Cr_shape=Cr_downsampled.shape)

    # Reconstruct the image for testing purposes
    Y_reconstructed = merge_blocks([idct_2d(block) for block in Y_quantized_blocks], Y.shape)
    Cb_reconstructed = merge_blocks([idct_2d(block) for block in Cb_quantized_blocks], Cb_downsampled.shape)
    Cr_reconstructed = merge_blocks([idct_2d(block) for block in Cr_quantized_blocks], Cr_downsampled.shape)
    Cb_resized = resize_channel_to_match(Y_reconstructed, Cb_reconstructed)
    Cr_resized = resize_channel_to_match(Y_reconstructed, Cr_reconstructed)
    reconstructed_ycbcr = merge_ycbcr(Y_reconstructed, Cb_resized, Cr_resized)
    return ycbcr2rgb(reconstructed_ycbcr)

# Decompression with Huffman Decoding
def decompress_image(file_path):
    data = np.load(file_path, allow_pickle=True)
    Y_encoded = data['Y']
    Cb_encoded = data['Cb']
    Cr_encoded = data['Cr']
    Y_codes = data['Y_codes'].item()
    Cb_codes = data['Cb_codes'].item()
    Cr_codes = data['Cr_codes'].item()
    Y_shape = data['Y_shape']
    Cb_shape = data['Cb_shape']
    Cr_shape = data['Cr_shape']

    # Reverse the Huffman codes to build decoding dictionaries
    Y_reverse_codes = {code: value for value, code in Y_codes.items()}
    Cb_reverse_codes = {code: value for value, code in Cb_codes.items()}
    Cr_reverse_codes = {code: value for value, code in Cr_codes.items()}

    # Decode the data
    Y_flat = decode_data(Y_encoded, Y_reverse_codes)
    Cb_flat = decode_data(Cb_encoded, Cb_reverse_codes)
    Cr_flat = decode_data(Cr_encoded, Cr_reverse_codes)

    # Reshape the data
    Y_quantized = np.array(Y_flat).reshape(Y_shape)
    Cb_quantized = np.array(Cb_flat).reshape(Cb_shape)
    Cr_quantized = np.array(Cr_flat).reshape(Cr_shape)

    # Reconstruct the image
    Y_reconstructed = idct_2d(Y_quantized)
    Cb_reconstructed = idct_2d(Cb_quantized)
    Cr_reconstructed = idct_2d(Cr_quantized)
    Cb_resized = resize_channel_to_match(Y_reconstructed, Cb_reconstructed)
    Cr_resized = resize_channel_to_match(Y_reconstructed, Cr_reconstructed)
    reconstructed_ycbcr = merge_ycbcr(Y_reconstructed, Cb_resized, Cr_resized)
    return ycbcr2rgb(reconstructed_ycbcr)

# Example Usage
if __name__ == "__main__":
    # Load an input image
    input_image = cv2.imread("input_image.jpg")

    # Compress the image
    compressed_image = compress_image(input_image, quality=1)
    cv2.imwrite("compressed_image.jpg", compressed_image)

    # Decompress the image
    decompressed_image = decompress_image("compressed_data.npz")
    cv2.imwrite("decompressed_image.jpg", decompressed_image)
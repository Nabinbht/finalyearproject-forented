import cv2
import numpy as np

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

def quantize(block, quality):
    Q = np.ones(block.shape) * quality
    return np.round(block / Q) * Q

def downsample(channel, factor=4):
    height, width = channel.shape
    new_height, new_width = height // factor, width // factor
    downsampled = channel[:new_height * factor, :new_width * factor]
    downsampled = downsampled.reshape(new_height, factor, new_width, factor).mean(axis=(1, 3))
    return downsampled

def rgb_to_ycbcr_manual(image):
    R, G, B = image[:, :, 0].astype(float), image[:, :, 1].astype(float), image[:, :, 2].astype(float)
    Y  = (0.299 * R + 0.587 * G + 0.114 * B)
    Cb = 128 + (-0.168736 * R - 0.331264 * G + 0.5 * B)
    Cr = 128 + (0.5 * R - 0.418688 * G - 0.081312 * B)
    return np.clip(Y, 0, 255), np.clip(Cb, 0, 255), np.clip(Cr, 0, 255)

def resize_channel_to_match(y_channel, channel_to_resize):
    new_height, new_width = y_channel.shape
    return cv2.resize(channel_to_resize, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

def merge_ycbcr(Y, Cb, Cr):
    return np.stack((Y, Cb, Cr), axis=-1)

def ycbcr2rgb(ycbcr_image):
    Y, Cb, Cr = ycbcr_image[:, :, 0], ycbcr_image[:, :, 1], ycbcr_image[:, :, 2]
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)
    rgb_image = np.stack((R, G, B), axis=-1)
    return np.clip(rgb_image, 0, 255).astype(np.uint8)

def compress_image(input_image, quality=1):
    Y, Cb, Cr = rgb_to_ycbcr_manual(input_image)
    Cb_downsampled = downsample(Cb)
    Cr_downsampled = downsample(Cr)
    Y_dct = dct_2d(Y)
    Cb_dct = dct_2d(Cb_downsampled)
    Cr_dct = dct_2d(Cr_downsampled)
    Y_quantized = quantize(Y_dct, quality)
    Cb_quantized = quantize(Cb_dct, quality)
    Cr_quantized = quantize(Cr_dct, quality)
    np.savez("compressed_data.npz", Y=Y_quantized, Cb=Cb_quantized, Cr=Cr_quantized)
    Y_reconstructed = idct_2d(Y_quantized)
    Cb_reconstructed = idct_2d(Cb_quantized)
    Cr_reconstructed = idct_2d(Cr_quantized)
    Cb_resized = resize_channel_to_match(Y_reconstructed, Cb_reconstructed)
    Cr_resized = resize_channel_to_match(Y_reconstructed, Cr_reconstructed)
    reconstructed_ycbcr = merge_ycbcr(Y_reconstructed, Cb_resized, Cr_resized)
    return ycbcr2rgb(reconstructed_ycbcr)

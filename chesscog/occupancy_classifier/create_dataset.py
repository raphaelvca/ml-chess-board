# from pathlib import Path
# import matplotlib.pyplot as plt
import cv2
# from PIL import Image, ImageDraw
# import json
import numpy as np
import chess
# import os
# import shutil
from recap import URI
# import argparse

# RENDERS_DIR = URI("data://render")
# OUT_DIR = URI("data://occupancy")
SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE
IMG_SIZE = BOARD_SIZE + 2 * SQUARE_SIZE

from chesscog.core.init import sort_corner_points

def crop_square(img: np.ndarray, square: chess.Square, turn: chess.Color) -> np.ndarray:
    """Crop a chess square from the warped input image for occupancy classification.

    Args:
        img (np.ndarray): the warped input image
        square (chess.Square): the square to crop
        turn (chess.Color): the current player

    Returns:
        np.ndarray: the cropped square
    """
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    if turn == chess.WHITE:
        row, col = 7 - rank, file
    else:
        row, col = rank, 7 - file

    # Debug
    # creates square of big img
    # result = img[int(SQUARE_SIZE * (row + .5)): int(SQUARE_SIZE * (row + 2.5)),
    #            int(SQUARE_SIZE * (col + .5)): int(SQUARE_SIZE * (col + 2.5))]
    # cv2.imshow('image', result)

    return img[int(SQUARE_SIZE * (row + .5)): int(SQUARE_SIZE * (row + 2.5)),
               int(SQUARE_SIZE * (col + .5)): int(SQUARE_SIZE * (col + 2.5))]

def warp_chessboard_image(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp the image of the chessboard onto a regular grid.

    Args:
        img (np.ndarray): the image of the chessboard
        corners (np.ndarray): pixel locations of the four corner points

    Returns:
        np.ndarray: the warped image
    """

    src_points = sort_corner_points(corners)
    dst_points = np.array([[SQUARE_SIZE, SQUARE_SIZE],  # top left
                           [BOARD_SIZE + SQUARE_SIZE, SQUARE_SIZE],  # top right
                           [BOARD_SIZE + SQUARE_SIZE, BOARD_SIZE + \
                            SQUARE_SIZE],  # bottom right
                           [SQUARE_SIZE, BOARD_SIZE + SQUARE_SIZE]  # bottom left
                           ], dtype=np.float)
    transformation_matrix, mask = cv2.findHomography(src_points, dst_points)

    # Debug help: img
    # Input img
    # cv2.imshow('image', img)
    # Result img
    # result = cv2.warpPerspective(img, transformation_matrix, (IMG_SIZE, IMG_SIZE))
    # cv2.imshow('image', result)

    return cv2.warpPerspective(img, transformation_matrix, (IMG_SIZE, IMG_SIZE))
import pygame
import numpy as np
import cv2
from scipy.ndimage import shift
import random

color = (255, 255, 255)
width = 13

pygame.init()
screen = pygame.display.set_mode((280, 280))
clock = pygame.time.Clock()


def draw_line(start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int(start[0] + float(i) / distance * dx)
        y = int(start[1] + float(i) / distance * dy)
        pygame.draw.circle(screen, color, (x, y), width)


def center_image(image):
    col_sum = np.nonzero(np.sum(image, axis=0))
    row_sum = np.nonzero(np.sum(image, axis=1))
    y1, y2 = row_sum[0][0], row_sum[0][-1]
    x1, x2 = col_sum[0][0], col_sum[0][-1]
    row_mov = ((image.shape[1] - 1 - y2) - y1) / 2
    col_mov = ((image.shape[0] - 1 - x2) - x1) / 2
    image = shift(image, shift=[row_mov, col_mov])
    return image


def add_noise(image, noise_amount):
    for _ in range(noise_amount):
        random_coords = np.random.randint(0, 27, size=2)
        image[tuple(random_coords)] = random.randint(60, 130)
    return image


def process_image(surface):
    pixel_array = pygame.surfarray.pixels2d(surface)
    pixel_array = np.array(pixel_array, dtype=np.uint8)
    pixel_array = np.transpose(pixel_array)
    resized_image = cv2.resize(pixel_array, (28, 28))
    centered_image = center_image(resized_image)
    blured_image = cv2.GaussianBlur(centered_image, (3, 3), 0)
    noisy_image = add_noise(blured_image, noise_amount=10)
    return noisy_image


screen.fill("black")


def start_drawing():
    image = None
    drawing = False
    last_pos = None
    while True:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return image
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.circle(screen, color, mouse_pos, width)
                last_pos = mouse_pos
                drawing = True
            elif event.type == pygame.MOUSEBUTTONUP:
                image = process_image(screen)
                drawing = False
            elif event.type == pygame.MOUSEMOTION and drawing:
                draw_line(last_pos, mouse_pos)
                last_pos = mouse_pos

        pygame.display.flip()

        clock.tick(60)

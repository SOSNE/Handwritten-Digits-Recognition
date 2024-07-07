import pygame
import sys
import numpy as np
import cv2

color = (255, 255, 255)
width = 7


pygame.init()
screen = pygame.display.set_mode((280, 280))
clock = pygame.time.Clock()


def draw_line(screen, start, end, width, color):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int(start[0] + float(i) / distance * dx)
        y = int(start[1] + float(i) / distance * dy)
        pygame.draw.circle(screen, color, (x, y), width)


def process_image(surface):
    pixel_array = pygame.surfarray.pixels2d(surface)
    pixel_array = np.array(pixel_array, dtype=np.uint8)
    resized_image = cv2.resize(pixel_array, (28, 28))
    return resized_image


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
            elif event.type == pygame.MOUSEMOTION:
                if drawing:
                    draw_line(screen, last_pos, mouse_pos, width, color)
                    last_pos = mouse_pos

        pygame.display.flip()

        clock.tick(60)


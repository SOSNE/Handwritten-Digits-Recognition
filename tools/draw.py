import pygame
import sys

color = (255, 255, 255)
width = 10
drawing = False
last_pos = None

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


screen.fill("black")

while True:
    mouse_pos = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pygame.draw.circle(screen, color, mouse_pos, width)
            last_pos = mouse_pos
            drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                draw_line(screen, last_pos, mouse_pos, width, color)
                last_pos = mouse_pos

    pygame.display.flip()

    clock.tick(60)

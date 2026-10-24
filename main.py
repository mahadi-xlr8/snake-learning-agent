import pygame
from game.constants import UP, DOWN, LEFT, RIGHT, TICKS_PER_SECOND, WINDOW_FPS
from game.snake_game import SnakeGame
from game.renderer import Renderer

KEY_DIRECTIONS = {
    pygame.K_UP: UP, pygame.K_DOWN: DOWN,
    pygame.K_LEFT: LEFT, pygame.K_RIGHT: RIGHT,
}


def main():
    game = SnakeGame()
    renderer = Renderer(game.size)
    clock = pygame.time.Clock()
    elapsed_ms = 0
    tick_ms = 1000 / TICKS_PER_SECOND
    running = True
    while running:
        dt = clock.tick(WINDOW_FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()
                    elapsed_ms = 0
                elif event.key in KEY_DIRECTIONS:
                    game.change_direction(KEY_DIRECTIONS[event.key])
        if not game.done:
            elapsed_ms += dt
            if elapsed_ms >= tick_ms:
                game.step()
                elapsed_ms %= tick_ms
        renderer.draw(game)
    renderer.close()


if __name__ == "__main__":
    main()

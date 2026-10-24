import pygame
from game.constants import (
    CELL_SIZE, EMPTY, WALL, SNAKE_BODY, SNAKE_HEAD, FOOD, OBSTACLE,
)

COLORS = {
    EMPTY: (18, 24, 35), WALL: (94, 105, 122),
    SNAKE_BODY: (35, 178, 95), SNAKE_HEAD: (125, 242, 151),
    FOOD: (239, 68, 68), OBSTACLE: (245, 158, 11),
}


class Renderer:
    def __init__(self, size):
        pygame.init()
        self.size = size
        self.screen = pygame.display.set_mode((size * CELL_SIZE, size * CELL_SIZE + 58))
        pygame.display.set_caption("Snake Learning Agent | Human Mode")
        self.font = pygame.font.SysFont("arial", 23, bold=True)
        self.small_font = pygame.font.SysFont("arial", 16)

    def draw(self, game):
        self.screen.fill((18, 24, 35))
        board = game.get_board()
        for y in range(game.size):
            for x in range(game.size):
                value = int(board[y, x])
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, COLORS[value], rect.inflate(-2, -2), border_radius=4)
        footer_y = game.size * CELL_SIZE + 9
        label = f"Score: {game.score}    Arrow keys: turn    R: restart    Esc: quit"
        self.screen.blit(self.small_font.render(label, True, (235, 240, 246)), (10, footer_y))
        if game.done:
            overlay = pygame.Surface((game.size * CELL_SIZE, game.size * CELL_SIZE), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 155))
            self.screen.blit(overlay, (0, 0))
            message = "YOU WIN!" if game.won else "GAME OVER"
            text = self.font.render(message, True, (255, 255, 255))
            hint = self.small_font.render("Press R to restart", True, (255, 255, 255))
            center_x = game.size * CELL_SIZE // 2
            center_y = game.size * CELL_SIZE // 2
            self.screen.blit(text, text.get_rect(center=(center_x, center_y - 12)))
            self.screen.blit(hint, hint.get_rect(center=(center_x, center_y + 22)))
        pygame.display.flip()

    def close(self):
        pygame.quit()

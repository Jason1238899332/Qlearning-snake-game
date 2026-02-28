import pygame
import random
import numpy as np
import time
from dataclasses import dataclass

# Directions as (dx, dy)
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

TURN_RIGHT = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}
TURN_LEFT  = {UP: LEFT,  LEFT: DOWN, DOWN: RIGHT, RIGHT: UP}


def draw_triangle_head(screen, x, y, cell, direction, color):
    """
    Draw a triangular snake head pointing in `direction` at grid cell (x, y),
    plus two small eyes.
    """
    px = x * cell
    py = y * cell

    # Triangle points + eye positions (inside the cell)
    if direction == UP:
        pts = [(px + cell * 0.5, py + cell * 0.08),
               (px + cell * 0.12, py + cell * 0.92),
               (px + cell * 0.88, py + cell * 0.92)]
        eye1 = (px + cell * 0.38, py + cell * 0.30)
        eye2 = (px + cell * 0.62, py + cell * 0.30)
    elif direction == DOWN:
        pts = [(px + cell * 0.5, py + cell * 0.92),
               (px + cell * 0.12, py + cell * 0.08),
               (px + cell * 0.88, py + cell * 0.08)]
        eye1 = (px + cell * 0.38, py + cell * 0.70)
        eye2 = (px + cell * 0.62, py + cell * 0.70)
    elif direction == LEFT:
        pts = [(px + cell * 0.08, py + cell * 0.5),
               (px + cell * 0.92, py + cell * 0.12),
               (px + cell * 0.92, py + cell * 0.88)]
        eye1 = (px + cell * 0.30, py + cell * 0.38)
        eye2 = (px + cell * 0.30, py + cell * 0.62)
    else:  # RIGHT
        pts = [(px + cell * 0.92, py + cell * 0.5),
               (px + cell * 0.08, py + cell * 0.12),
               (px + cell * 0.08, py + cell * 0.88)]
        eye1 = (px + cell * 0.70, py + cell * 0.38)
        eye2 = (px + cell * 0.70, py + cell * 0.62)

    # Head polygon
    pygame.draw.polygon(screen, color, pts)
    pygame.draw.polygon(screen, (0, 0, 0), pts, 2)  # outline

    # Eyes (white + black pupil)
    r_white = max(2, int(cell * 0.09))
    r_pupil = max(1, int(cell * 0.04))

    pygame.draw.circle(screen, (255, 255, 255), (int(eye1[0]), int(eye1[1])), r_white)
    pygame.draw.circle(screen, (255, 255, 255), (int(eye2[0]), int(eye2[1])), r_white)
    pygame.draw.circle(screen, (0, 0, 0), (int(eye1[0]), int(eye1[1])), r_pupil)
    pygame.draw.circle(screen, (0, 0, 0), (int(eye2[0]), int(eye2[1])), r_pupil)


@dataclass
class StepInfo:
    score: int
    steps: int
    elapsed: float


class SnakeEnv:
    """
    Action space (3):
      0: straight
      1: turn right
      2: turn left

    State (11):
      [danger_straight, danger_right, danger_left,
       dir_up, dir_down, dir_left, dir_right,
       food_left, food_right, food_up, food_down]
    """

    def __init__(self, grid_w=20, grid_h=15, cell=24, fps=10, render=True):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.cell = cell
        self.fps = fps
        self.enable_render = render

        # speed bounds (for +/- control)
        self.fps_min = 5
        self.fps_max = 60

        self.steps = 0
        self.score = 0
        self.direction = RIGHT
        self.snake = []
        self.food = None

        self.start_time = None
        self.mode_text = "AI"

        # expose turn maps
        self.TURN_RIGHT = TURN_RIGHT
        self.TURN_LEFT = TURN_LEFT

        # Pygame lazy init
        self._pg_inited = False
        self._screen = None
        self._clock = None
        self._font = None

    def _init_pygame(self):
        if self._pg_inited:
            return
        pygame.init()
        w, h = self.grid_w * self.cell, self.grid_h * self.cell
        self._screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Snake RL")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont(None, 22)
        self._pg_inited = True

    def close(self):
        if self._pg_inited:
            pygame.quit()
            self._pg_inited = False

    def change_speed(self, delta: int):
        self.fps = int(max(self.fps_min, min(self.fps_max, self.fps + delta)))

    def reset(self):
        self.steps = 0
        self.score = 0
        self.direction = RIGHT

        cx, cy = self.grid_w // 2, self.grid_h // 2
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]

        self._place_food()
        self.start_time = time.time()
        return self._get_state()

    def _elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def _place_food(self):
        while True:
            p = (random.randrange(self.grid_w), random.randrange(self.grid_h))
            if p not in self.snake:
                self.food = p
                return

    def _in_bounds(self, p):
        x, y = p
        return 0 <= x < self.grid_w and 0 <= y < self.grid_h

    def _is_collision(self, p):
        if not self._in_bounds(p):
            return True
        if p in self.snake:
            return True
        return False

    def _get_state(self):
        head = self.snake[0]
        dir_now = self.direction

        straight = (head[0] + dir_now[0], head[1] + dir_now[1])
        right_dir = TURN_RIGHT[dir_now]
        left_dir = TURN_LEFT[dir_now]
        p_right = (head[0] + right_dir[0], head[1] + right_dir[1])
        p_left = (head[0] + left_dir[0], head[1] + left_dir[1])

        danger_straight = 1.0 if self._is_collision(straight) else 0.0
        danger_right = 1.0 if self._is_collision(p_right) else 0.0
        danger_left = 1.0 if self._is_collision(p_left) else 0.0

        dir_up = 1.0 if dir_now == UP else 0.0
        dir_down = 1.0 if dir_now == DOWN else 0.0
        dir_left = 1.0 if dir_now == LEFT else 0.0
        dir_right = 1.0 if dir_now == RIGHT else 0.0

        food_left = 1.0 if self.food[0] < head[0] else 0.0
        food_right = 1.0 if self.food[0] > head[0] else 0.0
        food_up = 1.0 if self.food[1] < head[1] else 0.0
        food_down = 1.0 if self.food[1] > head[1] else 0.0

        return np.array([
            danger_straight, danger_right, danger_left,
            dir_up, dir_down, dir_left, dir_right,
            food_left, food_right, food_up, food_down
        ], dtype=np.float32)

    def step(self, action: int):
        """
        Reward:
          +10 eat
          -10 die
          -0.1 step
        """
        self.steps += 1

        # update direction by 3-action
        if action == 1:
            self.direction = TURN_RIGHT[self.direction]
        elif action == 2:
            self.direction = TURN_LEFT[self.direction]
        # else 0 = straight

        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        reward = -0.1
        done = False

        if self._is_collision(new_head):
            reward = -10.0
            done = True
            info = StepInfo(score=self.score, steps=self.steps, elapsed=self._elapsed())
            return self._get_state(), reward, done, info

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            reward = 10.0
            self._place_food()
        else:
            self.snake.pop()

        info = StepInfo(score=self.score, steps=self.steps, elapsed=self._elapsed())
        return self._get_state(), reward, done, info

    def render(self):
        """Draw current frame + clean HUD."""
        if not self.enable_render:
            return
        self._init_pygame()
        self._clock.tick(self.fps)

        W, H = self.grid_w * self.cell, self.grid_h * self.cell
        screen = self._screen
        screen.fill((10, 10, 10))

        # grid
        for x in range(0, W, self.cell):
            pygame.draw.line(screen, (40, 40, 40), (x, 0), (x, H))
        for y in range(0, H, self.cell):
            pygame.draw.line(screen, (40, 40, 40), (0, y), (W, y))

        # food
        fx, fy = self.food
        pygame.draw.rect(
            screen, (220, 60, 60),
            pygame.Rect(fx * self.cell, fy * self.cell, self.cell, self.cell)
        )

        # snake (head = green triangle with eyes, body = darker green rectangles)
        head_color = (0, 210, 0)
        body_color = (0, 140, 0)

        for i, (x, y) in enumerate(self.snake):
            if i == 0:
                draw_triangle_head(screen, x, y, self.cell, self.direction, head_color)
            else:
                rect = pygame.Rect(x * self.cell, y * self.cell, self.cell, self.cell)
                pygame.draw.rect(screen, body_color, rect)
                pygame.draw.rect(screen, (0, 0, 0), rect, 1)

        # HUD
        elapsed = self._elapsed()
        hud = self._font.render(
            f"Mode:{self.mode_text}  Speed:{self.fps}FPS  Time:{elapsed:5.1f}s  Score:{self.score}  Steps:{self.steps}",
            True, (230, 230, 230)
        )
        screen.blit(hud, (8, 8))
        pygame.display.flip()
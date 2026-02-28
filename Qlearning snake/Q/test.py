import pygame
import torch

from env_snake import SnakeEnv, UP, DOWN, LEFT, RIGHT
from model import DQN
from utils import get_device


def _wrap_lines(font, text, max_width):
    """Wrap a single line into multiple lines to fit max_width."""
    words = text.split(" ")
    lines = []
    cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        if font.size(test)[0] <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def show_start_screen(screen, clock, title="Snake"):
    """Start screen: press Y to start, ESC to quit. Auto-wrap & fit."""
    W, H = screen.get_size()

    big_font = pygame.font.SysFont(None, max(48, min(96, W // 6)), bold=True)
    mid_font = pygame.font.SysFont(None, max(20, min(34, W // 22)), bold=True)
    small_font = pygame.font.SysFont(None, max(18, min(28, W // 26)))

    raw_lines = [
        "In this game, you control the snake.",
        "Eat food to score and avoid crashing into walls or yourself!",
        "",
        "Press Y to start the game",
        "Press R to restart at any time",
        "Press H / TAB to toggle: AI <-> HUMAN",
        "HUMAN: up down left right Keys to move",
        "AI: The agent plays automatically",
        "Press - / = (or keypad +/-) to decrease / increase speed",
        "Press ESC to quit",
    ]

    max_text_width = int(W * 0.86)

    # ensure clean screen
    screen.fill((245, 245, 245))
    pygame.display.flip()

    while True:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_y:
                    return True

        screen.fill((245, 245, 245))

        # Title (shrink if too wide)
        title_font = big_font
        while title_font.size(title)[0] > int(W * 0.9) and title_font.get_height() > 30:
            title_font = pygame.font.SysFont(None, title_font.get_height() - 4, bold=True)

        title_surf = title_font.render(title, True, (0, 0, 0))
        screen.blit(title_surf, title_surf.get_rect(center=(W // 2, int(H * 0.22))))

        # Prepare wrapped lines
        rendered = []
        for t in raw_lines:
            if t == "":
                rendered.append(("", None, None))
                continue

            is_key = t.startswith("Press") or t.startswith("HUMAN") or t.startswith("AI:")
            font = mid_font if is_key else small_font
            color = (0, 120, 0) if is_key else (0, 0, 0)

            for seg in _wrap_lines(font, t, max_text_width):
                rendered.append((seg, font, color))

        # Total height for vertical placement
        line_gap = 10
        total_h = 0
        for seg, font, _ in rendered:
            if seg == "":
                total_h += 14
            else:
                total_h += font.get_height() + line_gap

        y = int(H * 0.34)
        if y + total_h > H - 20:
            y = max(20, H - 20 - total_h)

        # Draw lines
        for seg, font, color in rendered:
            if seg == "":
                y += 14
                continue
            surf = font.render(seg, True, color)
            rect = surf.get_rect(center=(W // 2, y))
            screen.blit(surf, rect)
            y += font.get_height() + line_gap

        pygame.display.flip()


def desired_dir_to_action(env: SnakeEnv, desired_dir):
    """Absolute direction -> relative 3-action."""
    cur = env.direction
    if desired_dir == cur:
        return 0
    if desired_dir == env.TURN_RIGHT[cur]:
        return 1
    if desired_dir == env.TURN_LEFT[cur]:
        return 2
    return 0


def test():
    device = get_device()
    print("Using device:", device)

    # slower default speed
    env = SnakeEnv(render=True, fps=10)
    env._init_pygame()

    # start screen
    started = show_start_screen(env._screen, env._clock, title="Snake RL")
    if not started:
        env.close()
        return

    # load model
    model = DQN().to(device)
    model.load_state_dict(torch.load("checkpoints/snake_dqn.pt", map_location=device))
    model.eval()

    s = env.reset()

    mode_ai = True
    env.mode_text = "AI"
    desired_dir = env.direction  # for HUMAN mode

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                # quit
                if event.key == pygame.K_ESCAPE:
                    running = False

                # restart
                if event.key == pygame.K_r:
                    s = env.reset()
                    desired_dir = env.direction

                # toggle mode
                if event.key in (pygame.K_h, pygame.K_TAB):
                    mode_ai = not mode_ai
                    env.mode_text = "AI" if mode_ai else "HUMAN"
                    desired_dir = env.direction

                # speed control
                if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    env.change_speed(-2)
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    env.change_speed(+2)

                # HUMAN arrow keys
                if not mode_ai:
                    if event.key == pygame.K_UP and env.direction != DOWN:
                        desired_dir = UP
                    elif event.key == pygame.K_DOWN and env.direction != UP:
                        desired_dir = DOWN
                    elif event.key == pygame.K_LEFT and env.direction != RIGHT:
                        desired_dir = LEFT
                    elif event.key == pygame.K_RIGHT and env.direction != LEFT:
                        desired_dir = RIGHT

        # choose action
        if mode_ai:
            with torch.no_grad():
                qs = model(torch.tensor(s, device=device).unsqueeze(0))
                a = int(torch.argmax(qs, dim=1).item())
        else:
            a = desired_dir_to_action(env, desired_dir)

        # step
        s, r, done, info = env.step(a)

        # render (HUD already clean)
        env.render()

        if done:
            s = env.reset()
            desired_dir = env.direction

    env.close()


if __name__ == "__main__":
    test()
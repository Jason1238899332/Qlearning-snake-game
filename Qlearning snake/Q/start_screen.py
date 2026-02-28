import pygame

def show_start_screen(screen, clock, title="Snake RL"):
    """
    Block here until user starts (Y) or quits (ESC / window close).
    Returns True if start, False if quit.
    """
    W, H = screen.get_size()
    big_font = pygame.font.SysFont(None, 84, bold=True)
    mid_font = pygame.font.SysFont(None, 28, bold=True)
    small_font = pygame.font.SysFont(None, 26)

    lines = [
        "In this game, you control the snake.",
        "Try to eat as much food as possible and avoid crashing!",
        "",
        "Press Y to start the game",
        "Press R to restart (when game is running)",
        "Press H / TAB to toggle: AI <-> HUMAN",
        "HUMAN: Use Arrow Keys to move",
        "AI: The agent plays automatically",
        "Press - / =  to decrease / increase speed",
        "Press ESC to quit",
    ]

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

        # background
        screen.fill((245, 245, 245))

        # title
        title_surf = big_font.render(title, True, (0, 0, 0))
        screen.blit(title_surf, title_surf.get_rect(center=(W // 2, H // 2 - 120)))

        # subtitle / instructions
        y = H // 2 - 30
        for i, text in enumerate(lines):
            if text == "":
                y += 18
                continue

            # green bold for key instruction lines like your sample
            is_key_line = text.startswith("Press") or text.startswith("HUMAN") or text.startswith("AI:")
            font = mid_font if is_key_line else small_font
            color = (0, 120, 0) if is_key_line else (0, 0, 0)

            surf = font.render(text, True, color)
            rect = surf.get_rect(center=(W // 2, y))
            screen.blit(surf, rect)
            y += 34 if is_key_line else 30

        pygame.display.flip()
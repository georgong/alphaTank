import pygame

class TankSidebar:
    def __init__(self, env, width=300, scroll_speed=30):
        self.env = env
        self.width = width
        self.scroll_speed = scroll_speed
        self.scroll_offset = 0
        self.font = pygame.font.SysFont(None, 22)
        self.section_height = 140
        self.max_bar_height = 60

        # 拖动状态
        self.dragging_scrollbar = False
        self.scrollbar_thumb_rect = pygame.Rect(0, 0, 0, 0)
        self.scrollbar_drag_start_y = 0
        self.scrollbar_initial_offset = 0

    def handle_event(self, event, screen_height):
        if event.type == pygame.MOUSEWHEEL:
            self.scroll_offset -= event.y * self.scroll_speed

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                x, y = pygame.mouse.get_pos()
                if self.scrollbar_thumb_rect.collidepoint(x, y):
                    self.dragging_scrollbar = True
                    self.scrollbar_drag_start_y = y
                    self.scrollbar_initial_offset = self.scroll_offset

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging_scrollbar = False

        elif event.type == pygame.MOUSEMOTION:
            if self.dragging_scrollbar:
                _, y = pygame.mouse.get_pos()
                delta = y - self.scrollbar_drag_start_y
                content_height = sum(1 for t in self.env.tanks) * self.section_height
                max_scroll = max(1, content_height - screen_height)
                scroll_area_height = screen_height - self.scrollbar_thumb_rect.height
                if scroll_area_height > 0:
                    self.scroll_offset = self.scrollbar_initial_offset + (delta / scroll_area_height) * max_scroll

    def draw(self, screen):
        SCENE_WIDTH = self.env.game_configs.WIDTH
        HEIGHT = screen.get_height()

        pygame.draw.rect(screen, (40, 40, 60), (SCENE_WIDTH, 0, self.width, HEIGHT))

        x_start = SCENE_WIDTH + 10
        y_start = 20 - self.scroll_offset

        visible_idx = 0
        for tank in self.env.tanks:
            y_offset = y_start + visible_idx * self.section_height
            if y_offset > HEIGHT or y_offset + self.section_height < 0:
                visible_idx += 1
                continue

            self._draw_tank_block(screen, tank, visible_idx, x_start, y_offset)
            visible_idx += 1

        self._draw_scrollbar(screen, HEIGHT)

        total_height = sum(1 for t in self.env.tanks) * self.section_height
        self.scroll_offset = max(0, min(self.scroll_offset, total_height - HEIGHT))

    def _draw_scrollbar(self, screen, screen_height):
        content_height = sum(1 for t in self.env.tanks) * self.section_height
        if content_height <= screen_height:
            return

        track_x = self.env.game_configs.WIDTH + self.width - 10
        track_height = screen_height
        thumb_height = max(int(screen_height / content_height * track_height), 30)
        scroll_ratio = self.scroll_offset / max(1, (content_height - screen_height))
        thumb_y = int(scroll_ratio * (track_height - thumb_height))

        pygame.draw.rect(screen, (80, 80, 100), (track_x, 0, 6, track_height), border_radius=3)
        self.scrollbar_thumb_rect = pygame.Rect(track_x, thumb_y, 6, thumb_height)
        pygame.draw.rect(screen, (160, 160, 255), self.scrollbar_thumb_rect, border_radius=3)

    def _draw_tank_block(self, screen, tank, idx, x_start, y_offset):
        name = self.font.render(f"Tank {idx+1}", True, (255, 255, 255))
        screen.blit(name, (x_start, y_offset))

        reward_val = getattr(tank, "reward", 0.0)
        reward_color = (255, 50, 50) if not tank.alive else (255, 255, 255)
        reward = self.font.render(f"Reward: {reward_val:+.2f}", True, reward_color)
        screen.blit(reward, (x_start, y_offset + 20))

        if hasattr(tank, "frames") and tank.frames:
            icon = pygame.transform.scale(tank.frames[0], (32, 32))
            screen.blit(icon, (x_start, y_offset + 45))

        if hasattr(tank, "action_distribution"):
            action_order = ["rotate", "move", "shoot"]
            action_labels = {
                "rotate": ["<", "-", ">"],
                "move":   ["v", "-", "^"],
                "shoot":  ["0", "1"]
            }

            bar_w = 10
            spacing = 5
            group_spacing = 60
            base_y = y_offset + 110

            for i, action in enumerate(action_order):
                probs = tank.action_distribution.get(action, [])
                if not tank.alive:
                    probs = [0 for _ in probs]
                group_x = x_start + 90 + i * group_spacing

                for j, prob in enumerate(probs):
                    h = int(prob * self.max_bar_height)
                    x = group_x + j * (bar_w + spacing)
                    y = base_y - h

                    color = {
                        "rotate": (200, 100, 100),
                        "move":   (100, 200, 100),
                        "shoot":  (100, 150, 250),
                    }.get(action, (200, 200, 200))

                    pygame.draw.rect(screen, color, (x, y, bar_w, h))
                    label = self.font.render(action_labels[action][j], True, (240, 240, 240))
                    screen.blit(label, (x, base_y + 5))

                group_label = self.font.render(action.capitalize(), True, (220, 220, 220))
                screen.blit(group_label, (group_x, base_y + 30))

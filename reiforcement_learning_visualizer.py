import pygame


class ReinforcementLearningVisualizer:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.environment = algorithm.environment
        self.rewards = []
        self.average_rewards = []
        self.episode = 0
        self.__init_pygame()

    def __init_pygame(self):
        pygame.init()
        pygame.font.init()
        pygame.display.init()
        pygame.display.set_caption(self.algorithm.get_title())

        self.width = 960
        self.height = 320
        self.info_height = 50
        self.screen = pygame.display.set_mode([self.width, self.height + self.info_height])
        self.font_size = min(self.width // 40, 12)
        self.font = pygame.font.SysFont('Arial', self.font_size)

    def reset(self):
        self.rewards.clear()
        self.average_rewards.clear()
        self.algorithm.reset()
        self.episode = 0

    def render_environment(self):
        self.render_info()
        self.environment.render(self.screen, self.width // 2, self.height)

    def render_text(self, text: str, x: int, y: int, text_align: str, text_baseline: str):
        text_surf = self.font.render(text, False, (0, 0, 0))
        text_rect = text_surf.get_rect()

        if text_align == 'right':
            x -= text_rect.right - text_rect.left
        elif text_align == 'center':
            x -= (text_rect.right - text_rect.left) // 2

        if text_baseline == 'bottom':
            y -= text_rect.bottom - text_rect.top
        elif text_baseline == 'middle':
            y -= (text_rect.bottom - text_rect.top) // 2

        self.screen.blit(text_surf, (x, y))

    def render_rewards(self, min_rewards: int = 10):
        if len(self.rewards) < 2:
            return

        width = self.width // 2
        height = self.height
        x0 = self.width // 2
        y0 = 0

        padding = 15
        count = max(len(self.rewards), min_rewards)

        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(x0, y0, width, height), 0)
        pygame.draw.aalines(self.screen, (0, 0, 0), False, [
            [x0 + padding, y0 + padding],
            [x0 + padding, y0 + height - padding],
            [x0 + width - padding, y0 + height - padding]
        ])

        min_reward, max_reward = min(self.rewards), max(self.rewards)
        reward_lines = []
        average_reward_lines = []

        for i, (reward, avg_reward) in enumerate(zip(self.rewards, self.average_rewards)):
            x = x0 + padding + i / (count - 1) * (width - 2 * padding)
            y = y0 + height - padding - ((reward - min_reward) / (max_reward - min_reward)) * (height - 2 * padding)
            y_avg = y0 + height - padding - ((avg_reward - min_reward) / (max_reward - min_reward)) * (height - 2 * padding)

            reward_lines.append([x, y])
            average_reward_lines.append([x, y_avg])

        pygame.draw.aalines(self.screen, (0, 150, 136), False, reward_lines)
        pygame.draw.aalines(self.screen, (244, 67, 54), False, average_reward_lines)

        self.render_text(f'{max_reward:.2f}', x0 + 2, y0 + padding, 'left', 'bottom')
        self.render_text(f'{min_reward:.2f}', x0 + 2, y0 + height - padding + 2, 'left', 'top')
        self.render_text(f'{len(self.rewards)}', reward_lines[-1][0], y0 + height - padding + 2, 'right', 'top')

        pygame.display.flip()

    def render_info(self):
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(0, self.height, self.width, self.info_height), 0)

        for i, text in enumerate(self.environment.get_info()):
            self.render_text(text, self.width // 2, self.height + 5 + i * (self.font_size + 3), 'center', 'top')

    def step(self):
        reward = self.algorithm.step(self.render_environment)
        self.rewards.append(reward)
        self.average_rewards.append(sum(self.rewards[-50:]) / len(self.rewards[-50:]))
        self.render_rewards()
        self.episode += 1
        print(f'End episode {self.episode} with reward {reward}')

import matplotlib.pyplot as plt
import pygame


class ReinforcementLearningVisualizer:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.environment = algorithm.environment
        self.rewards = []
        self.__init_pygame()

    def __init_pygame(self):
        pygame.init()
        pygame.font.init()
        pygame.display.init()
        pygame.display.set_caption(self.algorithm.get_title())

        self.width = 960
        self.height = 320
        self.screen = pygame.display.set_mode([self.width, self.height])
        self.font = pygame.font.SysFont('Arial', min(self.width // 40, 12))

    def reset(self):
        self.rewards.clear()
        self.algorithm.reset()

    def render_environment(self):
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
        for i, reward in enumerate(self.rewards):
            x = x0 + padding + i / (count - 1) * (width - 2 * padding)
            y = y0 + height - padding - ((reward - min_reward) / (max_reward - min_reward)) * (height - 2 * padding)
            reward_lines.append([x, y])

        pygame.draw.aalines(self.screen, (0, 150, 136), False, reward_lines)

        self.render_text(f'{max_reward:.2f}', x0 + 2, y0 + padding, 'left', 'bottom')
        self.render_text(f'{min_reward:.2f}', x0 + 2, y0 + height - padding + 2, 'left', 'top')
        self.render_text(f'{len(self.rewards)}', reward_lines[-1][0], y0 + height - padding + 2, 'right', 'top')

        pygame.display.flip()

    def step(self):
        step = self.algorithm.step()
        self.render_environment()

        if step['done']:
            self.rewards.append(step['reward'])
            self.environment.print_info()
            self.algorithm.reset_episode()
            self.render_rewards()

            plt.figure()
            plt.plot(self.rewards)
            plt.savefig("rewards.jpg")
            plt.close()

            print('End epidode')

        return step['done']

from typing import Union, List, Tuple
import numpy as np
import cv2
import pygame
from envs.abstract_environment import AbstractEnvironment
from spaces.discrete_space import DiscreteSpace
from spaces.uniform_space import UniformSpace


class Snake(AbstractEnvironment):
    INITIAL_LENGTH = 3

    TURN_LEFT = 1
    TURN_RIGHT = 2

    EAT_SELF = 'eat self'
    WALL = 'wall'
    EAT_FOOD = 'eat food'
    NO_EAT = 'no eat'
    DEFAULT = 'default'

    HEAD_CELL = 0
    SNAKE_CELL = 1
    FOOD_CELL = 2

    def __init__(self, use_conv: bool = False, field_width: int = 14, field_height: int = 9):
        self.use_conv = use_conv
        self.field_width = field_width
        self.field_height = field_height

        self.snake = None
        self.food = None
        self.direction = None
        self.steps_without_food = 0

        self.action_space = DiscreteSpace(3)
        self.observation_space = UniformSpace(-1, 1, 43)

        self.reset_info()

    def get_observation_space_shape(self) -> Union[int, List[int]]:
        return (self.field_height + 1, self.field_width + 1, 3) if self.use_conv else self.observation_space.shape

    def get_action_space_shape(self) -> int:
        return self.action_space.shape

    def sample_action(self) -> int:
        return self.action_space.sample()

    def reset_info(self):
        self.max_length = Snake.INITIAL_LENGTH
        self.wall = 0
        self.eat_self = 0
        self.no_eat = 0

    def reset(self):
        self.snake = self.__init_snake()
        self.food = self.__init_food()
        self.direction = {'dx': 0, 'dy': -1}
        self.steps_without_food = 0

        return self.__get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is invalid")

        if self.snake is None:
            raise ValueError("State is none. Call reset before using step method")

        dx, dy = self.__get_direction(action)
        self.direction['dx'], self.direction['dy'] = dx, dy

        move = self.__move_snake(dx, dy)
        done = move in [Snake.WALL, Snake.EAT_SELF, Snake.NO_EAT]
        state = self.__get_state()
        reward = self.__get_reward(move)

        return state, reward, done

    def draw(self, width: int = 600, height: int = 400):
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        cell_width = width // (self.field_width + 1)
        cell_height = height // (self.field_height + 1)

        for p in self.snake:
            color = (136, 150, 0) if p == self.snake[0] else (80, 175, 76)
            self.__draw_cell(img, cell_width, cell_height, p['x'], p['y'], color, True)

        self.__draw_cell(img, cell_width, cell_height, self.food['x'], self.food['y'], (54, 67, 244), True)

        for i in range(self.field_width + 1):
            for j in range(self.field_height + 1):
                self.__draw_cell(img, cell_width, cell_height, i, j, (204, 204, 204))

        return img

    def render(self, screen: pygame.display, width: int, height: int):
        cell_width = width // (self.field_width + 1)
        cell_height = height // (self.field_height + 1)

        pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(0, 0, width, height), 0)

        for p in self.snake:
            color = (0, 150, 136) if p == self.snake[0] else (76, 175, 80)
            self.__render_cell(screen, cell_width, cell_height, p['x'], p['y'], color, True)

        self.__render_cell(screen, cell_width, cell_height, self.food['x'], self.food['y'], (244, 67, 54), True)

        for i in range(self.field_width + 1):
            for j in range(self.field_height + 1):
                self.__render_cell(screen, cell_width, cell_height, i, j, (204, 204, 204))

        pygame.event.pump()
        pygame.display.flip()

    def print_info(self):
        print(f"Snake length: {len(self.snake)} (max: {self.max_length})")
        total = self.wall + self.eat_self + self.no_eat

        if total == 0:
            return

        ends = [
            f'wall: {self.wall} ({self.wall / total * 100:.2f}%)',
            f'self: {self.eat_self} ({self.eat_self / total * 100:.2f}%)',
            f'no eat: {self.no_eat} ({self.no_eat / total * 100:.2f}%)'
        ]

        print(f"Ends: {', '.join(ends)}")

    def __init_snake(self):
        snake = []
        x0 = self.field_width // 2
        y0 = self.field_height // 2

        for i in range(Snake.INITIAL_LENGTH):
            snake.append({'x': x0, 'y': y0 + i})

        return snake

    def __is_inside_snake(self, x, y, start: int = 0):
        for p in self.snake[start:]:
            if p['x'] == x and p['y'] == y:
                return True

        return False

    def __init_food(self):
        food = {'x': 0, 'y': 0}

        while True:
            food['x'] = np.random.randint(self.field_width + 1)
            food['y'] = np.random.randint(self.field_height + 1)

            if not self.__is_inside_snake(food['x'], food['y']):
                return food

    def __move_snake(self, dx, dy) -> str:
        head = self.snake[0]
        head_x, head_y = head['x'], head['y']

        if head_x + dx < 0 or head_y + dy < 0 or head_x + dx > self.field_width or head_y + dy > self.field_height:
            self.wall += 1
            return Snake.WALL

        if self.__is_inside_snake(head_x + dx, head_y + dy, 1):
            self.eat_self += 1
            return Snake.EAT_SELF

        if head_x + dx == self.food['x'] and head_y + dy == self.food['y']:
            self.snake.insert(0, {'x': head_x + dx, 'y': head_y + dy})
            self.food = self.__init_food()
            self.max_length = max(len(self.snake), self.max_length)
            self.steps_without_food = 0
            return Snake.EAT_FOOD

        self.steps_without_food += 1

        if self.steps_without_food > self.field_width * self.field_height * 2:
            self.no_eat += 1
            return Snake.NO_EAT

        for i in reversed(range(1, len(self.snake))):
            self.snake[i]['x'] = self.snake[i - 1]['x']
            self.snake[i]['y'] = self.snake[i - 1]['y']

        self.snake[0]['x'] += dx
        self.snake[0]['y'] += dy

        return Snake.DEFAULT

    def __state_to_tensor(self):
        state = np.zeros(self.get_observation_space_shape())
        state[self.food['y'], self.food['x'], Snake.FOOD_CELL] = 1
        state[self.snake[0]['y'], self.snake[0]['x'], Snake.HEAD_CELL] = 1

        for cell in self.snake:
            state[cell['y'], cell['x'], Snake.SNAKE_CELL] = 1

        return state

    def __is_collision(self, point):
        if point['x'] < 0 or point['x'] > self.field_width:
            return True

        if point['y'] < 0 or point['y'] > self.field_height:
            return True

        return self.__is_inside_snake(point['x'], point['y'])

    def __distance_to_collision(self, x0, y0, dx, dy, from_head: bool = False):
        x = x0 + (dx if from_head else 0)
        y = y0 + (dy if from_head else 0)
        i = 1

        while 0 <= x <= self.field_width and 0 <= y <= self.field_height and not self.__is_inside_snake(x, y, 1):
            x += dx
            y += dy
            i += 1

        return [dx * i / self.field_width, dy * i / self.field_height]

    def __state_to_vector(self):
        head_x = self.snake[0]['x']
        head_y = self.snake[0]['y']

        food_x = self.food['x']
        food_y = self.food['y']

        pointL = {'x': head_x - 1, 'y': head_y}
        pointR = {'x': head_x + 1, 'y': head_y}
        pointU = {'x': head_x, 'y': head_y - 1}
        pointD = {'x': head_x, 'y': head_y + 1}

        dirL = self.direction['dx'] == -1
        dirR = self.direction['dx'] == 1
        dirU = self.direction['dy'] == -1
        dirD = self.direction['dy'] == 1

        distances = [
            *self.__distance_to_collision(head_x, head_y, self.direction['dx'], self.direction['dy'], True),
            *self.__distance_to_collision(head_x - self.direction['dy'], head_y + self.direction['dx'], self.direction['dx'], self.direction['dy']),
            *self.__distance_to_collision(head_x + self.direction['dy'], head_y - self.direction['dx'], self.direction['dx'], self.direction['dy']),

            *self.__distance_to_collision(head_x, head_y, -self.direction['dx'], -self.direction['dy'], True),
            *self.__distance_to_collision(head_x - self.direction['dy'], head_y + self.direction['dx'], -self.direction['dx'], -self.direction['dy']),
            *self.__distance_to_collision(head_x + self.direction['dy'], head_y - self.direction['dx'], -self.direction['dx'], -self.direction['dy']),

            *self.__distance_to_collision(head_x, head_y, self.direction['dy'], -self.direction['dx'], True),
            *self.__distance_to_collision(head_x + self.direction['dx'], head_y + self.direction['dy'], self.direction['dy'], -self.direction['dx']),
            *self.__distance_to_collision(head_x - self.direction['dx'], head_y - self.direction['dy'], self.direction['dy'], -self.direction['dx']),

            *self.__distance_to_collision(head_x, head_y, -self.direction['dy'], self.direction['dx'], True),
            *self.__distance_to_collision(head_x + self.direction['dx'], head_y + self.direction['dy'], -self.direction['dy'], self.direction['dx']),
            *self.__distance_to_collision(head_x - self.direction['dx'], head_y - self.direction['dy'], -self.direction['dy'], self.direction['dx'])
        ]

        vector = [
            (dirU and self.__is_collision(pointU)) or
            (dirD and self.__is_collision(pointD)) or
            (dirL and self.__is_collision(pointL)) or
            (dirR and self.__is_collision(pointR)),

            (dirU and self.__is_collision(pointR)) or
            (dirD and self.__is_collision(pointL)) or
            (dirU and self.__is_collision(pointU)) or
            (dirD and self.__is_collision(pointD)),

            (dirU and self.__is_collision(pointR)) or
            (dirD and self.__is_collision(pointL)) or
            (dirR and self.__is_collision(pointU)) or
            (dirL and self.__is_collision(pointD)),

            dirL,
            dirR,
            dirU,
            dirD,

            food_x < head_x,
            food_x > head_x,
            food_y < head_y,
            food_y > head_y,

            self.direction['dx'],
            self.direction['dy'],

            (head_x - 0) / self.field_width,
            (head_y - 0) / self.field_height,
            (head_x - self.field_width) / self.field_width,
            (head_y - self.field_height) / self.field_height,

            (head_x - food_x) / self.field_width,
            (head_y - food_y) / self.field_height,

            *distances
        ]

        return np.array(vector)

    def __get_state(self):
        return self.__state_to_tensor() if self.use_conv else self.__state_to_vector()

    def __get_reward(self, move: str) -> float:
        if move == Snake.WALL:
            return -1

        if move == Snake.EAT_SELF:
            return -2

        if move == Snake.NO_EAT:
            return -4

        if move == Snake.EAT_FOOD:
            return len(self.snake)

        head = self.snake[0]
        prev_dx = head['x'] - self.direction['dx'] - self.food['x']
        prev_dy = head['y'] - self.direction['dy'] - self.food['y']
        prev_dst = abs(prev_dx) + abs(prev_dy)

        curr_dx = head['x'] - self.food['x']
        curr_dy = head['y'] - self.food['y']
        curr_dst = abs(curr_dx) + abs(curr_dy)

        if curr_dst < prev_dst:
            return 0.5 / len(self.snake)

        return -1 / len(self.snake)

    def __get_direction(self, action: int):
        dx, dy = self.direction['dx'], self.direction['dy']

        if action == Snake.TURN_LEFT:
            return dy, -dx

        if action == Snake.TURN_RIGHT:
            return -dy, dx

        return dx, dy

    def __draw_cell(self, img: np.ndarray, cell_width: int, cell_height: int, x: int, y: int, color, filled: bool = False):
        cv2.rectangle(img, (x * cell_width, y * cell_height), ((x + 1) * cell_width, (y + 1) * cell_height), color, -1 if filled else 1)

    def __render_cell(self, screen: pygame.display, cell_width: int, cell_height: int, x: int, y: int, color, filled: bool = False):
        pygame.draw.rect(screen, color, pygame.Rect(x * cell_width, y * cell_height, cell_width, cell_height), 0 if filled else 1)

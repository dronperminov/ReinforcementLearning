import abc
from typing import Union, List, Tuple, Optional
import numpy as np
import pygame


class AbstractEnvironment:
    @abc.abstractmethod
    def get_observation_space_shape(self) -> Union[int, List[int]]:
        pass

    @abc.abstractmethod
    def get_action_space_shape(self) -> int:
        pass

    @abc.abstractmethod
    def sample_action(self, probs: Optional[np.array] = None) -> int:
        pass

    @abc.abstractmethod
    def reset_info(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        pass

    @abc.abstractmethod
    def render(self, screen: pygame.display, width: int, height: int):
        pass

    @abc.abstractmethod
    def get_info(self) -> List[str]:
        pass

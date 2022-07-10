import json
import tensorflow as tf
import argparse
from envs.snake import Snake
from algorithms.dqn import DeepQNetwork
from algorithms.a2c import AdvancedActorCritic
from algorithms.reinforce import ReinforcePolicyGradients
from reiforcement_learning_visualizer import ReinforcementLearningVisualizer


def init_gpu():
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def main():
    parser = argparse.ArgumentParser(description="Reinforcement learning sandbox")

    parser.add_argument("config", type=str, help="Path to json with sandbox config")
    args = parser.parse_args()

    init_gpu()
    environment = Snake(use_conv=True)

    with open(args.config) as f:
        config = json.load(f)

    algorithm_name = config.get('algorithm', '')
    if algorithm_name == 'dqn':
        algorithm = DeepQNetwork(environment, config)
    elif algorithm_name == 'a2c':
        algorithm = AdvancedActorCritic(environment, config)
    elif algorithm_name == 'reinforce':
        algorithm = ReinforcePolicyGradients(environment, config)
    else:
        raise ValueError(f'Unknown algorithm name "{algorithm_name}"')

    visualizer = ReinforcementLearningVisualizer(algorithm)
    visualizer.reset()

    while True:
        visualizer.step()


if __name__ == '__main__':
    main()

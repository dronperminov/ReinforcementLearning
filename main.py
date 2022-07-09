from envs.snake import Snake
from algorithms.dqn import DeepQNetwork
from reiforcement_learning_visualizer import ReinforcementLearningVisualizer


def main():
    environment = Snake(use_conv=True)
    config = {
        'batch_size': 128,
        'min_replay_size': 1000,
        'max_replay_size': 10000,

        'learning_rate': 0.004,
        'optimizer': 'adam',

        'max_epsilon': 0.1,
        'min_epsilon': 0.005,
        'decay': 0.001,
        'gamma': 0.9,

        'train_model_period': 4,
        'update_target_model_period': 100,
        'save_model_path': 'dqn_cnn_snake.h5',

        'agent_architecture': [
            {'type': 'conv', 'fc': 16, 'fs': 3, 'stride': 1, 'padding': 'same', 'activation': 'relu'},
            {'type': 'maxpool', 'scale': 2},
            {'type': 'conv', 'fc': 32, 'fs': 3, 'stride': 1, 'padding': 'same', 'activation': 'relu'},
            {'type': 'maxpool', 'scale': 2},
            {'type': 'flatten'},
            {'type': 'dense', 'size': 256, 'activation': 'relu'}
        ],

        'agent_weights': 'dqn_cnn_snake_15.h5'
    }

    dqn = DeepQNetwork(environment, config)
    visualizer = ReinforcementLearningVisualizer(dqn)
    visualizer.reset()

    while True:
        visualizer.step()


if __name__ == '__main__':
    main()

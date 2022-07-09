from typing import List
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import Huber

from envs.abstract_environment import AbstractEnvironment
from common.model_builder import ModelBuilder
from common.optimizer_builder import OptimizerBuilder
from common.replay_buffer import ReplayBuffer


class DeepQNetwork:
    def __init__(self, environment: AbstractEnvironment, config: dict):
        self.environment = environment
        self.batch_size = config.get('batch_size', 128)
        self.min_replay_size = config.get('min_replay_size', 1000)
        self.replay_buffer = ReplayBuffer(config.get('max_replay_size', 10000))

        self.ddqn = config.get('ddqn', False)
        self.save_model_path = config.get("save_model_path", "model.h5")

        self.model = self.__init_agent(config['agent_architecture'], config['optimizer'], config['learning_rate'])

        if 'agent_weights' in config:
            self.model.load_weights(config['agent_weights'])
            print(f'Loaded weights from "{config["agent_weights"]}"')

        self.target_model = self.__init_agent(config['agent_architecture'], config['optimizer'], config['learning_rate'])
        self.target_model.set_weights(self.model.get_weights())
        print("DQN agent model:")
        print(self.model.summary())

        self.max_epsilon = config.get('max_epsilon', 1)
        self.min_epsilon = config.get('min_epsilon', 0.01)
        self.decay = config.get('decay', 0.004)
        self.gamma = config.get('gamma', 0.9)

        self.train_model_period = config.get('train_model_period', 4)
        self.update_target_model_period = config.get('update_target_model_period', 100)

    def __init_agent(self, architecture: List[dict], optimizer_name: str, learning_rate: float) -> Sequential:
        inputs = self.environment.get_observation_space_shape()
        outputs = self.environment.get_action_space_shape()
        last_layer = {'type': 'dense', 'size': outputs}

        agent = ModelBuilder.build(inputs, architecture + [last_layer])
        optimizer = OptimizerBuilder.build(optimizer_name, learning_rate)
        loss = Huber()
        agent.compile(optimizer=optimizer, loss=loss)

        return agent

    def get_title(self) -> str:
        return f'DQN (gamma: {self.gamma}, batch_size: {self.batch_size})'

    def get_action(self, state: np.ndarray):
        if np.random.random() < self.epsilon:
            return self.environment.sample_action()

        q = self.model.predict_on_batch(np.array([state]))[0]
        return np.argmax(q)

    def train(self):
        if len(self.replay_buffer) < self.min_replay_size:
            return

        mini_batch = self.replay_buffer.sample(self.batch_size)
        curr_states = np.array([v['state'] for v in mini_batch])
        next_states = np.array([v['next_state'] for v in mini_batch])

        target = self.model.predict_on_batch(curr_states)
        target_next = self.model.predict_on_batch(next_states)
        target_val = self.target_model.predict_on_batch(next_states)

        for i, info in enumerate(mini_batch):
            action = info['action']
            target[i][action] = info['reward']

            if not info['done']:
                if self.ddqn:
                    target[i][action] += self.gamma * target_val[i][np.argmax(target_next[i])]
                else:
                    target[i][action] += self.gamma * np.amax(target_val[i])

        self.model.train_on_batch(curr_states, target)

    def reset(self):
        self.environment.reset_info()
        self.replay_buffer.clear()
        self.episode = 0
        self.steps_to_update_target_model = 0
        self.best_reward = float('-inf')
        self.reset_episode()

    def reset_episode(self):
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay * self.episode)
        self.episode += 1

        self.state = self.environment.reset()
        self.done = False
        self.total_training_rewards = 0

    def step(self):
        action = self.get_action(self.state)
        state, reward, done = self.environment.step(action)

        self.replay_buffer.add(self.state, action, reward, state, done)
        self.steps_to_update_target_model += 1
        self.total_training_rewards += reward
        self.state = state

        if done and self.total_training_rewards > self.best_reward:
            self.best_reward = self.total_training_rewards
            self.model.save(self.save_model_path)
            print(f'Model was saved to "{self.save_model_path}"')

        if self.steps_to_update_target_model % self.train_model_period == 0:
            self.train()

        if not done:
            return {'done': False}

        if self.steps_to_update_target_model > self.update_target_model_period:
            self.target_model.set_weights(self.model.get_weights())
            self.steps_to_update_target_model = 0

        return {'done': True, 'reward': self.total_training_rewards}
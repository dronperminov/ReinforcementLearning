from typing import List, Tuple
import numpy as np
from tensorflow.keras.models import Sequential
from envs.abstract_environment import AbstractEnvironment
from common.model_builder import ModelBuilder
from common.optimizer_builder import OptimizerBuilder


class ReinforcePolicyGradients:
    def __init__(self, environment: AbstractEnvironment, config: dict):
        self.environment = environment

        self.optimizer = OptimizerBuilder.build(config['optimizer'], config['learning_rate'])
        self.agent = self.__init_agent(config['agent_architecture'])

        if 'agent_weights' in config:
            self.agent.load_weights(config['agent_weights'])
            print(f'Model weights were loaded from "{config["agent_weights"]}"')

        print("Reinforce agent:")
        print(self.agent.summary())

        self.gamma = config.get('gamma', 0.9)
        self.save_model_path = config.get("save_model_path", "reinforce.h5")

    def __init_agent(self, architecture: List[dict]) -> Sequential:
        inputs = self.environment.get_observation_space_shape()
        outputs = self.environment.get_action_space_shape()
        last_layer = {'type': 'dense', 'size': outputs, 'activation': 'softmax'}

        agent = ModelBuilder.build(inputs, architecture + [last_layer])
        agent.compile(loss="categorical_crossentropy", optimizer=self.optimizer)

        return agent

    def get_title(self) -> str:
        return f'Reinforce (gamma: {self.gamma})'

    def get_action(self, state: np.ndarray) -> int:
        probs = self.agent.predict_on_batch(np.array([state]))[0]
        action = self.environment.sample_action(probs)

        return action

    def remember(self, state, action, reward):
        self.states.append(state)
        action_onehot = np.zeros([self.environment.get_action_space_shape()])
        action_onehot[action] = 1
        self.actions.append(action_onehot)
        self.rewards.append(reward)

    def get_discounted_rewards(self):
        discounted_rewards = []
        discounted_reward = 0

        for reward in reversed(self.rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            discounted_rewards.append(discounted_reward)

        discounted_rewards.reverse()
        discounted_rewards = np.array(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

        return discounted_rewards

    def update_policy(self):
        states = np.array(self.states)
        actions = np.array(self.actions)
        discounted_rewards = self.get_discounted_rewards()

        self.agent.fit(states, actions, sample_weight=discounted_rewards, verbose=0)
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

    def reset(self):
        self.environment.reset_info()
        self.best_reward = float('-inf')

        self.states = []
        self.actions = []
        self.rewards = []

    def step(self, render=None):
        state = self.environment.reset()
        episode_reward = 0
        done = False

        while not done:
            action = self.get_action(state)
            next_state, reward, done = self.environment.step(action)
            self.remember(state, action, reward)

            if render:
                render()

            state = next_state
            episode_reward += reward

        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.agent.save(self.save_model_path)
            print(f'Model was saved to "{self.save_model_path}"')

        self.update_policy()

        return episode_reward

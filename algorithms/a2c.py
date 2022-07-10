from typing import List, Tuple
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import Huber

from envs.abstract_environment import AbstractEnvironment
from common.model_builder import ModelBuilder
from common.optimizer_builder import OptimizerBuilder


class AdvancedActorCritic:
    def __init__(self, environment: AbstractEnvironment, config: dict):
        self.environment = environment

        self.optimizer = OptimizerBuilder.build(config['optimizer'], config['learning_rate'])
        self.actor = self.__init_actor(config['actor_architecture'])
        self.critic = self.__init_critic(config['critic_architecture'])

        if 'actor_weights' in config:
            self.actor.load_weights(config['actor_weights'])
            print(f'Actor weights were loaded from "{config["actor_weights"]}"')

        if 'critic_weights' in config:
            self.critic.load_weights(config['critic_weights'])
            print(f'Critic weights were loaded from "{config["critic_weights"]}"')

        self.gamma = config.get('gamma', 0.9)

        self.save_actor_path = config.get("save_actor_path", "a2c_actor.h5")
        self.save_critic_path = config.get("save_critic_path", "a2c_critic.h5")

    def __init_actor(self, architecture: List[dict]) -> Sequential:
        inputs = self.environment.get_observation_space_shape()
        outputs = self.environment.get_action_space_shape()
        last_layer = {'type': 'dense', 'size': outputs, 'activation': 'softmax'}

        actor = ModelBuilder.build(inputs, architecture + [last_layer])
        actor.compile(loss='categorical_crossentropy', optimizer=self.optimizer)

        print("Actor model:")
        print(actor.summary())

        return actor

    def __init_critic(self, architecture: List[dict]) -> Sequential:
        inputs = self.environment.get_observation_space_shape()
        last_layer = {'type': 'dense', 'size': 1}

        critic = ModelBuilder.build(inputs, architecture + [last_layer])
        critic.compile(loss=Huber(), optimizer=self.optimizer)

        print("Critic model:")
        print(critic.summary())

        return critic

    def get_title(self) -> str:
        return f'A2C (gamma: {self.gamma})'

    def get_action(self, state: np.ndarray) -> int:
        probs = self.actor.predict_on_batch(np.array([state]))[0]
        action = self.environment.sample_action(np.squeeze(probs))
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

        values = self.critic.predict(states)[:, 0]
        advantages = discounted_rewards - values

        self.actor.fit(states, actions, sample_weight=advantages, verbose=0)
        self.critic.fit(states, discounted_rewards, verbose=0)
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

    def reset(self):
        self.environment.reset_info()
        self.best_reward = float('-inf')

        self.states = []
        self.actions = []
        self.rewards = []

    def save_models(self):
        self.actor.save(self.save_actor_path)
        self.critic.save(self.save_critic_path)
        print(f'Actor was saved to "{self.save_actor_path}"')
        print(f'Critic was saved to "{self.save_critic_path}"')

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
            self.save_models()

        self.update_policy()

        return episode_reward

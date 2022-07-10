from typing import List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Huber

from envs.abstract_environment import AbstractEnvironment
from common.model_builder import ModelBuilder
from common.optimizer_builder import OptimizerBuilder


class AdvancedActorCritic:
    def __init__(self, environment: AbstractEnvironment, config: dict):
        self.environment = environment

        self.model = self.__init_model(config['agent_architecture'])

        if 'agent_weights' in config:
            self.model.load_weights(config['agent_weights'])
            print(f'Model weights were loaded from "{config["agent_weights"]}"')

        self.optimizer = OptimizerBuilder.build(config['optimizer'], config['learning_rate'])
        self.loss = Huber()
        self.gamma = config.get('gamma', 0.9)

        self.save_model_path = config.get("save_model_path", "a2c.h5")

    def __init_model(self, architecture: List[dict]):
        inputs = self.environment.get_observation_space_shape()
        outputs = self.environment.get_action_space_shape()

        input_layer = {'type': 'input', 'inputs': inputs}
        actor_layer = {'type': 'dense', 'size': outputs, 'activation': 'softmax'}
        critic_layer = {'type': 'dense', 'size': 1}

        inputs = ModelBuilder.build_layer(input_layer)
        common = inputs

        for config in architecture:
            common = ModelBuilder.build_layer(config)(common)

        actor = ModelBuilder.build_layer(actor_layer)(common)
        critic = ModelBuilder.build_layer(critic_layer)(common)
        model = Model(inputs=inputs, outputs=[actor, critic])

        print("Model:")
        print(model.summary())

        return model

    def get_title(self) -> str:
        return f'A2C (gamma: {self.gamma})'

    def get_action(self, state: np.ndarray) -> Tuple[int, tf.Tensor, tf.Tensor]:
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        probs, critic_value = self.model(state)
        action = self.environment.sample_action(np.squeeze(probs))
        log_prob = tf.math.log(probs[0, action])

        return action, log_prob, critic_value[0, 0]

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

    def update_policy(self, tape: tf.GradientTape):
        discounted_rewards = self.get_discounted_rewards()

        actor_losses = []
        critic_losses = []

        for log_prob, value, reward in zip(self.probs, self.critics, discounted_rewards):
            diff = reward - value
            actor_losses.append(-log_prob * diff)
            critic_losses.append(self.loss(tf.expand_dims(value, 0), tf.expand_dims(reward, 0)))

        loss = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def reset(self):
        self.environment.reset_info()
        self.best_reward = float('-inf')

        self.probs = []
        self.critics = []
        self.rewards = []

    def step(self, render=None):
        state = self.environment.reset()
        episode_reward = 0
        done = False

        with tf.GradientTape() as tape:
            while not done:
                action, log_prob, critic = self.get_action(state)
                state, reward, done = self.environment.step(action)

                if render:
                    render()

                self.critics.append(critic)
                self.probs.append(log_prob)
                self.rewards.append(reward)
                episode_reward += reward

            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.model.save(self.save_model_path)
                print(f'Model was saved to "{self.save_model_path}"')

            self.update_policy(tape)
            self.critics.clear()
            self.probs.clear()
            self.rewards.clear()

        return episode_reward

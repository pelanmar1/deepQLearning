# -*- coding: utf-8 -*-
import random
import gym_2048
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import os
import tensorflow as tf
import game.browser_control as bc
import game.game_control as gc
import urllib.request




def append_to_csv(fn, csv_row):
    with open(fn, 'a') as fd:
        fd.write(csv_row)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    """Huber loss for Q Learning

    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * \
            K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def get_w():
    print('Beginning file download with urllib2...')
    # url = 'http://localhost:5000/w'
    url = "http://40.112.50.212:5000/w"
    local_fn = LOCAL_W_FN
    urllib.request.urlretrieve(url, local_fn)
    return local_fn


EPISODES = 10000
PROGRESS_FN = "temp_prog.csv"
WEIGHTS_FN = "./save/temp_w.h5"
LOCAL_W_FN = './dwl/pesos_az.h5'
azure_weights = True

if __name__ == "__main__":
    bc.init()
    state_size = 16  # Entrada de la red
    action_size = 4  # Salida de la red
    agent = DQNAgent(state_size, action_size)
    if azure_weights is True:
        print("Descargando pesos de Azure...")
        w_fn = get_w()
        agent.load(w_fn)
    done = False
    batch_size = 32

    for e in range(EPISODES):
        gc.restart_game()
        state = gc.getGrid()
        max_tile = 0
        max_score = 0
        state = np.reshape(state, [1, state_size])
        for time in range(10000):
            # env.render()
            # print("\n")

            action = agent.act(state)
            curr_score = gc.getScore()
            gc.performMove(action)
            done = gc.isOver()
            if not done:
                new_score = gc.getScore()
                reward = new_score - curr_score
                next_state = gc.getGrid()
                # done = gc.isOver()
                # next_state, reward, done, _ = env.step(action)
                max_score += reward
                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                max_tile = max(np.max(state), max_tile)
                state = next_state
            if done:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}, #moves: {}, max_tile: {}".format(
                    e, EPISODES, max_score, agent.epsilon, time, max_tile))
                csv_row = "{};{};{};{};{};{}\n".format(
                    e, EPISODES, max_score, agent.epsilon, time, max_tile)
                append_to_csv(PROGRESS_FN, csv_row)
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 10 == 0:
            print("Saving weights...")
            agent.save(WEIGHTS_FN)

import random
from collections import deque

import numpy as np
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from tqdm import tqdm

from constants import (SAVE_MODEL_FILE, GAMMA, EPSILON, EPSILON_DECAY, EPSILON_MIN,
                       BATCH_SIZE, LEARNING_RATE, EPOCHS, ACTIONS_THRESHOLD, START_NN_USING_THRESHOLD)


def create_model(lr):
    model = Sequential()
    model.add(Conv2D(32, (7, 7), activation='relu', input_shape=(240, 320, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (4, 4), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten()) # Add dropout
    model.add(Dense(800, activation='relu'))
    model.add(Dense(len(ACTIONS), activation='relu')) # Check activation (softmax?)
    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=lr))
    return model


class BlackScreenException(Exception):
    pass


ACTIONS = [[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0]]


class DQN:

    def __init__(self, game, use_saved=False):
        self.game = game
        self._memory = deque(maxlen=ACTIONS_THRESHOLD)

        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN

        self.batch_size = BATCH_SIZE
        self.learning_rate = LEARNING_RATE

        self.use_saved = use_saved
        self.is_trainable = not self.use_saved

        self.number_of_trainings = 0

        self.random_action_cnt = 0
        self.nn_action_cnt = 0

        if not self.use_saved:
            self.model = create_model(lr=self.learning_rate)
        else:
            self.model = load_model(SAVE_MODEL_FILE)

    def get_memory_size(self):
        return len(self._memory)

    def remember(self, previous_state, state, reward, action, done):
        if not self.is_trainable:
            return
        if not state.any():
            return
        _state = preprocess_image(state)
        _previous_state = preprocess_image(previous_state)
        _action = np.argmax(action)
        self._memory.append([_previous_state, _state, reward, _action, done])

    def act(self, state):
        try:
            state = preprocess_image(state)
        except BlackScreenException:
            self.random_action_cnt += 1
            return random.choice(ACTIONS)

        if self.use_saved or (self.number_of_trainings > START_NN_USING_THRESHOLD and (random.random() > self.epsilon)):
            self.nn_action_cnt += 1
            index = np.argmax(self.model.predict(state))
            return ACTIONS[index]
        else:
            self.random_action_cnt += 1
            return random.choice(ACTIONS)

    def train(self):
        if not self.is_trainable:
            return

        x = []
        y = []
        if len(self._memory) < BATCH_SIZE:
            training_batch = self._memory.copy()
        else:
            training_batch = random.sample(self._memory, BATCH_SIZE)
        print("Prepare for train: ")
        for previous_state, state, reward, action, done in tqdm(training_batch):
            target = self.model.predict(previous_state)

            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.gamma * self.model.predict(state).mean()  # Check std

            x.append(state)
            y.append(target)

        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)
        self.model.fit(x, y, epochs=EPOCHS, verbose=1, batch_size=20)
        self.model.save(SAVE_MODEL_FILE)
        self.number_of_trainings += 1
        self._memory.clear()
        if self.number_of_trainings > START_NN_USING_THRESHOLD:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            print("Epsilon: ", self.epsilon)


def preprocess_image(state):
    if not state.any():
        raise BlackScreenException
    state = state / 255
    state = np.transpose(state, [1, 2, 0])
    return np.expand_dims(state, axis=0)

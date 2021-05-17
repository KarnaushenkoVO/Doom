import random

from tqdm import tqdm
from vizdoom import *

from agent import DQN
from constants import TRAIN_GAMES

random.seed(1)
game = DoomGame()
game.load_config("/home/karnaushenko/ml_train/openaoi_doom/CustomCorridor.cfg")
game.init()
game.set_window_visible(False)


def calculate_additional_reward(previous_variables, variables):
    additional_reward = 0
    if previous_variables[0] != variables[0]:
        additional_reward += 20
    if previous_variables[2] != variables[2]:
        additional_reward -= 5
    return additional_reward


def train(game):
    agent = DQN(game)

    for i in tqdm(range(TRAIN_GAMES)):
        game.new_episode()
        previous_variables = None
        previous_img = None
        done = False
        local_history = []
        total_reward = 0
        while not done:
            state = game.get_state()

            img = state.screen_buffer
            variables = state.game_variables
            if previous_variables is None:
                previous_variables = variables
            if previous_img is None:
                previous_img = img

            action = agent.act(img)
            reward = game.make_action(action)
            done = game.is_episode_finished()
            reward = (reward + calculate_additional_reward(previous_variables, variables)) / 100
            total_reward += reward
            local_history.append([previous_img, img, reward, action, done])
            previous_variables = variables
            previous_img = img

        if total_reward >= 0:
            for previous_state, state, reward, action, done in local_history:
                agent.remember(previous_state, state, reward, action, done)
            agent.train()


if __name__ == '__main__':
    train(game)

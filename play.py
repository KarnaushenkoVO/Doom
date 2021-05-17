import random

from tqdm import tqdm
from vizdoom import *

from agent import DQN
from constants import PLAY_GAMES

random.seed(1)
game = DoomGame()
game.load_config("/home/karnaushenko/ml_train/openaoi_doom/CustomCorridor.cfg")
game.init()


def play(game):
    agent = DQN(game, use_saved=True)
    for i in tqdm(range(PLAY_GAMES)):
        game.new_episode()
        done = False
        while not done:
            state = game.get_state()
            img = state.screen_buffer
            action = agent.act(img)
            print(action)
            game.make_action(action)
            done = game.is_episode_finished()


if __name__ == '__main__':
    play(game)

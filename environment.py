import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy

# from pyboy.plugins.game_wrapper_pokemon_gen1 import GameWrapperPokemonGen1
from config import Config

"""
https://docs.pyboy.dk/index.html

GameWrapperPokemonGen1 - already returned, check if the methods defined in this work


load_state
save_state??
How to start game after first barrier

"""


class PokemonRedEnv(gym.Env):

    def __init__(self, path_to_rom, max_steps, action_space, render=False, save=True):
        super(PokemonRedEnv, self).__init__()

        # Initialize PyBoy with the ROM
        self.path_to_rom = path_to_rom
        self.render = render
        self.save = save

        # Define action space (buttons: A, B, Start, Up, Down, Left, Right)
        self.action_space = spaces.Discrete(len(action_space.keys()))

        # Max steps per episode
        self.max_steps = max_steps
        self.current_step = 0

        self.pyboy = PyBoy(path_to_rom)
        self.game_wrapper = self.pyboy.game_wrapper

        # Observation space (all Memory)
        self.observation_space = self.pyboy.memory

        # self.pyboy.set_emulation_speed(1)

    def reset(self):
        # Reset the game state
        self.pyboy.stop()
        self.pyboy = PyBoy(self.path_to_rom)
        self.game_wrapper = self.pyboy.game_wrapper
        self.current_step = 0

        # Return the initial observation
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        assert self.action_space.contains(action)

        # Execute the action (button press)
        self._take_action(action)

        # Increment step counter
        self.current_step += 1
        self.pyboy.tick(1, render=self.render)

        # Get the current observation
        observation = self._get_observation()

        # Calculate reward (customize this based on your needs)
        reward = self._get_reward()

        # Check if the episode is done
        done = self.current_step >= self.max_steps

        # Additional info (can be used for debugging)
        info = {
            # "player_position": self.game_wrapper.player_position,
            "game_area_collision": self.game_wrapper.game_area_collision,
            "game_area": self.game_wrapper.game_area,
        }

        truncated = False  # what is this??

        return observation, reward, done, truncated, info

    def start(self):
        # Start the PyBoy instance and game
        self.game_wrapper.start_game()

    def close(self):
        # Close the PyBoy instance
        self.pyboy.stop(save=self.save)

    def _take_action(self, action):
        # Map action to button press

        button = Config.button_map.get(action, None)

        if button:
            self.pyboy.button(button)

    def _get_observation(self):
        # Get the current memory state as an observation
        return np.array(self.pyboy.memory)

    def _get_reward(self):
        # Customize this function to calculate the reward based on the game state
        return 0

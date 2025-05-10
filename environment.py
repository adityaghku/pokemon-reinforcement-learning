import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
import random

# from pyboy.plugins.game_wrapper_pokemon_gen1 import GameWrapperPokemonGen1
from config import Config

"""
https://docs.pyboy.dk/index.html

GameWrapperPokemonGen1 - already returned, check if the methods defined in this work

"""


class PokemonRedEnv(gym.Env):

    def __init__(
        self, path_to_rom, max_steps, action_space, render=False, save=False, sound=True
    ):
        super(PokemonRedEnv, self).__init__()

        # Initialize PyBoy with the ROM
        self.path_to_rom = path_to_rom
        self.render = render
        self.save = save
        self.sound = sound

        # Define action space (buttons: A, B, Start, Up, Down, Left, Right)
        self.action_space = spaces.Discrete(len(action_space.keys()))

        # Max steps per episode
        self.max_steps = max_steps
        self.current_step = 0

        self.pyboy = PyBoy(
            path_to_rom, sound_emulated=self.sound, cgb=None, log_level="DEBUG"
        )

        self.game_wrapper = self.pyboy.game_wrapper

        assert self.pyboy.cartridge_title == "POKEMON RED"

        # Observation space (all Memory)
        self.observation_space = np.zeros(65536, dtype=np.uint8)

        # self.pyboy.set_emulation_speed(1)

    def start(self):
        # Start the PyBoy instance and game,
        self.game_wrapper.start_game()

        # Input sequence to skip intro and reach Pallet Town
        # Approximate sequence: title screen, name player/rival, skip dialogues
        input_sequence = [
            ("start", 60),  # Press Start at title screen
            ("a", 60),  # Confirm
            ("a", 60),  # Select default name (RED)
            ("a", 60),  # Confirm name
            ("a", 60),  # Select default rival name (BLUE)
            ("a", 60),  # Confirm rival name
            ("a", 60),  # Skip Oak's dialogue
            ("a", 60),
            ("a", 60),
            ("a", 60),
            ("a", 60),  # Ensure all dialogues are skipped
        ]

        for button, frames in input_sequence:
            self.pyboy.button(button)
            for _ in range(frames):
                self.pyboy.tick(render=self.render)

        # Inject randomness into the start state
        random_ticks = random.randint(0, 20)
        for _ in range(random_ticks):
            self.pyboy.tick(render=self.render)

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
        self.pyboy.tick(count=1, render=self.render)

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

    def close(self):

        # Close the PyBoy instance
        self.pyboy.stop(save=self.save)

    def _take_action(self, action):
        # Map action to button press

        button = Config.button_map.get(action, None)

        if button:
            self.pyboy.button(button, delay=1)

    def _get_observation(self):

        # Get the current memory state as an observation
        # https://docs.pyboy.dk/#pyboy.PyBoyMemoryView

        # mem1 = np.array(self.pyboy.memory[0x0000:0x10000], dtype=np.uint8)
        # mem2 = np.array(self.pyboy.memory[0x10000:0x20000], dtype=np.uint8)

        # identical = np.array_equal(mem1, mem2)
        # print("Is Echo RAM identical to WRAM?", identical)

        # They are idenrical , we go till hex 0x10000 and stop there (maybe truncate the memory space)

        self.observation_space = np.array(self.pyboy.memory[0x0000:0x10000])

        return self.observation_space

    def _get_reward(self):
        # Customize this function to calculate the reward based on the game state
        return np.random.random()


def create_env(render=True):
    return PokemonRedEnv(
        Config.path_to_rom,
        max_steps=Config.max_steps,
        action_space=Config.button_map,
        render=render,
    )

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

        # Observation space (all Memory)
        self.observation_space = np.zeros(65536, dtype=np.uint8)
        self.visited_tiles = set()

    def start(self):
        # Start the PyBoy instance and game,

        self.current_step = 0
        self.pyboy = PyBoy(
            self.path_to_rom, sound_emulated=self.sound, cgb=None, log_level="DEBUG"
        )

        self.game_wrapper = self.pyboy.game_wrapper

        assert self.pyboy.cartridge_title == "POKEMON RED"

        # 2x speed
        self.pyboy.set_emulation_speed(2)

        # Manually configured start point
        with open(Config.saved_state, "rb") as f:
            self.pyboy.load_state(f)

        # Inject randomness into the start state
        random_ticks = random.randint(0, 10)
        for _ in range(random_ticks):
            self.pyboy.tick(Config.tick, render=self.render)

    def reset(self):
        # Reset the game state
        self.pyboy.stop()
        self.start()

        # Return the initial observation
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        assert self.action_space.contains(action)

        # Execute the action (button press)
        self._take_action(action)

        # Increment step counter
        self.current_step += 1
        self.pyboy.tick(count=Config.tick, render=self.render)

        # Get the current observation
        observation = self._get_observation()

        # Calculate reward (customize this based on your needs)
        reward = self._get_reward()

        # Check if the episode is done
        done = self.current_step >= self.max_steps

        if done:
            print(
                f"Episode done: current_step={self.current_step}, max_steps={self.max_steps}"
            )

        # Additional info (can be used for debugging)
        info = {
            "game_area_collision": self.game_wrapper.game_area_collision,
            "game_area": self.game_wrapper.game_area,
            # "player_position": self.game_wrapper.player_position,
        }

        truncated = False  # what is this??

        return observation, reward, done, truncated, info

    def close(self):

        # Close the PyBoy instance
        self.pyboy.stop(save=self.save)

        # ONLY ONCE TO SKIP START SCREEN
        # with open(Config.saved_state, "wb") as f:
        #     self.pyboy.save_state(f)

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

        # This is the full observation space, I can try to reduce it maybe
        self.observation_space = np.array(self.pyboy.memory[0x0000:0x10000])

        return self.observation_space

    def _get_reward(self):
        # Get the current game area (tile map)
        game_area = self.game_wrapper.game_area()

        # Convert game area to a tuple for hashability (since it's a numpy array)
        game_area_tuple = tuple(game_area.flatten())

        # Track visited tiles in a set (persist across steps, initialize in __init__)
        if not hasattr(self, "visited_tiles"):
            self.visited_tiles = set()

        # Reward for new tiles: +1 if the current game area hasn't been seen before
        reward = 0
        if game_area_tuple not in self.visited_tiles:
            reward = 1.0
            self.visited_tiles.add(game_area_tuple)

        return reward


def create_env(render=True):
    return PokemonRedEnv(
        Config.path_to_rom,
        max_steps=Config.max_steps,
        action_space=Config.button_map,
        render=render,
    )

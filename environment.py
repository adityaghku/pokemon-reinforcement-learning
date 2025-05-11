import gymnasium as gym
from gymnasium import spaces
import numpy as np
from pyboy import PyBoy
import pandas as pd
from pyboy.utils import WindowEvent

# from pyboy.plugins.game_wrapper_pokemon_gen1 import GameWrapperPokemonGen1
from config import Config

"""
https://docs.pyboy.dk/index.html

GameWrapperPokemonGen1 - already returned, check if the methods defined in this work

"""


class PokemonRedEnv(gym.Env):

    def __init__(
        self,
        path_to_rom,
        max_steps,
        action_space,
        render=False,
        save=False,
        sound=False,
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

        self.HEX_START = 0xC000

        # Observation space (all Memory)
        # BANK 1 ONLY
        self.observation_space = np.zeros(8192, dtype=np.uint8)

        self.visited_tiles = set()
        self.prev_game_area_tuple = None

        self.ram_map = (
            pd.read_csv("ram_map.csv").set_index("HEX")["Description"].to_dict()
        )

    def start(self):
        # Start the PyBoy instance and game,

        self.current_step = 0
        self.pyboy = PyBoy(
            self.path_to_rom,
            sound_emulated=self.sound,
            cgb=None,
            log_level="DEBUG",
            no_input=False,
        )

        self.game_wrapper = self.pyboy.game_wrapper

        assert self.pyboy.cartridge_title == "POKEMON RED"

        # 6x speed
        self.pyboy.set_emulation_speed(6)

        # Manually configured start point
        with open(Config.saved_state, "rb") as f:
            self.pyboy.load_state(f)

        # Inject randomness into the start state
        # random_ticks = random.randint(0, 10)
        # for _ in range(random_ticks):
        #     self.pyboy.tick(Config.tick, render=self.render)

    def reset(self):
        # Reset the game state
        self.close()
        self.start()

        # Return the initial observation
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        assert self.action_space.contains(action)

        self._take_action(action)

        self.current_step += 1

        # Get the current observation
        observation = self._get_observation()

        # Calculate reward (customize this based on your needs)
        reward = self._get_reward()

        # Check if the episode is done
        done = self.current_step >= self.max_steps

        return observation, reward, done

    def _take_action(self, action):

        buttons = Config.button_map.get(action, None)

        # press button then release after some steps
        self.pyboy.send_input(buttons[0])

        for i in range(Config.tick):
            if i == 8:
                self.pyboy.send_input(buttons[1])
            self.pyboy.tick()

    def close(self):

        # Close the PyBoy instance
        self.pyboy.stop(save=self.save)

        # ONLY ONCE TO SKIP START SCREEN
        # with open(Config.saved_state, "wb") as f:
        #     self.pyboy.save_state(f)

    def _get_observation(self):

        # Get the current memory state as an observation
        # https://docs.pyboy.dk/#pyboy.PyBoyMemoryView

        # mem1 = np.array(self.pyboy.memory[0x0000:0x10000], dtype=np.uint8)
        # mem2 = np.array(self.pyboy.memory[0x10000:0x20000], dtype=np.uint8)

        # identical = np.array_equal(mem1, mem2)
        # print("Is Echo RAM identical to WRAM?", identical)

        # They are idenrical , we go till hex 0x10000 and stop there (maybe truncate the memory space)

        # This is the full observation space, I can try to reduce it maybe
        # I reduced it to bank 1
        self.observation_space = np.array(
            self.pyboy.memory[self.HEX_START : 0xE000], dtype=np.uint8
        )

        return self.observation_space

    def _get_reward(self):
        # Get the current game area (tile map)
        game_area = self.game_wrapper.game_area()

        # Convert game area to a tuple for hashability
        game_area_tuple = tuple(game_area.flatten())

        # Reward for new tiles: +1 if the current game area hasn't been seen before
        reward = 0
        if game_area_tuple not in self.visited_tiles:
            reward = 1.0
            self.visited_tiles.add(game_area_tuple)

        # Penalize staying in the same place
        if (
            self.prev_game_area_tuple is not None
            and game_area_tuple == self.prev_game_area_tuple
        ):
            reward -= 0.01  # Small negative reward for idling

        # Update previous game area
        self.prev_game_area_tuple = game_area_tuple

        # Reward for pokemon level sum
        reward += self.get_pokemon_levels_sum()

        return reward

    def get_pokemon_levels_sum(self):
        """
        Calculate the sum of the levels of the six Pokémon in the player's party.
        Uses memory addresses from the RAM map for Pokémon levels.
        """
        # Define the memory addresses for Pokémon levels
        level_addresses = ["D16E", "D19A", "D1C6", "D1F2", "D21E", "D24A"]

        # Initialize sum of levels
        total_level = 0

        # Access the observation space
        memory = self.observation_space

        # Iterate through the addresses
        for addr in level_addresses:
            if (
                addr in self.ram_map
                and "Pokemon" in self.ram_map[addr]
                and "Level" in self.ram_map[addr]
            ):
                # Convert hex address to decimal index
                decimal_addr = int(addr, 16) - self.HEX_START

                # Ensure the address is within the observation space
                if decimal_addr < len(memory):
                    level = memory[decimal_addr]
                    if level < 0:
                        level = 0
                    total_level += level
                else:
                    print(f"Warning: Address {addr} out of observation space range")
            else:
                print(
                    f"Warning: Address {addr} not found in RAM map or not a Pokémon level"
                )

        return total_level


def create_env(render=True):
    return PokemonRedEnv(
        Config.path_to_rom,
        max_steps=Config.max_steps,
        action_space=Config.button_map,
        render=render,
    )

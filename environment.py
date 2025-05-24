import gymnasium as gym
import numpy as np
from pyboy import PyBoy
import pandas as pd
from config import Config
import random
import os
from utils import clean_saves
import uuid

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
        process_id=0,
    ):
        super(PokemonRedEnv, self).__init__()

        # Initialize PyBoy with the ROM
        self.path_to_rom = path_to_rom
        self.render = render
        self.save = save
        self.sound = sound
        self.process_id = process_id

        # Define action space (buttons: A, B, Start, Up, Down, Left, Right)
        self.action_space = list(action_space.keys())

        # Max steps per episode
        self.max_steps = max_steps

        # to reduce observation space
        self.HEX_START = 0xC000

        # Observation space BANK 1 ONLY
        self.observation_space = np.zeros(8192, dtype=np.uint8)

        self.ram_map = (
            pd.read_csv("ram_map.csv").set_index("HEX")["Description"].to_dict()
        )

        clean_saves()

        self.save_files = [
            f"rom/{f}" for f in os.listdir("rom") if f.endswith("_save.state")
        ]

    def start(self):
        # Start the PyBoy instance and game,

        self.visited_tiles_by_map = {}  # Map ID (D35E) -> set of tile tuples
        self.prev_game_area_tuple = None
        self.prev_map_id = None
        self.prev_hp = None
        self.prev_event_flags = None
        self.prev_level_sum = None
        self.prev_pokemon_owned = None
        self.in_battle = 0

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

        random_save = random.choice(self.save_files + [Config.start_state])
        print(f"Loading {random_save}")

        # Manually configured start point, random pick from one of the saves
        with open(random_save, "rb") as f:
            self.pyboy.load_state(f)

        # Inject randomness into the start state
        random_ticks = random.randint(0, 10)
        for _ in range(random_ticks):
            self.pyboy.tick(Config.tick, render=self.render)

        # Initialize previous states for reward calculation
        self.prev_level_sum = self._get_pokemon_levels()
        self.prev_pokemon_owned = self._get_total_pokemon_owned()
        self.prev_hp = self._get_total_hp()

    def reset(self, episode):
        # Reset the game state

        self.close(episode)
        self.start()

        # Return the initial observation
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        assert action in self.action_space

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

        self.pyboy.tick(8)
        self.pyboy.send_input(buttons[1])
        self.pyboy.tick(Config.tick - 8)

    def close(self, episode):
        if self.pyboy is not None:
            # Save with unique filename per process
            if episode % 50 == 0 and self.save and episode > 0:
                self.save_files = [
                    f"rom/{f}" for f in os.listdir("rom") if f.endswith("_save.state")
                ]
                highest_index = 0
                if self.save_files:
                    for file in self.save_files:
                        if file != Config.start_state:
                            try:
                                index = int(file.split("_")[0].replace("rom/", ""))
                                highest_index = max(highest_index, index)
                            except ValueError:
                                continue
                # Use process_id and UUID for unique save file
                file_name = f"rom/{highest_index + 1}_save_p{self.process_id}_{uuid.uuid4().hex[:8]}.state"
                with open(file_name, "wb") as f:
                    self.pyboy.save_state(f)
            self.pyboy.stop(save=self.save)

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
        reward = 0.0

        # Exploration reward (already based on changes)
        exploration_reward = self.get_exploration_reward()
        reward += exploration_reward  # Scale to 0-25 range

        # Level increase reward
        level_change_reward = self.get_pokemon_levels_change()
        reward += level_change_reward * 2.0  # Scale to 0-10 per level

        # New Pokémon caught reward
        new_pokemon_reward = self.get_new_pokemon_owned()
        reward += new_pokemon_reward * 5.0  # Scale to 0-10 per Pokémon

        # Battle win reward
        battle_reward = self.won_battle()
        reward += battle_reward * 0.5  # Scale to 0-10

        # HP change reward
        hp_change = self.get_change_in_pokemon_hp()
        hp_reward = hp_change * 0.05  # Scale to ~0-5 (positive or negative)
        reward += hp_reward

        # Story event reward (already based on changes)
        story_reward = self.get_story_event_reward()
        reward += story_reward  # Scale to 0-10

        # Cap total reward to prevent extreme values
        reward = np.clip(reward, -10.0, 50.0)

        return reward

    def get_exploration_reward(self):
        # Get current map ID (D35E)
        map_id = self.get_ram_value_in_state("D35E")

        # Get current game area (tile map)
        game_area = self.game_wrapper.game_area()
        game_area_tuple = tuple(game_area.flatten())

        reward = 0.0

        # Initialize set for this map if not exists
        if map_id not in self.visited_tiles_by_map:
            self.visited_tiles_by_map[map_id] = set()
            reward += 50.0  # Bonus for entering a new map

        # Reward for new tiles on this map
        if game_area_tuple not in self.visited_tiles_by_map[map_id]:
            self.visited_tiles_by_map[map_id].add(game_area_tuple)
            reward += 10.0  # Reward for new tile configuration

        # Penalize staying in the same place
        if (
            self.prev_game_area_tuple is not None
            and game_area_tuple == self.prev_game_area_tuple
            and map_id == self.prev_map_id
        ):
            reward -= 0.1  # Small penalty for idling

        # Update previous state
        self.prev_game_area_tuple = game_area_tuple
        self.prev_map_id = map_id

        return reward

    def _get_pokemon_levels(self):
        """
        Calculate the sum of the levels of the six Pokémon in the player's party.
        Uses memory addresses from the RAM map for Pokémon levels.
        """
        # Define the memory addresses for Pokémon levels
        level_addresses = ["D16E", "D19A", "D1C6", "D1F2", "D21E", "D24A"]

        # Initialize sum of levels
        total_level = 0

        # Iterate through the addresses
        for addr in level_addresses:
            if (
                addr in self.ram_map
                and "Pokemon" in self.ram_map[addr]
                and "Level" in self.ram_map[addr]
            ):
                level = self.get_ram_value_in_state(addr)
                total_level += max(0, level)

            else:
                print(
                    f"Warning: Address {addr} not found in RAM map or not a Pokémon level"
                )

        return total_level

    def get_pokemon_levels_change(self):
        current_level_sum = self._get_pokemon_levels()
        if self.prev_level_sum is None:
            level_change = 0.0
        else:
            level_change = current_level_sum - self.prev_level_sum
        self.prev_level_sum = current_level_sum
        return max(0.0, level_change)  # Reward only increases

    def _get_total_pokemon_owned(self):
        total = 0
        for addr, value in self.ram_map.items():
            if value.startswith("Own "):
                count = self.get_ram_value_in_state(addr)
                total += max(0, count)
        return total

    def get_new_pokemon_owned(self):
        current_pokemon_owned = self._get_total_pokemon_owned()
        if self.prev_pokemon_owned is None:
            new_pokemon = 0.0
        else:
            new_pokemon = current_pokemon_owned - self.prev_pokemon_owned
        self.prev_pokemon_owned = current_pokemon_owned
        return max(0.0, new_pokemon)  # Reward only new Pokémon

    def get_number_of_party_pokemon(self):
        return self.get_ram_value_in_state("D163")

    def _get_total_hp(self):
        self.hp_addresses = [
            ("D16C", "D16D"),  # Pokémon 1
            ("D198", "D199"),  # Pokémon 2
            ("D1C4", "D1C5"),  # Pokémon 3
            ("D1F0", "D1F1"),  # Pokémon 4
            ("D21C", "D21D"),  # Pokémon 5
            ("D248", "D249"),  # Pokémon 6
        ]
        party_count = min(self.get_number_of_party_pokemon(), 6)
        total_hp = 0
        for i in range(party_count):
            low_byte_addr, high_byte_addr = self.hp_addresses[i]
            low_byte = int(self.get_ram_value_in_state(low_byte_addr))
            high_byte = int(self.get_ram_value_in_state(high_byte_addr))
            hp = (high_byte * 256) + low_byte
            total_hp += hp
        return total_hp

    def get_change_in_pokemon_hp(self):
        current_total_hp = self._get_total_hp()
        if self.prev_hp is None:
            hp_change = 0.0
        else:
            hp_change = current_total_hp - self.prev_hp
        self.prev_hp = current_total_hp
        return hp_change  # Positive for healing, negative for damage

    def won_battle(self):
        battle_state = self.get_ram_value_in_state("D057")
        if battle_state == 0 and self.in_battle == 1:
            self.in_battle = battle_state
            return 20.0  # Fixed reward for winning
        self.in_battle = battle_state
        return 0.0

    def get_story_event_reward(self):
        current_flags = np.zeros(8, dtype=np.uint8)
        i = 0
        for key, value in self.ram_map.items():
            if value.startswith("Event flag "):
                if i < 8:  # Ensure we don't exceed array bounds
                    current_flags[i] = self.get_ram_value_in_state(key)
                    i += 1
        current_bits = np.unpackbits(current_flags, bitorder="little")
        if self.prev_event_flags is None:
            self.prev_event_flags = current_bits
            return 0.0
        new_flags = (current_bits == 1) & (self.prev_event_flags == 0)
        num_new_flags = np.sum(new_flags)
        reward = min(num_new_flags * 200.0, 1000.0)
        self.prev_event_flags = current_bits.copy()
        return reward

    def get_ram_value_in_state(self, addr):
        decimal_addr = int(addr, 16) - self.HEX_START
        if decimal_addr < len(self.observation_space):

            res = self.observation_space[decimal_addr]

            return res
        else:
            print(f"Warning: Address {addr} out of observation space range")
            return 0


def create_env(render, process_id=0):
    return PokemonRedEnv(
        Config.path_to_rom,
        max_steps=Config.max_steps,
        action_space=Config.button_map,
        render=render,
        process_id=process_id,
    )

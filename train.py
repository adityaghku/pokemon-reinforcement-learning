# Example usage
from config import Config
from environment import PokemonRedEnv

if __name__ == "__main__":
    path_to_rom = Config.path_to_rom
    env = PokemonRedEnv(
        path_to_rom,
        max_steps=Config.max_steps,
        action_space=Config.button_map,
        render=True,
    )

    env.start()

    for i in range(10000000):
        print(i)
        action = env.action_space.sample()  # Random action
        print(action)

        observation, reward, done, _, info = env.step(action)

        if done:
            observation, _ = env.reset()

    env.close()

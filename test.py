from environment import create_env
import random

env = create_env()
env.start()

for i in range(3500):
    button = random.randint(0, 6)
    print(i, button)

    env.step(button)

env.close()

This project is an attempt to beat Pokemon Red using Reinforcement Learning, heavily inspired by: https://youtu.be/DcYLT37ImBY?si=qg6vcGQ_LsDB6EUa

I use a PPO algorithm and a custom reward function mapped from the game memory state and a memory map: https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Red_and_Blue/RAM_map#Bank_0

We only use Bank 1 and the map for the rewards because the other banks are useless

The environment is rendered using PyBoy

Disclaimer: I used LLMS such as ChatGPT and Grok for the code snippet generation
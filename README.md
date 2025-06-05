# Pokémon Red RL Agent

This project implements a reinforcement learning (RL) agent using the Proximal Policy Optimization (PPO) algorithm to play *Pokémon Red* on the PyBoy Game Boy emulator. The agent learns to navigate the game, explore the world, battle opponents, and optimize rewards based on exploration, Pokémon levels, battles won, and more.


It is heavily inspired by: https://youtu.be/DcYLT37ImBY?si=qg6vcGQ_LsDB6EUa

## Features
- **PPO Algorithm**: A robust RL algorithm for training the agent with stable policy updates.
- **PyBoy Integration**: Uses the PyBoy emulator to run *Pokémon Red* and extract game state.
- **Custom Reward System**: Rewards based on exploration, Pokémon levels, unique Pokémon owned, battle outcomes, and HP changes.
- **Logging and Checkpoints**: Detailed logging and periodic model checkpointing for training monitoring.
- **Evaluation Mode**: Evaluate trained models with visualization and performance metrics.

## Example Training

![Example Run](static/example_run.gif)


## Requirements
- Python 3.8+
- PyBoy
- PyTorch
- Gymnasium
- NumPy
- Pandas
- TQDM
- A *Pokémon Red* ROM file (`Pokemon Red.gb`)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/pokemon-red-rl.git
   cd pokemon-red-rl
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install pyboy torch gymnasium numpy pandas tqdm
   ```

4. **Prepare the ROM**:
   - Place the *Pokémon Red* ROM file (`Pokemon Red.gb`) in the `rom/` directory.
   - Optionally, add a starting state file (`start_state.state`) in the `rom/` directory for consistent training starts.

5. **Download RAM Map**:
   - Ensure `ram_map.csv` is in the project root. This file maps memory addresses to game variables (e.g., Pokémon levels, event flags).

## Usage
### Training the Agent
To train the PPO agent:
```bash
python train.py
```
- The script will train the agent for the number of episodes specified in `config.py` (`ppo_episodes`).
- Checkpoints are saved in the `checkpoints/` directory every `checkpoint_frequency` episodes.
- The best model (based on loss) is saved in the `model/` directory as `ppo_best.pth`.
- Logs are saved in the `checkpoints/` directory with timestamps.

### Evaluating a Trained Model
To evaluate a trained model:
```bash
python evaluate.py
```
- By default, it loads the best model (`model/ppo_best.pth`) and runs 10 episodes with rendering enabled.
- Customize evaluation by modifying `evaluate.py` or passing arguments (e.g., `model_path`, `num_episodes`).
- Evaluation logs include total rewards, steps, and game state details (e.g., player position, game area).

## Project Structure
- `config.py`: Configuration settings for the environment, training, PPO hyperparameters, and button mappings.
- `environment.py`: Defines the `PokemonRedEnv` Gym environment, interfacing with PyBoy and calculating rewards.
- `NeuralNetwork.py`: Implements the PPO network and agent, including policy and value networks.
- `train.py`: Main training script for collecting trajectories and updating the PPO agent.
- `evaluate.py`: Script for evaluating trained models and logging performance metrics.
- `utils.py`: Utility functions, including logging setup.
- `rom/`: Directory for the ROM file and save states.
- `ram_map.csv`: Maps memory addresses to game variables for observation and reward calculations. (https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Red_and_Blue/RAM_map#Bank_0 and https://bulbapedia.bulbagarden.net/wiki/Save_data_structure_(Generation_I))

 ## Reward Function (in progress)
 - Exploration Reward
 - Standing around penalty
 - Losing HP Penalty
 - Healing pokemon reward
 - Unique pokemon caught reward
 - Winning battles reward
 - Total party level reward


## Configuration
Modify `config.py` to adjust:
- **Environment Settings**: ROM path, start state, emulation speed (`tick`).
- **Training Settings**: Maximum steps per episode, number of episodes, save frequency.
- **PPO Hyperparameters**: Learning rate, clip epsilon, hidden dimensions, temperature decay, etc.
- **Logging**: Print intervals, checkpoint frequency, and directories.

Example:
```python
class Config:
    path_to_rom = "rom/Pokemon Red.gb"
    max_steps = 10000
    ppo_learning_rate = 3e-4
    checkpoint_dir = "checkpoints"
```

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure your code follows PEP 8 style guidelines and includes appropriate tests.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- [PyBoy](https://github.com/Baekalfen/PyBoy): Game Boy emulator for Python.
- [Gymnasium](https://gymnasium.farama.org/): RL environment framework.
- [PyTorch](https://pytorch.org/): Deep learning framework for PPO implementation.
import torch
import numpy as np
from config import Config
from NeuralNetwork import PPOAgent
from utils import setup_logging
import os
from environment import create_env


def evaluate_model(model_path, num_episodes=10, render=True, max_steps=1e4):
    """
    Evaluate a trained PPO agent on the Pokémon Red environment.

    Args:
        model_path (str): Path to the saved model (.pth file).
        num_episodes (int): Number of episodes to evaluate.
        render (bool): Whether to render the game (for visualization).
        max_steps (int): Maximum steps per episode.
    """
    # Setup logging
    logger = setup_logging("ppo_evaluation", Config.checkpoint_dir)

    # Initialize environment
    env = create_env(render=render, process_id=0)

    # Initialize agent
    input_dim = env.observation_space.shape[0]
    action_dim = len(Config.button_map)
    agent = PPOAgent(input_dim, action_dim)

    # Load model
    if not os.path.exists(model_path):
        logger.error(f"Model path {model_path} does not exist.")
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    agent.load(model_path)
    logger.info(f"Loaded model from {model_path}")

    # Start environment
    env.start()

    # Evaluation loop
    episode_rewards = []
    episode_steps = []
    for episode in range(num_episodes):
        state, _ = env.reset(episode)
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            # Get deterministic action (most likely action from policy)
            action_probs, _ = agent.network(torch.FloatTensor(state))
            action = torch.argmax(action_probs).item()

            # Step environment
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state

        # Log episode results
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        logger.info(
            f"Episode {episode + 1}/{num_episodes}: "
            f"Total Reward = {total_reward:.2f}, "
            f"Steps = {steps}, "
        )

    # Log summary statistics
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_steps = np.mean(episode_steps)
    logger.info(
        f"Evaluation Summary ({num_episodes} episodes):\n"
        f"Average Reward = {avg_reward:.2f} (±{std_reward:.2f})\n"
        f"Average Steps = {avg_steps:.2f}"
    )

    # Cleanup
    env.close()


if __name__ == "__main__":
    # Default to the best model saved during training
    model_path = os.path.join(Config.model_dir, f"{Config.model_name}_best.pth")

    # Run evaluation
    evaluate_model(
        model_path=model_path,
        num_episodes=10,
        render=True,  # Set to True for visualization
        max_steps=Config.max_steps,
    )

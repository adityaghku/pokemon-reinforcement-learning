import torch
from config import Config
from environment import create_env
from NeuralNetwork import PPOAgent
import os
from utils import setup_logging


def collect_trajectory(env, agent, max_steps):
    states, actions, log_probs, rewards, values = [], [], [], [], []
    state, _ = env.reset()

    for _ in range(max_steps):
        # Use agent.network.get_action instead of agent.get_action
        action, log_prob = agent.network.get_action(state)
        # Ensure state is on the correct device for network forward pass
        state_tensor = torch.FloatTensor(state).to(agent.network.device)
        action_probs, value = agent.network(state_tensor)

        next_state, reward, done, _, info = env.step(action)

        states.append(state)
        actions.append(action)
        log_probs.append(log_prob.item())
        rewards.append(reward)
        values.append(value.item())

        state = next_state

        if done:
            state, _ = env.reset()

    # Compute returns and advantages
    returns = []
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = (
            rewards[t] + Config.ppo_gamma * values[t + 1]
            if t + 1 < len(values)
            else rewards[t]
        )
        gae = delta + Config.ppo_gamma * Config.ppo_lambda * gae
        returns.insert(0, gae + values[t])
        advantages.insert(0, gae)

    return (states, actions, log_probs, returns, advantages)


def training():
    # Setup logging
    logger = setup_logging("ppo_training", Config.checkpoint_dir)

    # Initialize single environment and agent
    env = create_env()
    input_dim = env.observation_space.shape[0]
    action_dim = len(Config.button_map)
    agent = PPOAgent(input_dim, action_dim)

    # Start environment
    logger.info("Starting environment...")
    env.start()
    logger.info("Reached game start in environment...")

    best_loss = float("inf")

    try:
        for episode in range(Config.ppo_episodes):
            # Collect single trajectory
            logger.info(f"Collecting trajectory for episode {episode}...")
            trajectory = collect_trajectory(env, agent, Config.ppo_steps)

            # Update agent
            loss = agent.update(trajectory)

            # Logging
            if episode % Config.print_interval == 0:
                logger.info(f"Episode {episode}, Loss: {loss}")

            # Save checkpoint
            if episode % Config.checkpoint_frequency == 0:
                checkpoint_path = os.path.join(
                    Config.checkpoint_dir, f"ppo_checkpoint_{episode}.pth"
                )
                agent.save(checkpoint_path)

                # Keep only the latest checkpoints
                checkpoints = sorted(os.listdir(Config.checkpoint_dir))
                if len(checkpoints) > Config.keep_checkpoints:
                    for old_checkpoint in checkpoints[: -Config.keep_checkpoints]:
                        os.remove(os.path.join(Config.checkpoint_dir, old_checkpoint))

                # Save best model
                if Config.save_best and loss < best_loss:
                    best_loss = loss
                    agent.save(
                        os.path.join(Config.model_dir, f"{Config.model_name}_best.pth")
                    )

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Closing environment...")
        env.close()


if __name__ == "__main__":
    os.makedirs(Config.model_dir, exist_ok=True)
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    training()

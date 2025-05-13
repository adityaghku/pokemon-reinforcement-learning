import torch
from config import Config
from environment import create_env
from NeuralNetwork import PPOAgent
import os
from utils import setup_logging
import numpy as np
import logging
import multiprocessing as mp
import tqdm

# Main logger
logger = setup_logging("ppo_training", Config.checkpoint_dir, log_level=logging.DEBUG)


def collect_trajectory(env_id, agent_params, max_steps, episode, queue):

    try:

        # logger per thread
        logger = setup_logging(
            "ppo_training",
            Config.checkpoint_dir,
            log_level=logging.DEBUG,
            process_id=env_id,
        )

        env = create_env(render=False, process_id=env_id)
        input_dim = env.observation_space.shape[0]
        action_dim = len(Config.button_map)
        env.start()

        # Initialize agent and load shared parameters
        agent = PPOAgent(input_dim, action_dim)
        agent.network.load_state_dict(agent_params)
        agent.network.eval()  # Set to evaluation mode for inference

        logger.info(
            f"Process {env_id}: Starting trajectory collection for episode {episode}."
        )
        states, actions, log_probs, rewards, values = [], [], [], [], []
        state, _ = env.reset(episode)

        pbar = tqdm.tqdm(
            total=max_steps,
            desc=f"P{env_id} Ep{episode}",
            position=env_id
            + 1,  # Offset position to avoid overlap with main progress bar
            leave=False,  # Don't persist the bar after completion
        )

        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state).to(agent.network.device)
            action_probs, value = agent.network(state_tensor)
            action, log_prob = agent.network.get_action(
                state, temperature=agent.current_temperature
            )

            next_state, reward, done = env.step(action)

            # Update progress bar description with step and reward
            pbar.set_description(
                f"P{env_id} Ep{episode} Step {step} Reward: {reward:.2f}"
            )
            pbar.update(1)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob.item())
            rewards.append(reward)
            values.append(value.item())

            state = next_state

            if done:
                logger.info(f"Process {env_id}: Episode finished after {step} steps")
                state, _ = env.reset(episode)

        pbar.close()

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

        logger.info(f"Process {env_id}: Trajectory collected with {len(states)} steps.")
        logger.info(f"Process {env_id}: Total cumulative reward: {np.sum(rewards):.2f}")
        logger.info(f"Process {env_id}: Mean advantage: {np.mean(advantages):.4f}")

        # Put trajectory in queue
        queue.put((states, actions, log_probs, returns, advantages))

        # Clean up
        env.close(episode)

    except Exception as e:
        logger.error(f"Process {env_id}: Failed with error: {e}")
        queue.put(None)


def training():

    # Initialize single environment and agent
    input_dim = np.zeros(8192, dtype=np.uint8).shape[0]
    action_dim = len(Config.button_map)
    agent = PPOAgent(input_dim, action_dim)

    best_loss = float("inf")
    mp.set_start_method("spawn", force=True)

    episode_pbar = tqdm.tqdm(
        total=Config.ppo_episodes,
        desc="Training",
        position=0,  # Main progress bar at the top
    )

    try:
        for episode in range(Config.ppo_episodes):

            # Collect single trajectory
            logger.info(f"Collecting trajectory for episode {episode}...")
            queue = mp.Queue()
            processes = []

            # Share current network parameters
            agent_params = agent.network.state_dict()

            for env_id in range(Config.ppo_num_envs):
                p = mp.Process(
                    target=collect_trajectory,
                    args=(env_id, agent_params, Config.max_steps, episode, queue),
                )
                processes.append(p)
                p.start()

            trajectories = []
            for _ in range(Config.ppo_num_envs):
                traj = queue.get()
                if traj is not None:
                    trajectories.append(traj)

            # Wait for all processes to finish
            for p in processes:
                p.join()

            if not trajectories:
                logger.error("No valid trajectories collected.")
                episode_pbar.update(1)
                continue

            # Aggregate trajectories
            all_states, all_actions, all_log_probs, all_returns, all_advantages = (
                [],
                [],
                [],
                [],
                [],
            )
            for traj in trajectories:
                states, actions, log_probs, returns, advantages = traj
                all_states.extend(states)
                all_actions.extend(actions)
                all_log_probs.extend(log_probs)
                all_returns.extend(returns)
                all_advantages.extend(advantages)

            trajectory = (
                all_states,
                all_actions,
                all_log_probs,
                all_returns,
                all_advantages,
            )

            # Update agent
            logger.info(
                f"Main process: Updating agent with {len(all_states)} total steps..."
            )
            loss = agent.update(trajectory)

            # Update episode progress bar with loss
            episode_pbar.set_description(f"Training Ep{episode} Loss: {loss:.4f}")
            episode_pbar.update(1)

            # Logging
            if episode % Config.print_interval == 0:
                logger.info(f"Episode {episode}, Loss: {loss}")

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

    except KeyboardInterrupt:
        logger.info("Main process: Closing program due to KeyboardInterrupt")
    except Exception as e:
        logger.error(f"Main process: Training failed: {e}")
        raise
    finally:
        episode_pbar.close()
        logger.info("Main process: Training completed or interrupted.")


if __name__ == "__main__":
    os.makedirs(Config.model_dir, exist_ok=True)
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    training()

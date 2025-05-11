from pyboy.utils import WindowEvent


class Config:
    # Environment settings
    path_to_rom = "rom/Pokemon Red.gb"
    start_state = "rom/start_state.state"
    tick = 24

    # Training settings
    max_steps = 10000

    # Model saving
    save_frequency = 50
    model_dir = "model"
    model_name = "ppo"

    # Logging
    print_interval = 10
    checkpoint_dir = "checkpoints"
    checkpoint_frequency = 10
    keep_checkpoints = 10
    save_best = True

    # PPO hyperparameters
    ppo_hidden_dim = 512
    ppo_learning_rate = 3e-4
    ppo_clip_epsilon = 0.1
    ppo_gamma = 0.99
    ppo_lambda = 0.95
    ppo_value_loss_coef = 0.5
    ppo_entropy_coef = 0.01
    ppo_episodes = 1000
    ppo_initial_temperature = 1.5
    ppo_min_temperature = 0.1
    ppo_temperature_decay = 0.999

    # to thread?
    ppo_num_envs = 1

    button_map = {
        0: [WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A],
        1: [WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B],
        2: [WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START],
        3: [WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP],
        4: [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN],
        5: [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT],
        6: [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT],
        # 7: 'select'
    }

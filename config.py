class Config:
    # Environment settings
    path_to_rom = "rom/Pokemon Red.gb"

    # Training settings
    max_steps = 1e10

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
    ppo_hidden_dim = 256
    ppo_learning_rate = 3e-4
    ppo_clip_epsilon = 0.2
    ppo_gamma = 0.99
    ppo_lambda = 0.95
    ppo_value_loss_coef = 0.5
    ppo_entropy_coef = 0.01
    ppo_episodes = 1000  # need to swtich back to 1000
    ppo_steps = 2048

    # to thread?
    ppo_num_envs = 1

    button_map = {
        0: "a",
        1: "b",
        2: "start",
        3: "up",
        4: "down",
        5: "left",
        6: "right",
        # 7: 'select'
    }

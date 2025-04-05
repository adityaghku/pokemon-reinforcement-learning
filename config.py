class Config:
    # Environment settings
    path_to_rom = "rom/Pokemon Red.gb"

    # Training settings
    max_steps = 100000

    # Model saving
    save_frequency = 50
    model_dir = "model"
    model_name = "ppo"

    # Logging
    print_interval = 10  # More frequent printing for monitoring

    # Checkpoint settings
    checkpoint_dir = "checkpoints"
    checkpoint_frequency = 10  # More frequent checkpoints
    keep_checkpoints = 10
    save_best = True

    button_map = {
        0: "a",
        1: "b",
        2: "start",
        3: "up",
        4: "down",
        5: "left",
        6: "right",
        # 7: "select",
    }

import torch as th

# imports for file path handling
import os
import sys
from pathlib import Path

sys.path.append(Path(os.getcwd()).parent.as_posix())

misc_params = {
  "env_name": 'roar-e2e-ppo-v0',
  "run_fps": 8,  # TODO Link to the environment RUN_FPS
  "model_directory": Path("./output/PPOe2e_Run_5"),
  "run_name": "Run 5",
  "total_timesteps": int(1e6),
}

wandb_saves = {
  "gradient_save_freq": 512 * misc_params["run_fps"],
  "model_save_freq": 50 * misc_params["run_fps"],
}

PPO_params = dict(
  learning_rate=0.00001,  # be smaller 2.5e-4
  n_steps=1024 * misc_params["run_fps"],
  batch_size=64,  # mini_batch_size = 256?
  # n_epochs=10,
  gamma=0.99,  # rec range .9 - .99
  ent_coef=.00,  # rec range .0 - .01
  # gae_lambda=0.95,
  # clip_range_vf=None,
  # vf_coef=0.5,
  # max_grad_norm=0.5,
  # use_sde=True,
  # sde_sample_freq=5,
  # target_kl=None,
  # tensorboard_log=(Path(misc_params["model_directory"]) / "tensorboard").as_posix(),
  # create_eval_env=False,
  # policy_kwargs=None,
  verbose=1,
  seed=1,
  device=th.device('cuda:0' if th.cuda.is_available() else 'cpu'),
  # _init_setup_model=True,
)

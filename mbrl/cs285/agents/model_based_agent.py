from typing import Callable, Optional, Tuple
import numpy as np
import torch.nn as nn
import torch
import gymnasium as gym
from cs285.infrastructure import pytorch_util as ptu


class ModelBasedAgent(nn.Module):
    def __init__(
        self,
        env: gym.Env,
        make_dynamics_model: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[nn.ParameterList], torch.optim.Optimizer],
        ensemble_size: int,
        mpc_horizon: int,
        mpc_strategy: str,
        mpc_num_action_sequences: int,
        cem_num_iters: Optional[int] = None,
        cem_num_elites: Optional[int] = None,
        cem_alpha: Optional[float] = None,
    ):
        super().__init__()
        self.env = env
        self.mpc_horizon = mpc_horizon
        self.mpc_strategy = mpc_strategy
        self.mpc_num_action_sequences = mpc_num_action_sequences
        self.cem_num_iters = cem_num_iters
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        assert mpc_strategy in (
            "random",
            "cem",
        ), f"'{mpc_strategy}' is not a valid MPC strategy"

        # ensure the environment is state-based
        assert len(env.observation_space.shape) == 1
        assert len(env.action_space.shape) == 1

        self.ob_dim = env.observation_space.shape[0]
        self.ac_dim = env.action_space.shape[0]

        self.ensemble_size = ensemble_size
        self.dynamics_models = nn.ModuleList(
            [
                make_dynamics_model(
                    self.ob_dim,
                    self.ac_dim,
                )
                for _ in range(ensemble_size)
            ]
        )
        self.optimizer = make_optimizer(self.dynamics_models.parameters())
        self.loss_fn = nn.MSELoss()

        # keep track of statistics for both the model input (obs & act) and
        # output (obs delta)
        self.register_buffer(
            "obs_acs_mean", torch.zeros(self.ob_dim + self.ac_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_acs_std", torch.ones(self.ob_dim + self.ac_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_delta_mean", torch.zeros(self.ob_dim, device=ptu.device)
        )
        self.register_buffer(
            "obs_delta_std", torch.ones(self.ob_dim, device=ptu.device)
        )

    def update(self, i: int, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray):
        """
        Update self.dynamics_models[i] using the given batch of data.

        Args:
            i: index of the dynamics model to update
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
            next_obs: (batch_size, ob_dim)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        next_obs = ptu.from_numpy(next_obs)
        obs_acs_norm = torch.cat([obs, acs], dim=1)
        obs_acs_norm = (obs_acs_norm - self.obs_acs_mean) / (self.obs_acs_std + 1e-3)
        obs_delta_norm = ((next_obs - obs) - self.obs_delta_mean) / (self.obs_delta_std + 1e-3)
        obs_delta_pred = self.dynamics_models[i](obs_acs_norm)
        loss = self.loss_fn(obs_delta_norm, obs_delta_pred)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return ptu.to_numpy(loss)

    @torch.no_grad()
    def update_statistics(self, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray):
        """
        Update the statistics used to normalize the inputs and outputs of the dynamics models.

        Args:
            obs: (n, ob_dim)
            acs: (n, ac_dim)
            next_obs: (n, ob_dim)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        next_obs = ptu.from_numpy(next_obs)
        obs_acs = torch.cat([obs, acs], dim=1)

        self.obs_acs_mean = obs_acs.mean(dim=0)
        self.obs_acs_std = obs_acs.std(dim=0)
        self.obs_delta_mean = (next_obs - obs).mean(dim=0)
        self.obs_delta_std = (next_obs - obs).std(dim=0)

    @torch.no_grad()
    def get_dynamics_predictions(
        self, i: int, obs: np.ndarray, acs: np.ndarray
    ) -> np.ndarray:
        """
        Takes a batch of each current observation and action and outputs the
        predicted next observations from self.dynamics_models[i].

        Args:
            obs: (batch_size, ob_dim)
            acs: (batch_size, ac_dim)
        Returns: (batch_size, ob_dim)
        """
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        # get the model's predicted `next_obs`
        obs_acs_norm = torch.cat([obs, acs], dim=-1)
        obs_acs_norm = (obs_acs_norm - self.obs_acs_mean) / (self.obs_acs_std + 1e-3)
        pred_delta = self.dynamics_models[i](obs_acs_norm)
        pred_delta = pred_delta * (self.obs_delta_std + 1e-3) + self.obs_delta_mean # unnormalizing
        pred_next_obs = pred_delta + obs

        return ptu.to_numpy(pred_next_obs)

    def evaluate_action_sequences(self, obs: np.ndarray, action_sequences: np.ndarray):
        """
        Evaluate a batch of action sequences using the ensemble of dynamics models.

        Args:
            obs: starting observation, shape (ob_dim,)
            action_sequences: shape (mpc_num_action_sequences, horizon, ac_dim)
        Returns:
            sum_of_rewards: shape (mpc_num_action_sequences,)
        """
        sum_of_rewards = np.zeros(
            (self.ensemble_size, self.mpc_num_action_sequences), dtype=np.float32
        )
        obs = np.tile(obs, (self.ensemble_size, self.mpc_num_action_sequences, 1))

        for acs in action_sequences.transpose((1, 0, 2)):
            assert acs.shape == (self.mpc_num_action_sequences, self.ac_dim)
            assert obs.shape == (
                self.ensemble_size,
                self.mpc_num_action_sequences,
                self.ob_dim,
            )

            next_obs = np.array([self.get_dynamics_predictions(i, obs[i], acs) for i in range(self.ensemble_size)])
            assert next_obs.shape == (
                self.ensemble_size,
                self.mpc_num_action_sequences,
                self.ob_dim,
            )

            # get the reward for the current step in each rollout
            next_obs_reshaped = next_obs.reshape(self.ensemble_size * self.mpc_num_action_sequences, self.ob_dim)
            acs_repeated = np.tile(acs[np.newaxis, :, :], (self.ensemble_size, 1, 1))
            acs_reshaped = acs_repeated.reshape(self.ensemble_size * self.mpc_num_action_sequences, self.ac_dim)
            rewards, _ = self.env.get_reward(next_obs_reshaped, acs_reshaped)
            rewards = rewards.reshape(self.ensemble_size, self.mpc_num_action_sequences)

            assert rewards.shape == (self.ensemble_size, self.mpc_num_action_sequences)

            sum_of_rewards += rewards

            obs = next_obs

        # now we average over the ensemble dimension
        return sum_of_rewards.mean(axis=0)

    def get_action(self, obs: np.ndarray):
        """
        Choose the best action using model-predictive control.

        Args:
            obs: (ob_dim,)
        """
        # always start with uniformly random actions
        action_sequences = np.random.uniform(
            self.env.action_space.low,
            self.env.action_space.high,
            size=(self.mpc_num_action_sequences, self.mpc_horizon, self.ac_dim),
        )

        if self.mpc_strategy == "random":
            # evaluate each action sequence and return the best one
            rewards = self.evaluate_action_sequences(obs, action_sequences)
            assert rewards.shape == (self.mpc_num_action_sequences,)
            best_index = np.argmax(rewards)
            return action_sequences[best_index][0]
        elif self.mpc_strategy == "cem":
            elite_mean, elite_std = None, None
            for i in range(self.cem_num_iters):
                # the CEM algorithm
                if i == 0:
                    elite_mean = action_sequences.mean(axis=0)
                    elite_std = action_sequences.std(axis=0)

                sampled_actions = np.random.normal(
                    loc=elite_mean, 
                    scale=elite_std, 
                    size=(self.mpc_num_action_sequences, self.mpc_horizon, self.ac_dim)
                )

                rewards = self.evaluate_action_sequences(obs, sampled_actions)
                elite_acs = sampled_actions[rewards.argsort()[-self.cem_num_elites:]]

                elite_mean = self.cem_alpha * elite_acs.mean(axis=0) + (1 - self.cem_alpha) * elite_mean
                elite_std = self.cem_alpha * elite_acs.std(axis=0) + (1 - self.cem_alpha) * elite_std
            return elite_mean[0]
        else:
            raise ValueError(f"Invalid MPC strategy '{self.mpc_strategy}'")

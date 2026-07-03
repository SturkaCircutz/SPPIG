from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from cartpole_env import (
    PAPER_EVAL_ROLLOUTS,
    BangBangCartpolePSM,
    CartpoleEnv,
    Observation,
    cartpole_reward_spec,
    cartpole_space_spec,
    summarize_cartpole_results,
)


PAPER_PPO_TIMESTEPS = 10_000_000


@dataclass
class PPOConfig:
    policy_type: str = "mlp"
    total_timesteps: int = PAPER_PPO_TIMESTEPS
    rollout_steps: int = 1024
    update_epochs: int = 8
    minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_size: int = 64
    seed: int = 0
    initial_log_std: float = 0.0
    eval_rollouts: int = PAPER_EVAL_ROLLOUTS
    eval_test_max_steps: int = 15_000
    pretrain_steps: int = 0
    pretrain_batch_size: int = 256
    pretrain_learning_rate: float = 1e-3
    action_scale: float = 10.0
    num_envs: int = 8
    eval_interval: int = 0
    keep_best: bool = True
    verbose: bool = False
    metrics_output: Optional[str] = None


@dataclass
class PPOResult:
    train_success_rate: float
    test_success_rate: float
    train_reward_mean: float
    test_reward_mean: float
    train_steps_mean: float
    test_steps_mean: float
    train_survival_seconds_mean: float
    test_survival_seconds_mean: float
    timesteps: int


def result_to_metrics(result: PPOResult) -> Dict[str, object]:
    return {
        "timesteps": result.timesteps,
        "train_success_rate": result.train_success_rate,
        "test_success_rate": result.test_success_rate,
        "train_reward_mean": result.train_reward_mean,
        "test_reward_mean": result.test_reward_mean,
        "train_steps_mean": result.train_steps_mean,
        "test_steps_mean": result.test_steps_mean,
        "train_survival_seconds_mean": result.train_survival_seconds_mean,
        "test_survival_seconds_mean": result.test_survival_seconds_mean,
    }


def ppo_paper_protocol_status(cfg: PPOConfig) -> Dict[str, object]:
    train_env = CartpoleEnv.train_env()
    test_env = CartpoleEnv.test_env()
    paper_timestep_budget = cfg.total_timesteps == PAPER_PPO_TIMESTEPS
    paper_test_horizon = cfg.eval_test_max_steps == test_env.cfg.max_steps
    paper_eval_rollouts = cfg.eval_rollouts == PAPER_EVAL_ROLLOUTS
    lstm_minibatches_ok = cfg.policy_type != "lstm" or cfg.minibatches == 1
    single_run_matches_paper_budget = (
        paper_timestep_budget
        and paper_test_horizon
        and paper_eval_rollouts
        and lstm_minibatches_ok
    )
    return {
        "policy_type": cfg.policy_type,
        "train_horizon_seconds": train_env.cfg.horizon_seconds,
        "train_pole_length": train_env.cfg.pole_length,
        "train_horizon_steps": train_env.cfg.max_steps,
        "test_horizon_seconds": test_env.cfg.horizon_seconds,
        "test_pole_length": test_env.cfg.pole_length,
        "reward_spec": cartpole_reward_spec(),
        "space_spec": cartpole_space_spec(train_env.cfg),
        "paper_test_horizon_steps": test_env.cfg.max_steps,
        "selected_test_max_steps": cfg.eval_test_max_steps,
        "paper_eval_rollouts": PAPER_EVAL_ROLLOUTS,
        "selected_eval_rollouts": cfg.eval_rollouts,
        "uses_paper_eval_rollouts": paper_eval_rollouts,
        "paper_timestep_budget": paper_timestep_budget,
        "paper_test_horizon": paper_test_horizon,
        "ppo_lstm_minibatches_fixed_to_one": lstm_minibatches_ok,
        "single_run_matches_paper_budget": single_run_matches_paper_budget,
        "five_seed_hyperparameter_search": False,
        "paper_scale_baseline_protocol": False,
        "limitation": (
            "Standalone PPO training can match the paper timestep and test-horizon budget for one run, "
            "but it is not the paper's full five-seed hyperparameter-search baseline protocol."
        ),
    }


def rollout_to_update_metrics(rollout: "Rollout", update: int, timesteps: int) -> Dict[str, object]:
    horizon_truncations = int(rollout.horizon_truncations.sum().item())
    failure_terminations = int(rollout.failure_terminations.sum().item())
    return {
        "update": update,
        "timesteps": timesteps,
        "rollout_steps": int(rollout.rewards.numel()),
        "reward_mean": float(rollout.rewards.mean().item()),
        "horizon_truncations": horizon_truncations,
        "failure_terminations": failure_terminations,
        "episode_terminations": horizon_truncations + failure_terminations,
    }


class MLPActorCritic(nn.Module):
    def __init__(self, hidden_size: int, initial_log_std: float = 0.0, action_scale: float = 10.0) -> None:
        super().__init__()
        self.action_scale = float(action_scale)
        self.register_buffer("obs_scale", torch.tensor([2.4, 2.0, 0.2095, 2.0], dtype=torch.float32))
        self.actor = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.log_std = nn.Parameter(torch.tensor([initial_log_std], dtype=torch.float32))

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return obs / self.obs_scale

    def reset(self) -> None:
        pass

    def act(self, observation: Observation) -> float:
        obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean = self.action_mean(obs)
        return float(mean.squeeze(0).item())

    def action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self.action_scale * torch.tanh(self.actor(self.normalize(obs)))

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.action_mean(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(self.normalize(obs)).squeeze(-1)
        return action, log_prob, entropy, value


class LSTMActorCritic(nn.Module):
    def __init__(self, hidden_size: int, initial_log_std: float = 0.0, action_scale: float = 10.0) -> None:
        super().__init__()
        self.action_scale = float(action_scale)
        self.register_buffer("obs_scale", torch.tensor([2.4, 2.0, 0.2095, 2.0], dtype=torch.float32))
        self.encoder = nn.Linear(4, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.log_std = nn.Parameter(torch.tensor([initial_log_std], dtype=torch.float32))
        self._state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        return obs / self.obs_scale

    def reset(self) -> None:
        self._state = None

    def act(self, observation: Observation) -> float:
        obs = torch.tensor(observation, dtype=torch.float32).view(1, 1, 4)
        with torch.no_grad():
            features = torch.tanh(self.encoder(self.normalize(obs)))
            if self._state is None:
                output, self._state = self.lstm(features)
            else:
                output, self._state = self.lstm(features, self._state)
            mean = self.action_mean_from_output(output)
        return float(mean.squeeze().item())

    def action_mean_from_output(self, output: torch.Tensor) -> torch.Tensor:
        return self.action_scale * torch.tanh(self.actor(output))

    def initial_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(1, batch_size, self.lstm.hidden_size)
        c0 = torch.zeros(1, batch_size, self.lstm.hidden_size)
        return h0, c0

    def sequence_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = torch.tanh(self.encoder(self.normalize(obs)))
        state = initial_state if initial_state is not None else self.initial_state(obs.shape[1])
        outputs: List[torch.Tensor] = []
        for step in range(obs.shape[0]):
            # Replaying a rollout uses the previous transition's done flag to
            # decide whether the current observation starts a fresh episode.
            if dones is not None and step > 0:
                mask = (1.0 - dones[step - 1]).view(1, obs.shape[1], 1)
                state = (state[0] * mask, state[1] * mask)
            output, state = self.lstm(features[step : step + 1], state)
            outputs.append(output)
        output = torch.cat(outputs, dim=0)
        mean = self.action_mean_from_output(output).squeeze(-1)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        # Continuous PPO ratios are only valid if old and new log-probs are
        # evaluated on the same raw action tensor collected during rollout.
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(output).squeeze(-1)
        return action, log_prob, entropy, value

    def sequence_action_mean(
        self,
        obs: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        features = torch.tanh(self.encoder(self.normalize(obs)))
        state = initial_state if initial_state is not None else self.initial_state(obs.shape[1])
        means: List[torch.Tensor] = []
        for step in range(obs.shape[0]):
            # Match the recurrent reset convention used by PPO updates.
            if dones is not None and step > 0:
                mask = (1.0 - dones[step - 1]).view(1, obs.shape[1], 1)
                state = (state[0] * mask, state[1] * mask)
            output, state = self.lstm(features[step : step + 1], state)
            means.append(self.action_mean_from_output(output).squeeze(-1))
        return torch.cat(means, dim=0)


def train_ppo_cartpole(cfg: PPOConfig, output: Optional[str] = None) -> Tuple[nn.Module, PPOResult]:
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    envs = [CartpoleEnv.train_env(seed=cfg.seed + env_idx) for env_idx in range(cfg.num_envs)]
    model: nn.Module
    if cfg.policy_type == "mlp":
        model = MLPActorCritic(cfg.hidden_size, cfg.initial_log_std, cfg.action_scale)
    elif cfg.policy_type == "lstm":
        model = LSTMActorCritic(cfg.hidden_size, cfg.initial_log_std, cfg.action_scale)
    else:
        raise ValueError("policy_type must be 'mlp' or 'lstm'")

    if cfg.pretrain_steps > 0 and isinstance(model, MLPActorCritic):
        pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.pretrain_learning_rate)
        _pretrain_mlp_actor(model, pretrain_optimizer, cfg)
    elif cfg.pretrain_steps > 0 and isinstance(model, LSTMActorCritic):
        pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.pretrain_learning_rate)
        _pretrain_lstm_actor(model, pretrain_optimizer, cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    obs = torch.tensor([env.reset() for env in envs], dtype=torch.float32)
    episode_steps = torch.zeros(cfg.num_envs, dtype=torch.long)
    lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]]
    lstm_state = model.initial_state(cfg.num_envs) if isinstance(model, LSTMActorCritic) else None
    timesteps = 0
    best_state = copy.deepcopy(model.state_dict())
    best_timesteps = 0
    best_score = float("-inf")
    best_result: Optional[PPOResult] = None
    eval_history: List[Dict[str, object]] = []
    update_history: List[Dict[str, object]] = []
    while timesteps < cfg.total_timesteps:
        remaining_steps = cfg.total_timesteps - timesteps
        active_env_count = min(cfg.num_envs, remaining_steps)
        rollout_steps = min(cfg.rollout_steps, max(1, remaining_steps // active_env_count))
        rollout_lstm_state = None
        if lstm_state is not None:
            rollout_lstm_state = (
                lstm_state[0][:, :active_env_count, :].contiguous(),
                lstm_state[1][:, :active_env_count, :].contiguous(),
            )
        rollout = _collect_rollout(
            envs[:active_env_count],
            model,
            obs[:active_env_count],
            episode_steps[:active_env_count],
            rollout_lstm_state,
            cfg,
            rollout_steps,
        )
        # Rollouts are fixed-size chunks of longer vectorized environment
        # streams; carry boundary state forward instead of forcing resets.
        obs = obs.clone()
        obs[:active_env_count] = rollout.next_obs
        episode_steps = episode_steps.clone()
        episode_steps[:active_env_count] = rollout.next_episode_steps
        if lstm_state is not None and rollout.next_lstm_state is not None:
            h, c = lstm_state
            h = h.clone()
            c = c.clone()
            h[:, :active_env_count, :] = rollout.next_lstm_state[0]
            c[:, :active_env_count, :] = rollout.next_lstm_state[1]
            lstm_state = (h.detach(), c.detach())
        else:
            lstm_state = rollout.next_lstm_state
        timesteps += rollout.rewards.numel()
        update_history.append(rollout_to_update_metrics(rollout, len(update_history) + 1, timesteps))
        if cfg.policy_type == "mlp":
            _update_mlp(model, optimizer, rollout, cfg)
        else:
            _update_lstm(model, optimizer, rollout, cfg)
        if cfg.eval_interval > 0 and (timesteps >= cfg.total_timesteps or timesteps % cfg.eval_interval < rollout.rewards.numel()):
            current = evaluate_ppo_model(
                model,
                timesteps=timesteps,
                rollouts=cfg.eval_rollouts,
                test_max_steps=cfg.eval_test_max_steps,
            )
            score = current.train_success_rate * 1_000_000.0 + current.train_reward_mean
            eval_history.append(result_to_metrics(current))
            if cfg.verbose:
                print(
                    f"eval timesteps={timesteps} "
                    f"train_success={current.train_success_rate:.3f} "
                    f"train_reward={current.train_reward_mean:.1f} "
                    f"test_success={current.test_success_rate:.3f} "
                    f"test_reward={current.test_reward_mean:.1f}",
                    flush=True,
                )
            if score > best_score:
                best_score = score
                best_timesteps = timesteps
                best_state = copy.deepcopy(model.state_dict())
                best_result = current

    if cfg.keep_best and best_result is not None:
        model.load_state_dict(best_state)
        result = best_result
        result.timesteps = best_timesteps
    else:
        result = evaluate_ppo_model(
            model,
            timesteps=timesteps,
            rollouts=cfg.eval_rollouts,
            test_max_steps=cfg.eval_test_max_steps,
        )
    final_metrics = result_to_metrics(result)
    if output is not None:
        os.makedirs(os.path.dirname(output), exist_ok=True)
        torch.save({"config": cfg.__dict__, "state_dict": model.state_dict(), "result": result.__dict__}, output)
    metrics_output = cfg.metrics_output
    if metrics_output is None and output is not None:
        metrics_output = f"{os.path.splitext(output)[0]}_metrics.json"
    if metrics_output is not None:
        metrics_dir = os.path.dirname(metrics_output)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(metrics_output, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "config": cfg.__dict__,
                    "eval_history": eval_history,
                    "update_history": update_history,
                    "selected_result": final_metrics,
                    "reward_spec": cartpole_reward_spec(),
                    "space_spec": cartpole_space_spec(CartpoleEnv.train_env().cfg),
                    "paper_protocol_status": ppo_paper_protocol_status(cfg),
                    "selection_rule": "max train_success_rate, then train_reward_mean when eval_interval > 0 and keep_best is true",
                },
                handle,
                indent=2,
                sort_keys=True,
            )
    return model, result


@dataclass
class Rollout:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    horizon_truncations: torch.Tensor
    failure_terminations: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    next_obs: torch.Tensor
    next_episode_steps: torch.Tensor
    next_value: torch.Tensor
    initial_lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]]
    next_lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]]


def _collect_rollout(
    envs: List[CartpoleEnv],
    model: nn.Module,
    obs: torch.Tensor,
    episode_steps: torch.Tensor,
    lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    cfg: PPOConfig,
    rollout_steps: Optional[int] = None,
) -> Rollout:
    observations: List[torch.Tensor] = []
    actions: List[torch.Tensor] = []
    log_probs: List[torch.Tensor] = []
    rewards: List[torch.Tensor] = []
    dones: List[torch.Tensor] = []
    horizon_truncations: List[torch.Tensor] = []
    failure_terminations: List[torch.Tensor] = []
    values: List[torch.Tensor] = []

    num_envs = len(envs)
    if isinstance(model, LSTMActorCritic):
        state = lstm_state if lstm_state is not None else model.initial_state(num_envs)
        # PPO-LSTM updates replay the exact rollout sequence, so they need the
        # hidden state from before the first collected transition.
        initial_lstm_state = (state[0].detach().clone(), state[1].detach().clone())
    else:
        state = None
        initial_lstm_state = None
    steps_to_collect = cfg.rollout_steps if rollout_steps is None else max(1, rollout_steps)
    for _ in range(steps_to_collect):
        obs_tensor = obs
        observations.append(obs_tensor)
        with torch.no_grad():
            if isinstance(model, LSTMActorCritic):
                seq_obs = obs_tensor.view(1, num_envs, 4)
                features = torch.tanh(model.encoder(model.normalize(seq_obs)))
                output, state = model.lstm(features, state)
                mean = model.action_mean_from_output(output).view(num_envs)
                std = torch.exp(model.log_std).expand_as(mean)
                dist = Normal(mean, std)
                action = dist.sample()
                # Store the log-prob of the sampled Gaussian action; the update
                # later recomputes it for PPO's importance ratio.
                log_prob = dist.log_prob(action)
                value = model.critic(output).view(num_envs)
            else:
                action, log_prob, _, value = model.get_action_and_value(obs_tensor)
                action = action.view(num_envs)
        next_observations: List[Observation] = []
        next_episode_steps = episode_steps.clone()
        step_rewards: List[float] = []
        step_dones: List[float] = []
        step_horizon_truncations: List[float] = []
        step_failure_terminations: List[float] = []
        for env_idx, env in enumerate(envs):
            clipped_action = torch.clamp(action[env_idx], -10.0, 10.0)
            next_obs, reward, done = env.step(float(clipped_action.item()))
            next_episode_steps[env_idx] += 1
            truncated = next_episode_steps[env_idx].item() >= env.cfg.max_steps
            episode_done = done or truncated
            if episode_done:
                next_obs = env.reset()
                next_episode_steps[env_idx] = 0
            next_observations.append(next_obs)
            step_rewards.append(reward)
            step_dones.append(float(episode_done))
            step_horizon_truncations.append(float(truncated and not done))
            step_failure_terminations.append(float(done))
        done_tensor = torch.tensor(step_dones, dtype=torch.float32)
        if state is not None:
            # Reset recurrent memory for envs that terminated or hit the
            # configured horizon before their reset observation is reused.
            mask = (1.0 - done_tensor).view(1, num_envs, 1)
            state = (state[0] * mask, state[1] * mask)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(torch.tensor(step_rewards, dtype=torch.float32))
        dones.append(done_tensor)
        horizon_truncations.append(torch.tensor(step_horizon_truncations, dtype=torch.float32))
        failure_terminations.append(torch.tensor(step_failure_terminations, dtype=torch.float32))
        values.append(value.view(num_envs))
        obs = torch.tensor(next_observations, dtype=torch.float32)
        episode_steps = next_episode_steps

    obs_batch = torch.stack(observations)
    action_batch = torch.stack(actions)
    log_prob_batch = torch.stack(log_probs)
    reward_batch = torch.stack(rewards)
    done_batch = torch.stack(dones)
    horizon_truncation_batch = torch.stack(horizon_truncations)
    failure_termination_batch = torch.stack(failure_terminations)
    value_batch = torch.stack(values)
    with torch.no_grad():
        next_obs_tensor = obs
        if isinstance(model, LSTMActorCritic):
            # Bootstrap from the value of the post-rollout observation using
            # the carried recurrent state after any terminal masks.
            features = torch.tanh(model.encoder(model.normalize(next_obs_tensor.view(1, num_envs, 4))))
            output, _ = model.lstm(features, state)
            next_value = model.critic(output).view(num_envs)
        else:
            _, _, _, next_value = model.get_action_and_value(next_obs_tensor)
    next_lstm_state = None
    if state is not None:
        next_lstm_state = (state[0].detach(), state[1].detach())
    advantages, returns = _gae(reward_batch, done_batch, value_batch, next_value, cfg)
    return Rollout(
        obs_batch,
        action_batch,
        log_prob_batch,
        reward_batch,
        done_batch,
        horizon_truncation_batch,
        failure_termination_batch,
        value_batch,
        advantages,
        returns,
        obs,
        episode_steps,
        next_value,
        initial_lstm_state,
        next_lstm_state,
    )


def _gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    cfg: PPOConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    running_next_value = next_value
    for step in reversed(range(len(rewards))):
        # Terminal and truncated episodes stop both value bootstrapping and
        # advantage recursion for that environment.
        next_nonterminal = 1.0 - dones[step]
        delta = rewards[step] + cfg.gamma * running_next_value * next_nonterminal - values[step]
        last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae
        running_next_value = values[step]
    returns = advantages + values
    return advantages, returns


def _update_mlp(model: nn.Module, optimizer: torch.optim.Optimizer, rollout: Rollout, cfg: PPOConfig) -> None:
    observations = rollout.observations.reshape(-1, 4)
    actions = rollout.actions.reshape(-1)
    old_log_probs = rollout.log_probs.reshape(-1)
    returns = rollout.returns.reshape(-1)
    advantages = rollout.advantages.reshape(-1)
    batch_size = len(advantages)
    minibatch_size = max(1, batch_size // cfg.minibatches)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    for _ in range(cfg.update_epochs):
        # MLP policies can shuffle individual transitions because there is no
        # recurrent context to preserve.
        indices = torch.randperm(batch_size)
        for start in range(0, batch_size, minibatch_size):
            idx = indices[start : start + minibatch_size]
            _, new_log_prob, entropy, value = model.get_action_and_value(observations[idx], actions[idx].unsqueeze(-1))
            ratio = (new_log_prob - old_log_probs[idx]).exp()
            pg_loss1 = -advantages[idx] * ratio
            pg_loss2 = -advantages[idx] * torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
            policy_loss = torch.max(pg_loss1, pg_loss2).mean()
            value_loss = F.mse_loss(value, returns[idx])
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()


def _update_lstm(model: nn.Module, optimizer: torch.optim.Optimizer, rollout: Rollout, cfg: PPOConfig) -> None:
    advantages = (rollout.advantages - rollout.advantages.mean()) / (
        rollout.advantages.std(unbiased=False) + 1e-8
    )
    obs = rollout.observations
    actions = rollout.actions
    old_log_probs = rollout.log_probs
    returns = rollout.returns
    dones = rollout.dones
    for _ in range(cfg.update_epochs):
        # The recurrent policy is updated on the full time-major rollout so
        # hidden-state resets stay aligned with environment terminations.
        _, new_log_prob, entropy, value = model.sequence_action_and_value(
            obs,
            actions,
            dones,
            rollout.initial_lstm_state,
        )
        ratio = (new_log_prob - old_log_probs).exp()
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
        policy_loss = torch.max(pg_loss1, pg_loss2).mean()
        value_loss = F.mse_loss(value, returns)
        loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()


def _pretrain_mlp_actor(model: MLPActorCritic, optimizer: torch.optim.Optimizer, cfg: PPOConfig) -> None:
    env = CartpoleEnv.train_env(seed=cfg.seed + 10_000)
    teacher = BangBangCartpolePSM()
    observations: List[torch.Tensor] = []
    actions: List[float] = []
    for _ in range(64):
        obs = env.reset()
        teacher.reset()
        for _ in range(env.cfg.max_steps):
            action = teacher.act(obs)
            observations.append(torch.tensor(obs, dtype=torch.float32))
            actions.append(action)
            obs, _, done = env.step(action)
            if done:
                break
    obs_batch = torch.stack(observations)
    action_batch = torch.tensor(actions, dtype=torch.float32).unsqueeze(-1)
    for _ in range(cfg.pretrain_steps):
        idx = torch.randint(0, len(obs_batch), (min(cfg.pretrain_batch_size, len(obs_batch)),))
        pred = model.action_mean(obs_batch[idx])
        loss = F.mse_loss(pred, action_batch[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def _pretrain_lstm_actor(model: LSTMActorCritic, optimizer: torch.optim.Optimizer, cfg: PPOConfig) -> None:
    env = CartpoleEnv.train_env(seed=cfg.seed + 20_000)
    teacher = BangBangCartpolePSM()
    obs_sequences: List[torch.Tensor] = []
    action_sequences: List[torch.Tensor] = []
    for _ in range(64):
        obs = env.reset()
        teacher.reset()
        observations: List[torch.Tensor] = []
        actions: List[float] = []
        for _ in range(env.cfg.max_steps):
            action = teacher.act(obs)
            observations.append(torch.tensor(obs, dtype=torch.float32))
            actions.append(action)
            obs, _, done = env.step(action)
            if done:
                break
        if observations:
            obs_sequences.append(torch.stack(observations))
            action_sequences.append(torch.tensor(actions, dtype=torch.float32))

    for _ in range(cfg.pretrain_steps):
        seq_idx = torch.randint(0, len(obs_sequences), (1,)).item()
        obs = obs_sequences[seq_idx].unsqueeze(1)
        target = action_sequences[seq_idx].unsqueeze(1)
        pred = model.sequence_action_mean(obs)
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()


def evaluate_ppo_model(
    model: nn.Module,
    timesteps: int,
    rollouts: int = 20,
    test_max_steps: int = 15_000,
) -> PPOResult:
    train_env = CartpoleEnv.train_env(seed=100)
    test_env = CartpoleEnv.test_env(seed=200)
    train_results = [train_env.rollout(model) for _ in range(rollouts)]
    test_results = [test_env.rollout(model, max_steps=test_max_steps) for _ in range(rollouts)]
    train = summarize_cartpole_results(train_results)
    test = summarize_cartpole_results(test_results)
    return PPOResult(
        train_success_rate=train["success_rate"],
        test_success_rate=test["success_rate"],
        train_reward_mean=train["reward_mean"],
        test_reward_mean=test["reward_mean"],
        train_steps_mean=train["steps_mean"],
        test_steps_mean=test["steps_mean"],
        train_survival_seconds_mean=train["survival_seconds_mean"],
        test_survival_seconds_mean=test["survival_seconds_mean"],
        timesteps=timesteps,
    )

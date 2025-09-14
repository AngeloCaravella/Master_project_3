# File: ev2gym/rl_agent/custom_algorithms.py

import torch as th
from torch.nn import functional as F
import numpy as np

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.sac.sac import SAC
from stable_baselines3.ddpg.ddpg import DDPG

class CustomDDPG(DDPG):
    """
    Algoritmo Deep Deterministic Policy Gradient (DDPG) personalizzato per usare un Replay Buffer con Priorità.
    """

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            is_weights = self.replay_buffer.last_is_weights

            with th.no_grad():
                next_actions = self.actor_target(replay_data.next_observations)
                
                # CORREZIONE: Seleziona il primo output del critico (gestisce l'architettura Twin-Critic)
                target_q_values = self.critic_target(replay_data.next_observations, next_actions)[0]
                
                # CORREZIONE: Assicura che le dimensioni per il broadcasting siano corrette
                rewards = replay_data.rewards.reshape_as(target_q_values)
                dones = replay_data.dones.reshape_as(target_q_values)
                target_q_values = rewards + (1 - dones) * self.gamma * target_q_values

            # CORREZIONE: Seleziona il primo output del critico
            current_q_values = self.critic(replay_data.observations, replay_data.actions)[0]

            with th.no_grad():
                td_error = F.mse_loss(current_q_values, target_q_values, reduction="none")
                td_errors_per_sample = th.sqrt(td_error).squeeze().cpu().numpy()
                self.replay_buffer.update_priorities(td_errors_per_sample)

            critic_loss = (F.mse_loss(current_q_values, target_q_values, reduction="none") * is_weights).mean()
            
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if gradient_step % self.policy_delay == 0:
                # CORREZIONE: Usa il primo critico per calcolare la perdita dell'attore
                actor_loss = -self.critic(replay_data.observations, self.actor(replay_data.observations))[0].mean()
                actor_losses.append(actor_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

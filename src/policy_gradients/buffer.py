from typing import NamedTuple

import torch as ch


class BatchSamples(NamedTuple):
    states: ch.Tensor
    next_states: ch.Tensor
    actions: ch.Tensor
    victim_actions: ch.Tensor
    rewards: ch.Tensor
    not_dones: ch.Tensor


class AdvReplayBuffer:

    def __init__(self, buffer_size, state_dim, action_dim):

        # Adjust buffer size
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pos = 0
        self.size = 0

        self.states = ch.zeros((self.buffer_size, state_dim))
        self.next_states = ch.zeros((self.buffer_size, state_dim))
        self.actions = ch.zeros((self.buffer_size, state_dim))
        self.victim_actions = ch.zeros((self.buffer_size, action_dim))
        self.rewards = ch.zeros((self.buffer_size, 1))
        self.not_dones = ch.zeros((self.buffer_size, 1))

    def add(self, states: ch.Tensor, next_states: ch.Tensor, actions: ch.Tensor, victim_actions: ch.Tensor, rewards: ch.Tensor, not_dones: ch.Tensor):
        batch_size = states.shape[0]
        end_ptr = self.pos + batch_size

        rewards = rewards.unsqueeze(1)
        not_dones = not_dones.unsqueeze(1)

        if end_ptr <= self.buffer_size:
            self.states[self.pos : end_ptr] = states
            self.next_states[self.pos : end_ptr] = next_states
            self.actions[self.pos : end_ptr] = actions
            self.victim_actions[self.pos : end_ptr] = victim_actions
            self.rewards[self.pos : end_ptr] = rewards
            self.not_dones[self.pos : end_ptr] = not_dones
        else:
            overflow = end_ptr - self.buffer_size
            self.states[self.pos :] = states[: batch_size - overflow]
            self.next_states[self.pos :] = next_states[: batch_size - overflow]
            self.actions[self.pos :] = actions[: batch_size - overflow]
            self.victim_actions[self.pos :] = victim_actions[: batch_size - overflow]
            self.rewards[self.pos :] = rewards[: batch_size - overflow]
            self.not_dones[self.pos :] = not_dones[: batch_size - overflow]

            self.states[:overflow] = states[batch_size - overflow :]
            self.next_states[:overflow] = next_states[batch_size - overflow :]
            self.actions[:overflow] = actions[batch_size - overflow :]
            self.victim_actions[:overflow] = actions[: batch_size - overflow :]
            self.rewards[:overflow] = rewards[batch_size - overflow :]
            self.not_dones[:overflow] = not_dones[batch_size - overflow :]

        self.pos = end_ptr % self.buffer_size
        self.size = min(self.size + batch_size, self.buffer_size)

    def sample(self, batch_size):
        indices = ch.randint(0, self.size, (batch_size,))

        batch = BatchSamples(
            states=self.states[indices],
            next_states=self.next_states[indices],
            actions=self.actions[indices],
            victim_actions=self.victim_actions[indices],
            rewards=self.rewards[indices],
            not_dones=self.not_dones[indices],
        )
        return batch

    def get_size(self):
        return self.size

    def update_rewards_with_discriminator(self, disc_net, mode):
        if mode == "ilfd":
            # Prepare inputs by concatenating states and victim actions
            inputs = ch.cat((self.states[: self.size], self.victim_actions[: self.size]), dim=1)
        elif mode == "ilfo":
            # Prepare inputs by concatenating states and next states
            inputs = ch.cat((self.states[: self.size], self.next_states[: self.size]), dim=1)
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # Get discriminator output D(s, a) or D(s, s')
        D_output = disc_net(inputs)

        # Compute the rewards: log(D) - log(1 - D) and clip them to [-1.0, 1.0]
        # rewards = ch.clamp(ch.log(D_output) - ch.log(1 - D_output), min=-1.0, max=1.0)
        rewards = D_output

        # Debugging: Check if shapes match
        if rewards.shape != self.rewards[: self.size].shape:
            raise ValueError(f"Shape mismatch: rewards: {rewards.shape}, self.rewards[: self.size]: {self.rewards[: self.size].shape}")

        # Update rewards in the buffer
        self.rewards[: self.size] = rewards.detach()


class ExpertBuffer:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.not_dones = []

    def add_episodes(self, states: ch.Tensor, next_states: ch.Tensor, actions: ch.Tensor, rewards: ch.Tensor, not_dones: ch.Tensor):
        self.states.append(states)
        self.next_states.append(next_states)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.not_dones.append(not_dones)

    def create_buffer(self, n_episodes):
        buffer_size = sum(states.shape[0] for states in self.states[:n_episodes]) + 1
        adv_replay_buffer = AdvReplayBuffer(buffer_size, self.state_dim, self.action_dim)

        episodes_to_add = min(n_episodes, len(self.states))

        for i in range(episodes_to_add):
            states = self.states[i].clone()
            next_states = self.next_states[i].clone()
            actions = self.actions[i].clone()
            rewards = self.rewards[i].clone()
            not_dones = self.not_dones[i].clone()

            # In expert demonstrations, adverasry is not exist, so actions(adversary's actions) is equal to states
            adv_replay_buffer.add(
                states=states, next_states=next_states, actions=states, victim_actions=actions, rewards=rewards, not_dones=not_dones
            )

        return adv_replay_buffer

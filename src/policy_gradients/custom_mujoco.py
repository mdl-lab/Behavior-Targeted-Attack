import gymnasium as gym
from gym.spaces.box import Box


class custom_mujoco(gym.Env):
    """
    Mujoco custom env
    """

    def __init__(self, game, seed=1234, render_mode=None):
        super(custom_mujoco, self).__init__()

        self.game = game
        self.init_seed = seed
        self.render_mode = render_mode
        self.env = gym.make(game)

        self.observation_space = Box(
            self.env.observation_space.low,
            self.env.observation_space.high,
            shape=self.env.observation_space.shape,
            dtype=self.env.observation_space.dtype,
        )
        self.action_space = Box(self.env.action_space.low, self.env.action_space.high, shape=self.env.action_space.shape, dtype=self.env.action_space.dtype)

    def reset(self, seed=None, options=None):
        """
        reset the environment states

        Parameters
        ----------
        seed : int, optional
        options : Any, optional

        """

        super().reset(seed=seed, options=options)
        state, _ = self.env.reset()

        return state

    def step(self, action):
        """
        step the environment

        Parameters
        ----------
        action : np.ndarray
        """

        state, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        return state, reward, done, info

    def render(self):
        """
        render the environment
        """
        self.env.render()

    def close(self):
        """
        close the environment
        """
        self.env.close()

    def seed(self, seed):
        """
        Not set the seed for reset the environment
        """
        pass


if __name__ == "__main__":
    env = custom_mujoco("Ant-v4")

    print(type(env.observation_space), type(env.action_space))
    print(type(env.action_space) == Box)

    state = env.reset()
    action = env.env.action_space.sample()
    state, reward, is_done, info = env.step(action)

    print(state, reward, is_done, info)

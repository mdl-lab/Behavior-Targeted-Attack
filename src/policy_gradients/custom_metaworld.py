import gym
import metaworld.envs.mujoco.env_dict as _env_dict
from gym.spaces.box import Box


class custom_metaworld(gym.Env):
    """
    MetaWorld custom env
    reference from https://github.com/RyanLiu112/MRN/blob/main/utils.py#L79
    """

    def __init__(self, game, seed=1234, render_mode=None):
        super(custom_metaworld, self).__init__()

        self.game = game
        self.init_seed = seed
        self.render_mode = render_mode
        self.env = self.make_env(self.game, seed=self.init_seed, render_mode=self.render_mode)

        self.observation_space = Box(
            self.env.observation_space.low,
            self.env.observation_space.high,
            shape=self.env.observation_space.shape,
            dtype=self.env.observation_space.dtype,
        )
        self.action_space = Box(
            self.env.action_space.low, self.env.action_space.high, shape=self.env.action_space.shape, dtype=self.env.action_space.dtype
        )

    def reset(self, seed=None, options=None):
        """
        reset the environment states

        Parameters
        ----------
        seed : int, optional
        options : Any, optional

        """

        super().reset(seed=seed, options=options)
        # For training in various states, we need to recreate the environment
        self.env = self.make_env(game=self.game, seed=self.init_seed, render_mode=self.render_mode)
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
        set the seed for reset the environment
        """
        # print("Setting seed ", seed)
        self.env.seed(seed)
        pass

    def make_env(self, game, seed=1234, render_mode=None):
        if game in _env_dict.ALL_V2_ENVIRONMENTS:
            env_cls = _env_dict.ALL_V2_ENVIRONMENTS[game]
        else:
            env_cls = _env_dict.ALL_V1_ENVIRONMENTS[game]

        if render_mode is not None:
            env = env_cls(render_mode=render_mode)
        else:
            env = env_cls()

        env._freeze_rand_vec = False
        env._set_task_called = True
        env.seed(seed)

        return env


if __name__ == "__main__":
    env1 = custom_metaworld("window-close-v2")
    env2 = custom_metaworld("window-close-v2")

    state1 = env1.reset(seed=1234)
    state2 = env2.reset(seed=1234)
    print(state1 == state2)
    action = env1.action_space.sample()
    state1, reward1, is_done1, info1 = env1.step(action)
    state2, reward2, is_done2, info2 = env2.step(action)

    print(state1 == state2)

    print(reward1 == reward2)

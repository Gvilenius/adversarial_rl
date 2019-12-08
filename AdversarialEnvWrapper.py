import gym
class AdversarialEnvWrapper(gym.Wrapper):
    def __init__(self, env, pi, sess):
        self.pi = pi
        self.sess = sess
        self.obs = None
        gym.Wrapper.__init__(self, env)

    def _act(self, o, deterministic=False):
        if deterministic:
            return self.sess.run(self.pi.deterministic_action, {self.pi.obs_ph, o})
        return self.sess.run(self.pi.action, {self.pi.obs_ph, o})

    def step(self, perturb):
        _o = self.obs
        perturbed_o = _o + perturb
        a = self._act(perturbed_o)

        o_, r, d, _ = self.env.step(a)
        r = -r
        self.obs = o_
        return o_, r, d, _

    def reset(self):
        self.obs = gym.Wrapper.reset(self)
        return self.obs
        
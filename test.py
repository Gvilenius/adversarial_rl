from stable_baselines import PPO2, logger
from stable_baselines.common.cmd_util import make_adversarial_atari_env, make_atari_env, atari_arg_parser
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy
import tensorflow as tf

def test():
    model = PPO2.load("model.pkl")
    sess = model.sess
    
    env = VecFrameStack(make_atari_env("SpaceInvadersNoFrameskip-v0", 1, 123), 4)

    pi = model.act_model
    action_dist = pi.action
    action_one = pi.deterministic_action

    o = env.reset()

    while(True):
        env.render()
        # a, _, _, _ = pi.step(obs=o, deterministic=True)
        a = sess.run(action_one, {pi.obs_ph: o})
        o, r, d, _ = env.step(a)

def main():
    test()
    
if __name__ == '__main__':
    main()
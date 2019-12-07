from stable_baselines import PPO2, logger
from stable_baselines.common.cmd_util import make_atari_env, atari_arg_parser
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy
import tensorflow as tf

def test():
    tf_config = tf.ConfigProto(
        allow_soft_placement=True,
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)

    env = VecFrameStack(make_atari_env("SpaceInvadersNoFrameskip-v0", 1, 12), 4)
    model = PPO2.load("model.pkl", env)
    sess = model.sess

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
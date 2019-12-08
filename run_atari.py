from stable_baselines import PPO2, logger
from stable_baselines.common.cmd_util import make_atari_env, make_adversarial_atari_env,atari_arg_parser
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, MlpPolicy
    
def train(env_id, num_timesteps, seed, policy, attack = False,
          n_envs=8, nminibatches=4, n_steps=128):

    model = PPO2.load("model.pkl") 
    env = VecFrameStack(make_atari_env(env_id, n_envs, seed), 4)
    if attack:
        env = VecFrameStack(make_adversarial_atari_env(env_id, n_envs, seed, model), 4)

    policy = {'cnn': CnnPolicy, 'lstm': CnnLstmPolicy, 'lnlstm': CnnLnLstmPolicy, 'mlp': MlpPolicy}[policy]
#    model = PPO2(policy=policy, env=env, n_steps=n_steps, nminibatches=nminibatches,
#                lam=0.95, gamma=0.99, noptepochs=4, ent_coef=.01,
#                 learning_rate=lambda f: f * 2.5e-4, cliprange=lambda f: f * 0.1, verbose=1)
    model.learn(total_timesteps=num_timesteps)
    model.save("model")
    env.close()
    # Free memory
    del model

def main():
    """
    Runs the test
    """
    parser = atari_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='cnn')
    args = parser.parse_args()
    logger.configure()
    train(args.env, attack=False, num_timesteps=args.num_timesteps, seed=args.seed,
          policy=args.policy)


if __name__ == '__main__':
    main()

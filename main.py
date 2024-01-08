import numpy as np
import torch
import gym
import argparse
import os
import d4rl
import utils
import OACBVR
import pandas as pd
import datetime,time


def eval_policy(policy, env_name, seed, mean, std, seed_offset=100, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + seed_offset)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			state = (np.array(state).reshape(1,-1) - mean)/std
			action = policy.select_action(state)
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes
	d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes, D4RL score: {d4rl_score:.3f}")
	print("---------------------------------------")
	return d4rl_score



if __name__ == "__main__":

	starttime = datetime.datetime.now()
	parser = argparse.ArgumentParser()
	# Experiment
	parser.add_argument("--policy", default="OAC-BVR")               
	parser.add_argument("--env", default="hopper-medium-replay-v2")        
	parser.add_argument("--seed", default=0, type=int)              
	parser.add_argument("--eval_freq", default=5e3, type=int)       
	parser.add_argument("--max_timesteps", default=12e5, type=int)
	parser.add_argument("--behavior_timesteps", default=2e5, type=int)
	parser.add_argument("--behaviorvalue_timesteps", default=3e5, type=int)       
	parser.add_argument("--save_model", action="store_true")        
	parser.add_argument("--load_model", default="")                             
	parser.add_argument("--batch_size", default=200, type=int)      
	parser.add_argument("--discount", default=0.99)                 
	parser.add_argument("--tau", default=0.005)                     
	parser.add_argument("--policy_noise", default=0.2)              
	parser.add_argument("--noise_clip", default=0.5)                
	parser.add_argument("--policy_freq", default=2, type=int)       
	parser.add_argument("--alpha", default=0.03)
	parser.add_argument("--beta", default=8)
	parser.add_argument("--normalize", default=True)
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")



	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,
		"alpha": args.alpha,
		"beta": args.beta,
		"behaviorvalue_timesteps": args.behaviorvalue_timesteps
	}

	# Initialize policy
	policy = OACBVR.OAC_BVR(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
	if args.normalize:
		mean,std = replay_buffer.normalize_states() 
	else:
		mean,std = 0,1
	
	evaluations = []
	for t in range(int(args.max_timesteps)):
		if (t + 1)<int(args.behavior_timesteps):
			if (t + 1) % args.eval_freq == 0:
				print(f"Training behavior policy, Time steps: {t+1}")
			policy.train_beta(replay_buffer, args.batch_size)
		# Evaluate episode
		else:
			policy.train(replay_buffer, args.batch_size)
			if (t + 1) % args.eval_freq == 0:
				print(f"Time steps: {t+1}")
				evaluations.append(eval_policy(policy, args.env, args.seed, mean, std))
				log = pd.DataFrame(evaluations)
				log.to_csv(f"./results/{file_name}.csv")
	endtime = datetime.datetime.now()
	print (endtime - starttime).seconds

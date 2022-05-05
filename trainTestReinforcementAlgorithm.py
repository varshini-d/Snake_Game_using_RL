
import argparse
import time
from typing import Callable
import gym
import gym_snake
from stable_baselines3 import A2C
import numpy as np
from argparse import ArgumentParser
from datetime import datetime
import gym_snake.envs.snakeRewardFuncs as RewardFuncs


def trainRL(
    model_generator = lambda env: A2C('MlpPolicy', env, verbose=0),
    train_timesteps=1000,
    env_name='snake-v0',
    board_height=10,
    board_width=10,
    max_moves_no_fruit=0,
    visualize_training=False,
    visualization_fps=3000,
    reward_function=RewardFuncs.basic_reward_func,
    represent_border=False,
):

    env = gym.make(
        env_name, 
        board_height=board_height,
        board_width=board_width,
        max_moves_no_fruit=max_moves_no_fruit,
        use_pygame=visualize_training,
        fps=visualization_fps, 
        reward_func=reward_function,
        represent_border=represent_border,
    )  
    
    # Using model_generator and env to create the model
    model = model_generator(env)
    
    t0 = time.time()
    model.learn(
        total_timesteps=train_timesteps  # Number of actions the model should take in learning
    )
    t1 = time.time()
    print("Finished training in " + str(round(t1-t0, 2)) + " seconds")
    
    return model


def testRL(
    model,
    test_timesteps=100,
    env_name='snake-v0',
    board_height=10,
    board_width=10,
    max_moves_no_fruit=0,
    visualize_testing=True,
    visualization_fps=30,
    reward_function=RewardFuncs.basic_reward_func,
    represent_border=False,
):
    # Setying up the environment
    env = gym.make(
        env_name, 
        board_height=board_height,
        board_width=board_width, 
        max_moves_no_fruit=max_moves_no_fruit,
        use_pygame=visualize_testing,
        fps=visualization_fps, 
        reward_func=reward_function,
        represent_border=represent_border,
    )
    obs = env.reset()
    
    # Running the environment
    scores = []
    for i in range(test_timesteps):
        # TODO: consider whether we should try non-deterministic
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            scores.append(env.game.score)
            obs = env.reset()

    return scores



def analyzeRL(
    scores,  # completed game scores
):
    s_arr = np.array(scores)

    analysis = {
        "completed_games": len(s_arr),
        "high_score": -1,
        "mean_score": -1,
        "median_score": -1,
    }
    print("Number of completed games: ", len(s_arr))

    if len(s_arr) > 0:
        analysis["high_score"]= np.max(s_arr)
        analysis["mean_score"]= np.average(s_arr)
        analysis["median_score"]= np.median(s_arr)
        print("High Score over all games: ", analysis["high_score"])
        print("Mean Score over all games: ", analysis["mean_score"])
        print("Median Score over all games: ", analysis["median_score"])

    return analysis


def saveRL(
    model, 
    model_filename=""  # Filename to save model under. If empty, defaults to naming using datetime
):
    
    if len(model_filename) == 0:
        model_filename = "saved_models/"+str(datetime.now().strftime("[%Y-%m-%d %H:%M:%S%z]"))
    
    model.save(model_filename)  



def main():

    aparser = ArgumentParser("Snnake Reinforcement Learning")
    aparser.add_argument("--env_name", type=str, default="snake-v0")

    aparser.add_argument("--train_timesteps", type=int, default=1000)
    aparser.add_argument("--test_timesteps", type=int, default=100)

    aparser.add_argument("--board_height", type=int, default=10)
    aparser.add_argument("--board_width", type=int, default=10)
    aparser.add_argument("--max_moves_no_fruit", type=int, default=0)
    aparser.add_argument("--visualize_training", type=bool, default=False)
    aparser.add_argument("--visualize_testing", type=bool, default=True)
    aparser.add_argument("--visualization_fps", type=int, default=30)

    aparser.add_argument("--print_analysis", type=bool, default=True, help="bool to determine whether or not analysis of test scores is done")    
    
    aparser.add_argument("--save_model", type=bool, default=False, help="bool to determine whether or not to save the trained model")        
    aparser.add_argument("--model_filename", type=str, default="", help="filename for model if it is saved. Should probably start with 'saved_models/' directory")        
    
    args = aparser.parse_args()

    reward_function = RewardFuncs.basic_reward_func
    model_generator = lambda env: A2C('MlpPolicy', env, verbose=1)
    
    # Training
    model = trainRL(
        model_generator,
        args.train_timesteps, 
        args.env_name,
        args.board_height,
        args.board_width,
        args.max_moves_no_fruit,
        args.visualize_training,
        args.visualization_fps,
        reward_function
    )
    
    # Testing
    scores = testRL(
        model, 
        args.test_timesteps, 
        args.env_name,
        args.board_height,
        args.board_width,
        args.max_moves_no_fruit,
        args.visualize_testing,
        args.visualization_fps,
        reward_function)
    
    # Analysis of the model
    if args.print_analysis:
        analyzeRL(scores)
        
    # Saving the model analysis
    if args.save_model:
        saveRL(model, args.model_filename)


if __name__ == "__main__":
    main()

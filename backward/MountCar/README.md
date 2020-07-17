# Backward Q-Learning in MountainCar environment

This is an attempt to reproduce the second experiment from "[Backward Q-learning: The combination of Sarsa algorithm and Q-learning](https://www.sciencedirect.com/science/article/abs/pii/S0952197613001176?casa_token=0odLoh-gvywAAAAA:YELA1aKk3vl6u1fcTCtvYys1_fUl9PJ18Z6QCfC59ad04Fri-3uWlrAcgPm-ggUMfSXVN0U5TmyR)"

OpenAI Gym's [source code](https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py) for MountainCar environment was customized in this experiment.

6 algorithms presented in the above mentioned paper are tested here: Q-Learning, SARSA, SA-Q-Learning, EQL, ESL and Adaptive Q-Learning.
The experiment conducts 100 runs, each consisting of a learning phase and a testing phase on each of the algorithms. In the learning phase the agent trains continuously, until it has reached the goal position 300 times from 300 random starting states. In the testing phase the agent runs 40 times from randomly selected starting states. The randomly selected starting states are the same for each algorithm and are stored in states.py.

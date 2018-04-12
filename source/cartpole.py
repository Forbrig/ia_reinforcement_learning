import numpy as np
import gym
import random
import time

env = gym.make('CartPole-v0')
env.reset()

Q = np.zeros([40, 400, 40, 2])
#print(Q)

learning_rate = 0.8
y = 0.95
num_episodes = 1000

reward_list = []

# Q = [40][400][40][2]
# [40] = velocities from -2 to 2 (goes from -inf to inf)
# [400] = angles from -2 to 2 (goes from -4.8 to 4.8)
# [40] = pole velocities from -2 to 2 (goes from -inf to inf)
# [2] = actions (left, right) ({0, 1})
# gets the the second, third and forth args from observation and discretize
def discrete_indexes(state):
    #state = (int((state[1] * 10) + 20), int((state[2] * 100) + 200), int((state[3] * 10) + 20))

    car_velocity = int((state[1] * 10) + 20)
    pole_angle = int((state[2] * 100) + 200)
    pole_velocity = int((state[3] * 10) + 20)

    if (car_velocity > 40 - 1):
        car_velocity = 40 - 1
    elif (car_velocity < 0):
        car_velocity = 0

    if (pole_angle > 400 - 1):
        pole_angle = 400 - 1
    elif (pole_angle < 0):
        pole_angle = 0

    if (pole_velocity > 40 - 1):
        pole_velocity = 40 - 1
    elif (pole_velocity < 0):
        pole_velocity = 0

    state = (car_velocity, pole_angle, pole_velocity)
    return state

# (1./(episode + 1)) this decrease the chance of choosing a greedly action based on the num_episode
# we choose more greedly at the first ones
# here we need to give a chance to choose other actions because, if not, we always will use the first action
# that the reward is updated
def choose_action(state, episode):

    #reward_0 = Q[state[0], state[1], state[2], 0]
    #reward_1 = Q[state[0], state[1], state[2], 1]

    #print("reward 0: ", reward_0)
    #print("reward 1: ", reward_1)

    action = np.argmax(Q[state[0], state[1], state[2], :])
    if action == 0:
        if random.randint(1, 10) > 8:
            action = 1
    else:
        if random.randint(1, 10) > 8:
            action = 0

    #argmax(reward_0, reward_1)
    #print(action)
    #print(Q[state[0], state[1], state[2], :])

    #print(Q[state[0], state[1], state[2], 0])
    #print(Q[state[0], state[1], state[2], 1])
    return action


for i in range(num_episodes):
    state = env.reset()

    #state = (int((state[1] * 10) + 20), int((state[2] * 100) + 200), int((state[3] * 10) + 20))
    state = discrete_indexes(state)
    #print(int((state[1] * 10) + 20), int((state[2] * 100) + 200), int((state[3] * 10) + 20))
    #print(state)

    total_reward = 0
    done = 0
    for j in range(200):
        #env.render()


        action = choose_action(state, i)
        #action = np.argmax(Q[state[0], state[1], state[2], :])
        #print(action)
        #action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        #new_state = (int((new_state[1] * 10) + 20), int((new_state[2] * 100) + 200), int((new_state[3] * 10) + 20))
        new_state = discrete_indexes(new_state)
        Q[state[0], state[1], state[2], action] = Q[state[0], state[1], state[2], action] + learning_rate * (reward + y * np.max(Q[new_state[0], new_state[1], new_state[2], :]) - Q[state[0], state[1], state[2], action])
        total_reward = total_reward + reward
        state = new_state

        if (done == True):
            #print(total_reward)
            break

        reward_list.append(total_reward)

#print(max(reward_list))

for i in range(10):
    total_reward = 0
    state = env.reset()

    state = discrete_indexes(state)
    for j in range(200):
        env.render()
        action = np.argmax(Q[state[0], state[1], state[2], :])
        new_state, reward, done, info = env.step(action)
        new_state = discrete_indexes(new_state)
        total_reward = total_reward + reward
        state = new_state
        if (done == True):
            print(total_reward)
            break

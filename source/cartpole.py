import numpy as np
import gym
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


for i in range(num_episodes):
    state = env.reset()

    #state = (int((state[1] * 10) + 20), int((state[2] * 100) + 200), int((state[3] * 10) + 20))
    state = discrete_indexes(state)
    #print(int((state[1] * 10) + 20), int((state[2] * 100) + 200), int((state[3] * 10) + 20))
    #print(state)

    total_reward = 0
    done = 0
    for j in range(200):
        env.render()

        action = np.argmax(Q[state[0], state[1], state[2], :])
        #print(action)
        #action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        #new_state = (int((new_state[1] * 10) + 20), int((new_state[2] * 100) + 200), int((new_state[3] * 10) + 20))
        new_state = discrete_indexes(new_state)
        Q[state[0], state[1], state[2], action] = Q[state[0], state[1], state[2], action] + learning_rate * (reward + y * np.max(Q[new_state[0], new_state[1], new_state[2], :]) - Q[state[0], state[1], state[2], action])
        total_reward = total_reward + reward
        state = new_state

        if (done == True):
            break
            print(total_reward)

        reward_list.append(total_reward)

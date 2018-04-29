import numpy as np
import gym
import random
import cPickle as pickle
import time

env = gym.make('CartPole-v0')
env.reset()

Q = np.zeros([40, 400, 40, 2]) # q-table, where we store the states, actions and rewars
learning_rate = 0.8
y = 0.95
num_train_episodes = 1000 # with how many episodes our agent will train
num_test_episodes = 100 # how many episodes our agent will be tested
render = True # render our test (will slow the respose a lot)
slow_render = False # slows 0.2 secs each step the render of the test (don't work if render = False)
use_saved_qtable = False
print_rewards = True #print reward obtained after each test in an episode


# Q = [40][400][40][2]
# [40] = velocities from -2 to 2 (goes from -inf to inf)
# [400] = angles from -2 to 2 (goes from -4.8 to 4.8)
# [40] = pole velocities from -2 to 2 (goes from -inf to inf)
# [2] = actions (left, right) ({0, 1})
# gets the the second, third and forth args from observation and discretize
def discrete_indexes(state):
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

# save qtable and his average score (over 100 trials) on a data file
def save_qtable(score, qtable):
    file = open('qtable.data', 'wb')
    pickle.dump((score, qtable), file)
    file.close()

# load qtable and his average score from the file (if it exist)
def load_qtable():
    try:
        file = open('qtable.data')
        current_score, qtable = pickle.load(file)
    except:
        print("Can't open the stored qtable. Maybe it's missing.")

        return 0, 0
        #exit(0)

    return current_score, qtable

# (1./(episode + 1)) this decrease the chance of choosing a greedly action based on the num_episode
# we choose more greedly at the first ones
# here we need to give a chance to choose other actions because, if not, we always will use the first action
# that the reward is updated
def choose_action(state, episode):
    action = np.argmax(Q[state[0], state[1], state[2], :])
    if action == 0:
        if random.randint(1, 10) > 8:
            action = 1
    else:
        if random.randint(1, 10) > 8:
            action = 0
    return action

# train our agent with n episodes
def train_agent(num_episodes):
    print("Training new agent from scratch.")
    for i in range(num_episodes):
        state = env.reset()
        state = discrete_indexes(state)
        total_reward = 0
        done = 0
        for j in range(200):
            #env.render() #it slows the train a lot!
            action = choose_action(state, i)
            new_state, reward, done, info = env.step(action)
            new_state = discrete_indexes(new_state)
            Q[state[0], state[1], state[2], action] = Q[state[0], state[1], state[2], action] + learning_rate * (reward + y * np.max(Q[new_state[0], new_state[1], new_state[2], :]) - Q[state[0], state[1], state[2], action])
            total_reward = total_reward + reward
            state = new_state
            if (done == True):
                #print(total_reward)
                break

# test our agent to see how good is his bahavior in n epsodes with the current q-table
# return the average score in n epsides
def test_agent(num_tests, render, slow_render):
    for i in range(num_tests):
        total_reward = 0
        state = env.reset()
        state = discrete_indexes(state)
        for j in range(200):
            if render == True:
                if slow_render == True:
                    time.sleep(0.2)
                env.render()
            action = np.argmax(Q[state[0], state[1], state[2], :])
            new_state, reward, done, info = env.step(action)
            new_state = discrete_indexes(new_state)
            total_reward = total_reward + reward
            state = new_state
            if (done == True):
                if print_rewards == True:
                    print(total_reward)
                break
        reward_list.append(total_reward)
    return sum(reward_list)/float(len(reward_list))


reward_list = []
final_score = 0

if use_saved_qtable == True: # note that the test scores may difer
    saved_score, Q = load_qtable()
    if Q == 0: # if we couldnt open the qtable to test
        print 'Exiting.'
        exit(0)
    print 'Using stored Qtable. Note that the test score may differ from the saved score (it will not be updated).'
else:
    saved_score, _ = load_qtable()
    train_agent(num_train_episodes)

final_score = test_agent(num_test_episodes, render, slow_render)

#print 'Score of the current Qtable in this test: ', final_score
print 'Score of stored Qtable: ', saved_score

# if this trial gets a better score than the saved one it save the new qtable
if saved_score < final_score and use_saved_qtable == False:
    save_qtable(final_score, Q)
    print 'Saving current Qtable...'
else:
    # new score will never be saved
    print("Current Qtable will not be saved.")

print 'Final score: ', final_score

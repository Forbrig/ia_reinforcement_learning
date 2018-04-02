import numpy as np
import cPickle as pickle
#import _pickle as pickle #for python3+
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import gym
env = gym.make('CartPole-v0')
env.reset()
random_episodes = 0
reward_sum = 0
#
# Aqui ponha o seu codigo para ler os valores dos estados, se houver
#
while random_episodes < 100:
    observation = env.reset()
    for t in range(200):
        env.render()
        #print(observation)
#
# Aqui ponho o seu codigo para salvar a observacao, estado
#
#
# Aqui ponha o seu codigo para escolher a melhor acao dependendo do estado atual
#
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
#
# Aqui ponha o seu codigo para atualizar os valores dos estados apos uma execucao
#
            print("Episode finished after {} timesteps".format(t+1))
            break

#
#   Aqui ponha o seu codigo para salvar os valores dos estados calculados

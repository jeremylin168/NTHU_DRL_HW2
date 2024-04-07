# Environment
import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F


#
import numpy as np
import cv2
import random



class Agent():
    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # basic parameters
        self.action_range = 12
        self.epsilon = 0.1


        self.learning_rate = 3e-4
        self.gamma = 0.9

        self.batch_size = 64

        self.skip_frame = 4
        self.skip_frame_counter = 0
        self.action_buffer = 0

        self.update_iter = 32
        self.update_counter = 0



        # Networks
        self.target_Network = DQN_Model().to(self.device)
        #self.training_Network = DQN_Model().to(self.device)
        #self.loss_function = nn.MSELoss()

        #self.optimizer = torch.optim.Adam(self.training_Network.parameters(), lr=self.learning_rate)

        self.load_weight(path="./111062661_hw2_data")

        # memory
        #self.memory_counter = 0
        #self.memory_limit = 128 * 64
        #self.state_size = 84 * 84 * 4

        #self.memory = np.zeros((self.memory_limit, self.state_size * 2 + 2)) # two state frame & action & reward
        
        #state buffer
        self.state_buffer = np.zeros((4, 84, 84))
        self.next_state_buffer = np.zeros((4, 84, 84))


    def act(self, state):
        self.skip_frame_counter += 1
        if self.skip_frame_counter >= self.skip_frame:

            n_state = cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), (84, 84), interpolation=cv2.INTER_AREA)

            self.state_buffer[1:, :, :] = self.state_buffer[:-1, :, :]
            self.state_buffer[0, :, :] = n_state

            x = torch.tensor(np.reshape(self.state_buffer, (1, 4, 84, 84)), dtype=torch.float32).to(self.device)
            self.action_buffer = torch.argmax(self.target_Network(x), 1).cpu().numpy()[0]

            action = self.action_buffer
            self.skip_frame_counter = 0

        else:
            action = self.action_buffer

        return action


        
    """
    def store_memory(self, state, action, reward, next_state):
        n_state = cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), (84, 84), interpolation=cv2.INTER_AREA)
        #cv2.imshow("myphoto", n_state)
        n_next_state = cv2.resize(cv2.cvtColor(next_state, cv2.COLOR_RGB2GRAY), (84, 84), interpolation=cv2.INTER_AREA)

        self.next_state_buffer[1:, :, :] = self.next_state_buffer[:-1, :, :]
        self.next_state_buffer[0, :, :] = n_next_state

        new_record = np.hstack((self.state_buffer.ravel(), action, reward, self.next_state_buffer.ravel()))

        index = self.memory_counter % self.memory_limit
        self.memory[index, :] = new_record

        self.memory_counter +=1
        if self.memory_counter >= self.memory_limit:
            self.memory_counter = 0
            

    def update_network(self):
        # depatch memory

        sample_index = np.random.choice(self.memory_limit, self.batch_size)
        sampled_memory = self.memory[sample_index, :]

        states = torch.FloatTensor(np.reshape(sampled_memory[:, :self.state_size], (self.batch_size, 4, 84, 84))).to(self.device)
        actions = torch.LongTensor(sampled_memory[:, self.state_size:self.state_size + 1].astype(int)).to(self.device)
        rewards = torch.FloatTensor(sampled_memory[:, self.state_size + 1:self.state_size + 2]).to(self.device)
        next_states = torch.FloatTensor(np.reshape(sampled_memory[:, self.state_size+2:], (self.batch_size, 4, 84, 84))).to(self.device)

        #print(self.training_Network(states).shape)

        q_train = self.training_Network(states).gather(1, actions)
        q_next = self.target_Network(next_states).detach()

        q_target = rewards + self.gamma * torch.max(q_next, 1)[0].view(self.batch_size, 1)

        loss = self.loss_function(q_train, q_target)

        # backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter >= self.update_iter:
            self.target_Network.load_state_dict(self.training_Network.state_dict())
            self.update_counter = 0
    """
    def save_weight(self, p="none"):
        torch.save(self.training_Network.state_dict(), f"./test_model_weight_6_ep{p}")

    def load_weight(self, path):
        #self.training_Network.load_state_dict(torch.load(path))
        self.target_Network.load_state_dict(torch.load(path, map_location=self.device))
    

class DQN_Model(nn.Module):
    def __init__(self):
        super(DQN_Model, self).__init__()

        # conv 1
        # input (1, 84, 84)
        self.cnn1 = nn.Conv2d(4, 32, kernel_size=3, stride=1)


        self.cnn2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        #self.relu1 = nn.ReLU(inplace=True) 

        self.cnn3 = nn.Conv2d(64, 64, kernel_size=5, stride=2)

        self.cnn4 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=(1, 1))

        #self.cnn5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.cnn5 = nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=(2, 2))

        # input (32, 21, 21)
        

        # input (32, 15, 16)
        self.L1 = nn.Linear(128 * 4 * 4, 512)
        self.L2 = nn.Linear(512, 256)
        self.L3 = nn.Linear(256, 12)

    def forward(self, input):
        x = F.relu(self.cnn1(input))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        x = F.relu(self.cnn4(x))
        x = F.relu(self.cnn5(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        output = self.L3(x)

        return output

    


if __name__ == "__main__":
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    state = env.reset()
    env.render()
    episode_num = 20

    agent = Agent()

    episode_num = 50

    for eps in range(episode_num):
        # reset env
        state = env.reset()
        done = False
        rewards = 0

        while not done:
            action = agent.act(state)

            next_state, reward, done, info = env.step(action)
            env.render()

            rewards += reward

            state = next_state
        print(f"Episode: {eps}, rewards: {rewards}")

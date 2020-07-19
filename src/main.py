import numpy as np
import keras.backend.tensorflow_backend as backend
from keras.models import Model, Sequential
from keras.layers import Input, Dense, concatenate #, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
import bamboo as bam
import json
import copy

# Global variables
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 100  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 20  # Minimum number of steps in a memory to start training -> 20 by default
MINIBATCH_SIZE = 20  # How many steps (samples) to use for training -> 20 by default
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes) -> default 5
MIN_REWARD = 0  # For model save
MODEL_NAME = 'TFBAMBOO'

# Environment settings
EPISODES = 200
MAX_MOVES = 500

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = 0.9 * EPISODES
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

#  Stats settings
AGGREGATE_STATS_EVERY = 10  # episodes
FILE_NAME  = "Kaqchickel_complete.json"

# Import data of culms and members from grasshopper
IMPORT_CULMS = "culms_test_01.json"
IMPORT_MEMBERS = "members_test_01.json"

# Import data
culms = []
with open(f"export/{IMPORT_CULMS}") as read_file:
    data = json.load(read_file)
for culm in data:
    culms.append(bam.Culm(culm["id"], int(culm["d1"]), int(culm["d2"]), int(culm["t1"]), int(culm["t2"]), [int(x) for x in culm["nodes"]]))
members = []
with open(f"export/{IMPORT_MEMBERS}") as read_file:
    data = json.load(read_file)
for member in data:
    members.append(bam.Member(member["id"], [int(x) for x in member["s"]] , [int(x) for x in member["s0"]], [int(x) for x in member["s1"]]))

#NO BORRAR!!!!!
#Stützen_1
#culms_reordered = [65,34,74,96,100,23,147]
#members_reordered =  [0,0,3,4,1,1,3,4,2,2,3,4]
#Stützen_2
#culms_reordered = [47,89,0,5,11,29]
#members_reordered =  [5,5,8,9,6,6,8,9,7,7,8,9]
#Díagonalen
#culms_reordered = [59,60,53]
#members_reordered =  [10,11,10,11]
#Riegel
#culms_reordered = [52,102,78,97]
#members_reordered =  [12,13,14,15,12,13,14,15]
#Final
#culms_reordered = [65,34,74,96,100,23,147,47,89,0,5,11,29,52,102,78,97,59,60,53]
#members_reordered =  [0,0,3,4,1,1,3,4,2,2,3,4,5,5,8,9,6,6,8,9,7,7,8,9,12,13,14,15,12,13,14,15,10,11,10,11]

#bambu 74 - 638??

culms_reordered = [copy.deepcopy(culms[i]) for i in [65,34,74,96,100,23,147,47,89,0,5,11,29,52,102,78,97,59,60,53]]
for i in range(len(culms_reordered)):
    culms_reordered[i].id = i
culms = culms_reordered

members_reordered = [copy.deepcopy(members[i]) for i in [0,0,3,4,1,1,3,4,2,2,3,4,5,5,8,9,6,6,8,9,7,7,8,9,12,13,14,15,12,13,14,15,10,11,10,11]]
for i in range(len(members_reordered)):
    members_reordered[i].id = i
members = members_reordered

#Create environment
env = bam.StructureEnv()
env_list = [] # environment states to export for visualization

# For stats
ep_rewards = [0]

## For more repetitive results
#random.seed(1)
#np.random.seed(1)
#tf.set_random_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
#backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')
# Create export folder
if not os.path.isdir('export'):
    os.makedirs('export')

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.writer = tf.summary.FileWriter(self.log_dir, graph = self.sess.graph)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end

    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

# Agent class
class DQNAgent:
    def __init__(self, number_of_members):

        self.number_of_members = number_of_members

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())),write_graph=True)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):

        str_input = Input(shape=(self.number_of_members,), name='str_input')
        str_dense_1 = Dense(36, activation='relu', name='prelim_dense_1')(str_input)
        str_dense_2 = Dense(36, activation='relu', name='prelim_dense_2')(str_dense_1)
        str_output = Dense(12, activation='relu', name='prelim_output')(str_dense_2)

        member_culm_input = Input(shape=(2,), name='member_culm_input')
        merge = concatenate([member_culm_input, str_output], name='merge')

        dqn_dense_1 = Dense(24, activation='relu', name='main_dense_1')(merge)
        dqn_dense_2 = Dense(24, activation='relu', name='main_dense_2')(dqn_dense_1)
        dqn_dense_3 = Dense(24, activation='relu', name='main_dense_3')(dqn_dense_2)
        q_values = Dense(2, name='q_values')(dqn_dense_3) # 2 = how many choices (action = 0, action = 1)

        model = Model(inputs=[member_culm_input, str_input], outputs=q_values)

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['mae'])

        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition) # replay_memory is a dequeue -> older items will be removed if max lentgh is exceeded

    # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = [transition[0] for transition in minibatch]
        current_qs_list = self.model.predict([np.array([current_state[:2] for current_state in current_states]),
                                  np.array([current_state[2:] for current_state in current_states])])

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = [transition[3] for transition in minibatch]
        future_qs_list = self.target_model.predict([np.array([new_current_state[:2] for new_current_state in new_current_states]),
                                  np.array([new_current_state[2:] for new_current_state in new_current_states])])

        X1 = [] #member_culm_input
        X2 = [] #str_input
        Y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X1.append(current_state[:2])
            X2.append(current_state[2:])
            Y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit([np.array(X1), np.array(X2)], np.array(Y), batch_size=MINIBATCH_SIZE, epochs = 200, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # check new qs
        #new_qs = self.model.predict(current_states)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        #return self.model.predict(np.array(state).reshape(1,self.number_of_inupts))[0]
        return self.model.predict([np.array(state[:2]).reshape(1,2),
                                  np.array(state[2:]).reshape(1,self.number_of_members)])[0]

agent = DQNAgent(len(members))

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 0
    culm_sequence = []
    member_sequence = []

    # epsilon decaying
    if END_EPSILON_DECAYING > episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    # Reset environment
    env.reset(culms, members, MAX_MOVES)

    # Render result
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        env_list.append(env.export(episode, step, episode_reward, 0))

    # Reset flag and start iterating until episode ends
    current_state, done, _ = env.find_next_state()
    while not done:                          
            
        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q-Network
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, 2)
        # Compute reward and update environment
        if action == 1: # action -> extract solution from culm                            
            reward = env.extract_solution()
            culm_sequence.append(env.current_culm);
            member_sequence.append(env.current_member);
        else: # action -> pass culm     
            reward = 0.0
            env.members_found[env.current_culm][env.current_member] = 2
            env.switch_to_next_culm()

        new_state, done, extra_reward = env.find_next_state()
        reward += extra_reward # in case we´re done we profit from the unused bamboos

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        # Count reward
        episode_reward += reward        

        current_state = new_state
        step += 1    

        # Render result
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            env_list.append(env.export(episode, step, episode_reward, int(action)))
        
    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
      
        print (f"episode = {episode}")
        print (f"episode_reward = {episode_reward}")
        print (f"member_sequence = {member_sequence}")
        print (f"culm_sequence = {culm_sequence}")
        print (f"epsilon = {epsilon}")

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

# Export environments for visualization
outfile = open(f"export/{FILE_NAME}",'w', encoding="utf8")
outfile.write(json.dumps(env_list, sort_keys = True, ensure_ascii=False))
outfile.close()
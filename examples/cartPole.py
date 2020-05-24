# import numpy as np 
# import tensorflow as tf 
# import gym
# import os
# import datetime 
# from gym import wrappers

# # States (4): cart position, cart velocity, pole angle, pole velocity at its tip
# # Acciones (2): moverse a la derecha, o izquierda
# # Reward: Por cada step tomado +1 reward. El juego termina cuando cae el poste. 

# # El TargetNet es como un euxiliar


# class MyModel(tf.keras.Model):
# 	# Meto un estado y me devuelve una accion
# 	def __init__(self, num_states, hidden_units, num_actions):
# 		super(MyModel, self).__init__()
# 		self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
# 		self.hidden_layers = []
# 		for i in hidden_units:
# 			self.hidden_layers.append(tf.keras.layers.Dense(
# 				i, activation='tanh', kernel_initializer='RandomNormal'))
# 		self.output_layer = tf.keras.layers.Dense(
# 			num_actions, activation='linear', kernel_initializer='RandomNormal')	

# 	@tf.function # enable autograph and automatic control dependencies
# 	def call(self, inputs):
# 		# Implementar el forward pass
# 		# inputs shape: [batch_size, 4]
# 		z = self.input_layer(inputs)
# 		for layer in self.hidden_layers:
# 			z = layer(z)
# 		output = self.output_layer(z)
# 		return output # [batch size, 2]

# class DQN:
# 	def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
# 		self.num_actions = num_actions
# 		self.batch_size = batch_size
# 		self.optimizer = tf.optimizers.Adam(lr)

# 		# Multiplied by the Q-Value at the next step, the agent care less about
# 		# rewards in the distant feature than those in the immediate future
# 		self.gamma = gamma

# 		self.model = MyModel(num_states, hidden_units, num_actions)
# 		self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}

# 		# Experience replay buffer
# 		# The agent won´t start learning unless the size of the buffer is graeter
# 		# than self.min_experiences
# 		# Once the buffer reaches the max size it will delete the oldest values to make
# 		# room for the new values.
# 		self.max_experiences = max_experiences
# 		self.min_experiences = min_experiences

# 	def predict(self, inputs):
# 		# Accepta ya sea un solo estado (size 1,4) o un batch de estados como input
# 		# Retorna los resultados (accion)
# 		# atleast_2d porque queremos que sea 2d aunque solo sea batch de uno
# 		return self.model(np.atleast_2d(inputs.astype('float32')))

# 	def train(self, TargetNet):
# 		if len(self.experience['s']) < self.min_experiences:
# 			return 0
# 		# Randomly select a batch of (s, s´,a,r) values 
# 		ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)


# 		states = np.asarray([self.experience['s'][i] for i in ids])
#         actions = np.asarray([self.experience['a'][i] for i in ids])
#         rewards = np.asarray([self.experience['r'][i] for i in ids])
#         # Estado producido despues de la accion
#         states_next = np.asarray([self.experience['s2'][i] for i in ids])
#         dones = np.asarray([self.experience['done'][i] for i in ids])

#         # Get the actions at the next state
#         value_next = np.max(TargetNet.predict(states_next), axis=1)
#         # Ground truth values from the Bellman function
#         # done true es el terminal state, y solo regresa el reward
#         actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

#         # Calculate the square loss of the real target and prediction
#         with tf.GradientTape() as tape:
#             selected_action_values = tf.math.reduce_sum(
#                 self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
#             loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
#         variables = self.model.trainable_variables
#         gradients = tape.gradient(loss, variables)
#         self.optimizer.apply_gradients(zip(gradients, variables))
#         return loss # tensor

#     # epsilon is a value between 0 and 1 that decays over time. Larger epsilon at the beggining,
#     # because we want to explore more (selecting random actions). 
#     def get_action(self, states, epsilon):
#     	if np.random.random() < epsilon:
#     		return np.random.choice(self.num_actions)
#     	else:
#     		return np.argmax(self.predict(np.atleast_2d(states))[0])

#     # Experience replay buffer
#     def add_experience(self, exp):
#         if len(self.experience['s']) >= self.max_experiences:
#             for key in self.experience.keys():
#                 self.experience[key].pop(0)
#         for key, value in exp.items():
#             self.experience[key].append(value)

#     # We are going to have 2 instances of the DQN class: a training net and a target net
#     # training net: used to update the weights
#     # target net: only performs 2 tasks, predicting the value at next step Q(s´, a), and
#     # copying weights from the training net
#     def copy_weights(self, TrainNet):
#         variables1 = self.model.trainable_variables
#         variables2 = TrainNet.model.trainable_variables
#         for v1, v2 in zip(variables1, variables2):
#             v1.assign(v2.numpy())


# def play_game(env, TrainNet, TargetNet, epsilon, copy_step):
# 	rewards = 0
#     iter = 0
#     done = False
#     observations = env.reset() # get initial state
#     losses = list()
#     while not done:
#         action = TrainNet.get_action(observations, epsilon)
#         prev_observations = observations
#         observations, reward, done, _ = env.step(action)
#         rewards += reward
#         if done:
#             reward = -200
#             env.reset()

#         exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
#         TrainNet.add_experience(exp) # replay buffer
#         loss = TrainNet.train(TargetNet) # Entrenar con la prediccion
        
#         if isinstance(loss, int):
#             losses.append(loss)
#         else:
#             losses.append(loss.numpy())
        
#         iter += 1 # number of steps we´ve played in one game

#         # Copy the weights to the target net every "copy_step" steps
#         if iter % copy_step == 0:
#             TargetNet.copy_weights(TrainNet)

#     return rewards, mean(losses)


# def make_video(env, TrainNet):
# 	env = wrappers.Monitor(env, os.path.join(os.getcwd(), "videos"), force=True)
#     rewards = 0
#     steps = 0
#     done = False
#     observation = env.reset()
#     while not done:
#         action = TrainNet.get_action(observation, 0)
#         observation, reward, done, _ = env.step(action)
#         steps += 1
#         rewards += reward
#     print("Testing steps: {} rewards {}: ".format(steps, rewards))


# def main():
#     env = gym.make('CartPole-v0')
#     gamma = 0.99
#     copy_step = 25
#     num_states = len(env.observation_space.sample())
#     num_actions = env.action_space.n
#     hidden_units = [200, 200]
#     max_experiences = 10000
#     min_experiences = 100
#     batch_size = 32
#     lr = 1e-2
#     current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     log_dir = 'logs/dqn/' + current_time
#     summary_writer = tf.summary.create_file_writer(log_dir)

#     TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
#     TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
#     N = 50000
#     total_rewards = np.empty(N)
#     epsilon = 0.99
#     decay = 0.9999
#     min_epsilon = 0.1
#     for n in range(N):
#         epsilon = max(min_epsilon, epsilon * decay)
#         total_reward, losses = play_game(env, TrainNet, TargetNet, epsilon, copy_step)
#         total_rewards[n] = total_reward
#         avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
#         with summary_writer.as_default():
#             tf.summary.scalar('episode reward', total_reward, step=n)
#             tf.summary.scalar('running avg reward(100)', avg_rewards, step=n)
#             tf.summary.scalar('average loss)', losses, step=n)
#         if n % 100 == 0:
#             print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):", avg_rewards,
#                   "episode loss: ", losses)
#     print("avg reward for last 100 episodes:", avg_rewards)
#     make_video(env, TrainNet)
#     env.close()


# if __name__ == '__main__':
#     for i in range(3):
#         main()


import numpy as np
import tensorflow as tf
import gym
import os
import datetime
from statistics import mean
from gym import wrappers


class MyModel(tf.keras.Model):
    def __init__(self, num_states, hidden_units, num_actions):
        super(MyModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_states,))
        self.hidden_layers = []
        for i in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(
                i, activation='tanh', kernel_initializer='RandomNormal'))
        self.output_layer = tf.keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output


class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = MyModel(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())


def play_game(env, TrainNet, TargetNet, epsilon, copy_step):
    rewards = 0
    iter = 0
    done = False
    observations = env.reset()
    losses = list()
    while not done:
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations
        observations, reward, done, _ = env.step(action)
        rewards += reward
        if done:
            reward = -200
            env.reset()

        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        TrainNet.add_experience(exp)
        loss = TrainNet.train(TargetNet)
        if isinstance(loss, int):
            losses.append(loss)
        else:
            losses.append(loss.numpy())
        iter += 1
        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
    return rewards, mean(losses)

def make_video(env, TrainNet):
    env = wrappers.Monitor(env, os.path.join(os.getcwd(), "videos"), force=True)
    rewards = 0
    steps = 0
    done = False
    observation = env.reset()
    while not done:
        env.render()
        action = TrainNet.get_action(observation, 0)
        observation, reward, done, _ = env.step(action)
        steps += 1
        rewards += reward
    print("Testing steps: {} rewards {}: ".format(steps, rewards))


def main():
    env = gym.make('CartPole-v0')
    gamma = 0.99
    copy_step = 25
    num_states = len(env.observation_space.sample())
    num_actions = env.action_space.n
    hidden_units = [200, 200]
    max_experiences = 10000
    min_experiences = 100
    batch_size = 32
    lr = 1e-2
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = 'logs/dqn/' + current_time
    summary_writer = tf.summary.create_file_writer(log_dir)

    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    N = 50000
    total_rewards = np.empty(N)
    epsilon = 0.99
    decay = 0.9999
    min_epsilon = 0.1
    for n in range(N):
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward, losses = play_game(env, TrainNet, TargetNet, epsilon, copy_step)
        total_rewards[n] = total_reward
        avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
        with summary_writer.as_default():
            tf.summary.scalar('episode reward', total_reward, step=n)
            tf.summary.scalar('running avg reward(100)', avg_rewards, step=n)
            tf.summary.scalar('average loss)', losses, step=n)
        if n % 100 == 0:
            print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):", avg_rewards,
                  "episode loss: ", losses)
    print("avg reward for last 100 episodes:", avg_rewards)
    make_video(env, TrainNet)
    env.close()


if __name__ == '__main__':
    for i in range(3):
        main()

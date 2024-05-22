import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.layers import Dense
from collections import deque

#Enviornment
class Car: #Creat a car in the model
  def __init__(self):
    self.wait_time = 0

  def increment_wait_time(self):
    self.wait_time += 1

  def get_wait_time(self):
    return self.wait_time

class Lane: #Creat a lane in the model
  def __init__(self):
    self.cars = []

  def total_wait_time(self): #Get the total wait time from all of the cars in the lane
    wait = 0
    for car in self.cars:
      wait = wait + car.get_wait_time()
    return wait

  def add_car(self):
    self.cars.append(Car())

  def increment_wait_time(self):
    for car in self.cars:
      car.increment_wait_time()

  def emit_car(self):
    if len(self.cars) > 0:
      self.cars.pop(0)

  def get_number_of_cars(self):
    return len(self.cars)

class Intersection: #Creat the intersection in the model and control his behavior
  def __init__(self , delay_at_change = 2 , max_cars = 20 , min_green_tics = 2):
    self.lanes = {"NS" : Lane() , "SN" : Lane() , "EW" : Lane() , "WE" : Lane() , "SW" : Lane() , "NE" : Lane() , "WN" : Lane() , "ES" : Lane()}
    #lanes -> Sets the lanes, for example "NS" means lane North-South.
    self.action_to_lanes = {0 : ["EW" , "WE"] , 1 : ["NS" , "SN"] , 2 : ["SW" , "NE"] , 3 : ["WN" , "ES"] , 4 : ["SW" , "SN"] , 5 : ["EW" , "ES"] , 6 : ["NS" , "NE"] , 7 : ["WE" , "WN"]}
    #action_to_lanes -> Connecting between action number to lanes route combination
    self.delay_at_change = delay_at_change #the delay at the case of changing action number
    self.max_cars = max_cars #Overall max num of cars in the intersection
    self.min_green_tics = min_green_tics #min green tics between each action change
    self.prev_total_wait_time = 0
    self.total_wait_time_log = []
    self.change_interval =[]
    self.change_count = 0

  def __str__(self):
    s = "State: " + self.state + ", "
    s = s + "Total wait time: " + str(self.intersection_total_wait_time()) + " - "
    for k in self.lanes.keys():
      s = s + k
      s = s + ": (" + str(self.lanes[k].get_number_of_cars()) + " , " + str(self.lanes[k].total_wait_time()) + ") "
    return s

  def apply_action(self  , state , action): #Computing the selected action, calculating reward and total wait time
    self.add_cars()
    if (state[-1] != action):
      for i in range(self.delay_at_change):
        for v in self.lanes.values():
          v.increment_wait_time
      for j in range(self.min_green_tics):
        for ln in self.action_to_lanes[action]:
          self.lanes[ln].emit_car()

    for key in self.action_to_lanes.keys():
      if key != action:
        for ln in self.action_to_lanes[key]:
          self.lanes[ln].increment_wait_time()
    for ln in self.action_to_lanes[action]:
      self.lanes[ln].emit_car()

      #New state and reward

    wtl = []
    for k in self.lanes.keys():
      wtl.append(self.lanes[k].total_wait_time())
    wtl.append(action)
    if (state[-1] != action):
      if (self.intersection_total_wait_time() <= self.prev_total_wait_time):
        reward = 1;
      else:
        reward = -1
    else:
      reward = 0.1

    if (state[-1] != action):
      self.change_interval.append(self.change_count)
      self.change_count = 0
    else:
      self.change_count += 1

    self.total_wait_time_log.append(self.intersection_total_wait_time())
    self.prev_total_wait_time = self.intersection_total_wait_time()
    return reward , wtl

  def intersection_total_wait_time(self): #Returns the total wait time in the intersection
    wt = 0
    for k in self.lanes.keys():
      wt = wt + self.lanes[k].total_wait_time()
    return wt

  def get_number_of_cars(self):
    cr = 0
    for v in self.lanes.values():
      cr += v.get_number_of_cars()
    return cr

  def add_car(self , ln):
    self.lanes[ln].add_car()

  def add_cars(self): #Add cars to random lanes according to the max cars
    ld = {1: "NS" , 2 : "SN" , 3 : "EW" , 4 : "WE" , 5 : "SW" , 6 : "NE" , 7 : "WN" , 8 : "ES"}
    if self.get_number_of_cars() < self.max_cars:
      cars_to_add = max((self.max_cars - self.get_number_of_cars()) // 3 , 2)
      for i in range(cars_to_add):
        self.add_car(ld[random.randint(1 , 8)])

class Trafficlight: #Used as the agent of the model, managing the traffic flow in the intersection
  def __init__(self  , dqn , epsilon = 0.2 , epsilon_decay_rate = 0.9):
    self.intersection = Intersection()
    self.clock = 0
    self.epsilon = epsilon
    self.epsilon_decay_rate = epsilon_decay_rate
    self.dqn = dqn

  def select_epsilon_greedy_action(self , state): #Choosing an action regard epsilon-greedy policy
    greedy = np.random.uniform()
    if greedy < self.epsilon:
      return random.randint(0 , 7)
    else:
      q_values = self.dqn.predict([state])
      return int(np.argmax(np.array(q_values)))

  def get_initial_state(self):
    return [0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0]

  def epsilon_decay(self): #Computes the epsilon-decay process
    self.epsilon *= self.epsilon_decay_rate

  def get_epsilon(self):
    return self.epsilon

#Model
class DQN: #Creating the model of the project
  def __init__(self , num_actions , epochs_per_learning_session = 100 , batch_size = 16):
    self.model = Sequential([
    Dense(units = 64, input_shape = (16,), activation = 'relu'),
    Dense(units = 64, activation = 'relu'),
    Dense(units = 32, activation = 'relu'),
    Dense(units = 16, activation = 'relu'),
    Dense(units = 8, activation = 'linear')
    ])
    self.model.compile(loss= tf.keras.losses.MeanSquaredError(), optimizer=keras.optimizers.Adam(learning_rate=0.0001))
    self.loss_log = []
    self.epls = epochs_per_learning_session
    self.bs = batch_size

  def state_to_trainable_state(self , states):
    #Normalizing the lanes waiting time values in the input parameter according to minmax formula
    st = []
    for state in states:
      d = state[0 : 8]
      d = [float(i) for i in d]
      s = state[-1]
      mx = max(d)
      mn = min(d)
      if mx - mn != 0:
        d = [ (i - mn) / (mx - mn) for i in d]
      t = [0. , 0. , 0. , 0. , 0. , .0 , 0. , 0.]
      t[s] = 1.0
      for item in t:
        d.append(item)
      st.append(d)
    return np.array(st)

  def summary(self):
    return self.model.summary()

  def predict(self , X):
    return self.model.predict(self.state_to_trainable_state(X) , verbose = 0)

  def get_weights(self):
    return self.model.get_weights()

  def set_weights(self , weights):
    self.model.set_weights(weights)

  def fit(self , X , Y):
    X = self.state_to_trainable_state(X)
    h = self.model.fit(x=X, y=Y, validation_split = 0.0, batch_size = self.bs , epochs = self.epls , shuffle = True , verbose = 0)
    hl = h.history['loss']
    for item in hl:
      self.loss_log.append(item)

  def get_last_loss(self):
    if len(self.loss_log) > 0:
      return self.loss_log[-1]
    return 304

  def save_model(self , f):
    self.model.save_weights(f)

class ReplayBuffer: #Creat the ReplayBuffer element of the model
  def __init__(self, size):
    self.buffer = deque(maxlen=size)
    self.size = size

  def add(self, state, action, reward, next_state): #Adds sample to the ReplayBuffer memory
    if len(self.buffer) >= self.size:
      raise ValueError('Replay Buffer full')
    else:
      self.buffer.append((state, action, reward, next_state))

  def __len__(self):
    return len(self.buffer)

  def sample(self, num_samples): #Return number of smple from the ReplayBuffer
    states, actions, rewards, next_states = [], [], [], []
    idx = np.random.choice(len(self.buffer), num_samples)
    for i in idx:
      elem = self.buffer[i]
      state, action, reward, next_state = elem
      states.append(state)
      actions.append(action)
      rewards.append(reward)
      next_states.append(next_state)

    return states, actions, rewards, next_states

class DQNTrainer: #Combines all the classes in order to connect all and to train the model
  def __init__(self ,
               discount = 0.9 ,
               epsilon = 0.2 ,
               epsilon_decay = 0.99 ,
               qnn_train_time_tics_interval = 5 ,       # episode like
               tnn_update_time_tics_interval = 10 ,
               batch_size = 32 ,
               epsilon_decay_interval = 50 ,
               epochs_per_learning_session = 1 ,
               max_cars = 8 ,
               delay_at_change = 2 ,
               min_green_tics = 2
               ):

    self.env = Intersection(delay_at_change = delay_at_change , max_cars = max_cars , min_green_tics = min_green_tics)
    self.qnn = DQN(8 , epochs_per_learning_session = epochs_per_learning_session , batch_size = batch_size)
    self.tnn = DQN(8)
    self.tnn.set_weights(self.qnn.get_weights())
    self.agent = Trafficlight(self.qnn , epsilon = epsilon , epsilon_decay_rate = epsilon_decay)
    self.replay = ReplayBuffer(buffer_size)
    self.discount = discount
    self.epsilon = epsilon
    self.qnn_train_time_tics_interval = qnn_train_time_tics_interval
    self.tnn_update_time_tics_interval = tnn_update_time_tics_interval
    self.epsilon_decay = epsilon_decay
    self.batch_size = batch_size
    self.edi = epsilon_decay_interval
    self.epochs_per_learning_session = 1

  def qnn_train_step(self , states, actions, rewards, next_states): #Training the Q Neural Network
    Y = self.qnn.predict(states)
    next_qs = self.tnn.predict(next_states)
    next_qs_max = np.max(next_qs , axis = 1)
    for i in range(Y.shape[0]):
      Y[i , actions[i]] = rewards[i] + self.discount * next_qs_max[i]
    self.qnn.fit(states , Y)

  def train(self ,train_time_tics): #Training the model
    state = self.agent.get_initial_state()
    for t1 in range(1 , (train_time_tics // self.qnn_train_time_tics_interval) + 1):
      for t2 in range(self.qnn_train_time_tics_interval):
        action = self.agent.select_epsilon_greedy_action(state)
        reward , next_state = self.env.apply_action(state , action)
        self.replay.add(state, action, reward, next_state)
        state = next_state
      states , actions , rewards , next_states = self.replay.sample(self.batch_size)
      self.qnn_train_step(states, actions, rewards, next_states)

      if t1 % self.tnn_update_time_tics_interval == 0:
        self.tnn.set_weights(self.qnn.get_weights())

      if t1 % self.edi == 0:
        self.agent.epsilon_decay()

      if t1 % 10 == 0:
        print(f'Time tic: {t1*self.qnn_train_time_tics_interval} , qnn_loss: {self.qnn.get_last_loss(): .5f} , epsilon: {self.agent.get_epsilon()}')

      self.qnn.save_model(r'C:\Users\nirfo\RLModels\pycharm models\model8.weights.h5') #Save the model weights in order to use it later in SUMO
    return

def moving_average(a, n=3): #Helping finction that normalizing the values in moving avverage
  ret = np.cumsum(a, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  return ret[n - 1:] / n

if __name__ == '__main__':
  #Params
  num_actions = 8
  epochs_per_learning_session = 1
  batch_size = 64
  discount = 0.99 #suggested value 0.9 - 0.99 (lower: short-term thinking. higher: emphasizes long-term reward)
  epsilon = 1.0
  epsilon_decay = 0.999
  buffer_size = 300000
  qnn_train_time_tics_interval = 10
  tnn_update_time_tics_interval = 100
  epsilon_decay_interval = 10
  max_cars = 40
  delay_at_change = 2
  min_green_tics = 2
  train_time_tics = 200000

  #Training
  t = DQNTrainer(discount = discount ,
                    epsilon = epsilon ,
                    epsilon_decay = epsilon_decay ,
                    qnn_train_time_tics_interval = qnn_train_time_tics_interval ,
                    tnn_update_time_tics_interval = tnn_update_time_tics_interval ,
                    batch_size = batch_size ,
                    epsilon_decay_interval = epsilon_decay_interval ,
                    epochs_per_learning_session = epochs_per_learning_session ,
                    max_cars = max_cars ,
                    delay_at_change = delay_at_change ,
                    min_green_tics = min_green_tics
  )
  t.train(train_time_tics)

  #Ploting results
  loss_log = t.qnn.loss_log

  plt.figure(figsize = (10,6))
  plt.ylim(min(loss_log) - 0.01 , max(loss_log) + 0.01)
  plt.xlabel('Training sessions' , fontsize = 18)
  plt.ylabel('Loss' , fontsize = 18)
  plt.grid(True)
  plt.plot(loss_log, label='Cost')
  plt.savefig(r'C:\Users\nirfo\RLModels\Results\Model8\lost.png', dpi=300)
  plt.show(block=False)

  ma = moving_average(np.array(loss_log) , n = 100)
  mal = list(ma)

  plt.figure(figsize = (10,6))
  plt.ylim(min(mal) - 0.01 , max(mal) + 0.01)
  plt.xlabel('Training sessions' , fontsize = 18)
  plt.ylabel('Loss' , fontsize = 18)
  plt.grid(True)
  plt.plot(mal, label='Cost')
  plt.savefig(r'C:\Users\nirfo\RLModels\Results\Model8\ma_lost.png', dpi=300)
  plt.show(block=False)

  twt_log = t.env.total_wait_time_log

  plt.figure(figsize = (10,6))
  plt.ylim(min(twt_log) - 0.01 , max(twt_log) + 0.01)
  plt.xlabel('Training sessions' , fontsize = 18)
  plt.ylabel('Intersection wait time' , fontsize = 18)
  plt.grid(True)
  plt.plot(twt_log, label='Wait Time')
  plt.savefig(r'C:\Users\nirfo\RLModels\Results\Model8\twt.png', dpi=300)
  plt.show(block=False)

  twt_ma = moving_average(np.array(twt_log) , n = 400)

  plt.figure(figsize = (10,6))
  plt.ylim(min(twt_ma) - 0.01 , max(twt_ma) + 0.01)
  plt.xlabel('Training sessions' , fontsize = 18)
  plt.ylabel('Intersection wait time' , fontsize = 18)
  plt.grid(True)
  plt.plot(twt_ma, label='Cost')
  plt.savefig(r'C:\Users\nirfo\RLModels\Results\Model8\ma_twt.png', dpi=300)
  plt.show(block=False)

  ci = t.env.change_interval

  plt.figure(figsize = (10,6))
  plt.ylim(min(ci) - 0.01 , max(ci) + 0.01)
  plt.xlabel('Training sessions' , fontsize = 18)
  plt.ylabel('Change intervals' , fontsize = 18)
  plt.grid(True)
  plt.plot(ci, label='Change Intervals')
  plt.savefig(r'C:\Users\nirfo\RLModels\Results\Model8\ci.png', dpi=300)
  plt.show(block=False)

  ci_ma = moving_average(np.array(ci) , n = 100)

  plt.figure(figsize = (10,6))
  plt.ylim(min(ci_ma) - 0.01 , max(ci_ma) + 0.01)
  plt.xlabel('Training sessions' , fontsize = 18)
  plt.ylabel('Change intervals' , fontsize = 18)
  plt.grid(True)
  plt.plot(ci_ma, label='Change Intervals')
  plt.savefig(r'C:\Users\nirfo\RLModels\Results\Model8\ma_ci.png', dpi=300)
  plt.show()



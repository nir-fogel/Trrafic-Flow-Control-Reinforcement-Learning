import numpy as np
import traci
from keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense


def run():
    step = 0
    current_action = 0
    junction_id = 'J10'
    wait_penalty = 2 

    while traci.simulation.getMinExpectedNumber() > 0:
        lanes_wait_time = []
        for lane_id in lanes_id:
            sum_wait_time = 0
            vehicles_id = traci.lane.getLastStepVehicleIDs(lane_id)
            for vehicle_id in vehicles_id:
                sum_wait_time += traci.vehicle.getAccumulatedWaitingTime(vehicle_id)
            lanes_wait_time.append(sum_wait_time)
        lanes_input_time = state_to_input_state(lanes_wait_time)

        if step % 1 == 0:
            current_action = get_current_action()
            action_one_hot = np.array([0,0,0,0,0,0,0,0])
            print(action_one_hot)
            print(type(action_one_hot))
            if current_action != 0:
                action_one_hot[current_action-1] = 1
            print("Step: " + str(step) + " Sim Step: " + str(traci.simulation.getTime()))
            print(lanes_wait_time)
            input_params = np.concatenate((lanes_input_time, action_one_hot))
            input_params = input_params.reshape(1, -1)
            print(input_params)

            pred = model.predict(input_params)
            next_action = np.argmax(pred)+1
            print(next_action)

            if next_action != current_action:
                yellow1 = get_traffic_light_str(current_action)
                yellow1 = ''.join(['y' if char == 'g' else char for char in yellow1])
                traci.trafficlight.setRedYellowGreenState(junction_id, yellow1)
                for i in range(wait_penalty // 2):
                    traci.simulationStep()
                    step += 1

                yellow2 = get_traffic_light_str(next_action)
                yellow2 = ''.join(['y' if char == 'g' else char for char in yellow2])
                traci.trafficlight.setRedYellowGreenState(junction_id, yellow2)
                for i in range(wait_penalty // 2):
                    traci.simulationStep()
                    step += 1

                traci.trafficlight.setRedYellowGreenState(junction_id, get_traffic_light_str(next_action))

            current_action = next_action
        traci.simulationStep()
        step += 1
    return

def get_traffic_light_str(num_action):
    match num_action:
        case 0:
            return 'rrrrrrrrrrrr'
        case 1:
            return 'rrrggrrrrggr'
        case 2:
            return 'ggrrrrggrrrr'
        case 3:
            return 'rrgrrrrrgrrr'
        case 4:
            return 'rrrrrgrrrrrg'
        case 5:
            return 'rrrrrrgggrrr'
        case 6:
            return 'rrrgggrrrrrr'
        case 7:
            return 'gggrrrrrrrrr'
        case 8:
            return 'rrrrrrrrrggg'

def get_current_action():
    str_traffic_light = traci.trafficlight.getRedYellowGreenState(tlsID)
    match str_traffic_light:
        case 'rrrggrrrrggr':
            return  1
        case 'ggrrrrggrrrr':
            return  2
        case 'rrgrrrrrgrrr':
            return 3
        case 'rrrrrgrrrrrg':
            return  4
        case 'rrrrrrgggrrr':
            return 5
        case 'rrrgggrrrrrr':
            return 6
        case 'gggrrrrrrrrr':
            return 7
        case 'rrrrrrrrrggg':
            return 8
    return 0

def state_to_input_state(state):
    d = [float(i) for i in state]
    mx = max(d)
    mn = min(d)
    if mx - mn != 0:
        d = [(i - mn) / (mx - mn) for i in d]
    return np.array(d)

if __name__ == '__main__':
    num_actions = 8
    model = Sequential([
        Dense(units=128, input_shape=(16,), activation='relu'),
        Dense(units=128, activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=num_actions, activation='linear')
    ])
    model.load_weights(r'C:\Users\nirfo\RLModels\pycharm models\model3.weights.h5')

    tlsID = "J10"
    sumoBinary = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui"
    sumoCmd = [sumoBinary, "-c", r"C:\Users\nirfo\Sumo Stuff\conf_low_demand.sumocfg"]
    lanes_id = ["NI_0", "SI_0", "EI_0", "WI_0", "SI_1", "NI_1", "WI_1", "EI_1"]

    traci.start(sumoCmd)
    run()
    traci.close()

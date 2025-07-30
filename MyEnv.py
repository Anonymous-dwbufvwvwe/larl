import gymnasium as gym
from highway_env.envs.highway_env import *
from gym import spaces
import numpy as np
from typing import TypeVar
import pdb
from typing import Tuple,Union
from gymnasium import Wrapper
from gymnasium.utils import RecordConstructorArgs
from gymnasium.wrappers import RecordVideo

from highway_env.envs.common.action import Action, ActionType, action_factory
from highway_env.envs.common.finite_mdp import finite_mdp
from highway_env.envs.common.graphics import EnvViewer
from highway_env.envs.common.observation import ObservationType, observation_factory
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle
from typing import Callable, List, Sequence, Tuple, Union
Vector = Union[np.ndarray, Sequence[float]]
Matrix = Union[np.ndarray, Sequence[Sequence[float]]]
Interval = Union[
    np.ndarray,
    Tuple[Vector, Vector],
    Tuple[Matrix, Matrix],
    Tuple[float, float],
    List[Vector],
    List[Matrix],
    List[float],
]

Observation = TypeVar("Observation")

def lmap(v: float, x: Interval, y: Interval) -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])

def get_lane(y):
    if 14 <= y < 18:
        return +4
    if 10 <= y < 14:
        return +3
    elif 6 <= y < 10:
        return +2
    elif 2 <= y < 6:
        return +1
    elif -2 <= y < 2:
        return 0
    elif -6 <= y < -2:
        return -1
    elif -10 <= y < -6:
        return -2
    elif -14 <= y < -10:
        return -3
    else:
        return None


def classify_vehicles_by_lane(vehicles):
    ego_vehicle = vehicles[0]            
    other_vehicles = vehicles[1:]      
    ego_lane = get_lane(ego_vehicle.position[1])
    same_lane = []
    left_lanes = []
    right_lanes = []
    neighbor_lanes = []

    for vehicle in other_vehicles:
        lane = get_lane(vehicle.position[1])  
        if lane is None:
            continue
        if lane == ego_lane:
            same_lane.append(vehicle)
        elif lane-ego_lane == 1:
            right_lanes.append(vehicle)
            neighbor_lanes.append(vehicle)
        elif lane-ego_lane == -1:
            left_lanes.append(vehicle)
            neighbor_lanes.append(vehicle)
    
    return same_lane, left_lanes, right_lanes, neighbor_lanes



def ttc_calc(ego_vehicle,lanes):
    min_ttc = float("inf")
    delta_x_min = float("inf")
    vehicle_xs = [v.position[0] for v in lanes]
    diff_xs = vehicle_xs - ego_vehicle.position[0]
    
    if len(np.where(diff_xs>0)[0]) == 0:
        return min_ttc, delta_x_min
    leader_index = np.where(diff_xs>0)[0][0]

    vehicle = lanes[leader_index]
    other_x, other_vx = vehicle.position[0], vehicle.speed*np.cos(vehicle.heading)
    delta_x = other_x - ego_vehicle.position[0] - 4
    rel_vx = ego_vehicle.speed*np.cos(ego_vehicle.heading) - other_vx
    if delta_x > 0 and rel_vx > 0:
        ttc = delta_x / rel_vx
        min_ttc = min(min_ttc, ttc)
    if delta_x > 0 and delta_x < delta_x_min:    
        delta_x_min = delta_x
    return min_ttc, delta_x_min

def longi_same_ttc(vehicles,same_lanes):
    ego_vehicle = vehicles[0]                

    min_ttc, delta_x_min = ttc_calc(ego_vehicle,same_lanes)
    return min_ttc, delta_x_min

def longi_neighbor_ttc(vehicles,left_lanes, right_lanes,action):
    ego_vehicle = vehicles[0]                
    ego_lane = get_lane(ego_vehicle.position[1])
      
    min_ttc = float("inf")
    delta_x_min = float("inf")
    
    
    if action == 0 and ego_lane != 0:
        neighbor_lanes = left_lanes
    elif action == 2 and ego_lane != 2:
        neighbor_lanes = right_lanes
    else:
        return min_ttc, delta_x_min
    
    min_ttc, delta_x_min = ttc_calc(ego_vehicle,neighbor_lanes)
    return min_ttc, delta_x_min

def lateral_ttc(vehicles,other_lanes):
    ego_vehicle = vehicles[0]              
      
    min_ttc_lat = float("inf")
    min_dis_lat = float("inf")
    min_dis = float("inf")
    # min_ttc_lon = float("inf")
    for vehicle in other_lanes:
        other_x, other_vx = vehicle.position[0], vehicle.speed*np.cos(vehicle.heading)
        delta_x = abs(ego_vehicle.position[0] - other_x - 5)
        
        other_y, other_vy = vehicle.position[1], vehicle.speed*np.sin(vehicle.heading)
        delta_y = other_y - ego_vehicle.position[1] - 2
        
        rel_vy = other_vy - ego_vehicle.speed*np.sin(ego_vehicle.heading)
        
        rel_dis = np.sqrt((other_x-ego_vehicle.position[0])**2+(other_y-ego_vehicle.position[1])**2)

        signal = delta_y * rel_vy
        if delta_x < 15 and signal < 0:
            ttc_lat = abs(delta_y / rel_vy)
            min_ttc_lat = min(min_ttc_lat, ttc_lat)
            if abs(delta_y) < min_dis_lat:
                min_dis_lat = abs(delta_y)
                if rel_dis < min_dis:
                    min_dis = rel_dis
    return min_ttc_lat, min_dis_lat, min_dis

class MyEnv(HighwayEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """
    # def __init__(self):
    #     super().__init__()
        
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "simulation_frequency": 5,
                "lanes_count": 4,
                "vehicles_count": 20,
                "duration": 30,  # [s]
                "ego_spacing": 1.5,
                "vehicles_density": 2.0,
                "reward_speed_range":[20, 30],
                "actions":{},
                "overtake_num":0
            }
        )
        return cfg
        
    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False


    def set_values(self, jerk_weight, control_weight):
        self.jerk_weight = jerk_weight
        self.control_weight = control_weight
        
    def _reward(self, action: Action) -> float:
        
        
        # forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)

        # r_speed = np.clip(scaled_speed, 0, 1)
        self.config["actions"][self.time] = action
        
        
        r_speed = (1/10) * (self.vehicle.speed - 20)

        # # on road judge
        if_on_road = float(self.vehicle.on_road)
        
        
        # miss signal reward
        r_explor = 0
        ego_y = self.vehicle.position[1]
        if ego_y >= -2 and ego_y <= 2 and action == 0:
            r_explor = 0
        elif ego_y >= 6 and ego_y <= 10 and action == 2:
            r_explor = 0
        else:
            if self.time == 1.0:
                if action == 0 and action == 2:
                    r_explor = 0.4
            else:
                last_action = self.config["actions"][self.time-1.0]
                if action == 0 and last_action != 2:
                    r_explor = 0.4
                elif action == 2 and last_action != 0:
                    r_explor = 0.4
                else:
                    r_explor = -0.1
        
        
        # # collision reward
        if self.vehicle.crashed:
            r_collision = -2
        else:
            r_collision = 0
            
        # safety reward
        r_safety = 0
        for time in range((int(self.time)-1)*5,(int(self.time)-1)*5+5):
            vehicles_all = self.road.vehicles_save[time]
            same_lane, left_lanes, right_lanes, neighbor_lanes = classify_vehicles_by_lane(vehicles_all)
            lon_ttc_n, delta_x_min_n = longi_neighbor_ttc(vehicles_all,left_lanes, right_lanes,action)
            lon_ttc_s, delta_x_min_s = longi_same_ttc(vehicles_all,same_lane)
            lat_ttc, delta_y_min_n, min_dis = lateral_ttc(vehicles_all, neighbor_lanes)
            r_lon,r_lat,r_delta = 0, 0, 0
            
            if lon_ttc_n < 6: 
                r_lon += -1/lon_ttc_n*0.2
            if delta_x_min_n < 15:
                r_delta += -1/delta_x_min_n*0.2
            if lon_ttc_s < 3:
                r_lon += -1/lon_ttc_s*0.2
            if delta_x_min_s < 10:
                r_delta += -1/delta_x_min_s*0.2
            if lat_ttc < 6:
                r_lat += -1/lat_ttc*0.2   
            if delta_y_min_n < 2:
                r_delta += -1/delta_y_min_n*0.2
            if min_dis < 10:
                r_delta += -1/min_dis*0.2             
            r_safety += r_lon + r_lat + r_delta

        # overtake reward
        r_overtake = 0
        ego_x = self.vehicle.position[0]
        vehicles_pos= np.array([vehicle.position[0] for vehicle in self.road.vehicles[1:]])
        diff_x = ego_x - vehicles_pos
        overtake_num = np.sum(diff_x > 0)

        if overtake_num > self.config["overtake_num"]:            
            r_overtake += (overtake_num-self.config["overtake_num"])/vehicles_pos.shape[0]*5
            self.config["overtake_num"] = overtake_num

        elif overtake_num < self.config["overtake_num"]:
            r_overtake += (overtake_num-self.config["overtake_num"])/vehicles_pos.shape[0]*0.5
            self.config["overtake_num"] = overtake_num
            
        elif overtake_num == self.config["overtake_num"]:
            r_overtake = 0
        
        reward_nega = r_collision + r_safety
        reward_posi = 0
        if r_speed < 0:
            reward_nega +=  r_speed 
        else:
            reward_posi +=  r_speed 
        if r_explor < 0:
            reward_nega +=  r_explor
        else:
            reward_posi +=  r_explor
        
        if r_overtake < 0:
            reward_nega +=  r_overtake
        else:
            reward_posi +=  r_overtake 
        
        reward = (r_collision + r_overtake + r_speed + r_explor + r_safety)
        if reward >=0 and reward <=1:
            rewards = reward
        elif reward <=0 and reward >=-1:
            rewards = reward
        else:
            rewards = lmap(reward,[reward_nega,reward_posi],[-1,1])        
            if rewards == 1: 
                rewards = min(reward_posi,1)
            elif rewards == -1:
                rewards = max(reward_nega,-1)
      
        rewards = rewards*if_on_road
        
        return rewards
        
        
gym.register(
    id='MyEnv-v0',
    entry_point="MyEnv:MyEnv",
)

    
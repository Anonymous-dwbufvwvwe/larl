from __future__ import annotations

import copy
from collections import deque
import math
import numpy as np
import cvxpy
from highway_env.road.road import Road
from highway_env.utils import Vector
from highway_env.vehicle.objects import RoadObject
from enum import Enum
import pdb
from cvxpylayers.torch import CvxpyLayer
import torch
# import warnings
# warnings.filterwarnings('ignore')


def get_nparray_from_matrix(x):
    return np.array(x).flatten()

class Speed(Enum):
    SLOW = 20.0
    FAST = 30.0


class KinematicModel:
    
    def __init__(self, dt, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.dt = dt
        self.L = 5.0
        
    def get_state_space(self, v, phi, delta):
    
        A = np.matrix(np.zeros((4, 4)))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.dt * np.cos(phi)
        A[0, 3] = - self.dt * v * np.sin(phi)
        A[1, 2] = self.dt * np.sin(phi)
        A[1, 3] = self.dt * v * np.cos(phi)
        A[3, 2] = self.dt * np.tan(delta) / self.L

        B = np.matrix(np.zeros((4, 2)))
        B[2, 0] = self.dt
        B[3, 1] = self.dt * v / (self.L * math.cos(delta) ** 2)

        C = np.zeros(4)
        C[0] = self.dt * v * np.sin(phi) * phi
        C[1] = - self.dt * v * np.cos(phi) * phi
        C[3] = v * delta / (self.L * np.cos(delta) ** 2)

        return A, B, C
    
    def update_state(self, a, delta,MAX_STEER):

        # input check
        if delta >= MAX_STEER:
            delta = MAX_STEER
        elif delta <= -MAX_STEER:
            delta = -MAX_STEER

        self.x = self.x + self.v * np.cos(self.yaw) * self.dt
        self.y = self.y + self.v * np.sin(self.yaw) * self.dt
        self.yaw = self.yaw + self.v / self.L * np.tan(delta) * self.dt
        self.v = self.v + a * self.dt

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

def get_leader(lane_objects):
    leaders = np.array([v.position[0] for v in lane_objects])
    if len(leaders) == 0:
        leader_idx = None
        leader = None
    else:
        leader_idx = np.argmin(leaders)
        leader = lane_objects[leader_idx]
    return leader

def classify_vehicles_by_lane(vehicles):
    ego_vehicle = vehicles[0]  
    other_vehicles = vehicles[1:]        # 其余周车状态
    ego_lane = get_lane(ego_vehicle.position[1])
    left_lane,right_lane,same_lane = [],[],[]

    for vehicle in other_vehicles:
        lane = get_lane(vehicle.position[1])  
        if lane is None:
            continue
        if lane == ego_lane:
            diff_x = vehicle.position[0] - ego_vehicle.position[0]
            if diff_x > 0:
                same_lane.append(vehicle)
        
        if ego_lane is None:
            print("ego lane is None")
        elif abs(lane-ego_lane) == 1:
            diff_x = abs(vehicle.position[0] - ego_vehicle.position[0])
            diff_y = vehicle.position[1] - ego_vehicle.position[1]
            if diff_x <= 15:
                if diff_y < 0:
                    left_lane.append(vehicle)
                else:
                    right_lane.append(vehicle)
    return same_lane,left_lane,right_lane



def lane_pos_mapping(target_lane_index):
    lane_mapping = {0:0,1:4,2:8,3:12,4:16}
    desired_y = lane_mapping[target_lane_index]
    return desired_y


class Vehicle(RoadObject):
    """
    A moving vehicle on a road, and its kinematics.

    The vehicle is represented by a dynamical system: a modified bicycle model.
    It's state is propagated depending on its steering and acceleration actions.
    """

    LENGTH = 5.0
    """ Vehicle length [m] """
    WIDTH = 2.0
    """ Vehicle width [m] """
    DEFAULT_INITIAL_SPEEDS = [23, 25]
    """ Range for random initial speeds [m/s] """
    MAX_SPEED = 40.0
    """ Maximum reachable speed [m/s] """
    MIN_SPEED = -40.0
    """ Minimum reachable speed [m/s] """
    HISTORY_SIZE = 30
    """ Length of the vehicle state history, for trajectory display"""

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        predition_type: str = "constant_steering",
    ):
        super().__init__(road, position, heading, speed)
        self.prediction_type = predition_type
        self.action = {"steering": 0, "acceleration": 0}
        self.crashed = False
        self.impact = None
        self.log = []
        self.history = deque(maxlen=self.HISTORY_SIZE)
        self.T = 10
        self.u_lim = np.array([4.0,math.radians(10.0)])
        self.v_lim = np.array([35.0,4.0])
        self.du_lim = np.array([-4.0,math.radians(6.0)])
        self.NX = 4
        self.NU = 2
        self.acc_records = []
        self.delta_records = []

    @classmethod
    def create_random(
        cls,
        road: Road,
        speed: float = None,
        lane_from: str | None = None,
        lane_to: str | None = None,
        lane_id: int | None = None,
        spacing: float = 1,
    ) -> Vehicle:
        """
        Create a random vehicle on the road.

        The lane and /or speed are chosen randomly, while longitudinal position is chosen behind the last
        vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param speed: initial speed in [m/s]. If None, will be chosen randomly
        :param lane_from: start node of the lane to spawn in
        :param lane_to: end node of the lane to spawn in
        :param lane_id: id of the lane to spawn in
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: A vehicle with random position and/or speed
        """
        _from = lane_from or road.np_random.choice(list(road.network.graph.keys()))
        _to = lane_to or road.np_random.choice(list(road.network.graph[_from].keys()))
        _id = (
            lane_id
            if lane_id is not None
            else road.np_random.choice(len(road.network.graph[_from][_to]))
        )
        lane = road.network.get_lane((_from, _to, _id))
        if speed is None:
            if lane.speed_limit is not None:
                speed = road.np_random.uniform(
                    0.7 * lane.speed_limit, 0.8 * lane.speed_limit
                )
            else:
                speed = road.np_random.uniform(
                    Vehicle.DEFAULT_INITIAL_SPEEDS[0], Vehicle.DEFAULT_INITIAL_SPEEDS[1]
                )
        default_spacing = 12 + 1.0 * speed
        offset = (
            spacing
            * default_spacing
            * np.exp(-5 / 40 * len(road.network.graph[_from][_to]))
        )
        x0 = (
            np.max([lane.local_coordinates(v.position)[0] for v in road.vehicles])
            if len(road.vehicles)
            else 3 * offset
        )
        x0 += offset * road.np_random.uniform(0.9, 1.1)
        v = cls(road, lane.position(x0, 0), lane.heading_at(x0), speed)
        return v

    @classmethod
    def create_from(cls, vehicle: Vehicle) -> Vehicle:
        """
        Create a new vehicle from an existing one.

        Only the vehicle dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, vehicle.heading, vehicle.speed)
        if hasattr(vehicle, "color"):
            v.color = vehicle.color
        return v

    def act(self, action: dict | str = None) -> None:
        """
        Store an action to be repeated.

        :param action: the input action
        """
        if action:
            self.action = action

    def step(self, dt: float) -> None:
        """
        Propagate the vehicle state given its actions.

        Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
        If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
        The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        if str(self)[0:10] == "MDPVehicle":
            desired_y = lane_pos_mapping(self.target_lane_index[2])
            # if self.signal_action == "SLOWER":
            #     TARGET_SPEED = 20.0
            # elif self.signal_action == "FASTER":
            #     TARGET_SPEED = 30.0
            # elif self.signal_action == "IDLE":
            #     TARGET_SPEED = 25.0
            # else:
            TARGET_SPEED = 30.0
            
            if self.time == 1:
                oa, od = [0.0] * self.T, [0.0] * self.T  
            else:
                oa, od = self.oa, self.od   
            self.save_prior = (oa,od)    
            oa, od, = self.Iterative_linear_mpc(desired_y, TARGET_SPEED, dt, oa, od)   

            self.oa, self.od = oa, od
            if oa is None:
                delta_f = self.action["steering"]
                acc = self.action["acceleration"]  
            else:                         
                oa, od = oa[0], od[0]
                # oa, od = oa[0], od[0]
                delta_f = od
                acc = oa
            self.acc_records.append(acc)
            self.delta_records.append(delta_f)
        else:
            self.clip_actions()
            delta_f = self.action["steering"]
            acc = self.action["acceleration"]
            
        beta = np.arctan(1 / 2 * np.tan(delta_f))
        v = self.speed * np.array(
            [np.cos(self.heading + beta), np.sin(self.heading + beta)]
        )
        self.position += v * dt
        if self.impact is not None:
            self.position += self.impact
            self.crashed = True
            self.impact = None
        self.heading += self.speed * np.sin(beta) / (self.LENGTH / 2) * dt
        self.speed += acc * dt
        self.on_state_update()

    def clip_actions(self) -> None:
        if self.crashed:
            self.action["steering"] = 0
            self.action["acceleration"] = -1.0 * self.speed
        self.action["steering"] = float(self.action["steering"])
        self.action["acceleration"] = float(self.action["acceleration"])
        if self.speed > self.MAX_SPEED:
            self.action["acceleration"] = min(
                self.action["acceleration"], 1.0 * (self.MAX_SPEED - self.speed)
            )
        elif self.speed < self.MIN_SPEED:
            self.action["acceleration"] = max(
                self.action["acceleration"], 1.0 * (self.MIN_SPEED - self.speed)
            )

    def on_state_update(self) -> None:
        if self.road:
            self.lane_index = self.road.network.get_closest_lane_index(
                self.position, self.heading
            )
            self.lane = self.road.network.get_lane(self.lane_index)
            if self.road.record_history:
                self.history.appendleft(self.create_from(self))

    def predict_trajectory_constant_speed(
        self, times: np.ndarray
    ) -> tuple[list[np.ndarray], list[float]]:
        if self.prediction_type == "zero_steering":
            action = {"acceleration": 0.0, "steering": 0.0}
        elif self.prediction_type == "constant_steering":
            action = {"acceleration": 0.0, "steering": self.action["steering"]}
        else:
            raise ValueError("Unknown predition type")

        dt = np.diff(np.concatenate(([0.0], times)))

        positions = []
        headings = []
        v = copy.deepcopy(self)
        v.act(action)
        for t in dt:
            v.step(t)
            positions.append(v.position.copy())
            headings.append(v.heading)
        return (positions, headings)

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.direction  # TODO: slip angle beta should be used here

    @property
    def destination(self) -> np.ndarray:
        if getattr(self, "route", None):
            last_lane_index = self.route[-1]
            last_lane_index = (
                last_lane_index
                if last_lane_index[-1] is not None
                else (*last_lane_index[:-1], 0)
            )
            last_lane = self.road.network.get_lane(last_lane_index)
            return last_lane.position(last_lane.length, 0)
        else:
            return self.position

    @property
    def destination_direction(self) -> np.ndarray:
        if (self.destination != self.position).any():
            return (self.destination - self.position) / np.linalg.norm(
                self.destination - self.position
            )
        else:
            return np.zeros((2,))

    @property
    def lane_offset(self) -> np.ndarray:
        if self.lane is not None:
            long, lat = self.lane.local_coordinates(self.position)
            ang = self.lane.local_angle(self.heading, long)
            return np.array([long, lat, ang])
        else:
            return np.zeros((3,))

    def to_dict(
        self, origin_vehicle: Vehicle = None, observe_intentions: bool = True
    ) -> dict:
        d = {
            "presence": 1,
            "x": self.position[0],
            "y": self.position[1],
            "vx": self.velocity[0],
            "vy": self.velocity[1],
            "heading": self.heading,
            "cos_h": self.direction[0],
            "sin_h": self.direction[1],
            "cos_d": self.destination_direction[0],
            "sin_d": self.destination_direction[1],
            "long_off": self.lane_offset[0],
            "lat_off": self.lane_offset[1],
            "ang_off": self.lane_offset[2],
        }
        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ["x", "y", "vx", "vy"]:
                d[key] -= origin_dict[key]
        return d

    def __str__(self):
        return "{} #{}: {}".format(
            self.__class__.__name__, id(self) % 1000, self.position
        )

    def __repr__(self):
        return self.__str__()

    def predict_trajectory(
        self,
        actions: list,
        action_duration: float,
        trajectory_timestep: float,
        dt: float,
    ) -> list[Vehicle]:
        """
        Predict the future trajectory of the vehicle given a sequence of actions.

        :param actions: a sequence of future actions.
        :param action_duration: the duration of each action.
        :param trajectory_timestep: the duration between each save of the vehicle state.
        :param dt: the timestep of the simulation
        :return: the sequence of future states
        """
        states = []
        v = copy.deepcopy(self)
        t = 0
        for action in actions:
            v.act(action)  # Low-level control action
            for _ in range(int(action_duration / dt)):
                t += 1
                v.step(dt)
                if (t % int(trajectory_timestep / dt)) == 0:
                    states.append(copy.deepcopy(v))
        return states


    def predict_motion(self,dt,x0,oa,od):
        xbar = np.zeros((self.NX, self.T + 1))
        
        for i in range(len(x0)):
            xbar[i, 0] = x0[i]
        ugv =  KinematicModel(dt,x=self.position[0], y=self.position[1], yaw=self.heading, v=self.speed)
        if oa is not None:
            for (ai, di, i) in zip(oa, od, range(1, self.T + 1)):
                ugv.update_state(ai, di,self.u_lim[1])
                xbar[0, i] = ugv.x
                xbar[1, i] = ugv.y
                xbar[2, i] = ugv.v
                xbar[3, i] = ugv.yaw
        elif oa is None:
            oa = [0.0] * self.T
            od = [0.0] *self. T
            
            for (ai, di, i) in zip(oa, od, range(1, self.T + 1)):
                ugv.update_state(ai, di,self.u_lim[1])
                xbar[0, i] = ugv.x
                xbar[1, i] = ugv.y
                xbar[2, i] = ugv.v
                xbar[3, i] = ugv.yaw            
        return xbar

    def calc_profile(self,desired_y,TARGET_SPEED):
        xref = np.zeros((self.NX-1, self.T + 1))
        dref = np.zeros((1, self.T + 1))
        for i in range(self.T + 1):
            xref[0, i] = 0
            xref[1, i] = desired_y
            xref[2, i] = TARGET_SPEED
            dref[0, i] = 0.0
        return xref,dref
    
    def get_prediction(self,dt,vehicle):
        onlane = np.zeros((4,self.T+1))
        onlane[0,0] = vehicle.position[0]
        onlane[1,0] = vehicle.position[1]
        onlane[2,:] = vehicle.speed
        onlane[3,:] = vehicle.heading
        
        delta_x = onlane[2, :-1] * np.cos(onlane[3, :-1]) * dt
        delta_y = onlane[2, :-1] * np.sin(onlane[3, :-1]) * dt
        onlane[0, 1:] = onlane[0, 0] + np.cumsum(delta_x)
        onlane[1, 1:] = onlane[1, 0] + np.cumsum(delta_y)
        return onlane
    
    def get_constraint_feature(self,dt,x):
        sigma = 0.3
        vehicles = self.road.vehicles
        ego_vehicle = self.road.vehicles[0]
        same_lane,left_lane,right_lane = classify_vehicles_by_lane(vehicles)
        leader = get_leader(same_lane)
        
        diffs_onlane = []
        if leader is None:
            pass
        else:
            onlane_x = self.get_prediction(dt,leader) # [4,T+1]
            diff_onlane = (onlane_x[0,:] - x[0,:])- sigma*x[2,:]
            diffs_onlane.append(diff_onlane)
            
        if len(diffs_onlane) > 0:
            x_signal = 1
        else:
            x_signal = 0
        
        diffs_left = []  
        for vehicle in left_lane:
            left_y = self.get_prediction(dt,vehicle)
            diff_left = (x[1,:] - left_y[1,:]) 
            diffs_left.append(diff_left)
        
        diffs_right = []  
        for vehicle in right_lane:
            right_y = self.get_prediction(dt,vehicle)
            diff_right = (right_y[1,:] - x[1,:]) 
            diffs_right.append(diff_right)
            
        diffs_neighbor = diffs_left + diffs_right
        if len(diffs_neighbor) > 0:
            y_signal = 1
        else:
            y_signal = 0
            
        return diffs_onlane, diffs_neighbor, x_signal, y_signal
    

    
    def Solve_MPC(self,desired_y,dt,oa,od,TARGET_SPEED): 
        
        d_safe = {"x":10,"y":4}
        gamma_ycbf = 0.8
        
        R = np.diag([8.0,0.1])
        Q = np.diag([0.05,0.05]) 
        P = np.diag([0.2,0.2])
        
        
        S = np.diag([0.0,5.0])
        Rs_lon = np.diag([10.0,0.0])
        Rs_lat = np.diag([1000.0,0.0])
        
        x0 = [self.position[0], self.position[1], self.speed, self.heading]
        
        xref,dref = self.calc_profile(desired_y,TARGET_SPEED)
        xbar = self.predict_motion(dt,x0,oa,od)
        x = cvxpy.Variable((self.NX, self.T + 1))
        u = cvxpy.Variable((self.NU, self.T))
        
        slack_cbf_y = cvxpy.Variable((2, self.T))
        slack_cbf_x = cvxpy.Variable((2, self.T))
        
        diffs_onlane, diffs_neighbor, x_signal, y_signal = self.get_constraint_feature(dt,x)
        
        cost = 0.0
        constraints = []
        for t in range(self.T):
            cost += cvxpy.quad_form(u[:, t], Q)
            if t != 0:
                cost += cvxpy.quad_form(xref[1:, t] - x[1:3, t], R)
           
            self.dyn = KinematicModel(dt,x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
            A, B, C = self.dyn.get_state_space(xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]
            
            if t < (self.T - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], P)
                constraints += [u[1, t + 1] - u[1, t]-self.du_lim[1] * dt<= 0]
                constraints += [u[0, t + 1] - u[0, t]-self.du_lim[0] * dt>= 0]
                constraints += [u[1, t + 1] - u[1, t]+self.du_lim[1] * dt>= 0]
            
            # y_signal = False 
            if y_signal:
                for i in range(len(diffs_neighbor)):
                    cost += cvxpy.quad_form(slack_cbf_y[:, t], Rs_lat)
                    constraints += [diffs_neighbor[i][t+1]-d_safe["y"]-(diffs_neighbor[i][t]-d_safe["y"])+gamma_ycbf*(diffs_neighbor[i][t]-d_safe["y"]) + slack_cbf_y[0,t]>= 0]#-d3[0,t]
            if x_signal:
                for i in range(len(diffs_onlane)):
                    cost += cvxpy.quad_form(slack_cbf_x[:, t], Rs_lon)
                    constraints += [diffs_onlane[i][t+1]-d_safe["x"]-(diffs_onlane[i][t]-d_safe["x"])+gamma_ycbf*(diffs_onlane[i][t]-d_safe["x"]) + slack_cbf_x[0,t]>=0]#-d1[0,t]

        cost += cvxpy.quad_form(x[2:,t], S)
        cost += cvxpy.quad_form(xref[1:, self.T] - x[1:3, self.T], R)
        constraints += [x[:, 0] == x0]
        constraints += [x[2, :] <= self.v_lim[0]]
        constraints += [x[2, :] >= self.v_lim[1]]

        constraints += [u[0, :] <= self.u_lim[0]]
        constraints += [u[0, :] >= -8.0]
        constraints += [cvxpy.abs(u[1, :]) <= self.u_lim[1]]
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.ECOS, verbose=False)
        
                  
        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox = get_nparray_from_matrix(x.value[0, :])
            oy = get_nparray_from_matrix(x.value[1, :])
            ov = get_nparray_from_matrix(x.value[2, :])
            oyaw = get_nparray_from_matrix(x.value[3, :])
            oa = get_nparray_from_matrix(u.value[0, :])
            odelta = get_nparray_from_matrix(u.value[1, :])
        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None
  
        return oa, odelta

    
    def Iterative_linear_mpc(self,desired_y,TARGET_SPEED,dt,oa,od):
        if oa is None or od is None:
            oa = [0.0] * self.T
            od = [0.0] * self.T


        oa, od = self.Solve_MPC(desired_y,dt,oa,od,TARGET_SPEED)
        

        return oa, od
    
    
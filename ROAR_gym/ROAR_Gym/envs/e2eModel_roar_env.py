try:
    from ROAR_Gym.envs.roar_env import ROAREnv
except:
    from ROAR_Gym.ROAR_Gym.envs.roar_env import ROAREnv

from ROAR.utilities_module.vehicle_models import VehicleControl
from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.vehicle_models import Vehicle
from typing import Tuple
import numpy as np
from typing import List, Any
from ROAR_Sim.carla_client.carla_runner import CarlaRunner
import gym
import math
from collections import OrderedDict
from gym.spaces import Discrete, Box
import cv2
import wandb

mode='baseline'
if mode=='no_map':
    FRAME_STACK = 1
else:
    FRAME_STACK = 4

if mode=='baseline':
    CONFIG = {
        "x_res": 84,
        "y_res": 84
    }
else:
    CONFIG = {
        # max values are 280x280
        # original values are 80x80
        "x_res": 80,
        "y_res": 80
    }



class ROARppoEnvE2E(ROAREnv):
    def __init__(self, params):
        super().__init__(params)
        #self.action_space = Discrete(len(DISCRETE_ACTIONS))
        low=np.array([-2.5, -5.0, 1.0])
        high=np.array([-0.5, 5.0, 3.0])
        # low=np.array([100, 0, -1])
        # high=np.array([1, 0.12, 0.5])
        self.mode=mode
        if self.mode=='baseline':
            self.action_space = Box(low=low, high=high, dtype=np.float32)
        else:
            self.action_space = Box(low=np.tile(low,(FRAME_STACK)), high=np.tile(high,(FRAME_STACK)), dtype=np.float32)

        if self.mode=='no_map':
            self.observation_space = Box(low=np.tile([-1],(13)), high=np.tile([1],(13)), dtype=np.float32)
        elif self.mode=='combine':
            self.observation_space = Box(-10, 1, shape=(FRAME_STACK,3, CONFIG["x_res"], CONFIG["y_res"]), dtype=np.float32)
        elif self.mode=='baseline':
            self.observation_space = Box(-10, 1, shape=(FRAME_STACK,3, CONFIG["x_res"], CONFIG["y_res"]), dtype=np.float32)
        else:
            self.observation_space = Box(-10, 1, shape=(FRAME_STACK, CONFIG["x_res"], CONFIG["y_res"]), dtype=np.float32)
        self.prev_speed = 0
        self.prev_cross_reward = 0
        self.crash_check = False
        self.ep_rewards = 0
        self.frame_reward = 0
        self.highscore = -1000
        self.highest_chkpt = 0
        self.speeds = []
        self.prev_int_counter = 0
        self.steps=0
        self.largest_steps=0
        self.highspeed=0
        self.complete_loop=False
        self.his_checkpoint=[]
        self.his_score=[]
        self.time_to_waypoint_ratio = 0.25
        # self.crash_step=0
        # self.reward_step=0
        # self.reset_by_crash=True
        self.fps=8
        # self.crash_tol=5
        # self.reward_tol=5
        # self.end_check=False
        self.death_line_dis = 5
        ## used to check if stalled
        self.stopped_counter = 0
        self.stopped_max_count = 10
        # used to track episode highspeed
        self.speed = 0
        self.current_hs = 0
        # used to check laptime
        if self.carla_runner.world is not None:
            self.last_sim_time = self.carla_runner.world.hud.simulation_time
        else:
            self.last_sim_time = 0
        self.sim_lap_time = 0

        self.deadzone_trigger = True
        self.deadzone_level = 0.001

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        obs = []
        rewards = []
        self.steps+=1
        for i in range(1):
            # throttle=(action[i*3]+0.5)/2+1
            check = (action[i*3+0]+0.5)/2+1
            if check > 0.5:
                throttle = 1
                braking = 0
            else:
                throttle = 0
                braking = .8


            steering = action[i*3+1]/5

            if self.deadzone_trigger and abs(steering) < self.deadzone_level:
                steering = 0


            self.agent.kwargs["control"] = VehicleControl(throttle=throttle,
                                                          steering=steering,
                                                          braking=braking)
            ob, reward, is_done, info = super(ROARppoEnvE2E, self).step(action)
            obs.append(ob)
            rewards.append(reward)
            if is_done:
                break
        self.render()
        self.frame_reward = sum(rewards)
        self.ep_rewards += sum(rewards)

        self.speed = self.agent.vehicle.get_speed(self.agent.vehicle)
        if self.speed > self.current_hs:
            self.current_hs = self.speed

        if is_done:
            self.wandb_logger()
            self.crash_check = False
            self.update_highscore()
        return np.array(obs), self.frame_reward, self._terminal(), self._get_info()

    def _get_info(self) -> dict:
        info_dict = OrderedDict()
        info_dict["Current HIGHSCORE"] = self.highscore
        info_dict["Furthest Checkpoint"] = self.highest_chkpt*self.agent.interval
        info_dict["episode reward"] = self.ep_rewards
        info_dict["checkpoints"] = self.agent.int_counter*self.agent.interval
        info_dict["reward"] = self.frame_reward
        info_dict["largest_steps"] = self.largest_steps
        info_dict["current_hs"] = self.current_hs
        info_dict["highest_speed"] = self.highspeed
        info_dict["complete_state"]=self.complete_loop
        info_dict["avg10_checkpoints"]=np.average(self.his_checkpoint)
        info_dict["avg10_score"]=np.average(self.his_score)
        # info_dict["throttle"] = action[0]
        # info_dict["steering"] = action[1]
        # info_dict["braking"] = action[2]
        return info_dict

    def update_highscore(self):
        if self.ep_rewards > self.highscore:
            self.highscore = self.ep_rewards
        if self.agent.int_counter > self.highest_chkpt:
            self.highest_chkpt = self.agent.int_counter
        if self.current_hs > self.highspeed:
            self.highspeed = self.current_hs
        self.current_hs = 0

        if self.carla_runner.world is not None:
            current_time = self.carla_runner.world.hud.simulation_time
            if self.agent.int_counter * self.agent.interval < 5175:
                self.sim_lap_time = 400
            else:
                self.sim_lap_time = current_time - self.last_sim_time
            self.last_sim_time = current_time
        else:
            self.sim_lap_time = 0
            self.last_sim_time = 0
        return

    def _terminal(self) -> bool:
        if self.stopped_counter >= self.stopped_max_count:
            return True
        if not (self.agent.bbox_list[(self.agent.int_counter - self.death_line_dis) % len(self.agent.bbox_list)].has_crossed(self.agent.vehicle.transform))[0]:
            return True
        if self.carla_runner.get_num_collision() > self.max_collision_allowed:
            return True
        elif self.agent.finish_loop:
            self.complete_loop=True
            return True
        else:
            return False

    def get_reward(self) -> float:
        # prep for reward computation
        # reward = -0.1*(1-self.agent.vehicle.control.throttle+10*self.agent.vehicle.control.braking+abs(self.agent.vehicle.control.steering))*400/8
        reward=-1

        if self.crash_check:
            print("no reward")
            return 0


        if self.agent.cross_reward > self.prev_cross_reward:
            reward += (self.agent.cross_reward - self.prev_cross_reward)*self.agent.interval*self.time_to_waypoint_ratio



        if not (self.agent.bbox_list[(self.agent.int_counter - self.death_line_dis) % len(self.agent.bbox_list)].has_crossed(self.agent.vehicle.transform))[0]:
            reward -= 200
            self.crash_check = True

        if self.agent.int_counter > 5 and self.agent.vehicle.get_speed(self.agent.vehicle) < 1:
            self.stopped_counter += 1
            if self.stopped_counter >= self.stopped_max_count:
                reward -= 200
                self.crash_check = True

        if self.carla_runner.get_num_collision() > 0:
            reward -= 200
            self.crash_check = True


        # log prev info for next reward computation
        self.prev_speed = Vehicle.get_speed(self.agent.vehicle)
        self.prev_cross_reward = self.agent.cross_reward
        return reward

    def _get_obs(self) -> np.ndarray:
        if mode=='baseline':

            index_from=(self.agent.int_counter%len(self.agent.bbox_list))
            if index_from+10<=len(self.agent.bbox_list):
                # print(index_from,len(self.agent.bbox_list),index_from+10-len(self.agent.bbox_list))
                next_bbox_list=self.agent.bbox_list[index_from:index_from+10]
            else:
                # print(index_from,len(self.agent.bbox_list),index_from+10-len(self.agent.bbox_list))
                next_bbox_list=self.agent.bbox_list[index_from:]+self.agent.bbox_list[:index_from+10-len(self.agent.bbox_list)]
            assert(len(next_bbox_list)==10)
            map_list = self.agent.occupancy_map.get_map_baseline(transform_list=self.agent.vt_queue,
                                                    view_size=(CONFIG["x_res"], CONFIG["y_res"]),
                                                    bbox_list=self.agent.frame_queue,
                                                                 next_bbox_list=next_bbox_list
                                                    )


            cv2.imshow("data", np.hstack(np.hstack(map_list))) # uncomment to show occu map
            cv2.waitKey(1)

            return map_list[:,:-1]

        else:
            data = self.agent.occupancy_map.get_map(transform=self.agent.vehicle.transform,
                                                    view_size=(CONFIG["x_res"], CONFIG["y_res"]),
                                                    arbitrary_locations=self.agent.bbox.get_visualize_locs(),
                                                    arbitrary_point_value=self.agent.bbox.get_value(),
                                                    vehicle_velocity=self.agent.vehicle.velocity,
                                                    # rotate=self.agent.bbox.get_yaw()
                                                    )
            # data = cv2.resize(occu_map, (CONFIG["x_res"], CONFIG["y_res"]), interpolation=cv2.INTER_AREA)
            #cv2.imshow("Occupancy Grid Map", cv2.resize(np.float32(data), dsize=(500, 500)))

            # data_view=np.sum(data,axis=2)
            cv2.imshow("data", data) # uncomment to show occu map
            cv2.waitKey(1)
            # yaw_angle=self.agent.vehicle.transform.rotation.yaw
            # velocity=self.agent.vehicle.get_speed(self.agent.vehicle)
            # data[0,0,2]=velocity
            data_input=data.copy()
            data_input[data_input==1]=-10
            return data_input  # height x width x 3 array
    #3location 3 rotation 3velocity 20 waypoline locations 20 wayline rewards

    def reset(self) -> Any:
        if len(self.his_checkpoint)>=10:
            self.his_checkpoint=self.his_checkpoint[-10:]
            self.his_score=self.his_score[-10:]
        if self.agent:
            self.his_checkpoint.append(self.agent.int_counter*self.agent.interval)
            self.his_score.append(self.ep_rewards)
        self.ep_rewards = 0
        self.stopped_counter = 0
        if self.steps>self.largest_steps and not self.complete_loop:
            self.largest_steps=self.steps
        elif self.complete_loop and self.agent.finish_loop and self.steps<self.largest_steps:
            self.largest_steps=self.steps
        super(ROARppoEnvE2E, self).reset()
        self.steps=0
        # self.crash_step=0
        # self.reward_step=0
        return self._get_obs()

    def wandb_logger(self):
        wandb.log({
            "Episode reward": self.ep_rewards,
            "Checkpoint reached": self.agent.int_counter*self.agent.interval,
            "largest_steps" : self.largest_steps,
            "highest_speed" : self.highspeed,
            "Episode_Sim_Time": self.sim_lap_time,
            "episode Highspeed": self.current_hs,
            "avg10_checkpoints":np.average(self.his_checkpoint),
            "avg10_score":np.average(self.his_score),
        })
        return
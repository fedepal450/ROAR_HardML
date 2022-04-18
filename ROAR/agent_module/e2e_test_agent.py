from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.control_module.pid_controller import PIDController#TO REMOVE
from ROAR.planning_module.local_planner.loop_simple_waypoint_following_local_planner import \
    LoopSimpleWaypointFollowingLocalPlanner#TO REMOVE
from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner#TO REMOVE
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import WaypointFollowingMissionPlanner
from pathlib import Path
from ROAR.utilities_module.occupancy_map import OccupancyGridMap
from ROAR.perception_module.obstacle_from_depth import ObstacleFromDepth
from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import ViveTrackerData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
import numpy as np
from ROAR.utilities_module.data_structures_models import Transform, Location
import cv2
from typing import Optional
import scipy.stats
from collections import deque

class E2ETestAgent(Agent):
    def __init__(self, vehicle: Vehicle, agent_settings: AgentConfig, **kwargs):
        super.__init__()
        self.model_path = ""
        self.model = None# load your model from model_path

        self.vehicle = vehicle
        self.occupancy_map = None # load occuapncy map

        # frame skipping


    def step(self):
        # TODO: get current observation
        obs = None
        action, next_obs = self.model.predict(obs)
        pass
        return VehicleControl(action[0])


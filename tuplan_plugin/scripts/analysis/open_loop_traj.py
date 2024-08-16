import itertools
from typing import List, Optional

import os
from tqdm import tqdm
import msgpack
import lzma
import pickle
import json

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics
from nuplan.planning.metrics.utils.state_extractors import extract_ego_center_with_heading, extract_ego_time_point
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

def get_all_msgpack(dir):
    file_list = []
    for root_dir, _, files in os.walk(dir):
        for file in files:
            if file.endswith('.msgpack.xz'):
                file_name = os.path.join(root_dir, file)
                file_list.append(file_name)
    return file_list

def statese2point(traj):
    return [[x.array[0], x.array[1], x.heading] for x in traj]

class PlannerExpertAverageL2ErrorStatistics(MetricBase):
    """Average displacement error metric between the planned ego pose and expert."""

    def __init__(
        self,
        comparison_horizon: List[int],
        comparison_frequency: int,
    ) -> None:
        """
        Initialize the PlannerExpertL2ErrorStatistics class.
        :param comparison_horizon: List of horizon times in future (s) to find displacement errors.
        :param comparison_frequency: Frequency to sample expert and planner trajectory.
        """
        self.comparison_horizon = comparison_horizon
        self._comparison_frequency = comparison_frequency
        # Store the errors to re-use in high level metrics
        self.expert_timestamps_sampled: List[int] = []

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Return the estimated metric.
        :param history: History from a simulation engine.
        :param scenario: Scenario running this metric.
        :return the estimated metric.
        """
        scene_info ={
            "scenario_type": scenario.scenario_type,
            "scenario_token": scenario.token,
            "simulation_his": []
        }

        # Find the frequency at which expert trajectory is sampled and the step size for down-sampling it
        expert_frequency = 1 / scenario.database_interval
        step_size = int(expert_frequency / self._comparison_frequency)
        sampled_indices = list(range(0, len(history.data), step_size))
        # Sample the expert trajectory up to the maximum future horizon needed to compute errors
        expert_states = list(
            itertools.chain(
                list(scenario.get_expert_ego_trajectory())[0::step_size],
                scenario.get_ego_future_trajectory(
                    sampled_indices[-1],
                    max(self.comparison_horizon)+5,
                    max(self.comparison_horizon)+5 // self._comparison_frequency,
                ),
            )
        )
        expert_traj_poses = extract_ego_center_with_heading(expert_states)
        expert_timestamps_sampled = extract_ego_time_point(expert_states)

        # Extract planner proposed trajectory at each sampled frame
        planned_trajectories = list(history.data[index].trajectory for index in sampled_indices)

        # Find displacement error between the proposed planner trajectory and expert driven trajectory for all sampled frames during the scenario
        for curr_frame, curr_ego_planned_traj in enumerate(planned_trajectories):
            future_horizon_frame = int(curr_frame + max(self.comparison_horizon))
            # Interpolate planner proposed trajectory at the same timepoints where expert states are available
            planner_interpolated_traj = list(
                curr_ego_planned_traj.get_state_at_time(TimePoint(int(timestamp)))
                for timestamp in expert_timestamps_sampled[curr_frame : future_horizon_frame + 1]
                if timestamp <= curr_ego_planned_traj.end_time.time_us
            )
            if len(planner_interpolated_traj) < max(self.comparison_horizon) + 1:
                planner_interpolated_traj = list(
                    itertools.chain(planner_interpolated_traj, [curr_ego_planned_traj.get_sampled_trajectory()[-1]])
                )
                # If planner duration is slightly less than the required horizon due to down-sampling, find expert states at the final timepoint of the planner trajectory for the comparison.
                expert_traj = expert_traj_poses[curr_frame + 1 : future_horizon_frame] + [
                    InterpolatedTrajectory(expert_states).get_state_at_time(curr_ego_planned_traj.end_time).center
                ]
            else:
                expert_traj = expert_traj_poses[curr_frame + 1 : future_horizon_frame + 1]

            planner_interpolated_traj_poses = extract_ego_center_with_heading(planner_interpolated_traj)

            frame_info = {
                "ego_centor": history.extract_ego_state[curr_frame].center.array.tolist(),
                "ego_heading": history.extract_ego_state[curr_frame].center.heading,
                "ego_acceleration": history.extract_ego_state[curr_frame]._dynamic_car_state.center_acceleration_2d.array.tolist(),
                "ego_velocity": history.extract_ego_state[curr_frame]._dynamic_car_state.center_velocity_2d.array.tolist(),
                "pred_traj": statese2point(planner_interpolated_traj_poses[1:]),
                "gt_traj": statese2point(expert_traj)
            }

            scene_info["simulation_his"].append(frame_info)
        return scene_info

def ana_msgpack(msgpack_dir):

    with lzma.open(msgpack_dir, "rb") as f:
        data = msgpack.unpackb(f.read())
    data = pickle.loads(data)
    traj_class = PlannerExpertAverageL2ErrorStatistics(comparison_frequency=2, comparison_horizon=[2,4,6])
    scene_info = traj_class.compute(data.simulation_history, data.scenario)
    return scene_info

def excute(input_dir, output_dir):
    dir_list = get_all_msgpack(input_dir)
    info = {}
    for msgpack_dir in tqdm(dir_list):
        scenerio_token = msgpack_dir.split("/")[-1].split(".")[0]
        scenerio_info = ana_msgpack(msgpack_dir)
        info.update({scenerio_token: scenerio_info})
    json_data = json.dumps(info,indent=4)
    with open(output_dir, 'w') as json_file:
        json_file.write(json_data)

# PDMOpen
input_dir = "/data/haoruiyang_pro/nuplan/exp/exp/simulation/open_loop_boxes/pdm_open/simulation_log/PDMOpenPlanner"
output_dir = "/data/haoruiyang_pro/tuplan_garage/scripts/analysis/traj_pdm_open.json"
excute(input_dir, output_dir)

# PDM-Off
input_dir = "/data/haoruiyang_pro/nuplan/exp/exp/simulation/open_loop_boxes/pdm_corrected_states/simulation_log/PDMHybridPlanner"
output_dir = "/data/haoruiyang_pro/tuplan_garage/scripts/analysis/traj_pdm_corrected.json"
excute(input_dir, output_dir)

# PDM-projection
input_dir = "/data/haoruiyang_pro/nuplan/exp/exp/simulation/open_loop_boxes/pdm_corrected_states/simulation_log/PDMHybridPlanner"
output_dir = "/data/haoruiyang_pro/tuplan_garage/scripts/analysis/traj_pdm_projection.json"
excute(input_dir, output_dir)
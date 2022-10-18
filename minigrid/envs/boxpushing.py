from typing import Optional

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Box, ColoredGoal
from minigrid.minigrid_env import MiniGridEnv

import numpy as np

class BoxPushingEnv(MiniGridEnv):
    """
    ### Description

    This environment is an empty room, and the goal of the agent is to push 
    the box to reach every goal square, which provides a sparse reward. 
    A small penalty is subtracted for the number of steps to reach every goal.
    And there will be a large bonus once all boxes reach their goal squares.
    If the game used colored goals, then different colored boxes must reach the same colored goal,
    making it much harder to solve.

    ### Mission Space

    "push the box(es) to every goal square"

    ### Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ### Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ### Rewards

    A reward of '1' is given for success if , and '0' for failure.

    ### Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ### Registered Configurations

    - `MiniGrid-BoxPushing-8x8-v0`
    - `MiniGrid-BoxPushing-12x12-v0`
    - `MiniGrid-BoxPushing-16x16-v0`
    - `MiniGrid-ColoredBoxPushing-8x8-v0`
    - `MiniGrid-ColoredBoxPushing-12x12-v0`
    - `MiniGrid-ColoredBoxPushing-16x16-v0`

    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(2, 2),
        agent_start_dir=0,
        max_steps: Optional[int] = None,
        required_boxes_num=4, 
        colored=False,
        **kwargs
    ):
        self.agent_start_pos = agent_start_pos # Not used
        self.agent_start_dir = agent_start_dir
        self.colored = colored
        self.color_box = ["green"] * 4
        if self.colored:
            self.color_box = ["yellow", "green", "blue", "red"]

        self.required_boxes_num = required_boxes_num
        self.success_boxes_num = 0
        # Refresh after pushing every box into the goal
        self.every_box_step_count = 0

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs
        )

        from gymnasium import spaces
        # Only use 3 actions
        self.action_space = spaces.Discrete(3)
        image_observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 3),
            dtype="uint8",
        )
        self.observation_space = spaces.Dict(
            {
                "image": image_observation_space,
                "direction": spaces.Discrete(4),
                "mission": mission_space,
            }
        )
        print(self.observation_space)

    @staticmethod
    def _gen_mission():
        return "pushing the box(es) into goals"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in four corners
        self.put_obj(ColoredGoal(self.color_box[0]), width - 2, height - 2)
        self.put_obj(ColoredGoal(self.color_box[1]), 1, 1)
        self.put_obj(ColoredGoal(self.color_box[2]), width - 2, 1)
        self.put_obj(ColoredGoal(self.color_box[3]), 1, height - 2)

        box_pos_idx = np.random.choice((width - 4)*(height - 4), 4, replace=False)
        box_pos = []
        for i in range(4):
            box_pos.append([box_pos_idx[i] // (width - 4) + 2, box_pos_idx[i] % (width - 4) + 2])

        self.put_obj(Box(self.color_box[0]), box_pos[0][0], box_pos[0][1])
        self.put_obj(Box(self.color_box[1]), box_pos[1][0], box_pos[1][1])
        self.put_obj(Box(self.color_box[2]), box_pos[2][0], box_pos[2][1])
        self.put_obj(Box(self.color_box[3]), box_pos[3][0], box_pos[3][1])

        # Place the agent
        self.place_agent()

        self.mission = "push the box(es) to every goal square" 

    def gen_obs(self):
        """
        Generate the agent's view (fully observable, low-resolution encoding)
        """
         
        # Get the fully observable states
        grid = self.grid

        # Encode the fully observable view into a numpy array
        image = grid.encode(None)

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {"image": image, "direction": self.agent_dir, "mission": self.mission}

        return obs

    def reset(self, *args, **kwargs):
        self.success_boxes_num = 0
        return super().reset(*args, **kwargs)

    def is_success(self):
        return self.success_boxes_num == self.required_boxes_num

    def _reward(self):
        """
        Compute the reward to be given upon success
        """

        return max(0, 1 - 0.9 * (self.every_box_step_count / self.max_steps))

    def step(self, action):
        self.step_count += 1
        self.every_box_step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "box":
                new_box_obs = fwd_pos + self.dir_vec
                new_fwd_cell = self.grid.get(*new_box_obs)
                if new_fwd_cell is None or new_fwd_cell.can_overlap():
                    # Set the new position of the box
                    self.grid.set(new_box_obs[0], new_box_obs[1], fwd_cell)
                    self.grid.set(fwd_pos[0], fwd_pos[1], None)
                    # Move agent if box can move
                    self.agent_pos = tuple(fwd_pos)
                if new_fwd_cell is not None and new_fwd_cell.type == "goal" and new_fwd_cell.color == fwd_cell.color:
                    self.success_boxes_num += 1
                    reward = self._reward()
                    self.every_box_step_count = 0
                
        else:
            raise ValueError(f"Unknown action: {action}")
        
        if self.is_success():
            terminated = True
            reward = 10

        if self.step_count >= self.max_steps * self.required_boxes_num:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}
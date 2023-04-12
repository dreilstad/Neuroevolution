from simulation.environments.tartarus.minigrid.minigrid_env import MiniGridEnv
from simulation.environments.tartarus.minigrid.core.mission import MissionSpace
from simulation.environments.tartarus.minigrid.core.grid import Grid
from simulation.environments.tartarus.minigrid.core.world_object import Box

from random import choice
import numpy as np


class TartarusEnvironment(MiniGridEnv):
    """
    Tartarus environment
    """

    def __init__(self, configuration, N=6, K=6, moves=80, fixed_agent_pos=None, fixed_agent_dir=None):
        self.configuration = configuration
        self.fixed_agent_pos = fixed_agent_pos
        self.fixed_agent_dir = fixed_agent_dir
        self.N = N
        self.K = K
        super().__init__(
            mission_space=MissionSpace(mission_func=self._gen_mission),
            grid_size=None,
            width=self.N+2,
            height=self.N+2,
            max_steps=moves,
            render_mode="rgb_array"
        )

        self.blocks = None

    def to_str(self):
        AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

        tartarus_state = self.encode_tartarus_state()
        tartarus_state = np.char.mod('%d', tartarus_state)

        tartarus_state[self.agent_pos[1]-1][self.agent_pos[0]-1] = AGENT_DIR_TO_STR[self.agent_dir]
        return tartarus_state

    def _gen_grid(self, width, height):
        # generate an empty grid w*h
        self.grid = Grid(width, height)

        # generate the walls around
        self.grid.wall_rect(0, 0, width, height)

        self.blocks = []
        # place boxes
        boxes = []
        for i in range(1, self.N-1):
            for j in range(1, self.N-1):
                if self.configuration[i][j] == 1:
                    box = Box("brown")
                    self.put_obj(obj=box, i=j+1, j=i+1)

                    boxes.append((i, j))
                    self.blocks.append(box)

        # if not using fixed agent, use random position and direction
        if self.fixed_agent_pos is None:
            # get random position of agent in environment
            indices = np.arange(1, self.N-1)
            possible_positions = [(x, y) for x in indices for y in indices if (x, y) not in boxes]
            agent_pos = choice(possible_positions)

            # place agent with random position with random direction
            self.place_agent(
                top=(agent_pos[1] + 1, agent_pos[0] + 1),
                size=(1, 1)
            )
        else:
            self.place_agent(
                top=(self.fixed_agent_pos[1] + 1, self.fixed_agent_pos[0] + 1),
                size=(1, 1),
                rand_dir=False
            )
            self.agent_dir = self.fixed_agent_dir

        return

    @staticmethod
    def _gen_mission():
        return "Solve the Tartarus Problem"

    def gen_obs(self):
        grid = self.encode_tartarus_state_with_walls()
        y, x = self.agent_pos
        obs = grid[x-1:x+2, y-1:y+2]

        # Facing right
        if self.agent_dir == 0:
            obs = np.rot90(obs, k=1)
        # Facing down
        elif self.agent_dir == 1:
            obs = np.rot90(obs, k=2)
        # Facing left
        elif self.agent_dir == 2:
            obs = np.rot90(obs, k=3)

        # transform to 1d array and remove square with agent position (index 4)
        obs = np.delete(obs.ravel(), 4)
        return obs

    def state_evaluation(self):
        """
        Compute state evaluation
        """
        performance_score = 0.0

        # check corners
        corners = [(1,1), (1,self.N), (self.N, 1), (self.N, self.N)]
        for corner in corners:
            cell = self.grid.get(*corner)
            if cell is not None and cell.type == "box":
                performance_score += 2.0

        # check top and bottom borders
        for j in range(2, self.N):
            top_cell = self.grid.get(1, j)
            bottom_cell = self.grid.get(self.N, j)

            if top_cell is not None and top_cell.type == "box":
                performance_score += 1.0
            if bottom_cell is not None and bottom_cell.type == "box":
                performance_score += 1.0

        # check left and right borders
        for i in range(2, self.N):
            left_cell = self.grid.get(i, 1)
            right_cell = self.grid.get(i, self.N)

            if left_cell is not None and left_cell.type == "box":
                performance_score += 1.0
            if right_cell is not None and right_cell.type == "box":
                performance_score += 1.0

        return performance_score

    def encode_tartarus_state(self):
        tartarus_state = np.zeros((self.N, self.N))

        for i in range(1, self.width-1):
            for j in range(1, self.height-1):
                cell = self.grid.get(i, j)
                if cell is not None and cell.type == "box":
                    tartarus_state[j-1][i-1] = 1

        return tartarus_state

    def encode_tartarus_state_with_walls(self):
        pos_of_blocks = [b.cur_pos for b in self.blocks]

        tartarus_state = np.zeros((self.height, self.width))

        # set blocks
        for pos in pos_of_blocks:
            tartarus_state[pos[1]][pos[0]] = 1

        # set walls at edges
        tartarus_state[0,:] = -1
        tartarus_state[-1,:] = -1
        tartarus_state[:,0] = -1
        tartarus_state[:,-1] = -1

        return tartarus_state

    def get_final_block_positions_vector(self):
        standard_pos_coords = [(block.cur_pos[1]-1, block.cur_pos[0]-1) for block in self.blocks]
        unraveled_pos_coords = [index for pos in standard_pos_coords for index in pos]

        return unraveled_pos_coords


class DeceptiveTartarusEnvironment(TartarusEnvironment):

    def __init__(self, configuration, N=6, K=6, moves=80, fixed_agent_pos=None, fixed_agent_dir=None):
        super().__init__(
            configuration=configuration,
            N=N,
            K=K,
            moves=moves,
            fixed_agent_pos=fixed_agent_pos,
            fixed_agent_dir=fixed_agent_dir
        )

    def state_evaluation(self):
        """
        Compute state evaluation
        """
        performance_score = 0.0

        # check corners
        corners = [(1,1), (1,self.N), (self.N, 1), (self.N, self.N)]
        for corner in corners:
            cell = self.grid.get(*corner)
            if cell is not None and cell.type == "box":
                performance_score += 2.0

        # check top and bottom borders
        for j in range(2, self.N):
            top_cell = self.grid.get(1, j)
            bottom_cell = self.grid.get(self.N, j)

            if top_cell is not None and top_cell.type == "box":
                performance_score -= 1.0
            if bottom_cell is not None and bottom_cell.type == "box":
                performance_score -= 1.0

        # check left and right borders
        for i in range(2, self.N):
            left_cell = self.grid.get(i, 1)
            right_cell = self.grid.get(i, self.N)

            if left_cell is not None and left_cell.type == "box":
                performance_score -= 1.0
            if right_cell is not None and right_cell.type == "box":
                performance_score -= 1.0

        return performance_score

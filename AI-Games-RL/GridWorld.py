import numpy as np
import EnvironmentModel

class GridWorldState():

    def __init__(self, x, y, stateNumber):
        self.stateNumber = stateNumber


class GridWorld(EnvironmentModel.Environment):
    emoji_display = True

    def __init__(self, grid, max_steps, robotPos):
        self.gridWorldStates = dict()
        self.state = robotPos
        numberOfStates = 1
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                if grid[x][y] != 1:
                    self.gridWorldStates[(x, y)] = numberOfStates - 1
                    numberOfStates += 1
        EnvironmentModel.Environment.__init__(self, numberOfStates, 3, max_steps, None, None)
        self.grid = grid

    def p(self, next_state, state, action):

        if state == self.n_states - 1:
            if next_state == self.n_states - 1:
                return 1
            else:
                return 0
        loc = list(self.gridWorldStates.keys())[list(self.gridWorldStates.values()).index(state)]
        x = loc[0]
        y = loc[1]

        if self.grid[x][y] == 2 or self.grid[x][y] == 3:
            if next_state == self.n_states - 1:
                return 1
            else:
                return 0

        if action == 0:
            x -= 1
        elif action == 1:
            y += 1
        elif action == 2:
            x += 1
        else:
            y -= 1

        if x < 0 or x >= self.grid.shape[0]:
            if next_state == state:
                return 1
            else:
                return 0
        if y < 0 or y >= self.grid.shape[1]:
            if next_state == state:
                return 1
            else:
                return 0

        if self.grid[x][y] != 1:
            if self.gridWorldStates[(x, y)] == next_state:
                return 1
        else:
            if next_state == state:
                return 1
        return 0

    def r(self, next_state, state, action):
        if state == self.n_states - 1:
            return 0

        loc = list(self.gridWorldStates.keys())[list(self.gridWorldStates.values()).index(state)]
        x = loc[0]
        y = loc[1]
        if self.grid[x][y] == 2:
            return -1
        if self.grid[x][y] == 3:
            return 1
        return 0

    def render(self):
        if self.state == self.n_states - 1:
            print("Game Finished - Score = " + str(self.totalReward))
            return
        out = np.copy(self.grid)
        loc = list(self.gridWorldStates.keys())[list(self.gridWorldStates.values()).index(self.state)]
        x = loc[0]
        y = loc[1]
        out[x][y] = 4
        print("---------------------")
        if self.emoji_display:
            for row in range(out.shape[0]):
                rr = ""
                for c in out[row]:
                    rr += tile_draw[c]
                print(rr)
        else:
            print(out)
        print("---------------------")

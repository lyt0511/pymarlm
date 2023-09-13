# A Star Algorithm

class Map_grid:
    def __init__(self, max_x, max_y, passable_grid=None):  
        self.grid = []
        self.row = max_x
        self.col = max_y
        # Todo: replace passable_grid to pathing_grid in starcraft2.py
        self.passable_grid = passable_grid
        self.path_finder = Pathfinder(self.row, self.col)
        self.init_map()
        # self.passable_setting()

    def init_map(self):
        for row in range(self.row):
            # Add an empty array that will hold each node
            # in this row
            self.grid.append([])
            for column in range(self.col):
                node = Node()
                node.x = row
                node.y = column
                # node.color = "white"
                node.name = "My name is: ", node.x, node.y
                self.grid[row].append(node)  # Append a cell
        self.path_finder.update_neighbors(self.grid)

    def passable_setting(self):
        # randomly set a percentage of the tiles to be walls
        for row in range(self.row):
            for column in range(self.col):
                current = self.grid[row][column]
                if self.passable_grid != None:
                    if self.passable_grid[row][column] == False:
                        current.color = 'grey'
                        current.passable = False
        # set all the borders to walls, make them grey and make them not passable and mark the borders are border
        for row in range(self.row):
            self.grid[row][0].passable = False
            self.grid[row][self.col - 1].passaale = False
            self.grid[row][0].color = "grey"
            self.grid[row][self.col - 1].color = "grey"
            self.grid[row][0].border = True
            self.grid[row][self.col - 1].border = True
        for col in range(self.col):
            self.grid[0][col].passable = False
            self.grid[self.row - 1][col].passable = False
            self.grid[0][col].color = "grey"
            self.grid[self.row - 1][col].color = "grey"
            self.grid[0][col].border = True
            self.grid[self.row - 1][col].border = True

        # for iteration in range(ITERATIONS):
        #     for row in range(ROWS):
        #         for column in range(COLUMNS):
        #             if grid[row][column].border == False:
        #                 grey_color_count = 0
        #                 for node in grid[row][column].neighbor:
        #                     if node.color == 'grey':
        #                         grey_color_count = grey_color_count + 1
        #                 if grey_color_count >= 5:
        #                     grid[row][column].color = 'grey'
        #                     grid[row][column].passable = False
        #                 elif grey_color_count == 2:
        #                     grid[row][column].color = 'grey'
        #                     grid[row][column].passable = False
        #                 else:
        #                     grid[row][column].color = 'white'
        #                     grid[row][column].passable = True
        # #Iteration that tries to remove diagonal walls
        
        # for iteration in range(ITERATIONS):
        #     for row in range(ROWS):
        #         for column in range(COLUMNS):
        #             if grid[row][column].border == False:
        #                 grey_color_count = 0
        #                 for node in grid[row][column].neighbor:
        #                     if node.color == 'grey':
        #                         grey_color_count = grey_color_count + 1
        #                 if grey_color_count >= 5:
        #                     grid[row][column].color = 'grey'
        #                     grid[row][column].passable = False

class Node:
    def __init__(self, passable=True, fscore=0, gscore=0, hscore=0, x=0, y=0, name=None, parent=None, neighbor=None, border=False, color='white'):
        # 合法
        self.passable = passable
        self.fscore = fscore
        self.gscore = gscore
        self.hscore = hscore
        self.x = x
        self.y = y
        self.name = name
        self.parent = parent
        self.border = False
        self.color = color
        if neighbor is None:
            self.neighbor = []
        else:
            self.neighbor = [neighbor]

    def clear_node():
        fscore = 0
        gscore = 0
        hscore = 0
        parent = None


class Pathfinder:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

        self.openlist = {}
        self.closedlist = []
        self.path = []

    def calculate_h_score(self, start_x, start_y, end_x, end_y):
        x = abs(start_x - end_x)
        y = abs(start_y - end_y)
        answer = (x + y) * 10
        return answer

    def reset_color_and_path(self, grid):
        self.openlist.clear()
        self.closedlist = []
        self.path = []
        for row in range(self.rows):
            for column in range(self.cols):
                grid[row][column].clear_node
                # if grid[row][column].color == 'grey':
                #     grid[row][column].passable = False

    def find_path(self, grid, start_x, start_y, end_x, end_y):
        start_point = grid[start_x][start_y]
        end_point = grid[end_x][end_y]

        self.openlist[start_point] = 0

        while len(self.openlist) > 0:
            current = min(self.openlist, key=self.openlist.get)
            if current == end_point:
                while current.parent:
                    self.path.append(current)
                    current = current.parent
                self.path.append(current)
                actions = self.path_to_action()
                self.reset_color_and_path(grid)
                return actions
            self.closedlist.append(current)
            self.openlist.pop(current)
            for neighbor in current.neighbor:
                if neighbor in self.closedlist or neighbor.passable == False:
                    # print("neighbor", neighbor.name, "is on closed list or is not passable")
                    next
                else:
                    # if the neighbor is not on the openlist, meaning that this is a completly new unvisited and unscored node then do this
                    if (neighbor in self.openlist) == False:
                        # print("neighbor", neighbor.name, "is not on open list, adding it ")
                        # set the current node that we expecting  to the neighbors parent, so we can trace the path back if this is the optimal path
                        neighbor.parent = current
                        # set the gscore to the parents score + 10 if its a not a diagonal neighbor
                        neighbor.gscore = neighbor.parent.gscore + 10
                        # set the gscore to the parents score + 14 if its a diagonal neighbor
                        if neighbor.x != current.x and neighbor.y != current.y:
                            neighbor.gscore = neighbor.parent.gscore + 14
                        # calculate h score and set it
                        neighbor.hscore = self.calculate_h_score(
                            neighbor.x, neighbor.y, end_point.x, end_point.y)
                        # calculate f score and set it
                        neighbor.fscore = neighbor.hscore + neighbor.gscore
                        # add the neighbor to the openlist with the calculated f score
                        self.openlist[neighbor] = neighbor.fscore
                    # if the neighbor is on the openlist we need to recalculate the fscore again
                    elif neighbor in self.openlist:
                        # print("neighbor", neighbor.name, "is on the open list")
                        tempG = neighbor.parent.gscore + 10
                        neighbor.gscore = neighbor.parent.gscore + 10
                        if neighbor.x != current.x and neighbor.y != current.y:
                            neighbor.gscore = neighbor.parent.gscore + 14
                        if tempG < neighbor.gscore:
                            neighbor.gscore = tempG
                            neighbor.hscore = self.calculate_h_score(
                                neighbor.x, neighbor.y, end_point.x, end_point.y)
                            neighbor.fscore = neighbor.hscore + neighbor.gscore
                            self.openlist[neighbor] = neighbor.fscore

    # method that updates the nodes with the information of their neighbors
    def update_neighbors(self,grid):
        for row in range(self.rows):
            for column in range(self.cols):
                # 东、南、西、北邻居
                if row > 0:
                    grid[row][column].neighbor.append(grid[row - 1][column])
                if row < self.rows - 1:
                    grid[row][column].neighbor.append(grid[row + 1][column])
                if column > 0:
                    grid[row][column].neighbor.append(grid[row][column - 1])
                if column < self.cols - 1:
                    grid[row][column].neighbor.append(grid[row][column + 1])
                
                # 东南、东北、西南、西北邻居
                # if row > 0 and column > 0:
                #     grid[row][column].neighbor.append(grid[row - 1][column - 1])
                # if row > 0 and column < self.cols - 1:
                #     grid[row][column].neighbor.append(grid[row - 1][column + 1])
                # if row < self.rows - 1 and column > 0:
                #     grid[row][column].neighbor.append(grid[row + 1][column - 1])
                # if row < self.rows - 1 and column < self.cols - 1:
                #     grid[row][column].neighbor.append(grid[row + 1][column + 1])

    def path_to_action(self):
        path = self.path[::-1]
        pos = [(node.x, node.y) for node in path]
        action_queue = []
        prev = path[0]
        for cur in path[1:]:
            prev_x = prev.x
            prev_y = prev.y
            cur_x = cur.x
            cur_y = cur.y

            if cur_y - prev_y > 0:
                action_queue.append(2)
                continue
            elif cur_y - prev_y < 0:
                action_queue.append(3)
                continue
            if cur_x - prev_x > 0:
                action_queue.append(4)
                continue
            elif cur_x - prev_x < 0:
                action_queue.append(5)
                continue

        return action_queue

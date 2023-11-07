"""

Grid based Dijkstra planning

author: Atsushi Sakai(@Atsushi_twi)

"""

# import matplotlib.pyplot as plt
# import math

# show_animation = True

import math

import matplotlib.pyplot as plt

import sys
sys.path.append('./OccupancyAStar/examples/')
#from algorithms import plot_path, png_to_ogm, OccupancyGridMap
import utils
# from AStar import AStarPlanner

import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import json
import tracemalloc
import time
from collections import deque

show_animation = True

pause_time = 0.1

##part1
xwidth = 8

tree_result_list = list()

##find path from start to goal in tree
def path_from_to(root, target):
    queue = deque([(root, [root.data])])
    while (len(queue) > 0):
        node, path = queue.popleft()
        if node.data == target:
            return path
        
        ##compare cost of left and right, choose lesser 
        if node.left and node.right:
            if node.left.cost < node.right.cost :
                queue.append((node.left, path + [node.left.data]))
            else:
                queue.append((node.right, path + [node.right.data]))
                
        if node.left :
            queue.append((node.left, path + [node.left.data]))
        if node.right :
            queue.append((node.right, path + [node.right.data]))


##find parent of current coordinates(index) and add it to right of left
def findParentInsertChild(rootNode,currentIndex,parentIndex , cost):
    if rootNode is None:
        return
    findParentInsertChild(rootNode.left,currentIndex,parentIndex ,cost)
    
    if rootNode.data == parentIndex:
        if rootNode.left is None:
            rootNode.left = TreeNode(currentIndex , cost)
        else:
            rootNode.right = TreeNode(currentIndex , cost)
    
    findParentInsertChild(rootNode.right,currentIndex,parentIndex , cost)  
    
    
## declare TreeNode, add cost to store cost
class TreeNode:
    def __init__(self, data,cost):
        self.left = None
        self.right = None
        self.data = data
        self.cost = cost
        

    def __str__(self):
        return "(left: " + str(self.left) + ", right: " + str(self.right) + ", data: " + str(
            self.data) +", cost: " + str(self.cost) + ")"



##1

class Dijkstra:

    def __init__(self, ox, oy, resolution, robot_radius, model = '4n'):
        """
        Initialize map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.x_width = None
        self.y_width = None
        self.obstacle_map = None

        self.resolution = resolution
        self.robot_radius = robot_radius
        self.calc_obstacle_map(ox, oy)
        if model == '4n':
            self.motion = self.get_motion_model_4n()
        else:
            self.motion = self.get_motion_model_8n()
        # self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index  # index of previous Node

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        dijkstra path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gx: goal x position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """
        
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        plt.scatter(start_node.x, start_node.y, s=400, c='red', marker='s')
        plt.scatter(goal_node.x, goal_node.y, s=400, c='green', marker='s')
        
        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        ##create tree root 
        root = TreeNode(self.calc_grid_index(start_node) , 0) 
        
        while 1:
            c_id = min(open_set, key=lambda o: open_set[o].cost)
            current = open_set[c_id]

            
            ## create child node by finding parent
            key=lambda o: open_set[o].cost 
            findParentInsertChild(root, c_id ,current.parent_index , key(c_id) )
            
            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(self.calc_position(current.x, self.min_x),
                         self.calc_position(current.y, self.min_y), "xc")
                plt.scatter(self.calc_position(current.x, self.min_x),
                            self.calc_position(current.y, self.min_y),
                             s=300, c='yellow', marker='s')
                plt.pause(pause_time)   
                
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                plt.scatter(current.x,
                            current.y,
                            s=100, c='blue', marker='s') 
                plt.pause(pause_time)
                
                locx, locy = self.calc_final_path(current, closed_set)
                if len(locx)>2:
                    for i in range(len(locx)-1):
                        px = (locx[i], locx[i+1])
                        py = (locy[i], locy[i+1])
                        plt.plot(px, py, "-m", linewidth=4)
                        plt.pause(pause_time)    
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                
                print("cost:", goal_node.cost )
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            plt.scatter(current.x,
                        current.y,
                        s=100, c='blue', marker='s') 
            plt.pause(pause_time)             
        
            locx, locy = self.calc_final_path(current, closed_set)
            if len(locx)>2:
                for i in range(len(locx)-1):
                    px = (locx[i], locx[i+1])
                    py = (locy[i], locy[i+1])
                    plt.plot(px, py, "-m", linewidth=4)
                    plt.pause(pause_time)
            
            # plt.show() 
            
            # expand search grid based on motion model
            for move_x, move_y, move_cost in self.motion:
                node = self.Node(current.x + move_x,
                                 current.y + move_y,
                                 current.cost + move_cost, c_id)
                n_id = self.calc_grid_index(node)

                if n_id in closed_set:
                    continue

                if not self.verify_node(node):
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # Discover a new node
                else:
                    if open_set[n_id].cost >= node.cost:
                        # This path is the best until now. record it!
                        open_set[n_id] = node

        #part4 output result
        print("root",root)
        print("path generated by tree" )
        tree_result_list = path_from_to(root,self.calc_grid_index(goal_node))

        ##return as two list, x list and y list
        result_x = list();
        result_y = list();
        
        for x in tree_result_list:
            ##convert back from index to coordinates
            
            cur_x = x % self.x_width
            cur_y = (x - cur_x)/ self.x_width
            
            result_x.append(int(cur_x))
            result_y.append(int(cur_y))
            
        return result_x, result_y

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_position(goal_node.x, self.min_x)], [
            self.calc_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_position(n.x, self.min_x))
            ry.append(self.calc_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    ##get coordinates from index in grid system
    def calc_position(self, index, minp):
        pos = index * self.resolution + minp
        return pos

    ##get index from coordinates based on x or y
    ## use in determining start node
    def calc_xy_index(self, position, minp):
        return round((position - minp) / self.resolution)

    ##get index from coordinates based on x and y
    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

##verify node if it safe or not when exploring 
    def verify_node(self, node):
        px = self.calc_position(node.x, self.min_x)
        py = self.calc_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        if py < self.min_y:
            return False
        if px >= self.max_x:
            return False
        if py >= self.max_y:
            return False

        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.robot_radius:
                        self.obstacle_map[ix][iy] = True
                        break



    @staticmethod
    def get_motion_model_8n():
        # dx, dy, cost
        motion =[[-1, -1, math.sqrt(2)],
                [-1, 0, 1],
                [-1, 1, math.sqrt(2)],
                [0, 1, 1],
                [1, 1, math.sqrt(2)],
                [1, 0, 1],
                [1, -1, math.sqrt(2)],
                [0, -1, 1]
                ]
        return motion
    @staticmethod
    def get_motion_model_4n():
        """
        Get all possible 4-connectivity movements.
        :return: list of movements with cost [[dx, dy, cost]]
        """
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1]]
        return motion


def main():
    ##start to check storage memory
    ##staty to calculate time
    tracemalloc.start()
    start = time.time()
    print("start time" )

    # Obstacle and free grid positions
    ox, oy = [], [] #obstacle
    fx, fy = [], [] #free  

    config_file = 'OccupancyAStar/examples/MapConfig/config9x9.json'
    
    # start and goal position
    with open(config_file) as config_env:
        param = json.load(config_env)
    
    sx = param['sx']  # [m]
    sy = param['sy']   # [m]
    gx = param['gx']   # [m]
    gy = param['gy']   # [m]
    grid_size = param['grid_size']  # [m]
    robot_radius = param['robot_radius']   # [m]
    map_xlsx = param['map_xlsx']
    fig_dim = param['fig_dim']

    # padas is used to read map in .xlsx file
    gmap = pd.read_excel(map_xlsx,header=None)
    data = gmap.to_numpy()

    # Up side down: origin of image is up left corner, while origin of figure is bottom left corner
    data = data[::-1]   


    # Generate obstacles
    for iy in range(data.shape[0]):
        for ix in range(data.shape[1]):
            if data[iy, ix] == 1:
                ox.append(ix)
                oy.append(iy)
            else:
                fx.append(ix)
                fy.append(iy)

    if show_animation:  
        plt.figure(figsize=(fig_dim ,fig_dim))
        plt.xlim(-1, data.shape[1])
        plt.ylim(-1, data.shape[0])
        plt.xticks(np.arange(0,data.shape[1],1))
        plt.yticks(np.arange(0,data.shape[0],1))
        plt.grid(True)
        plt.scatter(ox, oy, s=300, c='gray', marker='s')
        plt.scatter(fx, fy, s=300, c='cyan', marker='s')  

    # dfs = DepthFirstSearchPlanner(ox, oy, grid_size, robot_radius, model = '8n')
    # rx, ry = dfs.planning(sx, sy, gx, gy)

    dijkstra = Dijkstra(ox, oy, grid_size, robot_radius)
    rx, ry = dijkstra.planning(sx, sy, gx, gy)

    new_rx = map(int, rx)
    new_ry = map(int, ry)
    
    road = [list(a) for a in zip(new_rx, new_ry)]
    
    ##output path 
    print("road from start to goal", road)
    
    ##calculate time to run 
    end = time.time()
    print("time end to calculate time")
    print("time used", end - start)
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    tracemalloc.stop()
    
    
    if show_animation:  
        for i in range(len(rx)-1):
            px = (rx[i], rx[i+1])
            py = (ry[i], ry[i+1])
            plt.plot(px, py, "-k")
            plt.pause(0.1)
        plt.show()   

if __name__ == '__main__':
    main()

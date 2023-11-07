import math

import matplotlib.pyplot as plt

import sys
sys.path.append('./OccupancyAStar/examples/')
import utils
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cm
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import json
import tracemalloc
import time

show_animation = True

pause_time = 0.1


class AStarPlanner:

    def __init__(self, ox, oy, resolution, rr, model = '8n'):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        if model == '4n':
            self.motion = self.get_motion_model_4n()
        else:
            self.motion = self.get_motion_model_8n()
        
        self.calc_obstacle_map(ox, oy)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """
        print(np.shape(self.obstacle_map))
        # cmap = colors.ListedColormap(['cyan','grey'])
        # plt.figure(figsize=(6,6))
        # plt.pcolor(self.obstacle_map[::-1],cmap=cmap,edgecolors='k', linewidths=1.5)
        # plt.show()

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        plt.scatter(start_node.x, start_node.y, s=400, c='red', marker='s')
        plt.scatter(goal_node.x, goal_node.y, s=400, c='green', marker='s')


        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))

            current = open_set[c_id]

            # show graph
            if show_animation:  # pragma: no cover
                # plt.plot(self.calc_grid_position(current.x, self.min_x),
                #          self.calc_grid_position(current.y, self.min_y), "xc")
                plt.scatter(self.calc_grid_position(current.x, self.min_x),
                            self.calc_grid_position(current.y, self.min_y),
                             s=300, c='yellow', marker='s')
                plt.pause(pause_time)                 


                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")
                plt.scatter(current.x,
                            current.y,
                            s=100, c='yellow', marker='s') 
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
                print("cost :" ,goal_node.cost)
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

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    plt.scatter(node.x,
                                node.y,
                                s=100, c='black', marker='s')
                    plt.pause(pause_time)                 
                    continue

                if n_id in closed_set:               
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                    plt.scatter(node.x,
                                node.y,
                                s=100, c='red', marker='s')
                    plt.pause(pause_time)
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

      

        return rx, ry

        # generate final course
    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    ## calculate heuristic value by using Pythagorean Theorem
    def calc_heuristic(n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    ##get coordinates from index in grid system
    def calc_grid_position(self, index, min_position):

        pos = index * self.resolution + min_position
        return pos

    ##get index from coordinates based on x or y
    ## use in determining start node
    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    # grid location as key
    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
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
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    
    @staticmethod
    def get_motion_model_8n():
        # dx, dy, cost
        # motion = [[1, 0, 1],
        #           [0, 1, 1],
        #           [-1, 0, 1],
        #           [0, -1, 1],
        #           [-1, -1, math.sqrt(2)],
        #           [-1, 1, math.sqrt(2)],
        #           [1, -1, math.sqrt(2)],
        #           [1, 1, math.sqrt(2)]]
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
        motion = [[-1, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
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

    ##get config setting from json file
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


    # '8n' for moving 8 directions, '4n' for moving 4 directions, default is '8n'
    a_star = AStarPlanner(ox, oy, grid_size, robot_radius, model='4n')  
    rx, ry = a_star.planning(sx, sy, gx, gy)

    # rx, ry are from goal to start, reverse the will make the path show from start to goal
    rx.reverse()  
    ry.reverse()
    
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

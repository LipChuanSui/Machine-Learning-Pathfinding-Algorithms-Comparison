"""

Depth-First grid planning

author: Erwin Lejeune (@spida_rwin)

See Wikipedia article (https://en.wikipedia.org/wiki/Depth-first_search)

"""

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
import time
import tracemalloc

show_animation = True

pause_time = 0.1

xwidth = 8

tree_result_list = list()

##preorder tree travesal 
def preorder(rootNode):
    #if root is None,return
        if rootNode is None:
            return
        
        #traverse current node
        print(rootNode.data)
    #traverse left subtree
        preorder(rootNode.left)
    
    #traverse right subtree
        preorder(rootNode.right)  


#find path for tree search
def hasPath(root, arr, x):
    
    if (not root):
        return False
    # else :
    #     print("root.data",root.data)
    
    cur_x = "";
    cur_y = "";
    
    #convert back to coordinates
    if root.data:
        cur_x = root.data % xwidth
        cur_y = (root.data - cur_x)/ xwidth
        
    arr.append("("+str(int(cur_x)) +","+ str(int(cur_y))+")")    
    ####
    
    if (root.data == x):    
        return True
     
    if (hasPath(root.left, arr, x) or
        hasPath(root.right, arr, x)):
        return True
     
    arr.pop(-1)
    return False

## print path from start to goal for tree search
def printPath(root, x):
    arr = []
    if (hasPath(root, arr, x)):
        for i in range(len(arr) - 1):
            print(arr[i], end = "-->")
            tree_result_list.append(arr[i])
        print(arr[len(arr) - 1])
        tree_result_list.append(arr[len(arr) - 1])
        
    else:
        print("No Path")


##### find parent to add child node in left or right, build tree
def findParentInsertChild(rootNode,currentIndex,parentIndex):
    if rootNode is None:
        return
    findParentInsertChild(rootNode.left,currentIndex,parentIndex )
    
    if rootNode.data == parentIndex:
        if rootNode.left is None:
            rootNode.left = TreeNode(currentIndex)
        else:
            rootNode.right = TreeNode(currentIndex)
    
    findParentInsertChild(rootNode.right,currentIndex,parentIndex)  
    

### tree node class
class TreeNode:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data
        

    def __str__(self):
        return "(left: " + str(self.left) + ", right: " + str(self.right) + ", data: " + str(
            self.data) +")"


class DepthFirstSearchPlanner:

    def __init__(self, ox, oy, reso, rr, model = '8n'):
        """
        Initialize grid map for Depth-First planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.reso = reso
        self.rr = rr
        self.calc_obstacle_map(ox, oy)

        if model == '4n':
            self.motion = self.get_motion_model_4n()
        else:
            self.motion = self.get_motion_model_8n()


    class Node:
        def __init__(self, x, y, cost, parent_index, parent):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index
            self.parent = parent

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, sx, sy, gx, gy):
        """
        Depth First search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                           self.calc_xy_index(sy, self.min_y), 0.0, -1, None)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                          self.calc_xy_index(gy, self.min_y), 0.0, -1, None)

        plt.scatter(start_node.x, start_node.y, s=400, c='red', marker='s')
        plt.scatter(goal_node.x, goal_node.y, s=400, c='green', marker='s')                           

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node
        closed_set[self.calc_grid_index(start_node)] = start_node

        ##create tree root
        root = TreeNode(self.calc_grid_index(start_node)) 
        
        
        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break
            

            current = open_set.pop(list(open_set.keys())[-1])

            c_id = self.calc_grid_index(current)

            ## create child node by finding parent
            findParentInsertChild(root, c_id ,current.parent_index)
            
            # show graph
            if show_animation:  # pragma: no cover
                # plt.plot(self.calc_grid_position(current.x, self.min_x),
                #          self.calc_grid_position(current.y, self.min_y), "xc")

                plt.scatter(self.calc_grid_position(current.x, self.min_x),
                            self.calc_grid_position(current.y, self.min_y),
                             s=300, c='yellow', marker='s')                         
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event:
                                             [exit(0) if event.key == 'escape'
                                              else None])
                plt.pause(pause_time)

                if len(closed_set)>=2:
                    locx, locy = self.calc_final_path(current, closed_set)
                    for i in range(len(locx)-1):
                        px = (locx[i], locx[i+1])
                        py = (locy[i], locy[i+1])
                        plt.plot(px, py, "-m", linewidth=4)
                        plt.pause(pause_time)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")

                plt.scatter(current.x,
                            current.y,
                            s=300, c='yellow', marker='s') 
                plt.pause(pause_time)
                
                locx, locy = self.calc_final_path(current, closed_set)
                if len(locx)>=2:
                    for i in range(len(locx)-1):
                        px = (locx[i], locx[i+1])
                        py = (locy[i], locy[i+1])
                        plt.plot(px, py, "-m", linewidth=4)
                        plt.pause(pause_time)


                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                
                print("cost:", goal_node.cost )
                
                break

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id, None)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    plt.scatter(node.x,
                                node.y,
                                s=100, c='black', marker='s')
                    plt.pause(pause_time)                      
                    continue

                if n_id not in closed_set:
                    open_set[n_id] = node
                    closed_set[n_id] = node
                    node.parent = current

                    # if open_set[n_id].parent == closed_set[n_id].parent:
                    #     print('ops')
                    plt.scatter(node.x,
                                node.y,
                                s=100, c='b', marker='s')
                    plt.pause(pause_time) 

        #part4 tree print path , return result two list x and y 
        print("preorder tree travesal")
        preorder(root)
        print("root",root)
        print("path generated by tree search" )
        printPath(root,self.calc_grid_index(goal_node))
        
        ##return as two list, x list and y list
        result_x = list();
        result_y = list();
        
        for x in tree_result_list:
            result_x.append(int(x[1]))
            result_y.append(int(x[3]))
            
        return result_x, result_y


    def calc_final_path(self, goal_node, closedset):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        n = closedset[goal_node.parent_index]
        while n is not None:
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            n = n.parent

        return rx, ry

    ##get coordinates from index in grid system
    def calc_grid_position(self, index, minp):
        """
        calc grid position

        :param index:
        :param minp:
        :return:
        """
        pos = index * self.reso + minp
        return pos


    ##get index from coordinates based on x or y
    ## use in determining start node
    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.reso)

    ##get index from coordinates based on x and y
    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.xwidth + (node.x - self.min_x)

##verify node if it safe or not when exploring 
    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.maxx:
            return False
        elif py >= self.maxy:
            return False

        # collision check
        if self.obmap[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.maxx = round(max(ox))
        self.maxy = round(max(oy))
        self.xwidth = round((self.maxx - self.min_x) / self.reso)
        self.ywidth = round((self.maxy - self.min_y) / self.reso)

        # obstacle map generation
        self.obmap = [[False for _ in range(self.ywidth)]
                      for _ in range(self.xwidth)]
        for ix in range(self.xwidth):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.ywidth):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obmap[ix][iy] = True
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

    dfs = DepthFirstSearchPlanner(ox, oy, grid_size, robot_radius, model = '4n')
    rx, ry = dfs.planning(sx, sy, gx, gy)
    
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

    # if show_animation:  # pragma: no cover
    #     plt.plot(rx, ry, "-r")
    #     plt.pause(0.01)
    #     plt.show()


if __name__ == '__main__':
    main()

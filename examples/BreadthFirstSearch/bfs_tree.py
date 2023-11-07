"""

Breadth-First grid planning

author: Erwin Lejeune (@spida_rwin)

See Wikipedia article (https://en.wikipedia.org/wiki/Breadth-first_search)

"""

import math
import pprint
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
from collections import deque

show_animation = True

pause_time = 0.1

xwidth = 8

tree_result_list = list()


def printLevelOrder(root):
    if root is None:
        return

    # Create an empty queue for level order traversal
    queue = []

    # Enqueue Root and initialize height
    queue.append(root)

    while(len(queue) > 0):

        # Print front of queue and
        # remove it from queue
        print(queue[0].data)
        node = queue.pop(0)

        # Enqueue left child
        if node.left :
            queue.append(node.left)

        # Enqueue right child
        if node.right :
            queue.append(node.right)


##get path from root to goal
def path_from_to(root, target):
    queue = deque([(root, [root.data])])
    while (len(queue) > 0):
        node, path = queue.popleft()
        if node.data == target:
            return path
        if node.left:
            queue.append((node.left, path + [node.left.data]))
        if node.right:
            queue.append((node.right, path + [node.right.data]))
            

##find parent of current coordinates(index) and add it to right of left
def findParentInsertChild(rootNode,currentIndex,parentIndex):
    #if root is None,return
    if rootNode is None:
        return
    findParentInsertChild(rootNode.left,currentIndex,parentIndex )
    
    if rootNode.data == parentIndex:
        if rootNode.left is None:
            rootNode.left = TreeNode(currentIndex)
        else:
            rootNode.right = TreeNode(currentIndex)
    
    findParentInsertChild(rootNode.right,currentIndex,parentIndex)  
    

## declare TreeNode
class TreeNode:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data
        

    def __str__(self):
        return "(left: " + str(self.left) + ", right: " + str(self.right) + ", data: " + str(
            self.data) +")"


        
class BreadthFirstSearchPlanner:
    def __init__(self, ox, oy, reso, rr, model = '8n'):
        """
        Initialize grid map for bfs planning

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
        Breadth First search based planning

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """
        xx = 0
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                           self.calc_xy_index(sy, self.min_y), 0.0, -1, None)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                          self.calc_xy_index(gy, self.min_y), 0.0, -1, None)
        
        plt.scatter(start_node.x, start_node.y, s=400, c='red', marker='s')
        plt.scatter(goal_node.x, goal_node.y, s=400, c='green', marker='s')  

        #plt.savefig("bfs_%d.png"%xx)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        ##create tree root
        root = TreeNode(self.calc_grid_index(start_node)) 
        
        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break
            
            current = open_set.pop(list(open_set.keys())[0])

            c_id = self.calc_grid_index(current)

            closed_set[c_id] = current

            ## create child node by finding parent
            findParentInsertChild(root, c_id ,current.parent_index)
        

            # show graph
            if show_animation:  # pragma: no cover

                xx = xx+1
                plt.scatter(self.calc_grid_position(current.x, self.min_x),
                            self.calc_grid_position(current.y, self.min_y),
                             s=300, c='yellow', marker='s')
                plt.pause(pause_time)       

                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event:
                                             [exit(0) if event.key == 'escape'
                                              else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)
            # print(len(closed_set))
                ##plt.savefig("bfs_%d.png"%xx)
                
                
            if len(closed_set) >= 2:
                locx, locy = self.calc_final_path(current, closed_set)
                
                for i in range(len(locx)-1):
                    px = (locx[i], locx[i+1])
                    py = (locy[i], locy[i+1])
                    plt.plot(px, py, "-m", linewidth=4)
                    plt.pause(pause_time) 
                    xx = xx + 1
                    ##plt.savefig("bfs_%d.png"%xx)

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Find goal")

                plt.scatter(current.x,
                            current.y,
                            s=300, c='yellow', marker='s') 
                plt.pause(pause_time)
                xx = xx + 1
                #plt.savefig("bfs_%d.png"%xx)
                
                if len(closed_set) >= 2:
                    locx, locy = self.calc_final_path(current, closed_set)                
                    for i in range(len(locx)-1):
                        px = (locx[i], locx[i+1])
                        py = (locy[i], locy[i+1])
                        plt.plot(px, py, "-m", linewidth=4)
                        plt.pause(pause_time)

                        xx = xx+1
                        #plt.savefig("bfs_%d.png"%xx)

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
                    xx = xx +1
                    #plt.savefig("bfs_%d.png"%xx)                    
                    continue

                if (n_id not in closed_set) and (n_id not in open_set):
                    node.parent = current
                    open_set[n_id] = node
                    plt.scatter(node.x,
                                node.y,
                                s=100, c='blue', marker='s')
                    plt.pause(pause_time) 
                    xx = xx +1
                    #plt.savefig("bfs_%d.png"%xx)               
                            
                if xx >= 108:
                    break      
        
        
        #tree travesal
        print("bfs queue tree travesal")
        printLevelOrder(root)
        print("root",root)
        print("path generated by tree search" )
        ###path from start to goal
        tree_result_list = path_from_to(root,self.calc_grid_index(goal_node))
        
        ##return as two list, x list and y list
        result_x = list();
        result_y = list();
        
        for x in tree_result_list:
            ##convert back from index to coordinates
            
            cur_x = x % self.xwidth
            cur_y = (x - cur_x)/ self.xwidth
            
            result_x.append(int(cur_x))
            result_y.append(int(cur_y))
            
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
 
    bfs = BreadthFirstSearchPlanner(ox, oy, grid_size, robot_radius, model='4n')
    rx, ry = bfs.planning(sx, sy, gx, gy)

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
    xx = 200
    #plt.savefig("bfs_%d.png"%xx)
    

if __name__ == '__main__':
     main()
    




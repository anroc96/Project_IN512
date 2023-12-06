__author__ = "Aybuke Ozturk Suri, Johvany Gustave"
__copyright__ = "Copyright 2023, IN512, IPSA 2023"
__credits__ = ["Aybuke Ozturk Suri", "Johvany Gustave"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"

from network import Network
from my_constants import *

from threading import Thread
import numpy as np
import matplotlib.pyplot as plt
import time
from random import randint

# Defining a function to update believes
def update_believes(x, y, cell_val, w, h, believes, item_found):
    # Updating believes for cells around the robot (1 or 2 cells away)
    for i in range(w): # For all columns in the map
        for j in range(h): # For all cells in one column
            if x-1 <= i and i <= x+1 and y-1 <= j and j <= y+1: # If that cell is one cell away from the robot
                if cell_val in [0, 0.25]: # Not 0.3 because the values of boxes are dominant over 0.25 and 0.5 of a key
                    believes[i, j] = 0
            elif x-2 <= i and i <= x+2 and y-2 <= j and j <= y+2: # If that cell is two cells away from the robot
                if cell_val in [0]:
                    believes[i, j] = 0
            else: # If that cell is far away from the robot and the robot found none-zero values (Focus on the item close to robot)
                if cell_val in [0.25, 0.3, 0.5, 0.6] and item_found == False:
                    believes[i, j] = 0
    return believes

# Defining a function to update known cell values
def update_known_values(x, y, w, h, found_item_type, found_cell_values):
    for i in range(w): # For all columns in the map
        for j in range(h): # For all cells in one column
            if x-1 <= i and i <= x+1 and y-1 <= j and j <= y+1: # If that cell is one cell away from the item
                if found_item_type == 0 and found_cell_values[i, j] not in [0.3, 0.6, 1]: # Other condition to consider that box values are dominant
                    found_cell_values[i, j] = 0.5
                elif found_item_type == 1 and found_cell_values[i, j] != 1: # Other condition if there is already another item there
                    found_cell_values[i, j] = 0.6
            elif x-2 <= i and i <= x+2 and y-2 <= j and j <= y+2: # If that cell is two cells away from the item
                if found_item_type == 0 and found_cell_values[i, j] not in [0.3, 0.5, 0.6, 1]: # Other condition to consider that box values are dominant
                    found_cell_values[i, j] = 0.25
                elif found_item_type == 1 and found_cell_values[i, j] not in [0.6, 1]: # Other condition if there is already another item there
                    found_cell_values[i, j] = 0.3
    found_cell_values[x, y] = 1
    return found_cell_values

# Defining a function to convert a cell coordinate into a move
def cell_to_move(x, y, chosen_next_cell):
    if chosen_next_cell[0] == x-1: # If next cell is on the left
        if chosen_next_cell[1] == y-1: # If next cell is above
            next_move = {"header": MOVE, "direction": UP_LEFT}
        elif chosen_next_cell[1] == y: # If next cell is on the same row
            next_move = {"header": MOVE, "direction": LEFT}
        elif chosen_next_cell[1] == y+1: # If next cell is under
            next_move = {"header": MOVE, "direction": DOWN_LEFT}
    elif chosen_next_cell[0] == x: # If next cell is on the same column
        if chosen_next_cell[1] == y-1: # If next cell is above
            next_move = {"header": MOVE, "direction": UP}
        elif chosen_next_cell[1] == y: # If next cell is on the same row
            next_move = {"header": MOVE, "direction": STAND}
        elif chosen_next_cell[1] == y+1: # If next cell is under
            next_move = {"header": MOVE, "direction": DOWN}
    elif chosen_next_cell[0] == x+1: # If next cell is on the right
        if chosen_next_cell[1] == y-1: # If next cell is above
            next_move = {"header": MOVE, "direction": UP_RIGHT}
        elif chosen_next_cell[1] == y: # If next cell is on the same row
            next_move = {"header": MOVE, "direction": RIGHT}
        elif chosen_next_cell[1] == y+1: # If next cell is under
            next_move = {"header": MOVE, "direction": DOWN_RIGHT}
    return next_move


class Agent:
    """ Class that implements the behaviour of each agent based on their perception and communication with other agents """
    def __init__(self, server_ip):

        #DO NOT TOUCH THE FOLLOWING INSTRUCTIONS
        self.network = Network(server_ip=server_ip)
        self.agent_id = self.network.id
        self.running = True
        self.network.send({"header": GET_DATA})
        env_conf = self.network.receive()
        self.x, self.y = env_conf["x"], env_conf["y"]   #initial agent position
        self.w, self.h = env_conf["w"], env_conf["h"]   #environment dimensions
        self.cell_val = env_conf["cell_val"] #value of the cell the agent is located in
        Thread(target=self.msg_cb, daemon=True).start()

        #TODO: DEFINE YOUR ATTRIBUTES HERE
        self.explo = np.zeros((env_conf["w"], env_conf["h"])) # Matrix of the explorated cells (0: not explorated, 1: explorated)
        self.believes = np.ones((env_conf["w"], env_conf["h"])) # Matrix of believes (values from 0 to 1)
        self.cell_values = np.zeros((env_conf["w"], env_conf["h"])) # Matrix of visited cell values
        self.found_cell_values = np.zeros((env_conf["w"], env_conf["h"])) # Matrix of known cell values to ignore already found items
        self.next_move = None # Next move chosen by the choose_next_move method
        self.found_item_type = None # Type of the item that has just been found (0: key, 1: box)
        self.found_item_owner = None # Id of then owner of the item that has just been found
        self.found_item_flag = False # Flag raised in the explore_cell method to trigger get item owner in choose_move method
        self.identified_item_flag = False # Flag raised in the msg_cb method to trigger a bloc in the explore_cell method
        self.broadcast_message_flag = False # Flag raised to broadcast that an item has been found 
        self.key_position = None # Coordinates of the robot's key
        self.box_position = None # Coordinates of the robot's box
        self.key_collected = False # Flag to tell if the key has been collected by the robot
        self.box_reached = False # Flag to tell if the box has been reached with the key


    def msg_cb(self): 
        """ Method used to handle incoming messages """
        while self.running:
            msg = self.network.receive()
            print(msg)
            # Update values
            if msg["sender"] == -1: # If the message is sent from the server
                if msg["header"] == 2: # If the message has a MOVE header
                    self.x, self.y = msg["x"], msg["y"]
                    self.cell_val = msg["cell_val"]
                
                elif msg["header"] == 5: # If the message has a GET ITEM OWNER header
                    self.found_item_type = msg["type"]
                    self.found_item_owner = msg["owner"]
                    self.identified_item_flag = True

                    if self.found_item_owner == self.agent_id: # If the item belongs to this robot
                        if self.found_item_type == 0: # If the item is a key
                            self.key_position = (self.x, self.y)
                            self.key_collected = True
                        elif self.found_item_type == 1: # If the item is a box
                            self.box_position = (self.x, self.y)
                            if self.key_collected is True:
                                self.box_reached = True
                    
                    else: # If the item belongs to another robot
                        self.broadcast_message_flag = True
            
            elif msg["header"] == 0: # If the message is a BROADCAST message coming from another robot
                if msg["Msg type"] in [1, 2]: # If the message says another robot found a key or a box
                    # Updating known cell values to ignore this newly found item
                    self.found_cell_values = update_known_values(msg["position"][0], msg["position"][1], self.w, self.h, msg["Msg type"]-1, self.found_cell_values)
                    # Resetting believes in case the robot was trying to find this item
                    self.believes = np.ones((self.w, self.h))
                    for i in range(self.w): # For all columns in the map
                        for j in range(self.h): # For all cells in one column
                            # Ajusting new believes considering cell visited before finding the target
                            if self.explo[i, j] == 1:
                                self.believes = update_believes(i, j, self.cell_values[i, j], self.w, self.h, self.believes, True)

                    if msg["owner"] == self.agent_id: # If the item belongs to this robot
                        if msg["Msg type"] == 1: # If the item is a key
                            self.key_position = msg["position"]
                        if msg["Msg type"] == 2: # If the item is a box
                            self.box_position = msg["position"]



    #TODO: CREATE YOUR METHODS HERE...

    def explore_cell(self):
        # Verify if the robot is collecting the key or reaching the box with the key
        if (self.x, self.y) == self.key_position:
            self.key_collected = True
        elif (self.x, self.y) == self.box_position and self.key_collected is True:
            self.box_reached = True

        # Updating cell_values
        if self.cell_val != 0:
            self.cell_values[self.x, self.y] = self.cell_val

        # Adjusting cell value if it corresponds to an item that has already been found
        if self.cell_val == self.found_cell_values[self.x, self.y]:
            self.cell_val = 0

        # Verify if the robot found a new item
        if self.cell_val == 1:
            if self.identified_item_flag is False: # If the item has not been identified yet
                self.found_item_flag = True
                return # Exit the function to choose a move
            
            else: # If the item found has been identified
                self.believes = np.ones((self.w, self.h)) # Resetting believes to search for a new item
                for i in range(self.w): # For all columns in the map
                    for j in range(self.h): # For all cells in one column
                        # Ajusting new believes considering cell visited before finding the target
                        if self.explo[i, j] == 1:
                            self.believes = update_believes(i, j, self.cell_values[i, j], self.w, self.h, self.believes, True)
                
                # Keeping known map values in memory to ignore found items
                self.found_cell_values = update_known_values(self.x, self.y, self.w, self.h, self.found_item_type, self.found_cell_values)
                self.cell_val = 0 # Adjusting cell value because item has been found
                self.identified_item_flag = False # Searching for a new item to identify

        # Marking the cell as explored
        self.explo[self.x, self.y] = 1 
        # Updating believes for cells around the robot (0 to 2 cells away)
        self.believes = update_believes(self.x, self.y, self.cell_val, self.w, self.h, self.believes, False)


    def choose_action(self):
        # Verify if an item has been found
        if self.found_item_flag is True:
            self.found_item_flag = False
            self.next_move = {"header": GET_ITEM_OWNER} # Get item owner (Ask the server for the type and the owner of the item)

        # Verify if a message must be broadcasted
        elif self.broadcast_message_flag is True:
            self.broadcast_message_flag = False
            self.next_move = {"header": BROADCAST_MSG, "Msg type": self.found_item_type+1, "position": (self.x, self.y), "owner": self.found_item_owner}

        # Go collect the key if it is another robot that found it first
        elif self.key_position is not None and self.key_collected is False:
            # Create list of possible next cells
            possible_next_cells = []
            for i in range(self.x-1, self.x+2): # For colums maximum one cell away
                for j in range(self.y-1, self.y+2): # For rows maximum one cell away
                    if i in range(self.w) and j in range(self.h) and (i, j): # If the indexes are inside the map
                        distance = np.linalg.norm([self.key_position[0]-i, self.key_position[1]-j]) # Calculate the distance to the key
                        possible_next_cells.append([i, j, distance])
            # Find best possible next cells based on the distance to the key
            possible_next_cells = np.array(possible_next_cells)
            [_, _, min_distance] = np.amin(possible_next_cells, axis = 0)
            best_next_cells = possible_next_cells[np.where((possible_next_cells[:,2] == min_distance))]
            # Randomly choose a cell among the best possible cells
            idx = randint(0,len(best_next_cells)-1)
            chosen_next_cell = (int(best_next_cells[idx][0]), int(best_next_cells[idx][1]))
            # Convert cell coordinates into move
            self.next_move = cell_to_move(self.x, self.y, chosen_next_cell)

        # Go to the box if the key is collected and the position of the box is known
        elif self.box_position is not None and self.key_collected is True:
            # Create list of possible next cells
            possible_next_cells = []
            for i in range(self.x-1, self.x+2): # For colums maximum one cell away
                for j in range(self.y-1, self.y+2): # For rows maximum one cell away
                    if i in range(self.w) and j in range(self.h) and (i, j): # If the indexes are inside the map
                        distance = np.linalg.norm([self.box_position[0]-i, self.box_position[1]-j]) # Calculate the distance to the box
                        possible_next_cells.append([i, j, distance])
            # Find best possible next cells based on the distance to the box
            possible_next_cells = np.array(possible_next_cells)
            [_, _, min_distance] = np.amin(possible_next_cells, axis = 0)
            best_next_cells = possible_next_cells[np.where((possible_next_cells[:,2] == min_distance))]
            # Randomly choose a cell among the best possible cells
            idx = randint(0,len(best_next_cells)-1)
            chosen_next_cell = (int(best_next_cells[idx][0]), int(best_next_cells[idx][1]))
            # Convert cell coordinates into move
            self.next_move = cell_to_move(self.x, self.y, chosen_next_cell)
        
        # Find the best next move to explore the map
        else:
            # Create list of possible next cells
            possible_next_cells = []
            for i in range(self.x-1, self.x+2): # For colums maximum one cell away
                for j in range(self.y-1, self.y+2): # For rows maximum one cell away
                    if i in range(self.w) and j in range(self.h) and (i, j) != (self.x, self.y): # If the indexes are inside the map and the cell is not the current cell
                        # Count number of cells with a belief of 1 around that cell (sum of 1/distances)
                        weighted_sum = 0 
                        for ii in range(self.w): # For colums maximum two cells away
                            for jj in range(self.h): # For rows maximum two cells away
                                if self.believes[ii, jj] == 1: # If that cell has a belief of 1
                                    distance = np.linalg.norm([ii-i, jj-j]) # Calculate the distance between these cells
                                    weighted_sum += 1/(distance+0.0000001) # Greater weight for closer cells
                        # Append the list of possible next cells with cell belief and nb visited cell as criterions
                        possible_next_cells.append([i, j, weighted_sum])
            # Find best possible next cells based on the weighted sum of cells with belief of 1
            possible_next_cells = np.array(possible_next_cells)
            [_, _, max_weighted_sum] = np.amax(possible_next_cells, axis = 0)
            best_next_cells = possible_next_cells[np.where((possible_next_cells[:,2] == max_weighted_sum))]
            # Randomly choose a cell among the best possible cells
            idx = randint(0,len(best_next_cells)-1)
            chosen_next_cell = (int(best_next_cells[idx][0]), int(best_next_cells[idx][1]))
            # Convert cell coordinates into move
            self.next_move = cell_to_move(self.x, self.y, chosen_next_cell)
        
        # Execute chosen action
        self.network.send(self.next_move)


    def plot_believes(self):
        plt.figure(self.agent_id+1, figsize=(6,6.5))
        plt.clf()   # Clear the matplotlib plot every time the robot moves
        plt.ion()
        plt.show()
        # Creating colormap with explored cells
        plt.pcolormesh(np.flip(self.explo.T, 0), cmap='Blues', edgecolors='k', vmin=0, vmax=2)
        # Creating array with cell values in visited cells and believes elsewhere
        cellval_or_belief = np.maximum(self.believes, self.cell_values)
        # Adding believes as annotations on every cell
        for i in range(self.h):
            for j in range(self.w):
                plt.annotate(str(round(np.flip(cellval_or_belief.T, 0)[j][i], 1)), xy=(i+0.5, j+0.5), ha='center', va='center', color='black')
        plt.axis('equal')
        plt.axis('off')
        plt.title(f'GridBelieves for robot {self.agent_id+1}')
        # Add total number of visited cells
        plt.annotate(f'Number of visited cells: {int(self.explo.sum())}', xy=(0,-1), color='black', annotation_clip=False)
        plt.annotate(f'Position of the key: {self.key_position} Key collected: {self.key_collected}', xy=(0,-2), color='black', annotation_clip=False)
        plt.annotate(f'Position of the box: {self.box_position} Box reached with key: {self.box_reached}', xy=(0,-3), color='black', annotation_clip=False)
        plt.tight_layout() 
        plt.draw()
        plt.pause(0.2)




if __name__ == "__main__":
    from random import randint
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--server_ip", help="Ip address of the server", type=str, default="localhost")
    args = parser.parse_args()

    agent = Agent(args.server_ip)
    try:    #Manual control test
        while True:
            agent.explore_cell()
            agent.plot_believes()
            time.sleep(1)
            agent.choose_action()
            #cmds = {"header": int(input("0 <-> Broadcast msg\n1 <-> Get data\n2 <-> Move\n3 <-> Get nb connected agents\n4 <-> Get nb agents\n5 <-> Get item owner\n"))}
            #if cmds["header"] == BROADCAST_MSG:
            #    cmds["Msg type"] = int(input("1 <-> Key discovered\n2 <-> Box discovered\n3 <-> Completed\n"))
            #    cmds["position"] = (agent.x, agent.y)
            #    cmds["owner"] = randint(0,3) # TODO: specify the owner of the item
            #elif cmds["header"] == MOVE:
            #    cmds["direction"] = int(input("0 <-> Stand\n1 <-> Left\n2 <-> Right\n3 <-> Up\n4 <-> Down\n5 <-> UL\n6 <-> UR\n7 <-> DL\n8 <-> DR\n"))
            #agent.network.send(cmds)
            time.sleep(0.5) # Added time sleep to allow for receiving incoming message in other thread before next iteration

    except KeyboardInterrupt:
        pass





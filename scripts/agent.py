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
def update_believes(x, y, cell_val, w, h, current_believes, explo, item_found):
    new_believes = current_believes
    # Creating multiplier matrix to update the believes
    believes_multiplier = np.ones((w, h))
    for i in range(w): # For all columns in the map
        for j in range(h): # For all cells in one column
            if explo[i, j] == 0: # If that cell has not been visited
                if x-1 <= i and i <= x+1 and y-1 <= j and j <= y+1: # If that cell is one cell away from the robot
                    if cell_val == 0:
                        believes_multiplier[i, j] = 0.8
                    elif cell_val == 0.3:
                        believes_multiplier[i, j] = 0.9
                    elif cell_val == 0.25:
                        believes_multiplier[i, j] = 0.9
                elif x-2 <= i and i <= x+2 and y-2 <= j and j <= y+2: # If that cell is two cells away from the robot
                    if cell_val == 0:
                        believes_multiplier[i, j] = 0.9
                    elif cell_val == 0.6:
                        believes_multiplier[i, j] = 0.9
                    elif cell_val == 0.5:
                        believes_multiplier[i, j] = 0.9
                elif x-3 <= i and i <= x+3 and y-3 <= j and j <= y+3: # If that cell is three cells away from the robot
                    if cell_val == 0.3:
                        believes_multiplier[i, j] = 0.9
                    elif cell_val == 0.25:
                        believes_multiplier[i, j] = 0.9
                    elif cell_val == 0.6:
                        believes_multiplier[i, j] = 0.8
                    elif cell_val == 0.5:
                        believes_multiplier[i, j] = 0.8
                elif x-4 <= i and i <= x+4 and y-4 <= j and j <= y+4: # If that cell is four cells away from the robot
                    if cell_val == 0.3:
                        believes_multiplier[i, j] = 0.8
                    elif cell_val == 0.25:
                        believes_multiplier[i, j] = 0.8
                else:
                    if cell_val in [0.25, 0.3, 0.5, 0.6] and item_found == False:
                        believes_multiplier[i, j] = 0
    # Updating believes for cells around the robot (1 or 2 cells away)
    new_believes = np.multiply(current_believes, believes_multiplier)
    return new_believes

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
        self.found_cell_values = np.zeros((env_conf["w"], env_conf["h"])) # Matrix of known cell values to ignore already found items
        self.next_move = None # Next move chosen by the choose_next_move method
        self.found_item_type = None # Type of the item that has just been found (0: key, 1: box)
        self.found_item_owner = None # Id of then owner of the item that has just been found
        self.found_item_flag = False # Flag raised in the explore_cell method to trigger get item owner in choose_move method


    def msg_cb(self): 
        """ Method used to handle incoming messages """
        while self.running:
            msg = self.network.receive()
            print(msg)
            # Update values
            if msg["sender"] == -1:
                if msg["header"] == 2:
                    self.x, self.y = msg["x"], msg["y"]   # Update agent position
                    self.cell_val = msg["cell_val"] # Value of the cell the agent is located in
                elif msg["header"] == 5:
                    self.found_item_type = msg["type"]
                    #self.found_item_owner = msg["owner"]



    #TODO: CREATE YOUR METHODS HERE...

    def explore_cell(self):
        # Adjusting cell value if it corresponds to an item that has already been found
        if self.cell_val == self.found_cell_values[self.x, self.y]:
            self.cell_val = 0

        # Verify if the robot found a new item
        if self.cell_val == 1:
            if self.found_item_type is None: # If the item has not been identified yet
                self.found_item_flag = True
                return # Exit the function to choose a move
            
            else: # If the item found has been identified
                old_believes = self.believes # Keeping old believes in memory
                self.believes = np.ones((self.w, self.h)) # Resetting believes to search for a new item
                for i in range(self.w): # For all columns in the map
                    for j in range(self.h): # For all cells in one column
                        # Ajusting new believes considering cell visited before finding the target
                        if self.explo[i, j] == 1:
                            self.believes[i, j] = old_believes[i, j]
                            self.believes = update_believes(i, j, old_believes[i, j], self.w, self.h, self.believes, self.explo, True)
                        # Keeping known map values in memory to ignore found items
                        if self.x-1 <= i and i <= self.x+1 and self.y-1 <= j and j <= self.y+1: # If that cell is one cell away from the robot
                            if self.found_item_type == 0:
                                self.found_cell_values[i, j] = 0.5
                            elif self.found_item_type == 1:
                                self.found_cell_values[i, j] = 0.6
                        elif self.x-2 <= i and i <= self.x+2 and self.y-2 <= j and j <= self.y+2: # If that cell is two cells away from the robot
                            if self.found_item_type == 0:
                                self.found_cell_values[i, j] = 0.25
                            elif self.found_item_type == 1:
                                self.found_cell_values[i, j] = 0.3
                self.found_cell_values[self.x, self.y] = 1
                self.cell_val = 0 # Adjusting cell value because item has been found
                self.found_item_type = None # Reseting this variable to search for a new item

        # Marking the cell as explored
        self.explo[self.x, self.y] = 1 
        # Updating believes with cell value
        self.believes[self.x, self.y] = self.cell_val
        # Updating believes for cells around the robot (1 or 2 cells away)
        self.believes = update_believes(self.x, self.y, self.cell_val, self.w, self.h, self.believes, self.explo, False)


    def choose_next_move(self):
        # Verify if an item has been found
        if self.found_item_flag is True:
            self.found_item_flag = False
            self.next_move = (5, 0) # Get item owner (Ask the server for the type and the owner of the item)
        else:
            # Create list of possible next cells
            possible_next_cells = []
            for i in range(self.w): # For all columns in the map
                for j in range(self.h): # For all cells in one column
                    if self.x-1 <= i and i <= self.x+1 and self.y-1 <= j and j <= self.y+1: # If that cell is one cell away from the robot
                        cell_belief = self.believes[i,j] # Get the belief from that cell
                        # Count number of cells with a belief of 1 around that cell
                        total_belief = 0 
                        for ii in range(i-2, i+3): # For colums maximum two cells away
                            for jj in range(j-2, j+3): # For rows maximum two cells away
                                if ii in range(self.w) and jj in range(self.h): # If the indexes are inside the map
                                    if self.believes[ii, jj] == 1: # If that cell has a belief of 1
                                        total_belief += 1
                        # Append the list of possible next cells with cell belief and nb visited cell as criterions
                        possible_next_cells.append([i, j, cell_belief, total_belief])
            
            # Find best possible next cells based on the belief of that cell and the total belief of surrounding cells
            possible_next_cells = np.array(possible_next_cells)
            [_, _, max_cell_belief, _] = np.amax(possible_next_cells, axis = 0)
            best_next_cells = possible_next_cells[np.where((possible_next_cells[:,2] == max_cell_belief))]
            [_, _, _, max_total_belief] = np.amax(best_next_cells, axis = 0)
            best_next_cells = best_next_cells[np.where((best_next_cells[:,3] == max_total_belief))]
            # Randomly choose a cell among the best possible cells
            idx = randint(0,len(best_next_cells)-1)
            chosen_next_cell = (int(best_next_cells[idx][0]), int(best_next_cells[idx][1]))

            # Convert cell coordinates into move
            if chosen_next_cell[0] == self.x-1: # If next cell is on the left
                if chosen_next_cell[1] == self.y-1: # If next cell is above
                    self.next_move = (2,5) # UL
                elif chosen_next_cell[1] == self.y: # If next cell is on the same row
                    self.next_move = (2,1) # Left
                elif chosen_next_cell[1] == self.y+1: # If next cell is under
                    self.next_move = (2,7) # DL
                else:
                    print('The agent chose an impossible move.')
            elif chosen_next_cell[0] == self.x: # If next cell is on the same column
                if chosen_next_cell[1] == self.y-1: # If next cell is above
                    self.next_move = (2,3) # Up
                elif chosen_next_cell[1] == self.y: # If next cell is on the same row
                    self.next_move = (2,0) # Stand
                elif chosen_next_cell[1] == self.y+1: # If next cell is under
                    self.next_move = (2,4) # Down
                else:
                    print('The agent chose an impossible move.')
            elif chosen_next_cell[0] == self.x+1: # If next cell is on the right
                if chosen_next_cell[1] == self.y-1: # If next cell is above
                    self.next_move = (2,6) # UR
                elif chosen_next_cell[1] == self.y: # If next cell is on the same row
                    self.next_move = (2,2) # Right
                elif chosen_next_cell[1] == self.y+1: # If next cell is under
                    self.next_move = (2,8) # DR
                else:
                    print('The agent chose an impossible move.')


    def move(self):
        if self.next_move:
            if self.next_move[0] == 5:
                self.network.send({"header": 5})
            elif self.next_move[0] == 2:
                self.network.send({"header": MOVE, "direction": self.next_move[1]})
        

    def plot_believes(self):
        plt.figure(self.agent_id+1, figsize=(6,6.5))
        plt.clf()   # Clear the matplotlib plot every time the robot moves
        plt.ion()
        plt.show()
        # Creating colormap with explored cells
        plt.pcolormesh(np.flip(self.explo.T, 0), cmap='Blues', edgecolors='k', vmin=0, vmax=2)
        # Adding believes as annotations on every cell
        for i in range(self.h):
            for j in range(self.w):
                plt.annotate(str(round(np.flip(self.believes.T, 0)[j][i], 1)), xy=(i+0.5, j+0.5), ha='center', va='center', color='black')
        plt.axis('equal')
        plt.axis('off')
        plt.title(f'GridBelieves for robot {self.agent_id+1}')
        # Add total number of visited cells
        plt.annotate(f'Number of visited cells: {int(self.explo.sum())}', xy=(0,-1), color='black', annotation_clip=False)
        plt.tight_layout() 
        plt.draw()
        plt.pause(0.5)




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
            agent.choose_next_move()
            time.sleep(1)
            agent.move()
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





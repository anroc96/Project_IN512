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

        #TODO: DEINE YOUR ATTRIBUTES HERE
        self.explo = np.zeros((env_conf["w"], env_conf["h"])) # Matrix of the explorated cells (0: not explorated, 1: explorated)
        self.believes = np.ones((env_conf["w"], env_conf["h"])) # Matrix of believes (values from 0 to 1)


    def msg_cb(self): 
        """ Method used to handle incoming messages """
        while self.running:
            msg = self.network.receive()
            print(msg)
            # Update values
            if msg["sender"] == -1:
                self.x, self.y = msg["x"], msg["y"]   #update agent position
                self.cell_val = msg["cell_val"] #value of the cell the agent is located in


    #TODO: CREATE YOUR METHODS HERE...

    def explore_cell(self):
        # Marking the cell as explored
        self.explo[self.x, self.y] = 1 
        # Updating believes with cell value
        self.believes[self.x, self.y] = self.cell_val
        # Creating multiplier matrix to update the believes
        believes_multiplier = np.ones((self.w, self.h))
        for i in range(self.w): # For all columns in the map
            for j in range(self.h): # For all cells in one column
                if self.explo[i, j] == 0: # If that cell has not been visited
                    if self.x-1 <= i and i <= self.x+1 and self.y-1 <= j and j <= self.y+1: # If that cell is one cell away from the robot
                        if self.cell_val == 0:
                            believes_multiplier[i, j] = 0.3
                        elif self.cell_val == 0.3:
                            believes_multiplier[i, j] = 0.6
                        elif self.cell_val == 0.25:
                            believes_multiplier[i, j] = 0.5
                    elif self.x-2 <= i and i <= self.x+2 and self.y-2 <= j and j <= self.y+2: # If that cell is two cells away from the robot
                        if self.cell_val == 0:
                            believes_multiplier[i, j] = 0.6
        # Updating believes for cells around the robot (1 or 2 cells away)
        self.believes = np.multiply(self.believes, believes_multiplier)


    def plot_believes(self):
        plt.figure(self.agent_id+1, figsize=(6,6.5))
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
        plt.show()




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
            cmds = {"header": int(input("0 <-> Broadcast msg\n1 <-> Get data\n2 <-> Move\n3 <-> Get nb connected agents\n4 <-> Get nb agents\n5 <-> Get item owner\n"))}
            if cmds["header"] == BROADCAST_MSG:
                cmds["Msg type"] = int(input("1 <-> Key discovered\n2 <-> Box discovered\n3 <-> Completed\n"))
                cmds["position"] = (agent.x, agent.y)
                cmds["owner"] = randint(0,3) # TODO: specify the owner of the item
            elif cmds["header"] == MOVE:
                cmds["direction"] = int(input("0 <-> Stand\n1 <-> Left\n2 <-> Right\n3 <-> Up\n4 <-> Down\n5 <-> UL\n6 <-> UR\n7 <-> DL\n8 <-> DR\n"))
            agent.network.send(cmds)
            time.sleep(0.01) # Added time sleep to allow for receiving incoming message in other thread before next iteration

    except KeyboardInterrupt:
        pass





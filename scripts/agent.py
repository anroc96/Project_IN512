__author__ = "Aybuke Ozturk Suri, Johvany Gustave"
__copyright__ = "Copyright 2023, IN512, IPSA 2023"
__credits__ = ["Aybuke Ozturk Suri", "Johvany Gustave"]
__license__ = "Apache License 2.0"
__version__ = "1.0.0"

from network import Network
from my_constants import *
import os
from threading import Thread
import numpy as np
import matplotlib.pyplot as plt
import time

# Defining a function to update believes
def update_believes(x, y, cell_val, w, h, believes, item_found):
    """
    Update the believes'grid of the robot

    Args:
        x (int): robot x-position
        y (int): robot y-positions
        cell_val (float): value of the cell in the environment
        w (int): environment largeur
        h (int): environment height
        believes (ndarray): believes' grid of the robot
        item_found (bool): whether an item as been found or not

    Returns:
        ndarray: the believes' grid of the robot
    """
    # Updating belief for the current cell
    believes[x, y] = 0
    # Updating believes for other cells in map
    for i in range(w): # For all columns in the map
        for j in range(h): # For all cells in one column
            if x-1 <= i and i <= x+1 and y-1 <= j and j <= y+1: # If that cell is one cell away from the robot
                if cell_val in [0, 0.25]: # No item can be one cell away of these cell values
                    believes[i, j] = 0
                elif cell_val in [0.3] and item_found == False: # Not when reupdating for 0.3 because box values are dominant over 0.25 and 0.5
                    believes[i, j] = 0
            elif x-2 <= i and i <= x+2 and y-2 <= j and j <= y+2: # If that cell is two cells away from the robot
                if cell_val in [0]: # No item can be two cells away of this cell value
                    believes[i, j] = 0
                elif cell_val in [0.5, 0.6] and item_found == False: # If a new item is found, focus on it
                    believes[i, j] = 0
            else: # If that cell is far away from the robot and the robot found none-zero values (Focus on the item close to robot)
                if cell_val in [0.25, 0.3, 0.5, 0.6] and item_found == False:
                    believes[i, j] = 0
    return believes

# Defining a function to update known cell values
def update_known_values(x, y, w, h, found_item_type, found_cell_values):
    """
    Update the known cell values in the map

    Args:
        x (int): robot x-position
        y (int): robot y-position
        w (int): environment largeur
        h (int): environment height
        found_item_type: type of the item 
        found_cell_values (ndarray): array containing the locations of the items

    Returns:
        ndarray: the found cell values 
    """
    for i in range(w): # For all columns in the map
        for j in range(h): # For all cells in one column
            if x-1 <= i and i <= x+1 and y-1 <= j and j <= y+1: # If that cell is one cell away from the item
                if found_item_type == KEY_TYPE and found_cell_values[i, j] not in [0.3, 0.6, 1]: # Box values are dominant
                    found_cell_values[i, j] = 0.5
                elif found_item_type == BOX_TYPE and found_cell_values[i, j] not in [1]: # 1 is dominant
                    found_cell_values[i, j] = 0.6
            elif x-2 <= i and i <= x+2 and y-2 <= j and j <= y+2: # If that cell is two cells away from the item
                if found_item_type == KEY_TYPE and found_cell_values[i, j] not in [0.3, 0.5, 0.6, 1]: # These values are dominant
                    found_cell_values[i, j] = 0.25
                elif found_item_type == BOX_TYPE and found_cell_values[i, j] not in [0.6, 1]: # These values are dominant
                    found_cell_values[i, j] = 0.3
    found_cell_values[x, y] = 1
    return found_cell_values

# Defining a function to reset+update believes and update found_cell_values when a new item is found
def new_item_update(x, y, w, h, found_item_type, believes, explo, cell_values, found_cell_values):
    """Reset and updtate the believes' grid and update the found cell values when a new item is found

    Args:
        x (int): robot x-position
        y (int): robot y-position
        w (int): environment largeur
        h (int): environment height
        found_item_type: type of the item 
        believes (ndarray): believes' grid of the robot
        explo (ndarray): exploration map of the robot
        cell_values (tuple): array containing every cell values of the environement
        found_cell_values (ndarray): array containing the locations of the items

    Returns:
        (ndarray, ndarray): believes' grid of the robot and the found cell values 
    """
    # Updating known cell values to ignore this newly found item
    found_cell_values = update_known_values(x, y, w, h, found_item_type, found_cell_values)
    # Resetting believes to serch for a new item
    believes = np.ones((w, h))
    for i in range(w): # For all columns in the map
        for j in range(h): # For all cells in one column
            # Ajusting new believes considering cell visited before finding the target
            if explo[i, j] == 1:
                believes = update_believes(i, j, cell_values[i, j], w, h, believes, True)
    return believes, found_cell_values

def delete_files(folder_path):
    """
    Delete all files in a given folder

    Args:
        folder_path (str): the absolute or relative path to the folder
    """
    import os, shutil
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
     
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    

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
       
        self.explo = np.zeros((env_conf["w"], env_conf["h"])) # Matrix of the explorated cells (0: not explorated, 1: explorated)
        self.believes = np.ones((env_conf["w"], env_conf["h"])) # Matrix of believes (0: no item can be there, 1: it is possible that there is an item)
        # Note: If the agent finds a new non zero value, it will focus on this item and put zeros everywhere far from its location even if there may be \
        # other items there. It will remove these "false zeros" after it found the item it was looking for.
        self.cell_values = np.zeros((env_conf["w"], env_conf["h"])) # Matrix of visited cell values to display on the map and to reupdate believes
        self.found_cell_values = np.zeros((env_conf["w"], env_conf["h"])) # Matrix of known cell values to ignore already found items
        self.found_item_type = None # Type of the item that has just been found (0: key, 1: box)
        self.found_item_owner = None # Id of the owner of the item that has just been found
        self.found_item_flag = False # Flag raised in the explore_cell method to trigger get item owner in choose_action method
        self.broadcast_message_flag = False # Flag raised in msg_cb method to trigger broadcast in choose_action method
        self.key_position = None # Coordinates of the robot's key
        self.box_position = None # Coordinates of the robot's box
        self.key_collected = False # Flag to tell if the key has been collected by the robot
        self.box_reached = False # Flag to tell if the box has been reached with the key and that the quest is completed


    def msg_cb(self): 
        """ Method used to handle incoming messages """
        while self.running:
            msg = self.network.receive()
            print(msg)

            if msg["header"] == MOVE:
                # Update variables
                self.x, self.y = msg["x"], msg["y"]
                self.cell_val = msg["cell_val"]
            
            elif msg["header"] == GET_ITEM_OWNER:
                # The robot found a new item
                self.found_item_type = msg["type"]
                self.found_item_owner = msg["owner"]
                # Updating believes and found_cell_values
                self.believes, self.found_cell_values = new_item_update(self.x, self.y, self.w, self.h, self.found_item_type, self.believes,
                                                                        self.explo, self.cell_values, self.found_cell_values)
                # Checking if the item belongs to this robot
                if self.found_item_owner == self.agent_id:
                    if self.found_item_type == KEY_TYPE:
                        self.key_position = (self.x, self.y)
                        self.key_collected = True
                    elif self.found_item_type == BOX_TYPE:
                        self.box_position = (self.x, self.y)
                        if self.key_collected is True:
                            self.box_reached = True
                # Tell the other robots about this item
                self.broadcast_message_flag = True
            
            elif msg["header"] == BROADCAST_MSG and msg["Msg type"] in [KEY_DISCOVERED, BOX_DISCOVERED]: # If another robot found an item
                # Updating believes and found_cell_values
                self.believes, self.found_cell_values = new_item_update(msg["position"][0], msg["position"][1], self.w, self.h, msg["Msg type"]-1, self.believes,
                                                                        self.explo, self.cell_values, self.found_cell_values)
                # Checking if the item belongs to this robot
                if msg["owner"] == self.agent_id:
                    if msg["Msg type"] == KEY_DISCOVERED:
                        self.key_position = msg["position"]
                    if msg["Msg type"] == BOX_DISCOVERED:
                        self.box_position = msg["position"]


    def cell2move(self, chosen_next_cell):
        """
        Convert a cell coordinate into a move to go to a specific position

        Args:
            chosen_next_cell (tuple): next coordinate the robot as to reach

        Returns:
            dict: the robot's next move message
        """
        if chosen_next_cell[0] == self.x-1: # If next cell is on the left
            if chosen_next_cell[1] == self.y-1: # If next cell is above
                next_move = {"header": MOVE, "direction": UP_LEFT}
            elif chosen_next_cell[1] == self.y: # If next cell is on the same row
                next_move = {"header": MOVE, "direction": LEFT}
            elif chosen_next_cell[1] == self.y+1: # If next cell is under
                next_move = {"header": MOVE, "direction": DOWN_LEFT}
        elif chosen_next_cell[0] == self.x: # If next cell is on the same column
            if chosen_next_cell[1] == self.y-1: # If next cell is above
                next_move = {"header": MOVE, "direction": UP}
            elif chosen_next_cell[1] == self.y: # If next cell is on the same row
                next_move = {"header": MOVE, "direction": STAND}
            elif chosen_next_cell[1] == self.y+1: # If next cell is under
                next_move = {"header": MOVE, "direction": DOWN}
        elif chosen_next_cell[0] == self.x+1: # If next cell is on the right
            if chosen_next_cell[1] == self.y-1: # If next cell is above
                next_move = {"header": MOVE, "direction": UP_RIGHT}
            elif chosen_next_cell[1] == self.y: # If next cell is on the same row
                next_move = {"header": MOVE, "direction": RIGHT}
            elif chosen_next_cell[1] == self.y+1: # If next cell is under
                next_move = {"header": MOVE, "direction": DOWN_RIGHT}
        return next_move
    

    def go_towards_cell(self, target_position):
        """
        Choose the best next move to go towards a certain cell

        Args:
            target_position (tuple): position of the target (key or box here)

        Returns:
            dict: the robot's best move message
        """
        max_distance = np.linalg.norm([self.w, self.h])
        for i in range(self.x-1, self.x+2): # For colums maximum one cell away
            for j in range(self.y-1, self.y+2): # For rows maximum one cell away
                if i in range(self.w) and j in range(self.h): # If the indexes are inside the map
                    distance = np.linalg.norm([target_position[0]-i, target_position[1]-j]) # Calculate the distance to the key
                    if distance < max_distance:
                        chosen_next_cell = (i, j)
                        max_distance = distance
        # Convert cell coordinates into move
        return self.cell2move(chosen_next_cell)
    
    
    def explore_cell(self):
        """ Explore the cell the robot is on """
        
        # Verify if the robot is collecting the key or reaching the box with the key
        if (self.x, self.y) == self.key_position:
            self.key_collected = True
        elif (self.x, self.y) == self.box_position and self.key_collected is True:
            self.box_reached = True

        # Updating cell_values
        self.cell_values[self.x, self.y] = self.cell_val

        # Creating a flag to ignore a cell value if it corresponds to an already found item
        known_cell_val = False 
        if self.cell_val == self.found_cell_values[self.x, self.y] and self.cell_val != 0:
            known_cell_val = True

        # Verify if the robot found a new item
        if self.cell_val == 1 and known_cell_val == False:
            self.found_item_flag = True
            return # Exit the function to choose a move and ask the server for the item type and owner

        # Marking the cell as explored
        self.explo[self.x, self.y] = 1

        # Updating believes
        self.believes = update_believes(self.x, self.y, self.cell_val, self.w, self.h, self.believes, known_cell_val)

    
    def explore_map(self):
        """
        Exploring algorithm of the robot

        Returns:
            tuple: the robot's move message
        """
        max_weighted_sum = 0
        for i in range(self.x-1, self.x+2): # For colums maximum one cell away
            for j in range(self.y-1, self.y+2): # For rows maximum one cell away
                if i in range(self.w) and j in range(self.h) and (i, j) != (self.x, self.y): # If the indexes are inside the map and the cell is not the current cell
                    
                    # Count number of cells with a belief of 1 around that cell (sum of 1/distances^2)
                    weighted_sum = 0 
                    
                    for ii in range(self.w): # For all columns in the map
                        for jj in range(self.h): # For all cells in this column
                            if self.believes[ii, jj] == 1: # If that cell has a belief of 1
                                distance2 = (ii-i)**2 + (jj-j)**2 # Calculate the squared distance between these cells
                                weighted_sum += 1/(distance2+0.0000001) # Greater weight for closer cells
                    
                    if weighted_sum > max_weighted_sum:
                        chosen_next_cell = (i, j)
                        max_weighted_sum = weighted_sum
                        
        # Convert cell coordinates into move
        return self.cell2move(chosen_next_cell)


    def choose_action(self):
        """ Choose a robot action """
        # Verify if the robot's quest is completed
        if self.box_reached is True:
            next_move = {"header": BROADCAST_MSG, "Msg type": COMPLETED, "position": (self.x, self.y), "owner": self.agent_id} # Broadcast that the quest is completed

        # Verify if an item has been found
        elif self.found_item_flag is True:
            self.found_item_flag = False
            next_move = {"header": GET_ITEM_OWNER} # Get item owner (Ask the server for the type and the owner of the item)

        # Verify if a message must be broadcasted
        elif self.broadcast_message_flag is True:
            self.broadcast_message_flag = False
            next_move = {"header": BROADCAST_MSG, "Msg type": self.found_item_type+1, "position": (self.x, self.y), "owner": self.found_item_owner}

        # Go collect the key if it is another robot that found it first
        elif self.key_position is not None and self.key_collected is False:
            next_move = self.go_towards_cell(self.key_position)

        # Go to the box if the key is collected and the position of the box is known
        elif self.box_position is not None and self.key_collected is True:
            next_move = self.go_towards_cell(self.box_position)
        
        # Find the best next move to explore the map     
        else:
            next_move = self.explore_map()
        
        # Execute chosen action
        self.network.send(next_move)


    def plot_believes(self, alpha=1.0, display=True):
        """
        Plot the believes' grid of the robot

        Args:
            alpha (float, optional): Opacity of the window (1.0=vivid, 0.0=transparent). Defaults to 1.0.
            display (bool, optional): display the grid on the screen. Defaults to True.
        """
        # Enable interactive mode to continue execution of the code after the plot is shown
        # Note: The "block" parameter on the "plt.show()" function doesn't seem to work on macOS.     
        if display: # If we want to display the plot         
            plt.ion()
        
        plt.figure(self.agent_id+1, figsize=(self.w/3, self.h/3))
        plt.clf() # Clear the matplotlib plot every time the robot moves
        
        # Creating colormap with explored cells and the agent's position
        colormap = np.copy(self.explo)
        colormap[self.x, self.y] = 3
        
        colors = ["Reds", "Blues", "Greens", "Oranges"]
        cmap = colors[self.agent_id]
        
        if self.key_position or self.box_position:
            if self.key_position:
                colormap[self.key_position[0], self.key_position[1]] = 2
            if self.box_position:
                colormap[self.box_position[0], self.box_position[1]] = 2
        
        plt.pcolormesh(np.flip(colormap.T, 0), cmap=cmap, edgecolors='k', vmin=0, vmax=3, alpha=alpha)
    
        # Creating array with cell values in visited cells and believes elsewhere
        cellval_or_belief = np.maximum(self.believes, self.cell_values)
        
        if not self.box_reached:   # If the box hasn't been reached yet    
            # Adding cell values or believes as annotations on every cell
            for i in range(self.w):
                for j in range(self.h):
                    if (i, self.w-(j+1)) == (self.x, self.y):
                        plt.annotate(str(round(np.flip(cellval_or_belief.T, 0)[j][i], 1)), xy=(i+0.5, j+0.5), ha='center', va='center', color='white')
                    else:
                        plt.annotate(str(round(np.flip(cellval_or_belief.T, 0)[j][i], 1)), xy=(i+0.5, j+0.5), ha='center', va='center', color='black')
            
            # Add total number of visited cells and other important information
            plt.annotate(f'Number of visited cells: {int(np.sum(self.explo))}', xy=(0,-1), color='black', annotation_clip=False)
            plt.annotate(f'Position of the key: {self.key_position}', xy=(0,-2), color='black', annotation_clip=False)
            plt.annotate(f'Position of the box: {self.box_position}', xy=(0,-3), color='black', annotation_clip=False)
            plt.annotate(f'Key collected: {self.key_collected}', xy=(0,-4), color='black', annotation_clip=False)
            plt.annotate(f'Box reached with key: {self.box_reached}', xy=(0,-5), color='black', annotation_clip=False)
            
        else:   # If the box has been reached
            text = "Congrats! The robot reached its box!\nPress any key in the terminal to exit..."
            plt.text(self.w/2, self.h/2, text, ha='center', va='center', color='black', fontsize=18, fontweight='bold')
            
        plt.axis('equal')
        plt.axis('off')
        plt.title(f'GridBelieves for robot {self.agent_id+1}')
        plt.tight_layout() # Reduce margins
        
        if display: # If we want to display the plot 
            plt.draw()
        
        foldername = "robot" + str(self.agent_id+1)
        full_path = os.path.join(IMG_PATH, foldername)
        
        if not os.path.exists(full_path):   # Check if the folder doesn't exist
            os.makedirs(full_path)          # Create the directory
        
        filename = "explo_" + str(int(np.sum(self.explo))) + ".jpg"
        plt.savefig(IMG_PATH + "/" + foldername + "/" + filename)
        
        if display:         # If we want to display the plot 
            plt.pause(0.2)  # Necessary for the plot to appear on macOS

    
    def pathGIF(self):
        import imageio
        import re
   
        foldername = "robot" + str(self.agent_id+1)
        full_path = os.path.join(IMG_PATH, foldername)
        
        if not os.path.exists(full_path):   # Check if the folder doesn't exist
            os.makedirs(full_path)          # Create the directory
        
        filenameGIF = "complete_path_robot_" + str(self.agent_id+1) + ".gif"
        GIF_path = IMG_PATH + "/" + foldername + "/" + filenameGIF   # GIF file relative path

        def natural_sort_key(string):    # Function to sort the images based on their number
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', string)]
        
        image_files = sorted(os.listdir(full_path), key=natural_sort_key) # List all files in the images directory
        images = [] # Create a list to store images
        
        # Read each image and append it to the images list
        for image_file in image_files:
            image_path = os.path.join(full_path, image_file)
            images.append(imageio.imread(image_path))
        
        imageio.mimsave(GIF_path, images)   # Create a GIF of the robot path
        print(f'GIF created and saved to {GIF_path}')


if __name__ == "__main__":
    import argparse
       
    IMG_PATH = "./Images"
    
    if not os.path.exists(IMG_PATH):    # Check if the folder doesn't exist
        os.makedirs(IMG_PATH)           # Create the directory

    for foldername in os.listdir(IMG_PATH): # Check if there is folders in the directory
        folder_path = os.path.join(IMG_PATH, foldername)
        delete_files(folder_path)   # Delete every files in the folder specified
        
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--server_ip", help="Ip address of the server", type=str, default="localhost")
    args = parser.parse_args()
    agent = Agent(args.server_ip)
    
    try:
        while not agent.box_reached:
            agent.explore_cell()
            agent.plot_believes(display=False)
            # time.sleep(0.5)
            agent.choose_action()
            time.sleep(0.1) # Added time sleep to allow for receiving incoming message in other thread before next iteration
        agent.plot_believes(alpha=0.5)
        input("Press any key to exit...")
        agent.pathGIF()
        
    except KeyboardInterrupt:
        pass
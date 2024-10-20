#'''
import os
import gymnasium as gym
from gymnasium.wrappers import FrameStack
import random
import matplotlib.pyplot as plt
import numpy as np
import pickle
import neat
import cv2 as cv
from matplotlib import pyplot as plt
from gymnasium.wrappers.resize_observation import ResizeObservation

N_FRAMES_TO_SKIP = 5 #Some odd number
PAUSE_BETWEEN_RENDERING_FRAMES = 0.01  # The number of seconds to pause between rendering frames

CROP_TOP_PIXELS = 30  # Adjust the number of pixels to crop from the top

STEPS_TAKEN = 2000 #should be 1000
NUMBER_OF_RUNS = 20

def find_asteroids(frame, num_outputs, show_plots=False):
    frame_cropped = frame[CROP_TOP_PIXELS:, :]
   
    # Convert the frame to grayscale
    gray_frame = cv.cvtColor(frame_cropped, cv.COLOR_RGB2GRAY)
    
    # Apply edge detection (you can use other edge detection methods as well)
    edges = cv.Canny(gray_frame, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if(show_plots):
        cv.imshow("Cropped Frame", frame_cropped)
        cv.imshow("Grayscale Frame", gray_frame)
        cv.waitKey(1)
        cv.imshow("Edges", edges)
        cv.waitKey(1)

    # Extract asteroid coordinates based on contour shape
    asteroid_coordinates = []
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.04 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)

        # Filter contours based on the number of vertices (asteroids might have certain shapes)
        if len(approx) >= 3:  # You may need to adjust this threshold based on the shape of asteroids
            # Get the bounding rectangle of the contour
            x, y, w, h = cv.boundingRect(contour)

            # Filter based on aspect ratio (you may need to adjust this threshold)
            aspect_ratio = w / h
            if 0.5 <= aspect_ratio <= 2.0:
                centroid_x = x + w // 2
                centroid_y = y + h // 2
                asteroid_coordinates.append(centroid_x)
                asteroid_coordinates.append(centroid_y+CROP_TOP_PIXELS)
    while(len(asteroid_coordinates) < num_outputs):
        asteroid_coordinates.append(0)
    return asteroid_coordinates


#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
with open('fitness2150.0.pickle', 'rb') as f:
    neat_data = pickle.load(f)
print(neat_data)
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

best_net = neat.nn.FeedForwardNetwork.create(neat_data, config)

observation_shape = (210,160)
env = gym.make('ALE/Asteroids-v5', frameskip=N_FRAMES_TO_SKIP, render_mode="human")
env = ResizeObservation(env, shape=observation_shape)
env = FrameStack(env, 2)
env.action_space.seed(42)


scores = [] #list to hold the scores of each game

for i in range(NUMBER_OF_RUNS): #play 20 games
    observation, info = env.reset() #reset the game to the starting point
    total_score = 0
    for _ in range(STEPS_TAKEN): #play for 1000 timesteps or until player dies
        env.render() #render the game
        current_observation = observation[0]
        previous_observation = observation[1]
        combined_observation = current_observation + previous_observation
        coordinates_astroids = find_asteroids(combined_observation,config.genome_config.num_inputs)#, show_plots=True)

        sel_control = best_net.activate(coordinates_astroids) #get the action from the neural network
        action = int(np.argmax(sel_control)) #convert the action to an integer
        observation, reward, terminated, truncated, info = env.step(action) #take a random action each step from the allotted controls
        total_score += reward #add the reward to the total score. reward is based on the action taken in the previous step such as shooting an asteroid

        if terminated or truncated: #if the player dies or the game is truncated, reset the game
            #save player score
            scores.append(total_score) #add the score to the list of scores
            observation, info = env.reset() #reset the game to the starting point


    #'''
    print("Player Score: ", total_score)
env.close() #close the game environment
avg_score = np.mean(scores) #calculate the average score
std_score = np.std(scores) #calculate the standard deviation of the scores

plt.bar("Bot",avg_score, yerr=std_score, align='center', alpha=0.5, ecolor='black', capsize=10) #plot the average score with the standard deviation
plt.title("Random Agent Scores")
plt.ylabel("Scores")
plt.xlabel("Agent")
plt.show()
#Uncomment if wanting to play the game directly
'''
import gymnasium as gym
import pygame
import shimmy
from gym.utils.play import play
mapping = {(pygame.K_LEFT,): 4, (pygame.K_UP,): 2, (pygame.K_RIGHT,): 3, (pygame.K_SPACE,): 1}
play(gym.make("ALE/Asteroids-v5", render_mode="rgb_array"), keys_to_action=mapping)
'''



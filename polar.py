#!/usr/local/bin/python3
#
# Authors
# Hithesh Busetty
# Dinesh challa
# Ujwala Shenoy
#
# Ice layer finder
# Based on skeleton code by D. Crandall, November 2021
#

from math import log
from copy import deepcopy
from PIL import Image       
from numpy import *
from scipy.ndimage import filters
import sys
import imageio

# calculate "Edge strength map" of an image                                                                                                                                      
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return sqrt(filtered_y**2)

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_boundary(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( int(max(y-int(thickness/2), 0)), int(min(y+int(thickness/2), image.size[1]-1 )) ):
            image.putpixel((x, t), color)
    return image

def draw_asterisk(image, pt, color, thickness):
    for (x, y) in [ (pt[0]+dx, pt[1]+dy) for dx in range(-3, 4) for dy in range(-2, 3) if dx == 0 or dy == 0 or abs(dx) == abs(dy) ]:
        if 0 <= x < image.size[0] and 0 <= y < image.size[1]:
            image.putpixel((x, y), color)
    return image


# Save an image that superimposes three lines (simple, hmm, feedback) in three different colors 
# (yellow, blue, red) to the filename
def write_output_image(filename, image, simple, hmm, feedback, feedback_pt):
    new_image = draw_boundary(image, simple, (255, 255, 0), 2)
    new_image = draw_boundary(new_image, hmm, (0, 0, 255), 2)
    new_image = draw_boundary(new_image, feedback, (255, 0, 0), 2)
    new_image = draw_asterisk(new_image, feedback_pt, (255, 0, 0), 2)
    imageio.imwrite(filename, new_image)



def simple_bayes(edge_strength, image_array):
    no_rows, no_cols = edge_strength.shape
    
    intensity_probab = image_array/255 #255 is the max value for intensity
    simple_air_ice = list()
    simple_ice_rock = list()
    for col in range(no_cols):
        # print(col)
        col_list = edge_strength[:, col]
        max_col_edge_strength = max(col_list)
        col_emission_prob = col_list/max_col_edge_strength  #emission probability
        # intensity_probab_col = intensity_probab[:,col]
        # col_list_prob = multiply(col_list_prob, intensity_probab_col)

        #sorting the emission probability array. enumerate helps to retain original index after sorting. 
        sorted_col_list = [i for i in sorted(
            enumerate(col_emission_prob), key=lambda x:x[1], reverse=True)]  
        air_ice = sorted_col_list[0] #first max is the air-ice boundary
        ice_rock = ()
        # iterating oveer next maximum elements in sorted and array 
        # to find a pixel which is 10 units away from air-ice boundary.
        for i in sorted_col_list[1:]: 
            if(i[0] > air_ice[0]+10):
                ice_rock = i
                break
        simple_air_ice.append(air_ice)
        simple_ice_rock.append(ice_rock)
    return [x[0] for x in simple_air_ice], [x[0] for x in simple_ice_rock]


# import math
def hmm_viterbi(edge_strength,image_array,simple_air_ice_initial,simple_ice_rock_intial):
    no_rows, no_cols = edge_strength.shape
    hmm_air_ice, hmm_ice_rock = list(), list()
    #taking initial state of HMM as output from simple bayes
    hmm_air_ice.append((simple_air_ice_initial,0))
    hmm_ice_rock.append((simple_ice_rock_intial,0))
    for col in range(1, no_cols):
        col_list = edge_strength[:, col]
        max_col_edge_strength = max(col_list)
        col_list_prob = col_list/max_col_edge_strength

        #air_ice
        hmm_prev_air_ice = hmm_air_ice[col-1]
        col_air_ice_hmm = list()
        for i in range(len(col_list_prob)):
            if(hmm_prev_air_ice[0] == i):
                transition_prob = 1
            else:
                #different emission probabilities are implemented
                transition_prob = (1/abs(hmm_prev_air_ice[0]-i))
                # transition_prob = -1*log(abs(hmm_prev_air_ice[0]-i)/no_rows)
            # transition_prob = (no_rows-abs(hmm_prev_air_ice[0]-i) /no_rows)
            transition_prob = abs(hmm_prev_air_ice[0]-i)/no_rows
            #col_ice_rock_hmm.append(col_list_prob[i] *transition_prob) #Viterbi implementation
            col_air_ice_hmm.append(
                col_list_prob[i] - transition_prob)
        sorted_col_list = [i for i in sorted(
            enumerate(col_air_ice_hmm), key=lambda x:x[1], reverse=True)]
        hmm_air_ice_current = sorted_col_list[0]
        hmm_air_ice.append(hmm_air_ice_current)

        #ice_rock
        hmm_prev_ice_rock = hmm_ice_rock[col-1]
        col_ice_rock_hmm = list()
        for i in range(len(col_list_prob)):
            #additional step to make all values until hmm_air_ice_current +10 pixels as 0
            if(i < hmm_air_ice_current[0]+10):
                col_ice_rock_hmm.append(-2222)
            else:
                if(hmm_prev_ice_rock[0] == i):
                    transition_prob = 1
                else:
                    # transition_prob=-1*(abs(hmm_prev_ice_rock[0]-i)/no_rows)
                    transition_prob = (1/abs(hmm_prev_ice_rock[0]-i))
                # transition_prob = (no_rows-abs(hmm_prev_ice_rock[0]-i) / no_rows)
                transition_prob = abs(hmm_prev_ice_rock[0]-i)/no_rows
                #col_ice_rock_hmm.append(col_list_prob[i] *transition_prob)
                col_ice_rock_hmm.append(col_list_prob[i] - transition_prob)
        sorted_col_list = [i for i in sorted(
            enumerate(col_ice_rock_hmm), key=lambda x:x[1], reverse=True)]
        hmm_ice_rock_current = sorted_col_list[0]
        hmm_ice_rock.append(hmm_ice_rock_current)
    return [x[0] for x in hmm_air_ice], [x[0] for x in hmm_ice_rock]


def hmm_human_feedback(edge_strength, image_array,human_air_ice_x, human_air_ice_y, human_ice_rock_x, human_ice_rock_y,simple_air_ice_initial,simple_ice_rock_initial):
    no_rows, no_cols = edge_strength.shape
    air_ice_list, ice_rock_list = simple_bayes(edge_strength=edge_strength,image_array=image_array)
    hmm_air_ice, hmm_ice_rock = list(), list()
    #taking initial state of HMM as output from simple bayes
    hmm_air_ice.append((simple_air_ice_initial,0))
    hmm_ice_rock.append((simple_ice_rock_initial,0))
    for col in range(1, no_cols):
        col_list = edge_strength[:, col]
        max_col_edge_strength = max(col_list)
        col_list_prob = col_list/max_col_edge_strength

        #air_ice
        hmm_prev_air_ice = hmm_air_ice[col-1]

        col_air_ice_hmm = list()
        for i in range(len(col_list_prob)):
            human_heuristic_airice = (abs(human_air_ice_y-i)/no_cols)
            emission_probab = (abs(hmm_prev_air_ice[0]-i)/no_rows)
            #col_ice_rock_hmm.append(col_list_prob[i] * emission_probab_icerock * human_heuristic_icerock)
            col_air_ice_hmm.append(
                col_list_prob[i] - emission_probab-human_heuristic_airice)
        sorted_col_list = [i for i in sorted(
            enumerate(col_air_ice_hmm), key=lambda x:x[1], reverse=True)]
        hmm_air_ice_current = sorted_col_list[0]
        hmm_air_ice.append(hmm_air_ice_current)

        #ice_rock
        hmm_prev_ice_rock = hmm_ice_rock[col-1]
        col_ice_rock_hmm = list()
        for i in range(len(col_list_prob)):
            #additional step to make all values until hmm_air_ice_current +10 pixels as -2222 
            # so that they wont be picked for ice-rock boundary
            if(i < hmm_air_ice_current[0]+10):
                # col_ice_rock_hmm.append(float(-inf))
                col_ice_rock_hmm.append(-2222)
            else:
                human_heuristic_icerock = (abs(human_ice_rock_y-i)/no_cols)
                emission_probab_icerock = (abs(hmm_prev_ice_rock[0]-i)/no_rows)
                #col_ice_rock_hmm.append(col_list_prob[i] * emission_probab_icerock * human_heuristic_icerock)
                col_ice_rock_hmm.append(
                    col_list_prob[i] - emission_probab_icerock-human_heuristic_icerock)
        sorted_col_list = [i for i in sorted(
            enumerate(col_ice_rock_hmm), key=lambda x:x[1], reverse=True)]
        hmm_ice_rock_current = sorted_col_list[0]
        hmm_ice_rock.append(hmm_ice_rock_current)
    return [x[0] for x in hmm_air_ice], [x[0] for x in hmm_ice_rock]


# main program
#
if __name__ == "__main__":

    if len(sys.argv) != 6:
        raise Exception("Program needs 5 parameters: input_file airice_row_coord airice_col_coord icerock_row_coord icerock_col_coord")

    input_filename = sys.argv[1]
    gt_airice = [ int(i) for i in sys.argv[2:4] ]
    gt_icerock = [ int(i) for i in sys.argv[4:6] ]

    # load in image 
    input_image = Image.open(input_filename).convert('RGB')
    image_array = array(input_image.convert('L'))

    # compute edge strength mask -- in case it's helpful. Feel free to use this.
    edge_strength = edge_strength(input_image)
    imageio.imwrite('edges.png', uint8(255 * edge_strength / (amax(edge_strength))))

    # You'll need to add code here to figure out the results! For now,
    # just create some random lines.
    airice_simple, icerock_simple = simple_bayes(
        edge_strength=edge_strength,
        image_array=image_array
    )
    airice_hmm, icerock_hmm = hmm_viterbi(
        edge_strength=edge_strength,
        image_array=image_array,
        simple_air_ice_initial=airice_simple[0],
        simple_ice_rock_intial=icerock_simple[0]
    )
    airice_feedback, icerock_feedback = hmm_human_feedback(
        edge_strength=edge_strength,
        image_array=image_array,
        human_air_ice_x=gt_airice[0],
        human_air_ice_y=gt_airice[1],
        human_ice_rock_x=gt_icerock[0],
        human_ice_rock_y=gt_icerock[1],
        simple_air_ice_initial=airice_simple[0],
        simple_ice_rock_initial=icerock_simple[0]
    )

    
    # Now write out the results as images and a text file
    write_output_image("air_ice_output.png", input_image, airice_simple, airice_hmm, airice_feedback, gt_airice)
    write_output_image("ice_rock_output.png", input_image, icerock_simple, icerock_hmm, icerock_feedback, gt_icerock)
    with open("layers_output.txt", "w") as fp:
        for i in (airice_simple, airice_hmm, airice_feedback, icerock_simple, icerock_hmm, icerock_feedback):
            fp.write(str(i) + "\n")


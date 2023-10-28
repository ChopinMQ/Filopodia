import Functions_c
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import math
from typing import Optional
from scipy.fftpack import fft2, ifft2
import networkx as nx
import matplotlib.pyplot as plt
from numba import jit
import os
import scipy.io
import matplotlib.colors as mcolors
import operator
from scipy.spatial.distance import euclidean

from skimage.morphology import skeletonize, medial_axis, thin
from skimage import morphology
from scipy.spatial.distance import euclidean
from collections import defaultdict
from itertools import chain
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# this code needs a whole image, and the predicted grayscale image of filopodia
# the position of the filopodia: 0, 1, 2, 3
# 0: top, 1: right, 2:bottom, 3: left
# and a parameter to control the usage of microglia cell removal, True: use, False: not use
# zoom factor
# training excel files
# colorful filopodia images to mark on
# image save directory
# a probability for different pictures

def process(path_whole, path_gray,
            pos, mircog, zoom,
            excel1, excel2,
            colorful_img, direct,
            p):

    img = Image.open(path_whole)
    img = np.asarray(img)
    image = img.astype("uint8")
    ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # get the noised whole image's topology information
    df_whole_uncleaned = Functions_c.extract_topo(binary,
                                                'Finish information extraction of whole image(uncleaned)')
    # first, clean the whole image
    whole_clean = Functions_c.isolated_pixesl_remove(binary,
                                                   df_whole_uncleaned,
                                                   'Finish cleaning of whole image')
    
    df_whole_cleaned = Functions_c.extract_topo(whole_clean,
                           'Finish information extraction of whole image(cleaned)')
    df_whole_cleaned['new_node1'] = df_whole_cleaned['node1'].apply(lambda x: x)
    df_whole_cleaned['new_node2'] = df_whole_cleaned['node2'].apply(lambda x: x)
    # find the largest ring
    G = nx.Graph()
    count = 0
    for coor in df_whole_cleaned.nodespair:
        temp_c = coor
        first = temp_c[0]
        second = temp_c[1]
        f = (first[1], first[0])
        s = (second[1], second[0])
        G.add_edge(f, s)
        G[f][s]['weight'] = float(df_whole_cleaned.length[count])
        count += 1

    # use area to compare
    contours, _ = cv2.findContours(image=whole_clean, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours, key=cv2.contourArea)
    # get the degree of nodes and keep the ones that near the contour
    node_degrees = dict(G.degree())
    degreeonetwo = []
    for node, degree in node_degrees.items():
        if degree < 5:
            # find if it's in or out
            inorout = cv2.pointPolygonTest(largest_contour, node, True)
            # 100 pixels to test
            if inorout < 100:
                degreeonetwo.append(node)
            
    # use a rectangular to estimate the center of the filopodia
    contour_rec, _ = cv2.findContours(whole_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_rect = cv2.minAreaRect(contour_rec[0])
    min_rect_box = cv2.boxPoints(min_rect)

    xm = (min(min_rect_box[:,0]) + max(min_rect_box[:,0])) / 2
    ym = (min(min_rect_box[:,1]) + max(min_rect_box[:,1])) / 2
    mid = (xm, ym)
    # not very far from the middle points
    # set diameter as
    d1 = -min(min_rect_box[:,0]) + max(min_rect_box[:,0])
    d2 = -min(min_rect_box[:,1]) + max(min_rect_box[:,1])
    diam = (d1 + d2) / 4 / 2

    tip = []
    for node in degreeonetwo:
        if cv2.norm(node, mid, cv2.NORM_L2) >= diam:
            tip.append(node)

    df_whole_cleaned['new_node1'] = df_whole_cleaned['node1'].apply(lambda x: x)
    df_whole_cleaned['new_node2'] = df_whole_cleaned['node2'].apply(lambda x: x)
    subG = G.subgraph(tip)

    # next step is to do template matching
    # use the zoomed image to match this image and mark the tip cells in the zoomed image
    # find the rectangular to include this and find the center area

    part = Image.open(path_gray)
    part = np.asarray(part)
    part = part.astype("uint8")

    # use whole process to get a clean filopodia image
    part1 = cv2.GaussianBlur(part, (3,3), 0)
    iter1 = Functions_c.iterative_thresh(1, part1,
                                'Finish the thresholding of uncleaned filopodia image with iteration 2')
    

    iter10 = Functions_c.iterative_thresh(10, part1,
                                'Finish the thresholding of uncleaned filopodia image with iteration 11')

    # use the uncleaned image to get topology information
    df_iter1_unclean = Functions_c.extract_topo(iter1,
                                                'Finish extracting topology information for uncleaned image')

    filo_clean = Functions_c.isolated_pixesl_remove(iter1,
                                                    df_iter1_unclean,
                                                    'Finish cleaning of filopodia image')
    df_clean = Functions_c.extract_topo(filo_clean,
                                        'Finish extracting the topology information of cleaned filopodia image')
    

    # now decide whether to use microglia cell removal
    if mircog == True:
        part = Functions_c.micro(filo_clean, df_clean, part,
                          'Finish microglia cell detection')
    else:
        pass

    # now we have a cleaned filopodia image and a removed/not removed grayscale image
    # use iter10 to drop the obvious microglia cell and do iterative thresholding again
    df_iter10 = Functions_c.extract_topo(iter10,
                                        'Finish extracting topology information for uncleaned image')
    
    filo10_clean = Functions_c.isolated_pixesl_remove(iter10,
                                                    df_iter10,
                                                    'Finish cleaning of filopodia image')
    part = Functions_c.replace(filo10_clean, iter10, part)
    part = part.astype(np.uint8)

    #clahe
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8,8))
    final_img = clahe.apply(part)
    final_img = final_img.astype(np.uint8)
    blur = Functions_c.adjust_gamma(final_img, 1.5)
    
    blur = cv2.GaussianBlur(blur,(3,3),0)
    filo_final = Functions_c.iterative_thresh(1, blur,
                                        'Finish thresholding of final filopodia image')
    
    # do microglia cell removal and feature extraction

    df_final = Functions_c.extract_topo(filo_final,
                                                'Finish extracting topology information for uncleaned filopodia image')

    filo_final = Functions_c.isolated_pixesl_remove(filo_final,
                                                    df_final,
                                                    'Finish cleaning of filopodia image, finally')
    df_final = Functions_c.extract_topo(filo_final,
                                        'Finish extracting the topology information of cleaned filopodia image, finally')
    
    search = filo_final.copy()
    # Calculate the scaling factor
    scaling_factor = 1 / zoom

    # Calculate the new dimensions of the resized image
    new_width = int(search.shape[1] * scaling_factor)
    new_height = int(search.shape[0] * scaling_factor)
    resized_image = cv2.resize(search, (new_width, new_height))

    # get the dimensions of image B (zoomed)
    height_B, width_B = resized_image.shape
    # template matching
    result = cv2.matchTemplate(binary, resized_image, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    top_left = max_loc
    top_right = (top_left[0] + width_B, top_left[1])
    bottom_right = (top_left[0] + width_B, top_left[1] + height_B)
    bottom_left = (top_left[0], top_left[1] + height_B)
    # clockwise order
    inside = []
    rectangle_point = np.array([top_left, top_right, bottom_right, bottom_left])

    # already change the sequence
    node_zoom = []
    # already change the sequence
    curve_zoom = []
    for coordinate in list(subG.edges):
        l = len(coordinate[0])
        for j in range(l-1):
            # get the polylines
            t1 = coordinate[j]
            t2 = coordinate[j+1]
            first = (df_whole_cleaned.new_node1 == (t1[1], t1[0])) & (df_whole_cleaned.new_node2 == (t2[1], t2[0]))
            second = (df_whole_cleaned.new_node1 == (t2[1], t2[0])) & (df_whole_cleaned.new_node2 == (t1[1], t1[0]))
            temp_data = df_whole_cleaned[first | second].curve
            string_data = temp_data.tolist()
            container = string_data[0]
            switched_data = [(y, x) for x, y in container]
            for c in switched_data:
                distance = cv2.pointPolygonTest(rectangle_point, c, False)
                if distance >= 0:
                    curve_zoom.append(c)

            for j in range(l):
                t = coordinate[j]
                node_zoom.append((t[0], t[1]))

        
    # change the coordinates of curves
    coor_zoomed_cur = []
    for coor in curve_zoom:
        x = coor[0] - top_left[0]
        y = coor[1] - top_left[1]
        coor_zoomed_cur.append((x, y))

    # change the coordinates for nodes
    coor_zoomed_nod = []
    for coor in node_zoom:
        x = coor[0] - top_left[0]
        y = coor[1] - top_left[1]
        coor_zoomed_nod.append((x, y))

    binary_p = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
    h, w, ch = binary_p.shape

    lh, lw = filo_final.shape
    # draw the curves on a black image
    black = np.zeros((h, w, ch))
    rad = 2
    black = black.astype("uint8")
    for (v, u) in coor_zoomed_cur:
        black = cv2.rectangle(black, (v, u), (v, u), (0, 0, 255), -1)
    for (v, u) in coor_zoomed_nod:
        black = cv2.rectangle(black, (v-rad, u-rad), (v+rad, u+rad), (0, 0, 255), -1)
    resize_black = cv2.resize(black, (lw, lh))
    uint_img = resize_black.astype('uint8')
    grayImage = cv2.cvtColor(uint_img, cv2.COLOR_BGR2GRAY)

    threshold_value = 0.000001  # Adjust this threshold value as needed
    _, binary_image = cv2.threshold(grayImage, threshold_value, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(binary_image,)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    result_image = cv2.bitwise_and(binary_image, mask)

    border_size = 1
    border_color = [255, 255, 255]
    img_with_border = cv2.copyMakeBorder(result_image,
                                        border_size,
                                        border_size,
                                        border_size,
                                        border_size,
                                        cv2.BORDER_CONSTANT,
                                        value=border_color)
    
    contours, _ = cv2.findContours(img_with_border, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # use different test point to find the boundary
    if pos == 0:
        # it's for the upper part
        test_point1 = [int(lw/2), 5]
        test_point2 = [int(lw/2), lh - 5]
    elif pos == 1:
        test_point2 = [5, int(lh/2)]
        test_point1 = [lw - 5, int(lh/2)]
    elif pos == 2:
        test_point2 = [int(lw/2), 5]
        test_point1 = [int(lw/2), lh - 5]
    elif pos == 3:
        test_point1 = [5, int(lh/2)]
        test_point2 = [lw - 5, int(lh/2)]

    need_ct = []
    # find the contour that this point is in
    for ct in contours:
        # +1 outside
        distance1 = cv2.pointPolygonTest(ct, test_point1, False)
        distance2 = cv2.pointPolygonTest(ct, test_point2, False)
        if distance1 > 0 and distance2 < 0:
            # choose this contour
            need_ct.append(ct)

    # now we have already found the tip cells' position
    # next step is to use the prelabeled data to connect the truncated filopodia
    # step 1 find the possible tip cells using ML methods
    # step 2 use the topology of a likelihood \cite{Xiaoling Hu, Fuxin Li, Dimitris Samaras, Chao Chen: "Topology-Preserving Deep Image Segmentation", in the Thirty-third Conference on Neural Information Processing Systems (NeurIPS), 2019.}
    # to guess if they should be connected, use the (gap distance, depth, variance)
    # step 3 use the fixed image to do filopodia detection (use the same random forest method)
    # step 4 if the detected nodes are connected then only keep the first node

    dist = cv2.distanceTransform(filo_final, cv2.DIST_L2, 3)
    df_final['new_node1'] = df_final['node1'].apply(lambda x: x)
    df_final['new_node2'] = df_final['node2'].apply(lambda x: x)
    G = nx.Graph()
    for coor in df_final.nodespair:
        temp_c = coor
        first = temp_c[0]
        second = temp_c[1]
        f = (first[1], first[0])
        s = (second[1], second[0])
        G.add_edge(f, s)

    df_c = pd.read_excel(excel1)
    df_c = df_c.iloc[:, 1:]
    df_n = pd.read_excel(excel2)
    df_test1 = df_c[np.isnan(df_c.Type)==True]
    df_train1 = df_c[df_c.Type==1]
    df_test2 = df_n[np.isnan(df_n.Type)==True]
    df_train2 = df_n[np.isnan(df_n.Type)==False]

    df_train2 = pd.concat([df_train1, df_train2])


    df_train_x = df_train2.iloc[:, :-1].to_numpy()
    df_train_y = df_train2.iloc[:, -1].to_numpy()

    clf = RandomForestClassifier(max_depth=15, random_state=7, n_estimators=100, class_weight='balanced')

    clf.fit(df_train_x, df_train_y)
    print('Finish training of random forest')
    print('processing...')
    need_ct.sort(key=len, reverse=True)
    print(len(need_ct[0]))
    # find all possible edges
    nodes_po = []
    for edge in G.edges:
        node1 = edge[0]
        node2 = edge[1]

        result1 = cv2.pointPolygonTest(need_ct[0], node1, False)
        result2 = cv2.pointPolygonTest(need_ct[0], node2, False)

        if (result1 >= 0 or result2 >= 0):
            # it's in or on the boundary, keep it
            nodes_po.append(edge)


    # get all of the characters
    flag = 0
    # if the curve is error then such curve is definitely not a filopodia, too long
    error_curves = []
    character = pd.DataFrame()
    pos = 0
    for edge in nodes_po:
        if flag == 0:
            info = character_function(edge, G, df_final, dist)
            if info != -1:
                character = pd.DataFrame([info])
                flag = 1
            else:
                error_curves.append(pos)
        else:
            info =  character_function(edge, G, df_final, dist)
            if info != -1:
                info = pd.DataFrame([info])
                character = pd.concat([character, info], ignore_index=True)
            else:
                error_curves.append(pos)
            pos += 1
    
    edge_all = [nodes_po[i] for i in range(len(nodes_po)) if i not in error_curves]
    print(len(edge_all))
    # get all nodes1
    final_nodes = []
    special = []
    position = 0
    two_point = []
    proba = []
    pos_list = []
    for edge in edge_all:
        # if any node is one degree then use a method to judge
        # if tow nodes are not one degree, then use another method
        nodea = edge[0]
        nodeb = edge[1]
        info = character_function(edge, G, df_final, dist)
        info = pd.DataFrame([info])
        info = info.to_numpy()
        pred = clf.predict(info.reshape(1, -1))
        pred_proba = clf.predict_proba(info.reshape(1, -1))
        deg1 = G.degree(nodea)
        deg2 = G.degree(nodeb)
        proba.append(pred_proba)
        if pred == 1:
            pos_list.append(position)
            if deg1 == 1:
                final_nodes.append(nodea)
            elif deg2 == 1:
                final_nodes.append(nodeb)
            else:
                node1 = nodea
                node2 = nodeb
                special.append(node1)
                special.append(node2)

        elif pred == 2:
            two_point.append(position)
            final_nodes.append(nodea)
            final_nodes.append(nodeb)

        position += 1
    
    subgraph_comp = G.subgraph(special)
    components = list(nx.connected_components(subgraph_comp))

    final_nodev2 = []
    for cn in components:
        length = len(cn)
        if length == 1:
            # keep it
            final_nodev2.append(list(cn)[0])
        else:
            distance_list = []
            for n in cn:
                distance_list.append(dist[n[1], n[0]])
            pos = np.argmin(distance_list)
            final_nodev2.append(list(cn)[pos])
        
    # for all nodes, get the nodes that within depth < 5 then if it degree is 1, regard as a filopodia
    nodes_from1 = []
    for node in final_nodes:
        # get the neighbors that depth is < 9
        ego_graph = nx.ego_graph(G, node, radius=8)
        for poss in ego_graph.nodes:
            if G.degree(poss) == 1:
                nodes_from1.append(poss)

    for node in final_nodev2:
        # get the neighbors that depth is < 9
        ego_graph = nx.ego_graph(G, node, radius=8)
        for poss in ego_graph.nodes:
            if G.degree(poss) == 1:
                nodes_from1.append(poss)

    final_node1 = []
    for n in set(nodes_from1):
        another = list(G.neighbors(n))[0]
        edge = [n, another]
        info = character_function(edge, G, df_final, dist)
        info = pd.DataFrame([info])
        info = info.to_numpy()
        pred_proba = clf.predict_proba(info.reshape(1, -1))
        if pred_proba[0][1] > p:
            final_node1.append(n)

    img = Image.open(colorful_img)
    imarray = np.array(img)
    height, width, channel = imarray.shape
    inteh = int(height/520)
    intew = int(width/520)
    tol = np.zeros((inteh*500, intew*500, channel))
    imarray = imarray[int((height-inteh*520)/2):int((height-inteh*520)/2)+1+inteh*520,int((width-intew*520)/2):int((width-intew*520)/2)+1+intew*520, :]
    for i in range(inteh):
        for j in range(intew):
            if (i == 0 and j != 0):
                imneed = imarray[0:520, 500*j-10:500*(j+1) +10, :]
            elif (i != 0 and  j == 0):
                imneed = imarray[500*i-10:500*(i+1)+10, 0:520, :]
            elif (i == 0 and j == 0):
                imneed = imarray[0:520, 0:520, :]
            else:
                imneed = imarray[500*i-10:500*(i+1)+10,500*j-10:500*(j+1) +10,:]

            if(i == 0 and j == 0):
                tol[i*500:500+i*500,j*500:j*500+500,:] = imneed[0:500, 0:500,:]
            elif(i == 0 and j != 0):
                tol[i*500:500+i*500,j*500:j*500+500,:] = imneed[0:500, 9:509,:]
            elif(i != 0 and j == 0):
                tol[i*500:500+i*500,j*500:j*500+500,:] = imneed[9:509, 0:500,:]
            else:
                tol[i*500:500+i*500,j*500:j*500+500,:] = imneed[9:509, 9:509,:]

    imarray = np.array(tol)
    image = imarray.astype("uint8")
    for final_n in final_node1:
        #for final_n in outsidenodes:
        image = cv2.circle(image, final_n, radius=2, color=(0, 0, 255), thickness=-1)


    cv2.imwrite(direct, image)

    print("Finish counting, perfect!!!!")
    return len(final_node1)




def character_function(edge, gr, data, dist):
    G = gr
    # the edge looks like this: ((1192, 136), (1185, 140))
    # return all variables

    node1 = edge[0]
    node2 = edge[1]

    judge1 = (data.new_node1 == (node1[1], node1[0])) & (data.new_node2 == (node2[1], node2[0]))
    judge2 = (data.new_node2 == (node1[1], node1[0])) & (data.new_node1 == (node2[1], node2[0]))

    pos = np.where(judge1 | judge2)
    try:
        curvep = data.iloc[pos[0][0]].curve

        width_var = float(data.iloc[pos[0][0]].width_var)
        tortuosity = float(data.iloc[pos[0][0]].tortuosity)

        points_list = []
        sum = 0
        for point in curvep:
            points_list.append(dist[point[0], point[1]])
            sum += dist[point[0], point[1]]

        length = len(points_list)
        # direct distance in euclidean space
        eud = np.linalg.norm(np.array(node1) - np.array(node2))

        # average width
        average_width = sum / length

        # median distance to black area
        median_width = np.median(points_list)

        # upper quantile
        uppql = np.quantile(points_list, 0.75)

        # bump size
        head = np.argmax(np.array(points_list) > 2)
        behind = np.argmax(np.array(points_list)[::-1] > 2)
        bumpsize = max(head, behind)
        if head == 0 and behind == 0:
            bumpsize = len(points_list)

        # length and width ratio of the bumpsize
        if max(head, behind) == head:
            ratio_lw = max(head, behind) / np.sum(points_list[:head+1])
        else:
            reve = points_list[::-1]
            ratio_lw = max(head, behind) / np.sum(reve[:behind+1])

        # topology characters
        deg1 = G.degree(node1)
        deg2 = G.degree(node2)

        dict = {'width': average_width,
                'width_var': width_var,
                'tortuosity': tortuosity,
                'length': length, 
                'direct': eud, 
                'median_width': median_width, 
                'upperq': uppql,
                'bumpsize': bumpsize, 
                'ratio': ratio_lw, 
                'degree1': deg1, 
                'degree2':deg2}
        return dict
    except:
        return -1
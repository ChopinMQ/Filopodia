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
import ast



# removal of isolated pixles functions
def inner_loop(pt, contours):
    cts = []
    flag = 0
    seq = 0
    for ct in contours:
        seq += 1
        result = cv2.pointPolygonTest(ct, pt, False)
        if result >= 0:
            # it's inside the contour or on the boundary
            cts.append(ct)
            flag = 1
        if seq == len(contours) and flag == 1:
            return [1, cts]
        elif seq == len(contours) and flag == 0:
            return [0]

# find whether we need fill this ct using black or white
def inside_coordinate(masks, ct, ima):
    u, v = np.where(masks == 0)
    pointsblack = []
    pointswhite = []
    allp = []
    for i in range(len(u)):
        pos = (int(v[i]), int(u[i]))
        pointswhite.append(pos)
        if cv2.pointPolygonTest(ct, pos, measureDist=False) > 0:
            pointsblack.append(pos)
            allp.append(ima[u, v])
    if np.mean(allp) > 250:
        # all white
        return pointswhite
    else:
        return pointsblack

def isolated_pixesl_remove(img, data, message):
    # this function returns a cleaned image using the topology information
    # and a binary image
    image = img.astype("uint8")
    ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # create a graph
    G = nx.Graph()

    for coor in data.nodespair:
        temp_c = coor
        first = temp_c[0]
        second = temp_c[1]
        f = (first[1], first[0])
        s = (second[1], second[0])
        G.add_edge(f, s)

    x = [ val[0] for val in list(G.nodes())]
    y = [ val[1] for val in list(G.nodes())]

    # get a list of connected small graphs
    components = list(nx.connected_components(G)) 
    # sort them and only keep the largest in the first position
    components.sort(key=len, reverse=True)
    largest = components.pop(0)
    # find the contours in this picture
    contours, _ = cv2.findContours(image=binary, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)

    binary_copy = binary.copy()
    count = 0
    for node_list in components:
        count += 1
        ele = list(node_list)
        try:
            for pt in ele:
                res = inner_loop(pt, contours)
                if res[0] == 1:
                    # it has a contour
                    # find the smallest
                    ar = []
                    tempc = res[1]
                    for ct in tempc:
                        area = cv2.contourArea(ct)
                        ar.append(area)
                    # get the index of the minimum
                    ind = np.argmin(ar)
                    cont_need = tempc[ind]
                    # fill this contour with black pixels
                    cv2.drawContours(binary_copy, [cont_need], -1, (0, 0, 0), thickness=cv2.FILLED)
        except:
            pass

    final_cont, _ = cv2.findContours(image=binary_copy, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    maxc = max(final_cont, key=cv2.contourArea)
    binary_f = binary_copy.copy()
    for ct in final_cont:
        if len(ct) != len(maxc):
            cv2.drawContours(binary_f, [ct], -1, (0, 0, 0), thickness=cv2.FILLED)

    final_cont, _ = cv2.findContours(image=binary_f, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    binary_final = binary_f.copy()
    for ct in final_cont:
        ar = cv2.contourArea(ct)
        if ar < 200 and ar > 4:
            img_m = np.ones(binary_final.shape)*255
            img_m = img_m.astype("uint8")
            masks = cv2.drawContours(img_m, [ct], -1, (0, 0, 0), thickness=cv2.FILLED)
            p = inside_coordinate(masks, ct, binary_final)
            for coor in p:
                v, u = coor
                binary_final = cv2.rectangle(binary_final, (v, u), (v, u), 0, -1)
        elif ar <= 4:
            cv2.drawContours(binary_final, [ct], -1, 0, thickness=cv2.FILLED)


    # use this message to show the step
    print(message)
    print('processing...')
    return binary_final

# iterative thresholding
# use numba to speeeeed up!

@jit(nopython=True)
def step1(h, w, mat1, mat2):
    for i in range(h):
        for j in range(w):
            if mat1[i][j] > 0:
                # drop it
                mat2[i][j] = 0
    return mat1, mat2

@jit(nopython=True)
def step2(h, w, mat1, mat2):
    for i in range(h):
        for j in range(w):
            if mat1[i][j] == 0 and mat2[i][j] > 0:
                mat1[i][j] = 255
    return mat1, mat2

# this is the iterative thresholding function
def iterative_thresh(t, im, message):
    # get a grayscale image and return a binary image
    # t is the time of iteration-1
    # im is the image
    normalized_img = cv2.normalize(im, im, 0, 255, cv2.NORM_MINMAX)
    ret, binary = cv2.threshold(normalized_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = binary.shape
    for x in range(t):
        binary, normalized_img = step1(h, w, binary, normalized_img)
        _, binary2 = cv2.threshold(normalized_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary, _ = step2(h, w, binary, binary2)
    print(message)
    print('processing...')
    return binary

# this function is for microglia cell
@jit(nopython=True)
def replace(fix_clean, res1, remove):
    for i in range(fix_clean.shape[0]):
        for j in range(fix_clean.shape[1]):
            if fix_clean[i][j] == 0 and res1[i][j] == 255:
                # means it is microglial
                remove[i][j] = 0
    return(remove)

# gamma correction
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# microglia cell detection
def micro(img, data, gray, message):
    # input is a binary image and its topology information
    # return a grayscale image excluding microglia cells
    image = img.astype("uint8")
    ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    G = nx.Graph()
    for coor in data.nodespair:
        temp_c = coor
        first = temp_c[0]
        second = temp_c[1]
        f = (first[1], first[0])
        s = (second[1], second[0])
        G.add_edge(f, s)

    nodes_lists = []
    degree1_set = []
    degree2_set = []
    ratio = []

    # for all nodes that degree >= 2
    # find if it connects too many nodes that degree == 1 for dehpth < 5
    # if yes, then drop this node as it may be on the boundary of microglia cells
    # we only hope to keep those points inside a microglia cell
    # create a boundary for microglia cell and filopodia cell

    for node in G.nodes():
        temp_list = []
        if G.degree(node) == 2:
            flag1 = 0
            flag2 = 0
            for j in range(5):
                for nd in nx.descendants_at_distance(G, node, distance=j):
                    temp_list.append(nd)
            for nd in temp_list:
                if G.degree(nd) == 1:
                    flag1 += 1
                if G.degree(nd) == 2:
                    flag2 += 1
            ratio.append(flag1 / len(temp_list))
            degree1_set.append(flag1)
            degree2_set.append(flag2)
            nodes_lists.append(node)

    for i in range(len(degree1_set)):
        if degree1_set[i] >= 6:
            G.remove_node(nodes_lists[i])

    # get the largest connected
    # get a list of connected small graphs
    components = list(nx.connected_components(G))
    components.sort(key=len, reverse=True)
    largest = components.pop(0)
    x = [ val[0] for val in list(G.nodes())]
    y = [ val[1] for val in list(G.nodes())]
    points = list(zip(x,y))
    fake_img = np.zeros(binary.shape)
    for y, x in points:
        fake_img[x][y] = 255
    # blur, gamma correction
    blur = cv2.blur(fake_img,(20,20))
    blur = cv2.blur(blur,(20,20))
    blur = blur / np.max(blur) * 255
    blur = blur.astype(np.uint8)
    blur = adjust_gamma(blur, 1.2)
    g = gray.copy()
    print(blur.shape)
    print(g.shape)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            subs = int(gray[i][j]) - int(blur[i][j])
            if subs < 0:
                subs = 0
            g[i][j] = subs

    print(message)
    print('processing...')
    return g

# feature extraction image
# cited from Wang et al. 2021 paper
def extract_graph(skeleton,image):
    
    skeleton=np.asarray(skeleton)
    skeleton=skeleton*255
    image=np.asarray(image)
    image=image.astype(np.uint8)
    image=image*255
    graph_nodes = zhang_suen_node_detection(skeleton)
    graph = breadth_first_edge_detection2(skeleton, image, graph_nodes)
    edges=[(u, v) for u, v, data in graph.edges(data=True)]
    data=[data for u, v, data in graph.edges(data=True)]

    for n1,n2,data in graph.edges(data=True):
        line=euclidean(n1,n2)
        graph[n1][n2]['line']=line

    return graph

def zhang_suen_node_detection(skel):

    def check_pixel_neighborhood(x, y, skel):

        accept_pixel_as_node = False
        item = skel.item
        p2 = item(x - 1, y) / 255
        p3 = item(x - 1, y + 1) / 255
        p4 = item(x, y + 1) / 255
        p5 = item(x + 1, y + 1) / 255
        p6 = item(x + 1, y) / 255
        p7 = item(x + 1, y - 1) / 255
        p8 = item(x, y - 1) / 255
        p9 = item(x - 1, y - 1) / 255

        components = (p2 == 0 and p3 == 1) + (p3 == 0 and p4 == 1) + \
                     (p4 == 0 and p5 == 1) + (p5 == 0 and p6 == 1) + \
                     (p6 == 0 and p7 == 1) + (p7 == 0 and p8 == 1) + \
                     (p8 == 0 and p9 == 1) + (p9 == 0 and p2 == 1)

        #components = (p2 == 1) + (p4 == 1) + (p6 == 1) + (p8 == 1)
        if (components >= 3) or (components == 1):
            accept_pixel_as_node = True
        return accept_pixel_as_node

    graph = nx.Graph()
    w, h = skel.shape
    item = skel.item
    for x in range(1, w - 1):
        for y in range(1, h - 1):
             if item(x, y) != 0 and check_pixel_neighborhood(x, y, skel):
                graph.add_node((x, y))
    return graph

def merge_nodes_2(G,nodes, attr_dict=None):
   
    if (nodes[0] in G.nodes()) & (nodes[1] in G.nodes()):
        for n1,n2,data in list(G.edges(data=True)):
            if (n1==nodes[0]) & (n2 != nodes[1]): 
                G.add_edges_from([(nodes[1],n2,data)])
            elif (n1!=nodes[1]) & (n2 == nodes[0]):
                G.add_edges_from([(n1,nodes[1],data)])
        G.remove_node(nodes[0])

def breadth_first_edge_detection2(skel, segmented, graph):

    def neighbors(x, y):
        item = skel.item
        width, height = skel.shape
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if (dx != 0 or dy != 0) and \
                                        0 <= x + dx < width and \
                                        0 <= y + dy < height and \
                                item(x + dx, y + dy) != 0:
                    yield x + dx, y + dy

    def distance_transform_diameter(edge_trace, segmented):

        dt = cv2.distanceTransform(segmented, 2, 0)
        edge_pixels = np.nonzero(edge_trace)
        diameters = defaultdict(list)
        for label, diam in zip(edge_trace[edge_pixels], 2.0 * dt[edge_pixels]):
            diameters[label].append(diam)
        return diameters

    label_node = dict()
    label_pixel=dict()
    queues = []
    label = 1
    label_length = defaultdict(int)
    for x, y in graph.nodes():
        for a, b in neighbors(x, y):
            label_node[label] = (x, y)
            label_length[label] = 1.414214 if abs(x - a) == 1 and \
                                              abs(y - b) == 1 else 1
            label_pixel[label]=[(a,b)]
            queues.append((label, (x, y), [(a, b)]))
            label += 1


    edges = set()
    edge_trace = np.zeros(skel.shape, np.uint32)
    edge_value = edge_trace.item
    edge_set_value = edge_trace.itemset
    label_histogram = defaultdict(int)

    while queues:
        new_queues = []
        for label, (px, py), nbs in queues:
            for (ix, iy) in nbs:
                value = edge_value(ix, iy)
                if value == 0:
                    edge_set_value((ix, iy), label)
                    label_histogram[label] += 1
                    label_length[label] += 1.414214 if abs(ix - px) == 1 and \
                                                       abs(iy - py) == 1 else 1
                    label_pixel[label]=label_pixel[label]+[(a,b) for a,b in neighbors(ix, iy)]
                    new_queues.append((label, (ix, iy), neighbors(ix, iy)))
                elif value != label:
                    edges.add((min(label, value), max(label, value)))
        queues = new_queues

    diameters = distance_transform_diameter(edge_trace, segmented)
    for l1, l2 in edges:
        u, v = label_node[l1], label_node[l2]
        if u == v:
            continue
        d1, d2 = diameters[l1], diameters[l2]
        diam = np.fromiter(chain(d1, d2), np.uint, len(d1) + len(d2))
        graph.add_edge(u, v, pixels=label_histogram[l1] + label_histogram[l2],
                       length=label_length[l1] + label_length[l2],
                       curve=label_pixel[l1] + label_pixel[l2][::-1],
                       width=np.median(diam),
                       width_var=np.var(diam))
    return graph


def extract_topo(img, message):
    # this function inputs are a binary image
    # returns a pandas dataframe about topology information
    image = np.asarray(img)
    image = image.astype("uint8")
    ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_img = Image.fromarray(binary)
    binary[binary == 255] = 1
    skeleton0 = morphology.skeletonize(binary)
    skeleton = skeleton0.astype(np.uint8)
    image = Image.fromarray(image)

    graph=extract_graph(skeleton,image)            
    k=True

    while k:
        tmp=graph.number_of_nodes()
        attribute="line"
        # attribute_threshold_value=10
        attribute_threshold_value=0
        to_be_removed = [(u, v) for u, v, data in
                                    graph.edges(data=True)
                                    if operator.le(data[attribute],
                                                attribute_threshold_value)]
        length=len(to_be_removed)
        for n in range(length):  
            nodes=to_be_removed[n]
            merge_nodes_2(graph,nodes)
        
        for n1,n2,data in graph.edges(data=True):
            line=euclidean(n1,n2)
            graph[n1][n2]['line']=line
        
        number_of_nodes=graph.number_of_nodes()
        k= tmp!=number_of_nodes
        
    #Check connected
    ####  keep the connected with 1 poitns
    compnt_size = 0
    compnt_size = 1
    operators ="smaller or equal"
    oper_str_value = operators
    operators = operator.le
    connected_components = sorted(
                    # new version: (G.subgraph(c) for c in connected_components(G))
                    list(graph.subgraph(c) for c in nx.connected_components(graph)),
                    key=lambda graph: graph.number_of_nodes())
    
    to_be_removed = [subgraph for subgraph in connected_components
                                if operators(subgraph.number_of_nodes(),
                                                        compnt_size)]
    for subgraph in to_be_removed:
        graph.remove_nodes_from(subgraph)
    nodes=[n for n in graph.nodes()]
    x=[x for (x,y) in nodes]
    y=[y for (x,y) in nodes]
    x1=int(np.min(x)+(np.max(x)-np.min(x))/2)
    y1=int(np.min(y)+(np.max(y)-np.min(y))/2)
    
    for n1,n2,data in graph.edges(data=True):
        centerdis1=euclidean((x1,y1),n2)
        centerdis2=euclidean((x1,y1),n1)
        #theta1=(math.atan2(-13,-14)/math.pi*180)%360
        #theta2=(math.atan2(-13,-14)/math.pi*180)%360
        
        if centerdis1>=centerdis2:
            centerdislow=centerdis2
            centerdishigh=centerdis1
        else:
            centerdislow=centerdis1
            centerdishigh=centerdis2
        graph[n1][n2]['centerdislow']=centerdislow
        graph[n1][n2]['centerdishigh']=centerdishigh

    alldata=save_data(graph,center=False)

    print(message)
    print("processing...")
    return alldata



def save_data(graph,center=False):
    lines=[]
    for n1,n2,data in graph.edges(data=True):
        lines.append(graph[n1][n2]['line'])
    
    length=[]
    for n1,n2,data in graph.edges(data=True):
        length.append(graph[n1][n2]['length'])
    
    width=[]
    for n1,n2,data in graph.edges(data=True):
        width.append(graph[n1][n2]['width'])
    
    width_var=[]
    for n1,n2,data in graph.edges(data=True):
        width_var.append(graph[n1][n2]['width_var'])
    
    nodespair=[(u, v) for u, v, data in graph.edges(data=True)]
    node1=[u for u, v, data in graph.edges(data=True)]
    node2=[v for u, v, data in graph.edges(data=True)]
    
    tortuosity=[x/y for x, y in zip(length,lines)]
    
    curve=[]
    for n1,n2,data in graph.edges(data=True):
        curve.append(graph[n1][n2]['curve'])

    
    if center:
        centerdislow=[]
        for n1,n2,data in graph.edges(data=True):
            centerdislow.append(graph[n1][n2]['centerdislow'])
            
        centerdishigh=[]
        for n1,n2,data in graph.edges(data=True):
            centerdishigh.append(graph[n1][n2]['centerdishigh'])
            
        thetalow=[]
        for n1,n2,data in graph.edges(data=True):
            thetalow.append(graph[n1][n2]['thetalow'])
            
        thetahigh=[]
        for n1,n2,data in graph.edges(data=True):
            thetahigh.append(graph[n1][n2]['thetahigh'])
            
        
    
        alldf=pd.DataFrame({'nodespair':nodespair,
                            'node1':node1,
                            'node2':node2,
                            'line':lines,
                            'length':length,
                            'width':width,
                            'width_var':width_var,
                            'tortuosity':tortuosity,
                            'centerdislow':centerdislow,
                            'centerdishigh':centerdishigh,
                            'thetalow':thetalow,
                            'thetahigh':thetahigh,
                            'curve':curve,})
    else:
        alldf=pd.DataFrame({'nodespair':nodespair,
                            'node1':node1,
                            'node2':node2,
                            'line':lines,
                            'length':length,
                            'width':width,
                            'width_var':width_var,
                            'tortuosity':tortuosity,
                            'curve':curve,})
    
    return alldf
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
from collections import defaultdict, deque
from itertools import chain
import ast


# -------------------------- helpers: isolated pixel removal --------------------------

def inner_loop(pt, contours):
    cts = []
    flag = 0
    seq = 0
    for ct in contours:
        seq += 1
        result = cv2.pointPolygonTest(ct, pt, False)
        if result >= 0:
            # inside the contour or on the boundary
            cts.append(ct)
            flag = 1
        if seq == len(contours) and flag == 1:
            return [1, cts]
        elif seq == len(contours) and flag == 0:
            return [0]

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
    # returns a cleaned image using topology info and a binary image
    image = img.astype("uint8")
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # make graph from node pairs
    G = nx.Graph()
    for coor in data.nodespair:
        first, second = coor[0], coor[1]
        f = (first[1], first[0])
        s = (second[1], second[0])
        G.add_edge(f, s)

    # get components (drop largest later via mask)
    components = list(nx.connected_components(G))
    components.sort(key=len, reverse=True)
    if components:
        _largest = components.pop(0)

    # contours
    contours, _ = cv2.findContours(image=binary, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    binary_copy = binary.copy()

    for node_list in components:
        ele = list(node_list)
        try:
            for pt in ele:
                res = inner_loop(pt, contours)
                if res[0] == 1:
                    # has contour -> find smallest
                    tempc = res[1]
                    ar = [cv2.contourArea(ct) for ct in tempc]
                    ind = int(np.argmin(ar))
                    cont_need = tempc[ind]
                    # fill with black
                    cv2.drawContours(binary_copy, [cont_need], -1, (0, 0, 0), thickness=cv2.FILLED)
        except Exception:
            pass

    final_cont, _ = cv2.findContours(image=binary_copy, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    if final_cont:
        maxc = max(final_cont, key=cv2.contourArea)
    else:
        maxc = None

    binary_f = binary_copy.copy()
    for ct in final_cont:
        if maxc is None or len(ct) != len(maxc):
            cv2.drawContours(binary_f, [ct], -1, (0, 0, 0), thickness=cv2.FILLED)

    final_cont, _ = cv2.findContours(image=binary_f, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    binary_final = binary_f.copy()
    for ct in final_cont:
        ar = cv2.contourArea(ct)
        if 4 < ar < 200:
            img_m = np.ones(binary_final.shape, dtype=np.uint8) * 255
            masks = cv2.drawContours(img_m, [ct], -1, (0, 0, 0), thickness=cv2.FILLED)
            p = inside_coordinate(masks, ct, binary_final)
            for (v, u) in p:
                binary_final = cv2.rectangle(binary_final, (v, u), (v, u), 0, -1)
        elif ar <= 4:
            cv2.drawContours(binary_final, [ct], -1, 0, thickness=cv2.FILLED)

    print(message)
    print('processing...')
    return binary_final


# -------------------------- iterative thresholding (numba) --------------------------

@jit(nopython=True)
def step1(h, w, mat1, mat2):
    for i in range(h):
        for j in range(w):
            if mat1[i][j] > 0:
                mat2[i][j] = 0
    return mat1, mat2

@jit(nopython=True)
def step2(h, w, mat1, mat2):
    for i in range(h):
        for j in range(w):
            if mat1[i][j] == 0 and mat2[i][j] > 0:
                mat1[i][j] = 255
    return mat1, mat2

def iterative_thresh(t, im, message):
    # grayscale -> binary
    normalized_img = cv2.normalize(im, im, 0, 255, cv2.NORM_MINMAX)
    _, binary = cv2.threshold(normalized_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = binary.shape
    for _ in range(t):
        binary, normalized_img = step1(h, w, binary, normalized_img)
        _, binary2 = cv2.threshold(normalized_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary, _ = step2(h, w, binary, binary2)
    print(message)
    print('processing...')
    return binary


# -------------------------- microglia detection --------------------------

@jit(nopython=True)
def replace(fix_clean, res1, remove):
    for i in range(fix_clean.shape[0]):
        for j in range(fix_clean.shape[1]):
            if fix_clean[i][j] == 0 and res1[i][j] == 255:
                remove[i][j] = 0
    return remove

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = (np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])).astype("uint8")
    return cv2.LUT(image, table)

def micro(img, data, gray, message):
    image = img.astype("uint8")
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    G = nx.Graph()
    for coor in data.nodespair:
        first, second = coor[0], coor[1]
        f = (first[1], first[0])
        s = (second[1], second[0])
        G.add_edge(f, s)

    # prune degree-2 nodes heavily connected to degree-1 nodes within depth 5
    nodes_lists = []
    degree1_set = []
    degree2_set = []
    ratio = []

    for node in G.nodes():
        if G.degree(node) == 2:
            temp_list = []
            for j in range(5):
                for nd in nx.descendants_at_distance(G, node, distance=j):
                    temp_list.append(nd)
            flag1 = sum(G.degree(nd) == 1 for nd in temp_list)
            flag2 = sum(G.degree(nd) == 2 for nd in temp_list)
            ratio.append(flag1 / max(1, len(temp_list)))
            degree1_set.append(flag1)
            degree2_set.append(flag2)
            nodes_lists.append(node)

    for i in range(len(degree1_set)):
        if degree1_set[i] >= 6:
            G.remove_node(nodes_lists[i])

    # seed image with nodes
    x = [val[0] for val in list(G.nodes())]
    y = [val[1] for val in list(G.nodes())]
    points = list(zip(x, y))
    fake_img = np.zeros(binary.shape, dtype=np.uint8)
    for yy, xx in points:
        fake_img[xx, yy] = 255

    # blur + gamma
    blur = cv2.blur(fake_img, (20, 20))
    blur = cv2.blur(blur, (20, 20))
    blur = (blur / max(1, np.max(blur)) * 255).astype(np.uint8)
    blur = adjust_gamma(blur, 1.2)

    g = gray.copy()
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            subs = int(gray[i][j]) - int(blur[i][j])
            if subs < 0:
                subs = 0
            g[i][j] = subs

    print(message)
    print('processing...')
    return g


# -------------------------- graph extraction --------------------------

def extract_graph(skeleton, image):
    skeleton = np.asarray(skeleton)
    skeleton = skeleton * 255
    image = np.asarray(image).astype(np.uint8)
    image = image * 255

    graph_nodes = zhang_suen_node_detection(skeleton)
    graph = breadth_first_edge_detection2(skeleton, image, graph_nodes)

    # compute straight line distance for each edge
    for n1, n2, data in graph.edges(data=True):
        line = euclidean(n1, n2)
        graph[n1][n2]['line'] = line
    return graph


def zhang_suen_node_detection(skel):
    def check_pixel_neighborhood(x, y, skel_):
        p2 = skel_[x - 1, y] / 255
        p3 = skel_[x - 1, y + 1] / 255
        p4 = skel_[x, y + 1] / 255
        p5 = skel_[x + 1, y + 1] / 255
        p6 = skel_[x + 1, y] / 255
        p7 = skel_[x + 1, y - 1] / 255
        p8 = skel_[x, y - 1] / 255
        p9 = skel_[x - 1, y - 1] / 255

        components = ((p2 == 0 and p3 == 1) + (p3 == 0 and p4 == 1) +
                      (p4 == 0 and p5 == 1) + (p5 == 0 and p6 == 1) +
                      (p6 == 0 and p7 == 1) + (p7 == 0 and p8 == 1) +
                      (p8 == 0 and p9 == 1) + (p9 == 0 and p2 == 1))

        return (components >= 3) or (components == 1)

    graph = nx.Graph()
    w, h = skel.shape
    for x in range(1, w - 1):
        for y in range(1, h - 1):
            if skel[x, y] != 0 and check_pixel_neighborhood(x, y, skel):
                graph.add_node((x, y))
    return graph


def merge_nodes_2(G, nodes, attr_dict=None):
    if (nodes[0] in G.nodes()) & (nodes[1] in G.nodes()):
        for n1, n2, data in list(G.edges(data=True)):
            if (n1 == nodes[0]) & (n2 != nodes[1]):
                G.add_edges_from([(nodes[1], n2, data)])
            elif (n1 != nodes[1]) & (n2 == nodes[0]):
                G.add_edges_from([(n1, nodes[1], data)])
        G.remove_node(nodes[0])


# -------------------------- BFS label growth (robust, NumPy-2.0 safe) --------------------------

def breadth_first_edge_detection2(skel, segmented, graph):
    """
    BFS label growth over skeleton pixels, then compute per-edge width via distance transform.
    NumPy >= 2.0 safe (no .item/.itemset). Queue holds explicit coordinates, no generator scoping.
    """

    def neighbors(x, y):
        H, W = skel.shape
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < H and 0 <= ny_ < W and skel[nx_, ny_] != 0:
                    yield nx_, ny_

    def distance_transform_diameter(edge_trace_u32, seg_u8_mask):
        # seg_u8_mask must be {0,255}
        dt = cv2.distanceTransform(seg_u8_mask, 2, 0)
        rr, cc = np.nonzero(edge_trace_u32)
        diameters = defaultdict(list)
        labs = edge_trace_u32[rr, cc]
        dist2 = (2.0 * dt[rr, cc]).astype(float)
        for lab, diam in zip(labs, dist2):
            diameters[int(lab)].append(float(diam))
        return diameters

    # ensure proper mask for DT
    seg_u8 = ((segmented.astype(np.uint8) > 0).astype(np.uint8)) * 255
    edge_trace = np.zeros(skel.shape, np.uint32)

    label_node      = {}                     # label -> seed node (x,y)
    label_length    = defaultdict(float)     # label -> path length
    label_histogram = defaultdict(int)       # label -> number of pixels
    label_pixels    = defaultdict(list)      # label -> pixels along curve
    edges = set()

    # seed queue with (lab, prev_x, prev_y, cur_x, cur_y)
    q = deque()
    label = 1
    for sx, sy in graph.nodes():
        for nx1, ny1 in neighbors(sx, sy):
            label_node[label] = (sx, sy)
            label_length[label] = 1.414214 if (abs(sx - nx1) == 1 and abs(sy - ny1) == 1) else 1.0
            label_pixels[label].append((nx1, ny1))
            q.append((label, sx, sy, nx1, ny1))
            label += 1

    # BFS growth
    while q:
        lab, prev_x, prev_y, cur_x, cur_y = q.popleft()
        val = edge_trace[cur_x, cur_y]

        if val == 0:
            edge_trace[cur_x, cur_y] = lab
            label_histogram[lab] += 1
            label_length[lab] += 1.414214 if (abs(cur_x - prev_x) == 1 and abs(cur_y - prev_y) == 1) else 1.0

            next_neighbors = list(neighbors(cur_x, cur_y))
            if next_neighbors:
                label_pixels[lab].extend(next_neighbors)
                for nx2, ny2 in next_neighbors:
                    q.append((lab, cur_x, cur_y, nx2, ny2))

        elif val != lab:
            l1, l2 = int(lab), int(val)
            if l1 != l2:
                edges.add((min(l1, l2), max(l1, l2)))

    # widths via distance transform
    diameters = distance_transform_diameter(edge_trace, seg_u8)

    # attach edges with attributes
    for l1, l2 in edges:
        u = label_node.get(l1)
        v = label_node.get(l2)
        if u is None or v is None or u == v:
            continue

        d1 = diameters.get(l1, [])
        d2 = diameters.get(l2, [])
        all_d = np.array(list(chain(d1, d2)), dtype=np.float64)

        if all_d.size:
            width = float(np.median(all_d))
            width_var = float(all_d.var()) if all_d.size > 1 else 0.0
        else:
            width = 0.0
            width_var = 0.0

        graph.add_edge(
            u, v,
            pixels=int(label_histogram[l1] + label_histogram[l2]),
            length=float(label_length[l1] + label_length[l2]),
            curve=label_pixels.get(l1, []) + list(reversed(label_pixels.get(l2, []))),
            width=width,
            width_var=width_var,
        )

    return graph


# -------------------------- topology extraction & saving --------------------------

def extract_topo(img, message):
    # inputs: binary/grayscale image
    # returns: pandas DataFrame with topology information
    image = np.asarray(img).astype("uint8")
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # skeletonize expects 0/1
    bin01 = binary.copy()
    bin01[bin01 == 255] = 1
    skeleton0 = morphology.skeletonize(bin01)
    skeleton = skeleton0.astype(np.uint8)

    graph = extract_graph(skeleton, image)
    k = True
    while k:
        tmp = graph.number_of_nodes()
        attribute = "line"
        attribute_threshold_value = 0
        to_be_removed = [(u, v) for u, v, data in graph.edges(data=True)
                         if operator.le(data[attribute], attribute_threshold_value)]
        for nodes in to_be_removed:
            merge_nodes_2(graph, nodes)

        for n1, n2, data in graph.edges(data=True):
            line = euclidean(n1, n2)
            graph[n1][n2]['line'] = line

        number_of_nodes = graph.number_of_nodes()
        k = tmp != number_of_nodes

    # keep components with > 1 nodes
    compnt_size = 1
    connected_components = sorted(
        (graph.subgraph(c) for c in nx.connected_components(graph)),
        key=lambda g: g.number_of_nodes()
    )
    to_remove = [sub for sub in connected_components if operator.le(sub.number_of_nodes(), compnt_size)]
    for sub in to_remove:
        graph.remove_nodes_from(sub)

    nodes = list(graph.nodes())
    if not nodes:
        print(message)
        print("processing...")
        return pd.DataFrame(columns=['nodespair','node1','node2','line','length','width','width_var','tortuosity','curve'])

    xs = [x for (x, y) in nodes]
    ys = [y for (x, y) in nodes]
    x1 = int(np.min(xs) + (np.max(xs) - np.min(xs)) / 2)
    y1 = int(np.min(ys) + (np.max(ys) - np.min(ys)) / 2)

    for n1, n2, data in graph.edges(data=True):
        centerdis1 = euclidean((x1, y1), n2)
        centerdis2 = euclidean((x1, y1), n1)
        if centerdis1 >= centerdis2:
            centerdislow, centerdishigh = centerdis2, centerdis1
        else:
            centerdislow, centerdishigh = centerdis1, centerdis2
        graph[n1][n2]['centerdislow'] = centerdislow
        graph[n1][n2]['centerdishigh'] = centerdishigh

    alldata = save_data(graph, center=False)

    print(message)
    print("processing...")
    return alldata


def save_data(graph, center=False):
    lines = [graph[u][v]['line'] for u, v, _ in graph.edges(data=True)]
    length = [graph[u][v]['length'] for u, v, _ in graph.edges(data=True)]
    width = [graph[u][v]['width'] for u, v, _ in graph.edges(data=True)]
    width_var = [graph[u][v]['width_var'] for u, v, _ in graph.edges(data=True)]

    nodespair = [(u, v) for u, v, _ in graph.edges(data=True)]
    node1 = [u for u, v, _ in graph.edges(data=True)]
    node2 = [v for u, v, _ in graph.edges(data=True)]
    tortuosity = [x / y if y != 0 else 0.0 for x, y in zip(length, lines)]

    curve = [graph[u][v]['curve'] for u, v, _ in graph.edges(data=True)]

    if center:
        centerdislow = [graph[u][v]['centerdislow'] for u, v, _ in graph.edges(data=True)]
        centerdishigh = [graph[u][v]['centerdishigh'] for u, v, _ in graph.edges(data=True)]
        thetalow = [graph[u][v]['thetalow'] for u, v, _ in graph.edges(data=True)]
        thetahigh = [graph[u][v]['thetahigh'] for u, v, _ in graph.edges(data=True)]

        alldf = pd.DataFrame({
            'nodespair': nodespair,
            'node1': node1,
            'node2': node2,
            'line': lines,
            'length': length,
            'width': width,
            'width_var': width_var,
            'tortuosity': tortuosity,
            'centerdislow': centerdislow,
            'centerdishigh': centerdishigh,
            'thetalow': thetalow,
            'thetahigh': thetahigh,
            'curve': curve,
        })
    else:
        alldf = pd.DataFrame({
            'nodespair': nodespair,
            'node1': node1,
            'node2': node2,
            'line': lines,
            'length': length,
            'width': width,
            'width_var': width_var,
            'tortuosity': tortuosity,
            'curve': curve,
        })
    return alldf

from ultralytics import YOLO
from matplotlib.pyplot import figure
import matplotlib.image as image
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from numpy import asarray
from PIL import Image
import cv2
from shapely.geometry import Polygon
import re

def order_points(pts):
    
    # order a list of 4 coordinates:
    # 0: top-left,
    # 1: top-right
    # 2: bottom-right,
    # 3: bottom-left
    
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

# calculates iou between two polygons

def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

def detect_corners(image):
    
    # YOLO model trained to detect corners on a chessboard
    model_trained = YOLO("best_corners.pt")
    results = model_trained.predict(source=image, line_width=1, conf=0.25, save_txt=True, save=True)

    # get the corners coordinates from the model
    boxes = results[0].boxes
    arr = boxes.xywh.numpy()
    points = arr[:,0:2]
    
    corners = order_points(points)
    
    return corners

# calculates chessboard grid

def plot_grid_on_transformed_image(image):
    
    corners = np.array([[0,0], 
                    [image.size[0], 0], 
                    [0, image.size[1]], 
                    [image.size[0], image.size[1]]])
    
    corners = order_points(corners)

    figure(figsize=(10, 10), dpi=80)

    # im = plt.imread(image)
    implot = plt.imshow(image)
    
    TL = corners[0]
    BL = corners[3]
    TR = corners[1]
    BR = corners[2]

    def interpolate(xy0, xy1):
        x0,y0 = xy0
        x1,y1 = xy1
        dx = (x1-x0) / 8
        dy = (y1-y0) / 8
        pts = [(x0+i*dx,y0+i*dy) for i in range(9)]
        return pts

    ptsT = interpolate( TL, TR )
    ptsL = interpolate( TL, BL )
    ptsR = interpolate( TR, BR )
    ptsB = interpolate( BL, BR )
        
    for a,b in zip(ptsL, ptsR):
        plt.plot( [a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--" )
    for a,b in zip(ptsT, ptsB):
        plt.plot( [a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--" )
        
    plt.axis('off')

    plt.savefig('chessboard_transformed_with_grid.jpg')
    return ptsT, ptsL

# detects chess pieces

def chess_pieces_detector(image):
    
    model_trained = YOLO("best_transformed_detection.pt")
    results = model_trained.predict(source=image, line_width=1, conf=0.5, augment=False, save_txt=True, save=True)
    
    boxes = results[0].boxes
    detections = boxes.xyxy.numpy()
    
    return detections, boxes

# connects detected piece to the right square

def connect_square_to_detection(detections, square, boxes):
    
    di = {0: 'b', 1: 'k', 2: 'n',
      3: 'p', 4: 'q', 5: 'r', 
      6: 'B', 7: 'K', 8: 'N',
      9: 'P', 10: 'Q', 11: 'R'}

    list_of_iou=[]
    
    for i in detections:

        box_x1 = i[0]
        box_y1 = i[1]

        box_x2 = i[2]
        box_y2 = i[1]

        box_x3 = i[2]
        box_y3 = i[3]

        box_x4 = i[0]
        box_y4 = i[3]
        
        #cut high pieces        
        if box_y4 - box_y1 > 60:
            box_complete = np.array([[box_x1,box_y1+40], [box_x2, box_y2+40], [box_x3, box_y3], [box_x4, box_y4]])
        else:
            box_complete = np.array([[box_x1,box_y1], [box_x2, box_y2], [box_x3, box_y3], [box_x4, box_y4]])
            
        #until here

        list_of_iou.append(calculate_iou(box_complete, square))

    num = list_of_iou.index(max(list_of_iou))

    piece = boxes.cls[num].tolist()
    
    if max(list_of_iou) > 0.15:
        piece = boxes.cls[num].tolist()
        return di[piece]
    
    else:
        piece = "empty"
        return piece

# perspective transforms an image with four given corners
def transform_image(image, corners):
    image = asarray(image)
    rect = order_points(corners)

    maxWidth = 1500; maxHeight = 1500

    # construct set of destination points to obtain a "birds eye view"
    dst = np.array([
        [0, 0],
        [maxWidth, 0],
        [maxWidth, maxHeight],
        [0, maxHeight]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    transformed_image = Image.fromarray(warped, "RGB")
    return transformed_image

def fix_fen(fen):
    rows = fen.split("/")
    fixed_rows = []
    for row in rows:
        # Replace consecutive 1s with their count
        fixed_row = re.sub(r"1+", lambda m: str(len(m.group())), row)
        fixed_rows.append(fixed_row)
    return "/".join(fixed_rows)

def get_fen(img: str) -> str:
    image = cv2.imread(img)

    corners = detect_corners(image)

    # Draw the points on the image
    for point in corners:
        cv2.circle(image, tuple(point.astype(int)), 5, (0, 0, 255), 10)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed_image = transform_image(image, corners)

    plt.figure(figsize =(10, 5))
    plt.subplot(121);plt.imshow(image)
    plt.subplot(122);plt.imshow(transformed_image)

    ptsT, ptsL = plot_grid_on_transformed_image(transformed_image)

    detections, boxes = chess_pieces_detector(transformed_image)

    #calculate the grid

    # Extract x and y coordinates
    x_coords = [ptsT[i][0] for i in range(9)]
    y_coords = [ptsL[i][1] for i in range(9)]

    # Generate all squares using loops
    FEN_annotation = []
    for row in range(8, 0, -1):  # Rows 8 to 1
        row_squares = []
        for col in range(8):  # Columns A to H (0 to 7 index)
            square = np.array([
                [x_coords[col], y_coords[row]],
                [x_coords[col + 1], y_coords[row]],
                [x_coords[col + 1], y_coords[row - 1]],
                [x_coords[col], y_coords[row - 1]]
            ])
            row_squares.append(square)
        FEN_annotation.append(row_squares)


    board_FEN = []
    corrected_FEN = []
    complete_board_FEN = []

    for line in FEN_annotation:
        line_to_FEN = []
        for square in line:
            piece_on_square = connect_square_to_detection(detections, square, boxes)    
            line_to_FEN.append(piece_on_square)
        corrected_FEN = [i.replace('empty', '1') for i in line_to_FEN]
        print(corrected_FEN)
        board_FEN.append(corrected_FEN)

    complete_board_FEN = [''.join(line) for line in board_FEN]
    to_FEN = '/'.join(complete_board_FEN)
    correct_fen = fix_fen(to_FEN)

    return correct_fen
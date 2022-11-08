import glob   
import os
import cv2
import functools
import chess
import json
import re
import csv

from pathlib import Path
from numpy import fabs, string_
from recap import URI, CfgNode as CN
from PIL import Image

from chesscog.corner_detection.detect_corners import find_corners, resize_image
from chesscog.occupancy_classifier.create_dataset import warp_chessboard_image, crop_square

class Creator_squares():

    _squares = list(chess.SQUARES)
    starting_creator_squares = True

    # path = "C:/MH-Entw/ml-chess-board-project/01_original_data\\test\\0046.png"
    # print(path)
    # img = cv2.imread(path)
    # # Debug help: show img
    # cv2.imshow('image', img)
    # cv2.waitKey()

    def __init__(self) -> None:
        # Get the current working directory
        self.current_working_dir = os.getcwd()
        self.current_working_dir = self.current_working_dir.replace('\\','/')

        path = self.current_working_dir + '/chesscog/corner_detection.yaml'
        self._corner_detection_cfg = CN.load_yaml_with_base(path)
        self.init_Figure_to_int()

    def create_squares_single_img(self):
        print("-"*20)
        print('Starting: Creator Squares')

        # Create path
        path_squares = self.current_working_dir + "/03_demo_data/squares/"
        
        # Create folder
        Path(path_squares).mkdir(parents=True, exist_ok=True)

        # Create label.csv file
        path_label_csv =  path_squares + "demo.csv"
        Path(path_label_csv).touch(exist_ok=True) 

        # Only write header once at beginning 
        with open(path_label_csv, mode='w', newline='') as label_file:
            employee_writer = csv.writer(label_file, delimiter=',')
            employee_writer.writerow(['file_name','label'])
            
        # Foreach Chess Board Img
        counter = 0
        for img_filename_found in glob.iglob(self.current_working_dir + "/03_demo"+ '**/*.png', recursive=True):

            try:
                img_filename = img_filename_found
                img_name_short = img_filename[-8:-4]

                # Step 0: import labels
                img_label = self.import_label_demo(img_name_short)
                turn = img_label["white_turn"]

                # Step 1: create labels squares
                label_occupacy, label_figure  = self.create_labels(img_label, turn)

                # Step 2: get img
                img = cv2.imread(img_filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Debug helper: show img
                # cv2.imshow('image', img)
                # cv2.waitKey()
                img, img_scale = resize_image(self._corner_detection_cfg, img)
                corners = find_corners(self._corner_detection_cfg, img)

                # Step 3: Wrap Img / Cut edges 
                warped = warp_chessboard_image(img, corners)

                # Step 4: Divide into 64 squares
                square_imgs = map(functools.partial(crop_square, warped, turn=turn), self._squares)

                # Step 5: Converts to PIL images
                square_imgs = map(Image.fromarray, square_imgs)

                # Step 6: Create list
                square_imgs = list(square_imgs)

                # Step 7: Export Squares + Labels
                # Squares Creation starts from right top if turn:white false -> numerate from right top
                positions = self.create_square_numeration_list(turn)
                
                # Export to directory
                counter2 = 0
                for img_single in square_imgs:

                    # Step 1: Save img + label to occupancy 
                    # save img as png
                    img_name = "img_" + img_name_short + "_square_" + "{:02d}".format(positions[counter2]) + ".png"
                    square_filename = path_squares + img_name 
                    img_single.save(square_filename)
                    
                    # save label to label csv
                    with open(path_label_csv,'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([img_name, "{:02d}".format(label_occupacy[positions[counter2]])])

                    counter2 += 1
                print(str(counter) + " img_" + img_name_short + ": done")
               

            except:
                print(str(counter) + " img_" + img_name_short + ": could not be split into squares")
            
        print(str(counter) + " images were sucessfully split into squares")
   
    def create_squares_all_img(self):
        print("-"*20)
        print('Starting: Creator Squares')

        classifier = ["occupancy" , "piece"]
        path_classifier = []
        path_label_csv = []

        # Create Pathes + Files 
        counter_modi = 0
        for modi in classifier:

            # Create Path
            path_classifier.append("02_created_data/" + modi + "/")  

            # Create folder
            Path(path_classifier[counter_modi]).mkdir(parents=True, exist_ok=True)

            # Create label.csv file
            path_label_csv.append(path_classifier[counter_modi] + str(classifier[counter_modi]) + ".csv")
            Path(path_classifier[counter_modi]).touch(exist_ok=True)

            # Only write header once at beginning 
            with open(path_label_csv[counter_modi], mode='w', newline='') as label_file:
                employee_writer = csv.writer(label_file, delimiter=',') #, quotechar='"', quoting=csv.QUOTE_MINIMAL
                employee_writer.writerow(['file_name','label'])
            
            counter_modi += 1
        
        counter = 0
        
        # Foreach Chess Board Img
        for img_filename_found in glob.iglob(self.current_working_dir + "/01_original_data/all"+ '**/*.png', recursive=True):

            try:
                img_filename = img_filename_found
                img_name_short = img_filename[-8:-4]

                # Debug helper
                # img_filename = "C:/MH-Entw/ml-chess-board-project/01_original_data/test/0094.png"
                # img_name = "0094"
                # print(img_filename)
                # print(img_name)

                # Step 0: import labels
                img_label = self.import_label("all",img_name_short)
                turn = img_label["white_turn"]

                # Step 1: create labels squares
                label_occupacy, label_figure  = self.create_labels(img_label, turn)

                # Step 2: get img
                img = cv2.imread(img_filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Debug helper: show img
                # cv2.imshow('image', img)
                # cv2.waitKey()
                img, img_scale = resize_image(self._corner_detection_cfg, img)
                corners = find_corners(self._corner_detection_cfg, img)

                # Step 3: Wrap Img / Cut edges 
                warped = warp_chessboard_image(img, corners)

                # Step 4: Divide into 64 squares
                square_imgs = map(functools.partial(crop_square, warped, turn=turn), self._squares)

                # Step 5: Converts to PIL images
                square_imgs = map(Image.fromarray, square_imgs)

                # Step 6: Create list
                square_imgs = list(square_imgs)

                # Step 7: Export Squares + Labels
                # Squares Creation starts from right top if turn:white false -> numerate from right top
                positions = self.create_square_numeration_list(turn)
                
                # Export to directory
                counter2 = 0
                for img_single in square_imgs:

                    # Step 1: Save img + label to occupancy 
                    # save img as png
                    img_name = "img_" + img_name_short + "_square_" + str(positions[counter2]) + ".png"
                    square_filename = "02_created_data/" + classifier[0] + "/" + img_name 
                    img_single.save(square_filename)
                    
                    # save label to label csv
                    with open(path_label_csv[0],'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([img_name, str(label_occupacy[positions[counter2]])])
                    

                    # Step 2: Save img + label to piece classification
                    if label_occupacy[positions[counter2]] > 0:
                        # save img as png
                        img_name = "img_" + img_name_short + "_square_" + str(positions[counter2]) + ".png"
                        square_filename = "02_created_data/" + classifier[1] + "/" + img_name 
                        img_single.save(square_filename)

                        # save label to label csv
                        with open(path_label_csv[1],'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([img_name, str(self.fen_dic[label_figure[positions[counter2]]])])

                    counter2 += 1
                if counter % 50 == 0:
                    print(str(counter) + " img_" + img_name_short + ": done")
               
                counter += 1
                # Debug helper
                if counter > 10000:
                    break
            except:
                print(str(counter) + " img_" + img_name_short + ": could not be split into squares")
            
        print(str(counter) + " images were sucessfully split into squares")

    def init_Figure_to_int(self):
        fen_posibilities = 'RNBQKPrnbqkp'
        self.fen_dic = {}
        counter = 0
        for char in fen_posibilities:
            self.fen_dic[char] = counter
            counter += 1

    def import_label_demo(self, img_name):
            path = self.current_working_dir + "/03_demo_data/" + img_name + ".json"
            file_json = open(path)
            label = json.load(file_json)
            file_json.close()
            return label

    def import_label(self, dir_name, img_name):
        # print("Starting: Import Labels")

        # path = "C:\\MH-Entw\\ml-chess-board-project\\01_original_data\\test\\0046.json"
        # path = "C:\\MH-Entw\\ml-chess-board-project\\01_original_data\\" + dir_name + "\\" + img_name + ".json"
        path = self.current_working_dir + "\\01_original_data\\" + dir_name + "\\" + img_name + ".json"

        file_json = open(path)
        label = json.load(file_json)
        file_json.close()
        return label

    def create_labels(self, label, turn_white):
        fen_original = label["fen"]

        # MH comment: maybe shorter solution possible
        fen_occupacy_repl = fen_original.replace("1", "-")
        fen_occupacy_repl = fen_occupacy_repl.replace("2", "-" * 2)
        fen_occupacy_repl = fen_occupacy_repl.replace("3", "-" * 3)
        fen_occupacy_repl = fen_occupacy_repl.replace("4", "-" * 4)
        fen_occupacy_repl = fen_occupacy_repl.replace("5", "-" * 5)
        fen_occupacy_repl = fen_occupacy_repl.replace("6", "-" * 6)
        fen_occupacy_repl = fen_occupacy_repl.replace("7", "-" * 7)
        fen_occupacy_repl = fen_occupacy_repl.replace("8", "-" * 8)

        # Create list
        fen_occupacy_split = re.split('/', fen_occupacy_repl)

        # Turn: White
        if(turn_white):
            fen_occupacy_rev_top_left = fen_occupacy_split.copy()

        # Turn: Black
        # Only for turn = black -> change start pos of fen
        elif(not turn_white):
            # Start from top
            fen_occupacy_split.reverse()
            fen_occupacy_rev_top = fen_occupacy_split.copy()

            # Start from left (top)
            fen_occupacy_rev_top_left = []
            for row in fen_occupacy_rev_top:
                fen_occupacy_rev_top_left.append(row[::-1])
        
        # Create list fen_occupacy
        label_occupacy = []
        label_figure =[]
        for row in fen_occupacy_rev_top_left:
            # print("row: " + str(row))
            for figure in row:
                occupied = 0
                if figure != "-":
                    occupied = 1
                # print("fig " + figure + "\tocc " +  str(occupied))
                label_figure.append(figure)
                label_occupacy.append(occupied)
        
        # Debug helper
        # print("\nfen_original")
        # print(fen_original)
        # print("\nfen_occupacy_rev_top_left")
        # print(fen_occupacy_rev_top_left)
        # print("\nfen_occupacy")
        # print(fen_occupacy)

        return label_occupacy, label_figure        

    def create_square_numeration_list(self, turn):
        dict = {}
        positions = []

        # Turn: White
        if(turn):
            for a in range(7, -1, -1):
                img_number = []
                for i in range(0, 8):
                    img_number.append(8 * (a + 1) - 8 + i)
                    # print("i:" + str(i))
                dict["row_" + str(a)] = img_number
                positions = positions + img_number
                # print(dict["row_" + str(a)])

        # Turn: Black
        elif(not turn):
            for a in range(0, 8):
                # print("a: " + str(a))
                img_number = []
                for i in range(7, -1, -1):
                    img_number.append(i + a * 8)
                    # print("i:" + str(i))
                dict["row_" + str(a)] = img_number
                positions = positions + img_number
                # print(dict["row_" + str(a)])
                # print(positions)


        return positions    


if __name__ == "__main__":

    creator_squares = Creator_squares() 
    # creator_squares.create_squares_all_img()
    creator_squares.create_squares_single_img()
import glob   
import os
import csv
import shutil
from pathlib import Path

# Get the current working directory
current_working_dir = os.getcwd()
current_working_dir = current_working_dir.replace('\\','/')

dic_counter_train = {"0": 0,"1": 0,"2": 0,"3": 0,"4": 0,"5": 0,"6": 0,"7": 0,"8": 0,"9": 0,"10": 0,"11": 0}
dic_counter_val = {"0": 0,"1": 0,"2": 0,"3": 0,"4": 0,"5": 0,"6": 0,"7": 0,"8": 0,"9": 0,"10": 0,"11": 0}
dic_counter_test = {"0": 0,"1": 0,"2": 0,"3": 0,"4": 0,"5": 0,"6": 0,"7": 0,"8": 0,"9": 0,"10": 0,"11": 0}
img_path = []
img_label = []
csv_rows_train = []
csv_rows_val = []
csv_rows_test = []

path_dir_not_needed = current_working_dir + "/02_created_data/piece/img_not_needed/"

with open(current_working_dir + "/02_created_data/piece/piece.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        # Create folder
        Path(path_dir_not_needed).mkdir(parents=True, exist_ok=True)
        if line_count == 0:
            csv_rows_train.append(row)
            csv_rows_val.append(row)
            csv_rows_test.append(row)
            print(f'Column names are {", ".join(row)}')
        else:
            path = row[0]
            label = row[1]
            # Train
            # Add img
            if dic_counter_train[str(label)] < 2600:
                dic_counter_train[str(label)] = dic_counter_train[str(label)] + 1
                csv_rows_train.append(row)
            # Val
            # Add img
            elif dic_counter_val[str(label)] < 250:
                dic_counter_val[str(label)] = dic_counter_val[str(label)] + 1
                csv_rows_val.append(row)
            # Test
            # Add img
            elif dic_counter_test[str(label)] < 250:
                dic_counter_test[str(label)] = dic_counter_test[str(label)] + 1
                csv_rows_test.append(row)
            # Val
            # Move img
            # else:
            #     original = current_working_dir + "/02_created_data/piece/" + path
            #     target = current_working_dir + "/02_created_data/piece/img_not_needed/" + path
            #     shutil.move(original, target)
            
            # print("key: " + str(label) + "\tvalue: " + str(dic_counter[str(label)]))
        line_count += 1
    print(f'Processed {line_count} lines.')

# Create label.csv file
path_piece_csv_train = current_working_dir + "/02_created_data/piece/piece_train.csv"
Path(path_piece_csv_train).touch(exist_ok=True)

# Create label.csv file
path_piece_csv_val = current_working_dir + "/02_created_data/piece/piece_val.csv"
Path(path_piece_csv_val).touch(exist_ok=True)

# Create label.csv file
path_piece_csv_test = current_working_dir + "/02_created_data/piece/piece_test.csv"
Path(path_piece_csv_test).touch(exist_ok=True)

# Only write header once at beginning 
with open(path_piece_csv_train, mode='w', newline='') as label_file:
    employee_writer = csv.writer(label_file, delimiter=',') #, quotechar='"', quoting=csv.QUOTE_MINIMAL
    for row in csv_rows_train:
        employee_writer.writerow(row)

# Only write header once at beginning 
with open(path_piece_csv_val, mode='w', newline='') as label_file:
    employee_writer = csv.writer(label_file, delimiter=',') #, quotechar='"', quoting=csv.QUOTE_MINIMAL
    for row in csv_rows_val:
        employee_writer.writerow(row)

# Only write header once at beginning 
with open(path_piece_csv_test, mode='w', newline='') as label_file:
    employee_writer = csv.writer(label_file, delimiter=',') #, quotechar='"', quoting=csv.QUOTE_MINIMAL
    for row in csv_rows_test:
        employee_writer.writerow(row)

# print(dic_counter)
# ML Chess Board Project

# Virtual Environment
1. venv-ml-project-rr-mh\Scripts\activate
2. work
1. deactivate

# Please install
1. pip install
2. pip install opencv-python
3. pip install recap
4. pip install -U scikit-learn
5. pip install chess
6. pip install Pillow
7. pip install --upgrade tensorflow
8. pip install pandas
9. pip install matplotlib
10. pip install tensorf
11. pip install pydot
Maybe environment file?

# to do:
1. import labels -> done
2. gitignore for imgs
3. readme download imgs
4. create datasets


# Steps
Download (01) Origninal Data

Creator Squares
* Split (01) Origninal Data in fields and safe as files with single fields + csv (02 Created Data/Occupancy or 02 Created Data/Piece)
* Which files should be included in the repo? (maybe all we need, also already splitted)

Occupancy classifier
* Use Creator Dataset (currently CreatorDatasetOccupancy.py) to create train, split and valid dataset from 02 Created Data/Occupancy
* Train, test and evaluate the model and safe it

Piece classifier
* Use Creator Dataset to create train, split and valid dataset from 02 Created Data/Piece
* Train, test and evaluate the model and safe it

Fen Creation Demo
* Split 03 Demodata in fields and write json (to know whose turn it is? this is currently not implemented)
* read squares
* classify occupancy using the saved model
* classify the piece using the saved model
* Create FEN and viszualize it

Other things
* piece classification was also tried with transfer learning (also occpancy?)


Report
Write a report in which you briefly describe:
The dataset used.
Chesscog data set
consists of

Any data pre-processing (if needed).
Splitting (at first tried with model, then used chesscog computer vision technique)
Creation of datasets

The architecture of your DL model.
Pictures and descriptions of CNN_occupancy, CNN_piece, TF_Piece

The performances of your model.
Train, Validation and Test Accuracies
Learning Graphs as pictures
Final result with demo (multiplied Accuracy)

How you applied the different techniques taught in class in your project.
CNN
* Conv Layer
* MaxPool
* DropOut
* L2 Regularization
* data augmentation (not profitable)
* Focal Loss
* binary Loss vs categorical loss
* diff activation functions

TL
* Xception pretrained model (also tried resnet and vgg but did not perform well)
* finetuning by unfreezing layers

The main challenges you faced (at least one per member).
* Square creation
* Input data (right dimenions; difference between import withcv2 and tf)
* Cluster, Learning Time
* Transfer learning -> everytime same result

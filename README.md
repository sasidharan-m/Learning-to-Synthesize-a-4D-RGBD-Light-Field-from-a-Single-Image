# Learning-to-Synthesize-a-4D-RGBD-Light-Field-from-a-Single-Image
PyTorch Implementation of the paper "Learning to Synthesize a 4D RGBD Light Field from a Single Image"

Paper: https://arxiv.org/abs/1708.03292 

Instructions for Training:
    --> Setup the python environment using pip install -r requirements.txt
    --> Prepare the dataset for training. Follow the below file structure:
        (Visualization shown for 2 image dataset with 13 x 13 grid of aub-aperture views)
            Directory containing the Light Field data 
            |
            --- Folder for image 1
            |   |
            |   --- All sub-aperture images named view_0_0.png to view_12_12.png
            |
            |
            ---- Folder for image 2
                |
                --- All sub-aperture images named view_0_0.png to view_12_12.png
    --> Call the train.py passing in the path of the directory containing the Light Field data and the path to where you want the weights to be stored
    --> Do train.py -h for information on how to pass in the paths as arguments and other optional arguments


Instructions for Inference:
    --> Setup the python environment using pip install -r requirements.txt 
    --> Call the test.py passing in the path to the pre-trained weights, path to the input image and the path to the directory where the generated Light Field data should be saved
    --> Do test.py -h for information on how to pass in the paths as arguments and other optional arguments
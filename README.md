# Learning-to-Synthesize-a-4D-RGBD-Light-Field-from-a-Single-Image
PyTorch Implementation of the paper "Learning to Synthesize a 4D RGBD Light Field from a Single Image"

Paper: https://arxiv.org/abs/1708.03292 

Instructions for Training:
    --> Setup the python environment using pip install -r requirements.txt <br>
    --> Prepare the dataset for training. Follow the below file structure: <br>
        (Visualization shown for 2 image dataset with 13 x 13 grid of aub-apert ure views) <br>
            Directory containing the Light Field data <br>
            | <br>
            --- Folder for image 1 <br>
            |   | <br>
            |   --- All sub-aperture images named view_0_0.png to view_12_12.png<br>
            |<br>
            |<br>
            ---- Folder for image 2<br>
                |<br>
                --- All sub-aperture images named view_0_0.png to view_12_12.png<br>
    --> Call the train.py passing in the path of the directory containing the Light Field data and the path to where you want the weights to be stored <br>
    --> Do train.py -h for information on how to pass in the paths as arguments and other optional arguments <br>


Instructions for Inference:<br>
    --> Setup the python environment using pip install -r requirements.txt <br>
    --> Call the test.py passing in the path to the pre-trained weights, path to the input image and the path to the directory where the generated Light Field data should be saved<br>
    --> Do test.py -h for information on how to pass in the paths as arguments and other optional arguments<br>
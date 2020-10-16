# syncrack_generator
## Description
The current repository provides a tool for massive generation of synthetic images emulating pavement and cracks on such pavement.

Even though real-life annotated datasets exist, the provided labels tend to be weak because a clean accurate annotation in real-life samples is very costful. The ultimate goal of this generator is to provide a tool for training and testing crack detection methods/models at pixel-wise level with clean accurate annotations (for each image, a ground truth annotation is provided).

An example of a generated pavement image, as well as a crack on that image and its corresponding ground-truth are contained in the "examples" folder.

![alt text](https://github.com/Sutadasuto/syncrack_generator/blob/main/examples/img.jpg?width=210) ![alt text](https://github.com/Sutadasuto/syncrack_generator/blob/main/examples/gt.jpg?raw=true)

A dataset of 500 pavement images with cracks and their corresponding ground-truths is provided in the "syncrack_dataset.zip" file (those images were generated, of course, using the code contained in this repository).

## How to run
### Needed packages
The current code was tested using Python 3.7.6. Additional packages needed:
* opencv=4.4.0
* scipy=1.4.1
* noise=1.2.2

For user convenience, a conda environment setup file (environment.yml) is provided in this repository.

### Running the program
To run with default parameters, you can simply run
```
python generate_datase.py
```

This will generate 500 images (with their corresponding ground-truths) with 420x360 pixels size (Width x Height) in a folder called "syncrack_dataset" inside the repository's root.

You can change number of images, image size and destination folder by providing additional arguments. Run
```
python generate_datase.py -h
```
for more info.

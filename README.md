# syncrack_generator
## Description
The current repository provides a tool for massive generation of synthetic images emulating pavement and cracks on such pavement.

Even though real-life annotated datasets exist, the provided labels tend to be weak because a clean accurate annotation in real-life samples is very costful. The ultimate goal of this generator is to provide a tool for training and testing crack detection methods/models at pixel-wise level with clean accurate annotations (for each image, a ground truth annotation is provided).

An example of a generated pavement image, as well as a crack on that image and its corresponding ground-truth are contained in the "examples" folder.

<p align="center">
![alt text](https://github.com/Sutadasuto/syncrack_generator/blob/main/examples/img.jpg?raw=true) ![alt text](https://github.com/Sutadasuto/syncrack_generator/blob/main/examples/gt.png?raw=true)
</p>

A dataset of 500 pavement images with cracks and their corresponding ground-truths is provided in the "syncrack_dataset.zip" file (those images were generated, of course, using the code contained in this repository).

Additionally, we include a function aimed to attack the precise pixel-wise annotations in order to make them look similar to how a human would make the annotations in real life: missing some parts of the cracks, drawing the cracks thicker or thinner than reality, etc. An example of an attacked ground-truth is contained in the "examples weak labels" folder.

The images show the pavement picture as well as a comparison between the actual ground-truth and the attacked annotation: in the annotations image, the left part is the actual ground-truth, tha middle one is the attacked annotation, and the right one is a visual comparison of the overlapping of both; white means agreement between actual and attacked annotations, blue means real crack pixels lacking in the attacked mask, and red means cracks in the attacked annotation that are not part of the actual ground-truth.

<p align="center">
![alt text](https://github.com/Sutadasuto/syncrack_generator/blob/main/examples_weak_labels/img.jpg?raw=true) ![alt text](https://github.com/Sutadasuto/syncrack_generator/blob/main/examples_weak_labels/gt_comparison.png?raw=true)
</p>

## How to run
### Needed packages
The current code was tested using Python 3.7.6. Additional packages needed:
* opencv=4.4.0
* scipy=1.4.1
* noise=1.2.2

For user convenience, a conda environment setup file (environment.yml) is provided in this repository.

### Running the data generation program
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

### Running the annotation attack program

You must run
```
python -c "from weak_labels import attack_dataset;attack_dataset('path/to/dataset')"
```
substituting 'path/to/dataset' by the path to a dataset folder generated with this repository (e.g. if running "generate_dataset.py" with default arguments, you can substitute 'path/to/dataset' with 'syncrack_dataset').

This command will create two new folders in the parent folder containing the dataset directory. 'directory_attacked' will contain a copy of the database, but replacing the ground-truth images with the attacked annotations; 'directory_label_comparison' will contain the comparisons of the actual and attacked annotations per image (like in the image above).

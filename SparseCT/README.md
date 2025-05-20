# SparseCT

## Dimensionality Notation
data.py contains functions to generate the synthetic dataset of ellipses. 
We suppose that the dataset has dimension (N, x, y, t) where:

* N -> number of images in the dataset.
* x, y -> the dimension of each slice.
* t -> number of slices.

We want the projector to rotate with axis of rotation x , since the segments added to each image is places on the plane
yt.

E.g. (N, x, y, t) = (N, 600, 450, 32). Then each projection has dimension (600, 450), with rotation axis x of dimension
600, and the line segment are on the plane of dimension (450 x 32).

This is coherent with the notation of the TIGRE Toolbox.
To help visualization, it is possible to use functions in visualize.py.

The data can be saved in .npy, .npz, .tif, and the code for AstraToolbox now works only image-per-image.

In the .tif format, the image needs to be reshaped in (N, t, x, y) to be correctly visualized by ImageJ.

## Geometry

Rotation Axis: x

Source moves on the plane: yt

Data dimensionality should be: (x, y, t) = (512, 512, 32).

## Examples
An example on how to use the functions is in the file main.py.

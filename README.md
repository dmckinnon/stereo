# Stereoscopy

This exercise is for learning all about stereoscopic depth maps, and, if it gets to it, scene reconstruction from multiple views.
The basic idea is that you have multiple views of the same scene, and you know where your cameras are relative to each other - using this
information, you can determine the depth (to within some error bound) of any point within the image. For this, the cameras must be
[calibrated](https://github.com/dmckinnon/calibration/blob/master/README.md) instrinsically, and extrinsically (meaning you know the
 rigid-body transformation from one camera to the other).
 
I won't be following any specific paper; rather, just bringing in the theory here and there where necessary. I'll try to explain as best
I can, but better than to read my explanations is to read the theory behind it. The dataset I'm working from is [here](https://vision.in.tum.de/data/datasets/3dreconstruction)
 - it's superb, it has a lot of high quality images of objects, and complete calibration data for each view. 

# Contents:
to add

# Overview
To get a depth map - that is, an image that contains the depth of every point in a scene at each pixel - one needs at least two views, and calibrated cameras. 
The method is then to detect common elements between the two images (commonly called 'Features' - see the next section), and for those
common elements that can easily be found in both, use the knowledge we have about how the two cameras are situated with respect to 
where each spot is to figure out where it is in 3D space. To put this more clearly, let's say two cameras are both looking at my face 
from an angle slightly off center - one to the left, one to the right. They can both see my nose. Since we know the transform from one camera
to the other, and each camera can project a ray through where my nose is, we can compute the intersection of those rays to say that my
nose is a depth __d__ from the cameras. Repeat for each point that can be detected in both images, interpolate for anything in between, 
and bam, there's your depth map. 

The following sections break this down into greater detail, and go over the major components of my code. 

# Feature Detection
I've written about [feature detection elsewhere](https://github.com/dmckinnon/stitch#feature-detection), but I'll go over it here too.


# Feature Description and Matching
I've written about [feature description and matching elsewhere](https://github.com/dmckinnon/stitch#feature-description), but I'll go over it again. 


# Triangulation
epipolar numpty bumpty

# Disparity map






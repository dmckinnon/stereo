# Stereoscopy

This exercise is for learning all about stereoscopic depth maps, and, if it gets to it, scene reconstruction from multiple views.
The basic idea is that you have multiple views of the same scene, and you know where your cameras are relative to each other - using this
information, you can determine the depth (to within some error bound) of any point within the image. For this, the cameras must be
[calibrated](https://github.com/dmckinnon/calibration/blob/master/README.md) instrinsically, and extrinsically (meaning you know the
 rigid-body transformation from one camera to the other).
 
I won't be following any specific paper; rather, just bringing in the theory here and there where necessary. I'll try to explain as best
I can, but better than to read my explanations is to read the theory behind it. The dataset I'm working from is [here](https://vision.in.tum.de/data/datasets/3dreconstruction) - it's superb, it has a lot of high quality images of objects, and complete calibration data for each view. 

# Contents:
to add

# Overview
To get a depth map - that is, an image that contains the depth of every point in a scene at each pixel - one needs at least two views, and calibrated cameras. 
The method is then to detect common elements between the two images (commonly called 'Features' - see the next section), and for those
common elements that can easily be found in both, use the knowledge we have about how the two cameras are situated with respect to 
where each spot is to figure out where it is in 3D space. To put this more clearly, let's say two cameras are both looking at my face 
from an angle slightly off center - one to the left, one to the right. They can both see my nose. Since we know the transform from one camera
to the other, and each camera can project a ray through where my nose is, we can compute the intersection of those rays to say that my
nose is a depth *d* from the cameras. Repeat for each point that can be detected in both images, interpolate for anything in between, 
and bam, there's your depth map. 

The following sections break this down into greater detail, and go over the major components of my code. 

# Feature Detection
I've written about [feature detection elsewhere](https://github.com/dmckinnon/stitch#feature-detection), but I'll go over it here too. TODO


# Feature Description and Matching
I've written about [feature description and matching elsewhere](https://github.com/dmckinnon/stitch#feature-description), but I'll go over it again. TODO

# Fundamental Matrix, or Essential Matrix
Since we have a calibration, we're technically finding the Essential matrix. These matrices are basically fancy names for "the 3D transform from one camera to another". If we have two cameras, *C_0* and *C_1*, situated at two points in space looking at the same thing, then there exists some rigid-body transform that takes coordinates in the frame of *C_0* and puts them into the frame of *C_1*. A rigid-body transform is basically a rotation and a translation - or a slide. You can imagine sitting two ye olde-timey cameras on a table, pointing at, say, a wall nearby. They might be at some angle to each other. Grab one, turn it a bit, then slide it, and bam, it's in exactly the same position as the other, facing the same way (assuming these are magical cameras that can move through each other). That's what this matrix does. The __Essential Matrix__ takes points projected from a calibrated camera to the projective plane for another calibrated camera (that is, you need the camera calibration matrix). The __Fundamental Matrix__ doesn't care about calibration, and takes points in the projective image plane for one camera to the projective image plane of the other camera. 

We compute this with an algorithm called the 8-Point Algorithm, designed by Hartley (link). 

# Triangulation
So triangulation is basically the idea of we have several known points *P_i*, and an unknown point *X*, and we know where *X* is relative to each *P_i*, so we use that to figure out our best approximation for *X*. I have a couple of ideas for this algorithm, and I'm going to try those before reading the literature, to see if I can figure it out. 

#### First idea:
We know that *X* lies on a ray from *C_i*, being camera *i*, through the point *P_i* in the image *i*. So if we have two images, from cameras *C_0* and *C_1*, and we know the transform from camera 0 to camera 1 (which we are given at the start or can compute), then surely we can just equate these two rays to find the point of intersection?

Good guess, but no, I was wrong with this. They may not be perfectly equal (in an ideal world, they are, but this is obviously not ideal). So they may never have intersection. Next!


#### Second idea:
We can pick a starting point of a known depth *d* on the ray *X_i* from *C_i* and project this into some image *j* using the Fundamental matrix between images *i* and *j*. Then compare the projected patch from image *i* to the patch at that location in *j*. If they match, bam, *d* is your depth. If not, increase *d* by some small delta and repeat, continuing until you hit your maximum depth. The best matching patch is where you stop and say that's your depth. 

Trouble is, this is very computationally expensive, and unnecessary. There is a better way. 


#### Third idea:
Going back to the first idea, we have those two rays. What we can do is take the difference of those two rays ... and minimise it! Surely the point where the rays are closest is the best depth for *X*. So, get the equation for their difference, put this in terms of a vector of the depth parameters, and get the Jacobian. Equate to zero, solve, and bam we have our depth parameters. Use these. 
Now this may still not be the best way, but it's a decent method I thought of. 


#### Oops
Turns out I'm rather wrong. The usual Euclidean geometric principles I'm relying on don't actually apply in Projective Geometry. (why?)
Let's see what the actual literature says. 

Firstly, there is [Hartley and Sturm's original 1997 paper on triangulation](https://users.cecs.anu.edu.au/~hartley/Papers/triangulation/triangulation.pdf), which details several methods and presents an optimal one. This is complicated to explain here, but the crux is that instead of adjusting depth for minimisation, they realised that there are a pair of true points that the feature points are approximating. Let's adjust the feature points such that we minimise the distance between the feature points and the theoretical true points. Their formulation for this problem results in a 6-degree polynomial, which must be solved to find the minimum. The advantage of their algorithm is that it is closed form and requires no iteration. 

Then came [Kanatani's paper on triangulation](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.220.724&rep=rep1&type=pdf), which aimed to improve on Hartley and Sturm with an iterative algorithm. 

Finally, [Peter Lindstrom improved on Kanatani's method](https://e-reports-ext.llnl.gov/pdf/384387.pdf) by designing a non-iterative quadratic solution that is faster than Hartley and Sturm, and more stable. 

# Disparity map
Once we've triangulated every matching pair of points, we can create a disparity map. BAsically an image where each pixel is the depth gotta add more here woo





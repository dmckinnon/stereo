# Stereoscopy

This exercise is for learning all about stereoscopic depth maps. The basic idea is that you have multiple views of the same scene, and you know where your cameras are relative to each other - using this
information, you can determine the depth (to within some error bound) of any point within the image. For this, the cameras must be
[calibrated](https://github.com/dmckinnon/calibration/blob/master/README.md) instrinsically. If you don't know where your cameras are relative to each other at the start, this can be derived from shared information across the images - up to a scale factor (this means that we might know that the cameras are 1 'unit' apart and a point is, say, 2 units away from the first camera. But the units could be metres, kilometres, parsecs, whatever, and we wouldn't be able to find out). So we can know relative distances. This is ok, because I'm trying to find relative depths in the scene. 
 
I won't be following any specific paper; rather, just bringing in the theory here and there where necessary. I'll try to explain as best
I can, but better than to read my explanations is to read the theory behind it. The dataset I'm working from is [this one](http://vision.middlebury.edu/stereo/data/scenes2014/) - it's superb, it has a lot of high quality images of objects, and complete calibration data for each view. [Here is another dataset](https://vision.in.tum.de/data/datasets/3dreconstruction) - this one does multiple views, not just two. It's harder for the features I'm going to be using, but it's worth a shot if you want to expand on this tutorial. 
For displaying images, I'm using OpenCV; [here's how to install it on Windows 10](https://www.youtube.com/watch?v=MXqpHIMdKfU&feature=youtu.be).

### Building and running
I've included the .vcxproj if you're a visual studio user, to just grab straight out of the box. You'll need to install opencv/get the headers and the libs and dlls for it, and modify the paths correctly - look in project properties, under includes and linker. 
While I did this on Windows in Visual Studio, the only thing that might be platform dependent is how I grab the file path for the input data - all the algorithms etc are independent of platform. The only dependencies otherwise are Eigen and OpenCV, just for a few things like the Mat types, Gaussian blur, the Sobel operator, etc. 

# Contents:

- [Overview](https://github.com/dmckinnon/stereo#overview)
- [Feature Description and Matching](https://github.com/dmckinnon/stereo#feature-description-and-matching)
- [Fundamental Matrix, or Essential Matrix](https://github.com/dmckinnon/stereo#fundamental-matrix-or-essential-matrix)
- [Triangulation](https://github.com/dmckinnon/stereo#triangulation)
- [Rectification](https://github.com/dmckinnon/stereo#rectification)
- [Conclusion](https://github.com/dmckinnon/stereo#results-and-conclusion)


# Overview
A quick note before we begin: I assume a basic knowledge of linear algebra, matrices, 3D transformations, etc. If you don't know this, that is fine and this can still be helpful. I'm just not going to explain everything from scratch - I'll link where I can for more clarity. You can code all this without knowing the underlying theory (but it helps).


To get a depth map - that is, an image that contains the depth of every point in a scene at each pixel, with respect to a camera and here I'm just picking one of the cameras arbitrarily - one needs at least two views, and calibrated cameras. 
One method is to detect common elements between the two images (commonly called 'Features' - see the next section), and for those
common elements that can easily be found in both, we can derive a relationship between the two views - namely, how one is situated relative to the other. From this and the camera calibration matrices we can figure out the transform (rotation and translation) in 3D between the cameras. Once we know this, we can figure out the depth of each feature in the cameras. To put this more clearly, let's say two cameras are both looking at my face 
from an angle slightly off center - one to the left, one to the right. They can both see my nose. Since we know the transform from one camera
to the other, and each camera can project a ray through where my nose is (that is, if one imagines the image sitting a small distance in front of the camera, the ray is a line from the camera centre through the pixel in the image, and going out from there), we can compute the intersection of those rays to say that my
nose is a depth *d* from the cameras. Repeat for each feature that can be detected in both images, interpolate for anything in between, 
and there's your depth map. This is called __Triangulation__.


Another method does something slightly different, and is called __Rectification__. Once again, we find how the two images relate, but now we warp both images. It's easy to picture a couple of cameras facing the same scene from different angles. Well, we transform - rotate and skew - the images so it's as if the cameras are both on a line facing exactly the same way, and they take very warped pictures (meaning the images, as the cameras face them, would not be rectangular, but a squashed rectangle) of the same thing from different angles. When we warp them the right way, we can get the images to line up perfectly so that horizontal lines in the images correspond (meaning a point in image 0 with a y coordinate of 0 (or whatever else) will have exactly the same y coordinate in image 1, just a different x coordinate), and the depth of a particular pixel is related to the difference between its *x* coordinates in each image. In this way we can get the depth of every pixel easily, and again, there's the depth map.

The following sections break this down into greater detail, and go over the major components of my code. 

# Feature Detection
I've written about [feature detection elsewhere](https://github.com/dmckinnon/stitch#feature-detection), but I'll go over it here too. "Features" can be a bit of a vague concept in computer vision, and it certainly was for me at the start. Just call some OpenCV function and magical dots appear on your image! But what are they? How do you get them? Why do we want them?

Features are basically identifiable parts of images. An image is an array of numbers. How do I know what is identifiable if I see it again in another image? How do I know what is important to track? A 'feature point' is this, and Feature Detection finds these points. These points are useful because we can find them again in the other image (see the paragraph below for a greater description of this). So we find a feature on a part of one image, and hopefully we can find the same feature in the other image. Using Feature Descriptors, the next section, we can compare features and know that we have found the same one. Multiple matched features then helps us in the later section Feature Matching, where we try to figure out how to go from one image to the other. If we have several feature points in one image, and have found the same in the other image, then we can figure out how the two images fit together ... and that, right there, is how panoramas work!


There are a lot of different types of features, based on how you look for them.

#### Some common types of features:

- FAST features
- SIFT features
- SURF features
- ORB features
- [Difference of Gaussian](https://en.wikipedia.org/wiki/Difference_of_Gaussians) features
- [Determinant of hessian](https://milania.de/blog/Introduction_to_the_Hessian_feature_detector_for_finding_blobs_in_an_image) features 


There are plenty more. Some are simple, some are ... rather complex (read the wikipedia page for SIFT features, and enjoy). They each might find slightly different things, but in general, what 'feature detectors' aim to do is find points in an image that are sufficiently distinct that you can easily find that same feature again in another image - a future one, or the other of a stereo pair, for example. Features are distinct things like corners (of a table, of an eye, of a leaf, whatever), or edges, or points in the image where there is a lot of change in more than just one direction. To give an example of what is not a feature, think of a blank wall. Take a small part of it. Could you find that bit again on the wall? That exact bit? It's not very distinct, so you likely couldn't. Then take a picture of someone's face. If I gave you a small image snippet containing just a bit of the corner of an eye, you'd find where it fit very quickly. AIShack has a [rather good overview](http://aishack.in/tutorials/features/) of the general theory of features.

In [my other tutorial](https://github.com/dmckinnon/stitch) I used FAST features, also called FAST corners. If you want a quick overview, see my other tutorial or OpenCV's [Explanation of FAST](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_fast/py_fast.html). 
Here's part of it copied and pasted:

The idea behind FAST is that corners usually have two lines leading away from them, and that the intensity of the pixels in one of the two angles created by those lines will be either lighter or darker than the other. For example, think of the corner of a roof with the sun above. Two lines (the roof edges) come out from the corner, and below will be darker than above. The way a FAST feature detector works is that for each pixel, it scans a circle of 16 pixels around it, about 3 pixels radius, and compares the intensities to the centre intensity (plus or minus a threshold). If there is a stretch of sequential pixels 12 or more in length that are all of greater intensity (plus a threshold) than the centre, or lesser intensity (minus a threshold) than the centre, this is deemed a FEATURE. (OpenCV's explanation has some better visuals)


# Feature Description and Matching
I've written about [feature description and matching elsewhere](https://github.com/dmckinnon/stitch#feature-description), but I'll go over it again. 

Your average image might have over a thousand features - this is quite a lot to process later, as you'll see. We don't need that many features to figure out how the panorama fits together (100 feature points per image is more than enough). So we should remove some features. How do we know which ones to remove? We compute a 'score' for each feature, that measures how strong that feature is, and we get rid of all features below a certain score. A 'strong' feature here means a feature point that is really clear and distinct, and easy to find again. A 'weak' feature is one that is vague, and would easily be mismatched. 

Once again, there are many methods of scoring features, and one of the most famous is the Shi-Tomasi score, invented by Shi and Tomasi in 1994. Here is their [original paper](http://www.ai.mit.edu/courses/6.891/handouts/shi94good.pdf).

AI Shack has a [good article](http://aishack.in/tutorials/shitomasi-corner-detector/) on the Shi Tomasi score, but it relies on some [background knowledge](http://aishack.in/tutorials/harris-corner-detector/), or having read the previous articles linked at the bottom (they're short and easy and good).

Essentially, for the feature point and a small patch surrounding it, a matrix called the [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) is computed. This is basically the two dimensional version of the [gradient](https://en.wikipedia.org/wiki/Sobel_operator) of a line. The way we compute this is documented [here](http://aishack.in/tutorials/harris-corner-detector/). Then, we compute the __eigenvalues__ for this matrix. Since this is a two-by-two matrix (see the previous link), the eigenvalues are just the solutions to a simple quadratic equation. The Shi-Tomasi score, then, is simply the minimum eigenvalue. If you're not sure what eigenvalues are, then these are values unique to a matrix, one per dimension for a square matrix, that ... are indicative of its scale in the direction of its eigenvectors - special vectors of a matrix. Think of a matrix as a transform. The eigenvectors are the vectors along which the matrix transform is maximised or minimised, and the eigenvalues tell how much this occurs. If we know the minimum eigenvalue, then we know that the gradient matrix, the Jacobian, will cause *at least* a certain amount of change. Since we want points of sharp change to class as strong features, looking for eigenvalues of a certain minimum or above will give us points that are strong.  

In other words, for a two-by-two jacobian matrix, the eigenvalues define how strong the gradient is in the direction of the two eigen__vectors__ of the matrix. Basically, how much change we have in each direction. For a good corner, you should have a sharp image gradient (difference in pixel intensity) in both directions, so the minimum eigenvalue won't be that small. For just an edge, you'll have a sharp gradient in one direction but not the other, meaning one eigenvalue will be small. 

We then have a cutoff threshold for this value - another tunable parameter - and everything with a score below this - that is to say, every feature with a minimal eigenvalue of value lower than this threshold - is thrown away and ignored. Every feature with a higher score is kept. 


The final stage of the feature scoring is to perform Non-Maximal Suppression (and unfortunately I can't find a good explanation online). The theory of Non-Maximal Suppression is that for where you have a group of points clustered in an area, like having, say, twenty features of a good score in the same little patch of the image ... you don't need all of these. You've already registered a strong feature there. So you suppress, that is to say, put aside, the weaker features within some radius of the strongest in that patch. In other words, you suppress the points that aren't maximal. Hence, non-maximal suppression. 

So we do this over our feature set that we've already cut down. For every feature, if there are any features in a 5x5 patch around it that are weaker, we suppress these too, just to reduce the amount we have to process over.

I'm now going to start referring to these feature points we've been talking about as 'keypoints' - it's a fancy word for 'important point' that is used often in literature and other code, so best to get used to it if you're going to do further research. We now want to create keypoint descriptors, which are unique numbers for each keypoint/feature so that if we ever saw this again, we could easily identify it, or at least say "look, this one is very similar to that one, as their identifying numbers are close". This is what I mentioned before in Feature Detection. We found feature points in each image. Now we want to try to see which ones are the same, to find matching points in each image. How do we know if two features are actually referring to the same image patch? We need some distinct identifier - a descriptor. Each feature point will have its own descriptor, and when we try to figure out which of the first image's features are also seen in the second image, we try to see which features actually have the same descriptor, or very similar descriptors. This works because the descriptor is based on the patch of the image around the feature point - so if we see this again, it should register as the same. 

So, for now, how do we make these 'keypoint IDs'? What identifies a keypoint uniquely, or rather, how can we describe this? 
As always, there are many ways of making descriptors for keypoints. BRIEF descriptors, SURF descriptors ... different descriptors have different strengths and weaknesses, like robustness to various conditions, or the descriptors might use fewer pixels but have similar identifying power ... but here, because I wanted to learn them, I chose SIFT descriptors. This stands for the [Scale-Invariant Feature Transform](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform). 

Once again, AI Shack has [quite a good description of SIFT descriptors](http://aishack.in/tutorials/sift-scale-invariant-feature-transform-features/) - in fact, a good description of the entire [SIFT feature detector](http://www.aishack.in/tutorials/sift-scale-invariant-feature-transform-introduction/) too. I encourage reading both (the second is the whole series, the first link is just the last page in the series. It's not a long series).


I'll attempt to write my own briefer description here. So what's unique about a feature? Well, the patch around and including it, up to a certain size. On an image of a face, the corner of an eye is pretty unique, if you get a decent-sized patch around it - a bit of eye, some crinkles if the face is smiling, etc. What makes these things unique? Or rather, how do we capture this uniqueness?

Once again, gradients. The change in intensity - going from mild skin (say. The skin could be darker, but this example works best on a really pale person), to the darker shadow of a crinkle, back to mild skin - can be measured best by the gradient of a line through that area. If we get a little patch over a face crinkle, then we could find a dominant gradient vector for that patch - pointing perpendicular to the crinkle since that's the direction of the greatest change (light to dark). This dominant gradient vector, both in angle and magnitude, can then be used to identify such a patch. 

Ok, so we can identify a little patch with one edge. But aren't features corners? So, let's do more patches. SIFT creates a 4x4 grid of patches, with the feature at the centre of it all. Each patch is 4x4 pixels (so 16x16 pixels total to scan over). We find the dominant gradient vector for each patch - the vector angles are bucketed into one of 8 buckets, just to simplify things - and then we list all these in sequence. This string of bits - magnitude, angle, magnitude, angle, etc, for 16 gradient vectors around the feature, is the unqiue descriptor that defines the feature. For reference, I strongly recommend looking at the AI Shack link above. 

There are a couple of things worth mentioning now. The angle of each gradient is taken relative to the angle of the feature (when we create the feature, we measure the angle of the gradient in a smaller patch centred on the feature), to make this descriptor rotationally invariant. Another thing is that all the magnitudes are normalised, capped at 0.2, and normalised again, so as to make the vector invariant to drastic illumination. This all means that if we look at the feature again from a different angle, or under mildly different lighting, we should still be able to uniquely match it to the original. 

All of this is explained with nice diagrams in the AI Shack link above. 

So far we have found features, cut out the ones we don't want, and then made unique descriptors, or IDs, for the remainder. What's next? Matching them! This is so that we know which parts of the first image are in the second image, and vice versa - basically, where do the images overlap and how do they overlap? Does a feature in a corner of image 1 match with a feature in the middle of image 2? Yes? Is that how the images should line up then? Feature Matching answers this question - which bits of the first image are in the second and where?

Now we have to get the features from the left image and features from the right image and ask "which of these are the same thing?" and then pair them up. There are some complicated ways to do this, that work fast and optimise certain scenarios (look up k-d trees, for example) but what I have done here is, by and large, pretty simple. I have two lists of features. For each feature in one list (I use the list from the left image), search through all features in the second list, and find the closest and second closest. 'Closest' here is defined by the norm of the vector difference between the descriptors. 

When we have found the closest and second closest right-image features for a particular left-image feature, we take the ratio of their distances to the left-image feature to compare them. If DistanceToClosestRightImageFeature / DistanceToSecondClosestRightImageFeature < 0.8, this is considered a strong enough match, and we store these matching left and right features together. What is this ratio test? This was developed by Lowe, who also invented SIFT. His reasoning was that for this match to be considered strong, the feature closest in the descriptor space must be the closest feature by a clear bound - that is, it stands out and is obviously the closest, and not like "oh, it's a tiny bit closer than this other one". Mathematically, the closest feature should be less than 80 percent of the next closest feature. 

# Fundamental Matrix, or Essential Matrix
Before I go into what the __Fundamental__ matrix is or what the __Essential__ matrix is, I'll talk briefly about what a __calibration__ or __intrinsic__ matrix is. 

### Camera Calibration
I have written on this topic [elsewhere](https://github.com/dmckinnon/calibration/blob/master/README.md), and there are [many](https://en.wikipedia.org/wiki/Camera_resectioning) [good](http://ksimek.github.io/2013/08/13/intrinsic/) resources out there for this topic. Honestly, I think both [wikipedia](https://en.wikipedia.org/wiki/Camera_resectioning) and the [github link](http://ksimek.github.io/2013/08/13/intrinsic/) are better than what I can write, but I'll give it a shot. If we want to transform points from pixel space - that is, in the 2D space of an image - to 3D space ... and we do, since we're trying to find the depth of real 3D points ... then we need to know the camera intrinsic parameters. Why? Well, with these we can form the __Camera matrix__ and use this to convert points between pixel space and 3D space. HOWEVER there is one important caveat: we can go from a 3D point to a 2D point easily enough, but when going from 2D to 3D, with only this matrix, *we don't know the depth of the point*. That's what this entire tutorial is about. So, since we don't know the depth, we project a 2D point to a __3D ray__. The true 3D point could be anywhere along that ray, going from the camera centre out into the world along the line of the pixel coordinate. 

Important caveat aside, the next question is how does this matrix work? Well, it contains the camera parameters like focal length in x, focal length in y, camera skew, and the pincipal point - which is the offset of the camera centre from the image centre (yeah, they might not be the same). So when we get a 3D point and convert it to image space, we multiply the *x* coordinate by *x* focal length, add the skew if any to skew the image, and offset by the principal point's *x* component. Then, for *y*, multiply by *y* focal length, and offset by *y* principal point. This gets the point in pixel coordinates. To go from pixels to the real world ... use the inverse of this matrix. 

This matrix is commonly denoted *K* in literature. 

### Essential Fundamentals
Since we have a calibration matrix, we're technically finding the [Essential matrix, but we can still find the Fundamental matrix](https://www.cc.gatech.edu/~afb/classes/CS4495-Fall2013/slides/CS4495-09-TwoViews-2.pdf). These matrices are basically fancy names for "the 3D transform from one camera to another". If we have two cameras, *C_0* and *C_1*, situated at two points in space looking at the same thing, then there exists some rigid-body transform that takes coordinates in the frame of *C_0* and puts them into the frame of *C_1*. A rigid-body transform is basically a rotation and a translation - or a slide. You can imagine sitting two ye olde-timey cameras on a table, pointing at, say, a wall nearby. They might be at some angle to each other. Grab one, turn it a bit, then slide it, and bam, it's in exactly the same position as the other, facing the same way (assuming these are magical cameras that can move through each other). That's what this matrix does. The __Essential Matrix__ takes 3D points from one camera frame, and puts them in the other's frame. The __Fundamental Matrix__ doesn't care about calibration, and takes points in the projective image plane for one camera to the projective image plane of the other camera. To convert between the two, you need the [intrinsic camera matrix](https://www.mathworks.com/help/vision/ug/camera-calibration.html). If we have an essential matrix *E* for a camera pair, a fundamental matrix *F* for the same cameras, and calibration matrices (or intrinsic matrices) *K* and *K'*, then these relate by [*E* = (*K'* ^T) *F* *K*](https://en.wikipedia.org/wiki/Fundamental_matrix_(computer_vision)), where *K*^T means *K* transposed. 


We compute this with an algorithm called the [8-Point Algorithm](https://en.wikipedia.org/wiki/Eight-point_algorithm), designed by [Hartley](http://www.cs.cmu.edu/afs/andrew/scs/cs/15-463/f07/proj_final/www/amichals/fundamental.pdf) ([Here's another way of framing it](https://cs.adelaide.edu.au/~wojtek/papers/pami-nals2.pdf)). 

TODO: explain the following better
This algorithm works in a very similar way to how we [find a homography between the images](https://github.com/dmckinnon/stitch#finding-the-best-transform), except that we have a different equation. If we have corresponding points *x* and *x'* and fundamental matrix *F* then these relate by *x**F**x'* = __0__. Using this constraint we can form a system of linear equations for each element of *F*. Then we use our old friend, [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition) to compute a result for *F*. Note that the points *x* and *x'* should be normalised to have zero mean and standard deviation of 1 to remove the possibility of numerical error. This can come from some points having values of 1000, or so, and others 0.1. An error of a half affects one point significantly more than the other, but shouldn't. 


# Triangulation
Once we have the Fundamental Matrix or Essential Matrix we can triangulate the points. So triangulation is basically the idea that we know where the same point is in each image, but not where it is in 3D space. But we can get the depthless ray for the point in each image space, and presumably, where those rays cross or somewhere near, is where the point is, and we can measure the depth along the ray and bam. 

#### First idea:
We know that the 3D point *X* lies on a ray from *C_i*, being camera *i*, through the point *P_i* in the image *i*. So if we have two images, from cameras *C_0* and *C_1*, and we know the transform from camera 0 to camera 1 (which we are given at the start or can compute), then surely we can just equate these two rays to find the point of intersection?

Good guess, but no, I was wrong with this. They may not be perfectly equal (in an ideal world, they are, but this is obviously not ideal). So they may never have intersection. Next!


#### Second idea:
We can pick a starting point of a known depth *d* on the ray *X_i* from *C_i* and project this into some image *j* using the Fundamental matrix between images *i* and *j*. Then compare the projected patch from image *i* to the patch at that location in *j*. If they match, bam, *d* is your depth. If not, increase *d* by some small delta and repeat, continuing until you hit your maximum depth. The best matching patch is where you stop and say that's your depth. 

Trouble is, this is very computationally expensive, and unnecessary. There is a better way. 


#### Third idea:
Going back to the first idea, we have those two rays. What we can do is take the difference of those two rays ... and minimise it! Surely the point where the rays are closest is the best depth for *X*. So, get the equation for their difference, put this in terms of a vector of the depth parameters, and get the Jacobian. Equate to zero, solve, and bam we have our depth parameters. Use these. 
Now this may still not be the best way, but it's a decent method I thought of. 


#### Oops
Turns out I'm rather wrong. Or rather, my first idea was close, my third idea was even closer, but there's more to it. 
Let's see what the actual literature says. The idea is basically *how can we tweak the points we are triangulating to minimise the distance between the rays, before we actually solve for it?*
That is, we could just find the minimal distance between the rays. But the points we are triangulating aren't necessarily correct. But they are probably close to the theoretical correct points. Can we adjust them closer to the theoretical best?

Firstly, there is [Hartley and Sturm's original 1997 paper on triangulation](https://users.cecs.anu.edu.au/~hartley/Papers/triangulation/triangulation.pdf), which details several methods and presents an optimal one. This is complicated to explain here, but the crux is that instead of adjusting depth for minimisation, they realised that there are a pair of true points that the feature points are approximating. Let's adjust the feature points such that we minimise the distance between the feature points and the theoretical true points. Their formulation for this problem results in a 6-degree polynomial, which must be solved to find the minimum. The advantage of their algorithm is that it is closed form and requires no iteration. 

Then came [Kanatani's paper on triangulation](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.220.724&rep=rep1&type=pdf), which aimed to improve on Hartley and Sturm with an iterative algorithm. Kanatani's idea is based, instead of estimating the point we are trying to emulate, estimating the delta we need to add to our detected point to get to the expected point. The diference is that framing the problem in this way allows one to algebraically rework it into a pair of equations in *F*, *x* and *x'* that can easily be iterated on. 

Finally, [Peter Lindstrom improved on Kanatani's method](https://e-reports-ext.llnl.gov/pdf/384387.pdf) by designing a non-iterative quadratic solution that is faster than Hartley and Sturm, and more stable. Technically, it is iterative, but in two iterations it gets numerically close enough for reasonable precision and so Lindstrom just optimised two iterations into the one closed-form algorithm. I'll be honest - I don't really understand this algorithm yet. But I can code it, and that matters more. Can always learn the theory well later. It's based, again, on minimising the delta between the detected points and expected corresponding points, subject to the epipolar constraint. But this time, Lindstrom reworks this equation using [Lagrange Multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier) to show that we are projecting the detected points onto epipolar lines through the expected points, and from this we can create a quadratic that when solves gives the update that forms the delta to add to the detected points to get the expected points. If this doesn't make sense, that's fine - I don't get it either.  

# Rectification
Another thing we can do to get the depth of a scene - and this is denser, as we get depth pixel-by-pixel. With this we create what's called a __Disparity Map__ which is essentially the image but every pixel represents the depth at that point. 

So, what is [Rectification](http://www.sci.utah.edu/~gerig/CS6320-S2012/Materials/CS6320-CV-F2012-Rectification.pdf)? This is the process of [rotating each image so that objects in the images align](https://en.wikipedia.org/wiki/Image_rectification#/media/File:2DRectificationBAG.jpg) - then, when we want to find the depth of corresponding pixels, they lie in the same y-coordinate in each image, and the depth is directly proportional to the difference in x-coordinates. So in the same y-coordinate in each image, find the pixel that is most similar, and compute the x difference - and make sure that it is positive. You've got depth! How easy is that?!

So let's go through the process of computing the __rectified__ images. Now that we have the Fundamental and Essential matrices, it's pretty simple.

Rectified images are just those that have a baseline parallel to the image planes (visually, both cameras are facing forward at the same up/down angle, not looking toward or away from each other at all), and corresponding points in each image have exactly the same vertical coordinates. 
To do this, we need to rotate and twist the images as seen [here](https://en.wikipedia.org/wiki/Image_rectification#/media/File:2DRectificationBAG.jpg). There are multiple strategies to compute these rotations and twists, as seen in the various links here; I'm going with the ever-popular [Zhang's method](http://dev.ipol.im/~morel/Dossier_MVA_2011_Cours_Transparents_Documents/2011_Cours7_Document2_Loop-Zhang-CVPR1999.pdf), as his papers are succinct, straightforward, and missing unnecessary fluff or obscurity. They're clearer than most I read. 

According to Zhang, we need a [homography](https://en.wikipedia.org/wiki/Homography_(computer_vision)) for each image. In layman's terms, a homography is simply a [transformation between two planes in 3D space](http://www.cs.toronto.edu/~jepson/csc2503/tutorials/homography.pdf) - here, the first plane is the image plane, and the second is the rotated and twisted version of that. Zhang computes these homographies by splitting them up into the constituent components - a [projective transform](http://www.geom.uiuc.edu/docs/reference/CRC-formulas/node16.html), a similarity transform (rotation, translation, or scaling), and a [shearing transform](http://mathworld.wolfram.com/Shear.html). To compute the projective transform, Zhang defines the amount of projective distortion the image currently has (compared to what it needs to be transformed to; if you look at the images linked), and we attempt to minimise this. This minimisation then gives us parameters for the projective transformation. Zhang then shows that the similarity - the rotation, translation, and/or scaling - can be defined in terms of the fundamental matrix and the projective transform. Alright, we can compute that now. Finally, the shearing transform isn't necessary, but is nice to clear up some final distortion. This is computed by using the constraints that a shear must give us. To undistort, the shear must preserve the aspect ration, and also make corners perpendicular. This allows us to give us some simultaneous equations, and bam we get the three transforms. Chain these, and you have your homography. 

I realise that the above paragraph may not be clear; the paper is difficult to get through. However, give it a read and see for yourself. At the very least, the math is detailed, so if my words don't make sense, compare the paper to the code, and you can see the equations there. 

Once we've rectified the images, we can compute the depth by matching pixels in each line. For each pixel in one image, we scan along the same horizontal line in the second image, looking for the closest match. Then compute the depth using (Szeliski's depth formula), and we use this to colour a depth image. The depth image is presented as a grayscale image where the darkness or lightness corresponds to depth - darker points are further away, and lighter points are closer. 

For the results, see the rectification section below. 

# Results and Conclusion
### Viewing the triangulated points
Once we've triangulated every matching pair of points, we transform each of these into the frame of the first camera (or second; the point is we pick one and stick with it). This is our 3D point cloud! We can render this in something like [MeshLab](http://www.meshlab.net/), which takes .txt files of points in the format:

Px Py Pz

and so on. When viewing this it can be a bit difficult to see that we have the right depths, but if you wiggle the point cloud around a bit with your mouse, then you can get an idea for the scene shape. 

### Viewing the rectified points
As explained above, the rectified depth image is a grayscale where darkness corresponds to depth. Since this dataset has ground truth images, we can compare our image to these to see how accurate this method was. I'm not bothered with this, since I'm trying to present just a method, not trying to get the most accurate method. But that's how I represent this data. 

### Reading the code
I've included what are hopefully explanatory comments where I consider necessary, and useful, and above every major function - and the major functions sort of follow the headings above, in main.cpp at least. If you scroll down through that, you'll see the Feature Extraction, the Matching, Triangulation, and then Rectification at least. Dive into those. I've tried to comment those well too, and each is located just below its helpers. 

### Wrapping up
Thanks for reading! I hope this was helpful and that you could learn from it, in whatever context - whether as just a quick reference, stealing some code, or fully following along. 



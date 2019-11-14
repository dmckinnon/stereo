#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/core.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include <algorithm>
#include "Features.h"
#include <Eigen/Dense>
#include <filesystem>
#include <Windows.h>
#include "Stereography.h"
#include "Estimation.h"
#include <stdlib.h>
//#include <GL/glew.h> // This must appear before freeglut.h
//#include <GL/freeglut.h>
#include <omp.h>

using namespace std;
using namespace cv;
using namespace Eigen;

#define STEREO_OVERLAP_THRESHOLD 20

#define BUFFER_OFFSET(offset) ((GLvoid *) offset)
//#define DEBUG_FEATURES
//#define DEBUG_MATCHES
//#define DEBUG_FUNDAMENTAL
//#define DEBUG_ESSENTIAL_MATRIX

#define ROTATION_STEP 0.2f
#define TRANSLATION_STEP 0.1f

#define TRIANGULATION_POINT_CLOUD
#define RECTIFICATION_DEPTH_MAP


/*
	This is an exercise in stereo depth-maps and reconstruction (stretch goal)
	See README for more detail

	The way this will work is for a given number of input images (start with two)
	and the associated camera calibration matrices, we'll derive the Fundamental matrices
	(although we could just do the essential matrix) between each pair of images. Once
	we have this, a depth-map for each pair of images with sufficient overlap will be computed.
	If there are enough images, we can try to do a 3D reconstruction of the scene

	Input:
	- at least two images of a scene
	- the associated camera matrices

	Output:
	- depth-map as an image

	Steps:

	Feature Detection, Scoring, Description, Matching
		Using my existing FAST Feature detecter
		FAST Features are really just not good for this. Need SIFT or something. Seriously tempted to just
		use opencv's implementation of SIFT, and save to my own format. 

	Derivation of the Fundamental Matrix 
		This implies we don't even need the camera matrices. This will be done with the normalised
		8-point algorithm by Hartley (https://en.wikipedia.org/wiki/Eight-point_algorithm#The_normalized_eight-point_algorithm)

	Triangulation
		This will be computed using Peter Lindstrom's algorithm

	Rectification
		
	Depth-map

	TODO:
	- rectification
	- write up Lindstrom

*/

// Support functions
vector<string> get_all_files_names_within_folder(string folder)
{
	vector<string> names;
	string search_path = folder + "/*.*";
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				names.push_back(fd.cFileName);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return names;
}
inline bool does_file_exist(const std::string& name) {
	ifstream f(name.c_str());
	return f.good();
}

// Debug function prototypes
void DebugMatches(
	const vector<std::pair<Feature, Feature>>& matches,
	const vector<ImageDescriptor>& images,
	const Matrix3f& fundamentalMatrix);
void DebugEpipolarLines(
	StereoPair stereo,
	const vector<std::pair<Feature, Feature>>& matches,
	const vector<ImageDescriptor>& images);

// Main
int main(int argc, char** argv)
{
	// first arg is the folder containing all the images
	if (argc < 2 || strcmp(argv[1], "-h") == 0)
	{
		cout << "Usage:" << endl;
		cout << "stereo.exe <Folder to images> <calibration file> -mask [mask image] -features [Folder to save/load features]" << endl;
		exit(1);
	}
	string featurePath = "";
	bool featureFileGiven = false;
	string pointCloudOutputPath = "";
	Mat maskImage;
	if (argc >= 3)
	{
		for (int i = 3; i < argc; i += 2)
		{
			if (strcmp(argv[i], "-mask") == 0)
			{
				maskImage = imread(argv[i+1], IMREAD_GRAYSCALE);
			}
			if (strcmp(argv[i], "-features") == 0)
			{
				featurePath = string(argv[i+1]);
				featureFileGiven = true;
			}
			if (strcmp(argv[i], "-output") == 0)
			{
				pointCloudOutputPath = string(argv[i + 1]);
			}
		}
	}

	// Create an image descriptor for each image file we have
	vector<ImageDescriptor> images;
	string imageFolder = argv[1];

	// We have the option of saving the feature descriptors out to a file
	// If we have done that, we can pull that in to avoid recomputing features every time
	bool featuresRead = false;
	if (featureFileGiven)
	{
		std::cout << "Attempting to load features from " << featurePath << std::endl;
		// If the feature file exists, read the image descriptors from it
		if (does_file_exist(featurePath))
		{
			if (ReadDescriptorsFromFile(featurePath, images))
			{
				featuresRead = true;
				cout << "Read descriptors from " << featurePath << endl;
			}
			else
			{
				std::cout << "Reading descriptors from file failed" << endl;
			}
		}
	}
	if (!featuresRead)
	{
		auto imageFiles = get_all_files_names_within_folder(imageFolder);
		for (auto& image : imageFiles)
		{
			ImageDescriptor img;
			img.filename = imageFolder + "\\" + image;
			images.push_back(img);
		}

		// Collect the camera matrices
		// Note that the camera matrices are not necessarily at the centre of their own coordinate system;
		// they may have encoded some rigid-body transform in there as well?
		ReadCalibrationMatricesFromFile(argv[2], images);

		GetImageDescriptorsForImages(images);
	}
	// If opted, check for a features file
	if (featureFileGiven && !featuresRead)
	{
		// If the features file does not exist, save the features out to it
		if (!does_file_exist(featurePath))
		{
			if (!SaveImageDescriptorsToFile(featurePath, images))
			{
				std::cout << "Saving descriptors to file failed" << std::endl;
			}
		}
	}

	// THis should be stored in a 2D matrix where the index in the matrix corresponds to array index
	// and the array holds the fundamental matrix
	int s = (int)images.size();
	cout << "Matching features for " << images[0].filename << " and " << images[1].filename << endl;
	vector<std::pair<Feature, Feature>> matches = MatchDescriptors(images[0].features, images[1].features, MAX_DIST_BETWEEN_MATCHES);

	if (matches.size() < STEREO_OVERLAP_THRESHOLD)
	{
		cout << matches.size() << " features - not enough overlap between " << images[0].filename << " and " << images[1].filename << endl;
	}
	cout << matches.size() << " features found between " << images[0].filename << " and " << images[1].filename << endl;

	StereoPair stereo;
	stereo.img1 = images[0];
	stereo.img2 = images[1];
	// Compute Fundamental matrix
	Matrix3f fundamentalMatrix;
	if (!FindFundamentalMatrixWithRANSAC(matches, fundamentalMatrix, stereo))
	{
		cout << "Failed to find fundamental matrix for pair " << images[0].filename << " and " << images[1].filename << endl;
	}
	cout << "Fundamental matrix found for pair " << images[0].filename << " and " << images[1].filename << endl;

	// Compute essential matrix
	// E = KT * F * K
	stereo.F = fundamentalMatrix;
	stereo.E = stereo.img2.K.transpose() * stereo.F * stereo.img1.K;

	// Cheeky debug if you want it
#ifdef DEBUG_MATCHES
	DebugMatches(matches, images, fundamentalMatrix);
#endif

#ifdef DEBUG_ESSENTIAL_MATRIX
	DebugEpipolarLines(stereo, matches, images);
#endif

#ifdef TRIANGULATION_POINT_CLOUD
	// So now, run feature detection again, but manually.
	// We'll define a looser bound on clustering and scoring.
	// Then get the depth of each point and build up a point cloud of the scene
	// This all rides on sufficiently many points for a good calibration
	// of the stereo matrix

	for (auto& image : images)
	{
		Mat img = imread(image.filename, IMREAD_GRAYSCALE);

		vector<Feature> features;
		FindFASTFeatures(img, features);
		if (features.empty())
		{
			cout << "No features were found in " << image.filename << endl;
		}
		// Refactor this to take params
		features = ScoreAndClusterFeatures(img, features, 500, 2);

		// Create descriptors with scale information for better matching

		// Create descriptors for each feature in the image
		std::vector<FeatureDescriptor> descriptors;
		if (!CreateSIFTDescriptors(img, features, descriptors))
		{
			cout << "Failed to create feature descriptors for image " << image.filename << endl;
			continue;
		}

		image.width = img.cols;
		image.height = img.rows;
		image.features = features;
	}

	// Now do some cheeky feature matching again
	// Some features won't match well, but that's ok
	vector<std::pair<Feature, Feature>> newMatches;
	newMatches = MatchDescriptors(images[0].features, images[1].features, MAX_DIST_BETWEEN_MATCHES*2);

	// COmpute transform between images
	Vector3f t(0, 0, 0);
	Matrix3f R;
	R.setZero();
	DecomposeEssentialMatrix(stereo.E, R, t);
	
	// Now for each match, get the depth in the frame of the first image
	vector<Vector3f> depthPoints;
	for (auto& match : newMatches)
	{
		Point2f xprime = match.first.p;
		Point2f x = match.second.p;
		Vector3f pointX = stereo.img2.K.inverse() * Vector3f(x.x, x.y, 1);
		Vector3f pointXPrime = stereo.img1.K.inverse() * Vector3f(xprime.x, xprime.y, 1);
		float d0 = 0;
		float d1 = 0;
		// TODO - do I need to do this the other way?
		// TODO - normalise or no? No?
		Matrix3f Einverse = stereo.E.inverse();
		if (!Triangulate(d0, d1, pointX, pointXPrime, stereo.E))
		{
			match.first.depth = BAD_DEPTH;
			match.second.depth = BAD_DEPTH;
			continue;
		}

		// Is this depth in first frame or second frame?
		// There is also a bug where depth is negative sometimes.
		// This can happen from triangulation
		d0 = abs(d0);
		d1 = abs(d1);
		match.first.depth = abs(d0);
		match.second.depth = abs(d1);

		Vector3f pointIn3D(xprime.x, xprime.y, 1);
		pointIn3D = images[0].K.inverse() * pointIn3D;
		pointIn3D = pointIn3D / pointIn3D[2];
		pointIn3D *= d1;

		depthPoints.push_back(pointIn3D);
	}


	
	// Write these points out to a text file
	if (pointCloudOutputPath.size() == 0)
	{
		cout << "No output path given! Ending now ..." << endl;
		exit(0);
	}
	std::ofstream pointFile(pointCloudOutputPath + "\\point_cloud.txt");
	if (pointFile.is_open())
	{
		for (int i = 0; i < depthPoints.size(); ++i)
		{
			auto& p = depthPoints[i];
			// get normals
			Vector3f normal = p;
			normal *= 1 / sqrt(normal[0] * normal[0] + normal[1]*normal[1] + normal[2]*normal[2]);
			pointFile << p[0] << " " << p[1] << " " << p[2] << " " << normal[0] << " " << normal[1] << " " << normal[2];
			if (i < depthPoints.size() - 1)
			{
				pointFile << endl;
			}
		}
		pointFile.close();
	}
	
#endif

#ifdef RECTIFICATION_DEPTH_MAP
	// So now, rectify the images so that epipolar lines are corresponding horizontal
	// lines. This means that to compute 'depth', we just need the horizontal
	// distance between two pixels. This can map to physical depth, but to create
	// a depth map, or a point cloud to display, we don't necessarily care about that

	// Compute rectification rotations

	// Apply to images
	
	// Compute depth map

	// Show depth map

	// save depth map?
#endif

	return 0;
}

/* #################################
    Section for debug functions
   ################################# */
void DebugMatches(
	const vector<std::pair<Feature, Feature>>& matches,
	const vector<ImageDescriptor>& images,
	const Matrix3f& fundamentalMatrix)
{
	// Draw matching features
	Mat matchImageScored;
	Mat img_i = imread(images[0].filename, IMREAD_GRAYSCALE);
	Mat img_j = imread(images[1].filename, IMREAD_GRAYSCALE);
	hconcat(img_i, img_j, matchImageScored);
	int offset = img_i.cols;
	// Draw the features on the image
	for (unsigned int i = 0; i < matches.size(); ++i)
	{
		Feature f1 = matches[i].first;
		Feature f2 = matches[i].second;


		auto f = Vector3f(matches[i].first.p.x, matches[i].first.p.y, 1);
		auto fprime = Vector3f(matches[i].second.p.x, matches[i].second.p.y, 1);


		auto result = fprime.transpose() * fundamentalMatrix * f;
		std::cout << "reprojection error: " << result << endl;

		f2.p.x += offset;

		circle(matchImageScored, f1.p, 2, (255, 255, 0), -1);
		circle(matchImageScored, f2.p, 2, (255, 255, 0), -1);
		line(matchImageScored, f1.p, f2.p, (0, 0, 0), 2, 8, 0);

		// Debug display
		imshow("matches", matchImageScored);
		waitKey(0);
	}
}

void DebugEpipolarLines(
	StereoPair stereo,
	const vector<std::pair<Feature, Feature>>& matches,
	const vector<ImageDescriptor>& images)
{
	// Debug the Essential Matrix now
	// We do this by drawing the epipolar line from the essential matrix at various depths,
	// and drawing the matching feature
	Vector3f t(0, 0, 0);
	Matrix3f R;
	R.setZero();
	DecomposeEssentialMatrix(stereo.E, R, t);

	// verify with the difference between t_skew * R and E
	std::cout << "Difference between E and t_skew * R:" << endl;
	Matrix3f t_skew;
	t_skew << 0, -t[2], t[1],
		t[2], 0, -t[0],
		-t[1], t[0], 0;
	Matrix3f residual = stereo.E - t_skew * R;
	cout << residual << endl;

	std::cout << "Rotation: \n" << R << "\nTranslation: \n" << t << endl;

	cout << "E is " << endl << stereo.E << endl << " and E inverse is " << endl << stereo.E.inverse() << endl;

	for (auto& m : matches)
	{
		Mat epipolarLines;
		Mat img_1 = imread(images[0].filename, IMREAD_GRAYSCALE);
		Mat img_2 = imread(images[1].filename, IMREAD_GRAYSCALE);
		hconcat(img_1, img_2, epipolarLines);
		int offset = img_1.cols;

		Point2f img1Point = m.first.p;
		Point2f img2Point = m.second.p;
		Feature f2 = m.second;
		f2.p.x += offset;
		circle(epipolarLines, img1Point, 6, (255, 255, 0), -1);
		circle(epipolarLines, f2.p, 6, (255, 255, 0), -1);
		//cout << "Features are " << img1Point << " and " << f2.p << endl;

		// Here we are NOT normalising
		// But we are going from image 0 into image 1, as that is the direction in which we computed the fundamental matrix
		Vector3f projectivePoint;
		projectivePoint[0] = img1Point.x;
		projectivePoint[1] = img1Point.y;
		projectivePoint[2] = 1;
		Vector3f point = images[0].K.inverse() * projectivePoint;
		point = point / point[2];
		//cout << "Starting with " << point << endl;
		for (double d = 1; d < 10; d += 0.2)
		{
			Vector3f eL = point * d;
			//cout << "depth vector:\n" << eL << endl;
			Vector3f transformedPoint = R.inverse() * eL - R.inverse() * t; // IS THIS RIGHT?
			//cout << "transformed:\n" << transformedPoint << endl;
			transformedPoint /= transformedPoint[2];
			//cout << "normalised:\n" << transformedPoint << endl;
			// now project:
			projectivePoint = images[1].K * transformedPoint;
			// get u, v from first to bits
			Point2f reprojection(projectivePoint[0], projectivePoint[1]);
			//reprojection /= 4;
			reprojection.x += offset;
			circle(epipolarLines, reprojection, 2, (255, 255, 0), -1);
			//cout << "Epipolar line point at depth " << d << " is " << reprojection << endl;
		}

		// Display
		imshow("Epipolar line", epipolarLines);
		waitKey(0);
	}
}








/*
	The following is a loop that draws the epipolar lines, so you can see what this
	looks like


for (auto& match : matches)
{
	Mat epipolarLines;
	Mat img_1 = imread(images[0].filename, IMREAD_GRAYSCALE);
	Mat img_2 = imread(images[1].filename, IMREAD_GRAYSCALE);
	hconcat(img_1, img_2, epipolarLines);
	int offset = img_1.cols;
	// depth is way the hell off. Fix this anohter time

	// TODO: triangulation isn't working.
	// or it gives negative depth
	// Theory 1: mathematics is wrong somehow, not sure where
	// Can show this in OpenGL to debug easier?

	// Potential issues:
	// - coordinate normalisation?
	// - need to anti-rotate t vector? (doubt it)
	// - decomposition of E into R and t? This seems weird and too convenient

	Point2f xprime = match.first.p;
	Point2f x = match.second.p;
	Vector3f pointX = stereo.img2.K.inverse() * Vector3f(x.x, x.y, 1);
	Vector3f pointXPrime = stereo.img1.K.inverse() * Vector3f(xprime.x, xprime.y, 1);
	float d0 = 0;
	float d1 = 0;
	// TODO - do I need to do this the other way?
	// TODO - normalise or no? No?
	Matrix3f Einverse = stereo.E.inverse();
	if (!Triangulate(d0, d1, pointX, pointXPrime, stereo.E))
	{
		match.first.depth = BAD_DEPTH;
		match.second.depth = BAD_DEPTH;
		continue;
	}

	circle(epipolarLines, xprime, 3, (255, 255, 0), -1);
	x.x += offset;
	circle(epipolarLines, x, 3, (255, 255, 0), -1);

	// Now transform to cam 0

	// Is this depth in first frame or second frame?
	d0 = abs(d0);
	d1 = abs(d1);
	match.first.depth = abs(d0);
	match.second.depth = abs(d1);// transform the point in 3D from first camera to second camera
	cout << "Depths are " << d0 << " and " << d1 << endl;

	Vector3f t(0, 0, 0);
	Matrix3f R;
	R.setZero();
	DecomposeEssentialMatrix(stereo.E, R, t);
	Vector3f projectivePoint;
	projectivePoint[0] = xprime.x;
	projectivePoint[1] = xprime.y;
	projectivePoint[2] = 1;
	Vector3f point = images[0].K.inverse() * projectivePoint;
	point = point / point[2];
	Vector3f eL = point * d1;
	Vector3f transformedPoint = R.inverse() * eL - R.inverse() * t; // IS THIS RIGHT?
	transformedPoint /= transformedPoint[2];
	//cout << "normalised:\n" << transformedPoint << endl;
	// now project:
	projectivePoint = images[1].K * transformedPoint;
	// get u, v from first to bits
	Point2f reprojection(projectivePoint[0], projectivePoint[1]);





	// Transform points into each camera frame

	// Copy the depths to the StereoPair array
	for (auto& f : images[0].features)
	{
		if (f == match.first)
		{
			f.depth = match.first.depth;
		}
	}
	for (auto& f : images[1].features)
	{
		if (f == match.second)
		{
			//f.depth = match.second.depth;
		}
	}

	reprojection.x += offset;
	circle(epipolarLines, reprojection, 5, (255, 255, 0), 2);
	cout << "Point at depth " << d0 << " is " << reprojection << endl;

	// Display
	imshow("Depths", epipolarLines);
	waitKey(0);
}
*/

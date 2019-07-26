#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include "Features.h"
#include <Eigen/Dense>
#include <filesystem>
#include <Windows.h>
#include "Stereography.h"
#include "Estimation.h"

using namespace std;
using namespace cv;
using namespace Eigen;

#define STEREO_OVERLAP_THRESHOLD 50

struct ImageDescriptor
{
	std::vector<Feature> features;
	std::string filename;
	MatrixXf K;
};

struct StereoPair
{
	ImageDescriptor img1;
	ImageDescriptor img2;
	Matrix3f F;
};

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

	Derivation of the Fundamental Matrix 
		This implies we don't even need the camera matrices. This will be done with the normalised
		8-point algorithm by Hartley (https://en.wikipedia.org/wiki/Eight-point_algorithm#The_normalized_eight-point_algorithm)

	Triangulation
		This will be computed using Peter Lindstrom's algorithm, once I figure it out

	Depth-map


	Log
	- 

	TODO:
	- how to pull in images
	- do we need camera matrices?
	- TEST NORMALISATION

	Question: why does a homography send points to points between two planes, but a fundamental matrix, 
	          still a 3x3, send a point to a line, when it is specified more?

*/
// Support function
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
// Main
int main(int argc, char** argv)
{
	// first arg is the folder containing all the images
	if (argc < 2 || strcmp(argv[1], "-h") == 0)
	{
		cout << "Usage:" << endl;
		cout << "stereo.exe <Folder to images> <Folder to calibration matrices>" << endl;
		exit(1);
	}
	string imageFolder = argv[1];
	auto imageFiles = get_all_files_names_within_folder(imageFolder);

	// Collect the camera matrices
	// Note that the camera matrices are not necessarily at the centre of their own coordinate system;
	// they may have encoded some rigid-body transform in there as well?
	vector<MatrixXf> calibrationMatrices;
	if (argc >= 3)
	{
		string calibFolder = argv[2];
		auto calibFiles = get_all_files_names_within_folder(calibFolder);
		for (const auto& calib : calibFiles)
		{
			ifstream calibFile;
			calibFile.open(calibFolder + "\\" + calib);
			if (calibFile.is_open())
			{
				string line;
				if (getline(calibFile, line))
				{
					if (strcmp(line.c_str(), "CONTOUR") == 0)
					{
						// This is a good calib file, so read in the data
						int i = 0;
						MatrixXf K(3,4);
						K.setZero();
						while (getline(calibFile, line))
						{
							stringstream ss(line);
							for (int j = 0; j < 4; ++j)
								ss >> K(i,j);
							i += 1;
						}
						calibrationMatrices.push_back(K);
						cout << K << endl;
					}
				}
			}
		}
	}

	// How shall we have the architecture?
	// open image, then store extracted features, name, in a structure
	// over each pair, if they have enough overlap, compute the Fundamental
	// for each pair, reopen the images, construct the depth-map
	// OR over all the images, triangulate and reconstruct


	// Need a structure to hold extracted features, name
	// Then a structure to hold two of these, and a fundamental matrix



	// Loop over the images to pull out features 
	vector<ImageDescriptor> images;
	int index = 0;
	for (const auto& imageName : imageFiles)
	{
		string imagePath = imageFolder + "\\" + imageName;
		Mat img = imread(imagePath, IMREAD_GRAYSCALE);

#ifndef ESSENTIAL_MATRIX
		// Scale the image to be square along the smaller axis
		int size = min(img.cols, img.rows);
		resize(img, img, Size(size, size), 0, 0, CV_INTER_LINEAR);
#endif

		// Find FAST features
		vector<Feature> features;
		if (!FindFASTFeatures(img, features))
		{
			cout << "Failed to find features in image " << imageName << endl;
			return 0;
		}
		// Score features with Shi-Tomasi score
		std::vector<Feature> goodFeatures = ScoreAndClusterFeatures(img, features);
		if (goodFeatures.empty())
		{
			cout << "Failed to score and cluster features in image " << imageName << endl;
			return 0;
		}
		// Create descriptors for each feature in the image
		std::vector<FeatureDescriptor> descriptors;
		if (!CreateSIFTDescriptors(img, goodFeatures, descriptors))
		{
			cout << "Failed to create feature descriptors for image " << imageName << endl;
			return 0;
		}

		cout << "Image descriptor created for image " << imageName << endl;
		ImageDescriptor i;
		i.K = calibrationMatrices[index];
		i.filename = imageName;
		i.features = goodFeatures;
		images.push_back(i);

		index++;
	}

	// Run over each possible pair of images and count how many features they have in common. If more
	// than some minimum threshold, say 30, compute the fundamental matrix for these two images
	vector<StereoPair> pairs;
	// THis should be stored in a 2D matrix where the index in the matrix corresponds to array index
	// and the array holds the fundamental matrix
	int s = images.size();
	StereoPair** matrices = new StereoPair * [s];
	for (int k = 0; k < s; ++k)
	{
		matrices[k] = new StereoPair[s];
		for (int l = 0; l < s; ++l)
		{
			StereoPair p;
			p.F.setZero();
			matrices[k][l] = p;
		}
	}
	for (int i = 0; i < s; ++i)
	{
		for (int j = i + 1; j < s; ++j)
		{
			cout << "Matching features for " << images[i].filename << " and " << images[j].filename << endl;
			std::vector<std::pair<Feature, Feature>> matches = MatchDescriptors(images[i].features, images[j].features);

			if (matches.size() < STEREO_OVERLAP_THRESHOLD)
			{
				cout << "Not enough overlap between " << images[i].filename << " and " << images[j].filename << endl;
				continue;
			}

#ifndef ESSENTIAL_MATRIX
			// We don't have camera calibration matrices yet
			// Compute Fundamental matrix
			Matrix3f fundamentalMatrix;
			if (!FindFundamentalMatrix(matches, fundamentalMatrix))
			{
				cout << "Failed to find fundamental matrix for pair " << images[i].filename << " and " << images[j].filename << endl;
				continue;
			}

			cout << "Fundamental matrix found for pair " << images[i].filename << " and " << images[j].filename << endl;

			matrices[i][j].img1 = images[i];
			matrices[i][j].img2 = images[j];
			matrices[i][j].F = fundamentalMatrix;
#endif

#ifdef DEBUG_FUNDAMENTAL
			// How do I verify that this is the fundamental matrix?
			// Surely it should transform matching feature points into each other?
			// It transforms points into lines. So we can transform one image's point into
			// a line in the other image, and then verify that the feature is on that line
			// But we can also use the epipolar constraint to check
			// verify the epipolar cinstraint
			// This seems to be working. Each match has a pixel error of <5
			for (auto& m : matches)
			{
				auto f = Vector3f(m.first.p.x, m.first.p.y, 1);
				auto fprime = Vector3f(m.second.p.x, m.second.p.y, 1);

				auto result = fprime.transpose() * fundamentalMatrix * f;
				cout << result << endl << endl;
			}
#endif

			// Now perform triangulation on each of those points to get the depth
			// TODO: figure out structure here
			// Need to find the depth of each point. Not just feature points, but every point. 
			// But for that, we need to match pixels. 
			// Perhaps for that we need the homography. 
			// So that for each pixel location we transform that into the other camera, and then find the closest
			// discrete pixel. 
			// Lindstrom's triangulation assumes pixels are square; we do too, even though this is false
			// we could stretch the image or just see what happens
			// We'll get a homography between the images to get matching pixels. 
			// Then for each pixel in image A, we project it into image B and get the closest point
			// use these two points for the triangulation to create the depth map

			// Get homography between the two images
			Matrix3f H;
			H.setZero();
			if (!FindHomography(H, matches))
			{
				cout << "Couldn't find homography between " << images[i].filename << " and " << images[j].filename << endl;
				continue;
			}

			// reopen the images
			Mat img1 = imread(images[i].filename, IMREAD_GRAYSCALE);
			Mat img2 = imread(images[j].filename, IMREAD_GRAYSCALE);
			int imageWidth = img1.cols;
			int imageHeight = img1.rows;
			// Note that all images should in theory be the same size for depth maps. 
			// For reconstruction this is irrelevant

			// initialise the inverse-depth map
			Mat inverseDepthMap = Mat(imageWidth, imageHeight, CV_32F, 0);

			// loop over pixels in image 1
			// project them to image 2
			// triangulate to get depth
			// save inverse depth at pixel location from image 1 in the inverse depth map
			// TODO: is this how to do it? This does it from one view. Can we artificially create a view?
			for (int y = 0; y < imageHeight; ++y)
			{
				for (int x = 0; x < imageWidth; ++x)
				{
					Vector3f pixel(x, y, 1);
					Vector3f pixelPrime = H * pixel;
					pixelPrime /= pixelPrime(2);

				    // Triangulate a single pair of points, which means back to the papers

					// Check that depth isn't zero - if it is, tings are wrong

					// inverseDepth = 1/d;
				}
			}

			// Display inverse depth image
		}
	}

	return 0;
}
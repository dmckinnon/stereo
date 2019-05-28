#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <math.h>
#include "Features.h"
#include <Eigen/Dense>
#include <filesystem>
#include <Windows.h>

using namespace std;
using namespace cv;
using namespace Eigen;

#define STEREO_OVERLAP_THRESHOLD 50

struct ImageDescriptor
{
	std::vector<Feature> features;
	std::string filename;
	// potentially camera matrix
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
		cout << "stereo.exe <FolderToImages>" << endl;
		exit(1);
	}
	string folder = argv[1];
	auto files = get_all_files_names_within_folder(folder);

	// How shall we have the architecture?
	// open image, then store extracted features, name, in a structure
	// over each pair, if they have enough overlap, compute the Fundamental
	// for each pair, reopen the images, construct the depth-map
	// OR over all the images, triangulate and reconstruct


	// Need a structure to hold extracted features, name
	// Then a structure to hold two of these, and a fundamental matrix



	// Loop over the images to pull out features 
	vector<ImageDescriptor> images;
	for (const auto& imageName : files)
	{
		string imagePath = folder + "\\" + imageName;
		Mat img = imread(imagePath, IMREAD_GRAYSCALE);

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

		ImageDescriptor i;
		i.filename = imageName;
		i.features = goodFeatures;
		images.push_back(i);
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
	for (int i = 0; i < images.size(); ++i)
	{
		for (int j = i+1; j < images.size(); ++j)
		{
			std::vector<std::pair<Feature, Feature>> matches = MatchDescriptors(images[i].features, images[j].features);

			if (matches.size() < STEREO_OVERLAP_THRESHOLD)
			{
				continue;
			}

			// Compute Fundamental matrix
			Matrix3f fundamentalMatrix;
			if (!FindFundamentalMatrix(matches, F))
			{
				continue;
			}
		
			matrices[i][j].img1 = images[i];
			matrices[i][j].img2 = images[j];
			matrices[i][j].F = fundamentalMatrix;
		}
	}

	return 0;
}
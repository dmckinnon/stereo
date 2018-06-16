#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	/*
	  So what I'm going to do is:
	    - bring in both the images
		- import the intrinsic camera settings

		- Use SIFT to compute feature descriptors
		- Use FLANN (?) to match the descriptors (or write own algo) between the images
		- Use the 8 point algorithm to find the essential matrix
		- Use an epipolar search and maybe some triangulation or P3P to find the depth of some points
	*/

	// Read in both image files

	// Read the image file
	Mat image = imread("C:\\Users\\d_mcc\\OneDrive\\Pictures\\pirate.png");

	// Check for failure
	if (image.empty())
	{
		cout << "Could not open or find the image" << endl;
		cin.get(); //wait for any key press
		return -1;
	}

	std::string windowName = "The Guitar"; //Name of the window

	namedWindow(windowName); // Create a window

	imshow(windowName, image); // Show our image inside the created window.

	waitKey(0); // Wait for any keystroke in the window

	destroyWindow(windowName); //destroy the created window

	return 0;
}
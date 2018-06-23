#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

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
		- Rectify? To get a disparity map? To optimise the epipolar search?
	*/

	// Read in both image files
	Mat leftImage = imread("C:\\Users\\d_mcc\\Projects\\vision\\Classroom1-perfect\\im0.png");
	Mat rightImage = imread("C:\\Users\\d_mcc\\Projects\\vision\\Classroom1-perfect\\im1.png");

	// Read in the camera settings
	// TODO: Export this to another function
	Mat leftIntrinsics(Size(3,3), CV_32F, Scalar(0));
	Mat rightIntrinsics(Size(3, 3), CV_32F, Scalar(0));
	float dOffset = 0.f;
	float baseline = 0.f;
	// Hardcoded from file cos file reading is hard????


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

// Read camera params
//void ReadCamera()
//{
/*ifstream cameraParams;
cameraParams.open("C:\\Users\\d_mcc\\Projects\\vision\\Classroom1-perfect\\calib.txt", ios::in);
if (cameraParams.is_open())
{
string line;
while (getline(cameraParams, line))
{
// disambiguate between the lines
// For now, screw it, hard code the params and do input later
if (line.find("cam0"))
{
string buf;
stringstream ss(line);
vector<string> values;

while (ss >> buf)
{
values.push_back(buf);
cout << buf << endl;
cin.get();
}
}
else if (line.find("cam1"))
{

}
else if (line.find("doffs"))
{

}
else if (line.find("baseline"))
{

}
}

cameraParams.close();
}
else
{
cout << "Could not open or find the calibration file" << endl;
cin.get(); //wait for any key press
return -1;
}*/
//}
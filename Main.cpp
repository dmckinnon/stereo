#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
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
        - import the intrinsic camera settings // lol trouble with this

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
    Mat leftIntrinsics = Mat::zeros(3, 3, CV_32F);
    Mat rightIntrinsics = Mat::zeros(3, 3, CV_32F);
    float dOffset = 0.f;
    float baseline = 0.f;

    // Hardcoded from file cos file reading is hard????
    // TODO: fix
    dOffset = 107.911;
    baseline = 237.604;
    leftIntrinsics.at<float>(0,0) = 3962.004f;
    leftIntrinsics.at<float>(0,1) = 0.f;
    leftIntrinsics.at<float>(0, 2) = 1146.717f;
    leftIntrinsics.at<float>(1, 0) = 0.f;
    leftIntrinsics.at<float>(1, 1) = 3962.004f;
    leftIntrinsics.at<float>(1, 2) = 975.476f;
    leftIntrinsics.at<float>(2, 0) = 0.f;
    leftIntrinsics.at<float>(2, 1) = 0.f;
    leftIntrinsics.at<float>(2, 2) = 1.f;

    rightIntrinsics.at<float>(0, 0) = 3962.004;
    rightIntrinsics.at<float>(0, 1) = 0;
    rightIntrinsics.at<float>(0, 2) = 1254.628;
    rightIntrinsics.at<float>(1, 0) = 0;
    rightIntrinsics.at<float>(1, 1) = 3962.004;
    rightIntrinsics.at<float>(1, 2) = 975.476;
    rightIntrinsics.at<float>(2, 0) = 0;
    rightIntrinsics.at<float>(2, 1) = 0;
    rightIntrinsics.at<float>(2, 2) = 1;
    //////////////////ugh/////////////////////////

    // Create feature descriptor vectors for each of the images
	// We use ORB features
	vector<KeyPoint> keypointsLeft, keypointsRight;
	Mat descLeft, descRight;
	Ptr<ORB> detector = ORB::create();
	detector->detectAndCompute(leftImage, noArray(), keypointsLeft, descLeft);
	detector->detectAndCompute(rightImage, noArray(), keypointsRight, descRight);

    // Brute force matcher, cos screw compute time, we'll figure this out later
    BFMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descLeft, descRight, matches);

    // debug - draw matches
	Mat output;
	drawMatches(leftImage, keypointsLeft, rightImage, keypointsRight, matches, output);


    std::string leftWindowName = "Left image";
    std::string rightWindowName = "Right image";
	std::string debugWindowName = "debug image";

    namedWindow(debugWindowName); // Create a window

    imshow(debugWindowName, output); // Show our image inside the created window.

    waitKey(0); // Wait for any keystroke in the window

    destroyWindow(debugWindowName); //destroy the created window

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
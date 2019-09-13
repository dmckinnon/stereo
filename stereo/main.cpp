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
#include <GL/glew.h> // This must appear before freeglut.h
#include <GL/freeglut.h>
#include <omp.h>

using namespace std;
using namespace cv;
using namespace Eigen;

#define STEREO_OVERLAP_THRESHOLD 20

#define BUFFER_OFFSET(offset) ((GLvoid *) offset)
#define DEBUG_FEATURES
//#define DEBUG

GLuint buffer = 0;
GLuint vPos;
GLuint program;

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
		This will be computed using Peter Lindstrom's algorithm, once I figure it out

	Depth-map


	TODO:
	- TEST NORMALISATION
	- bring in openGL for visualisation: https://sites.google.com/site/gsucomputergraphics/educational/set-up-opengl

	Question: why does a homography send points to points between two planes, but a fundamental matrix, 
	          still a 3x3, send a point to a line, when it is specified more?

*/
void init();
void reshape(int width, int height);
void display();
void DrawPoints(const vector<Vector3f>& points);
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
inline bool does_file_exist(const std::string& name) {
	ifstream f(name.c_str());
	return f.good();
}
// Main
int main(int argc, char** argv)
{
	/* Some opengl rubbish to test that I have this working */
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(640, 480);   // Set the window's initial width & height
	glutInitWindowPosition(50, 50); // Position the window's initial top-left corner
	glutCreateWindow("Point Cloud");          // Create window with the given title

	// Register the display callback function
	glutDisplayFunc(display);

	// Register the reshape callback function
	glutReshapeFunc(reshape);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Set background color to black and opaque
	glClearDepth(1.0f);                   // Set background depth to farthest
	glEnable(GL_DEPTH_TEST);   // Enable depth testing for z-culling
	glDepthFunc(GL_LEQUAL);    // Set the type of depth-test
	glShadeModel(GL_SMOOTH);   // Enable smooth shading
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);  // Nice perspective corrections
	// Start the event loop
	//glutMainLoopEvent();



	// first arg is the folder containing all the images
	if (argc < 2 || strcmp(argv[1], "-h") == 0)
	{
		cout << "Usage:" << endl;
		cout << "stereo.exe <Folder to images> <calibration file> -mask [mask image] -features [Folder to save/load features]" << endl;
		exit(1);
	}
	string featurePath = "";
	bool featureFileGiven = false;
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
		}
	}

	vector<ImageDescriptor> images;
	// Create an image descriptor for each image file we have
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
		vector<MatrixXf> calibrationMatrices;
		string calib = argv[2];
		ifstream calibFile;
		calibFile.open(calib);
		if (calibFile.is_open())
		{
			string line;
			while (getline(calibFile, line))
			{
				Matrix3f K;
				replace(line.begin(), line.end(), '[', ' ');
				replace(line.begin(), line.end(), ']', ' ');
				replace(line.begin(), line.end(), ';', ' ');
				// extract the numbers out of the stringstream
				vector<string> tokens;
				string token;
				stringstream stream;
				stream << line;
				while (!stream.eof())
				{
					stream >> token;
					tokens.push_back(token);
				}

				// if line contains cam
				if (tokens[0].find("cam0") != string::npos)
				{
					// Read calibration and assign to descriptor for image 0
					K << stod(tokens[1], nullptr), stod(tokens[2], nullptr), stod(tokens[3], nullptr),
						stod(tokens[4], nullptr), stod(tokens[5], nullptr), stod(tokens[6], nullptr),
						stod(tokens[7], nullptr), stod(tokens[7], nullptr), stod(tokens[9], nullptr);

					for (auto& img : images)
					{
						// Yeah, this could be a lot better, I know
						if (img.filename.find("0") != string::npos)
						{
							img.K = K;
							break;
						}
					}
				}
				else if (tokens[0].find("cam1") != string::npos)
				{
					// Read calibration and assign to image 1
					K << stod(tokens[1], nullptr), stod(tokens[2], nullptr), stod(tokens[3], nullptr),
						stod(tokens[4], nullptr), stod(tokens[5], nullptr), stod(tokens[6], nullptr),
						stod(tokens[7], nullptr), stod(tokens[7], nullptr), stod(tokens[9], nullptr);
					for (auto& img : images)
					{
						if (img.filename.find("1") != string::npos)
						{
							img.K = K;
							break;
						}
					}
				}
			}
		}

		for (auto& image : images)
		{
			Mat img = imread(image.filename, IMREAD_GRAYSCALE);
			vector<Feature> features = FindHarrisCorners(img, 20);
			if (features.empty())
			{
				cout << "No features were found in " << image.filename << endl;
			}

#ifdef DEBUG_FEATURES
			Mat img_i = imread(image.filename, IMREAD_GRAYSCALE);
			for (auto& f : features)
			{
				circle(img_i, f.p, 3, (255, 255, 0), -1);
			}

			// Display
			imshow("Image - best features", img_i);
			waitKey(0);
#endif

			// Create descriptors for each feature in the image
			std::vector<FeatureDescriptor> descriptors;
			if (!CreateSIFTDescriptors(img, features, descriptors))
			{
				cout << "Failed to create feature descriptors for image " << image.filename << endl;
				continue;
			}

			image.features = features;
		}
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
	StereoPair pair;
	cout << "Matching features for " << images[0].filename << " and " << images[1].filename << endl;
	std::vector<std::pair<Feature, Feature>> matches = MatchDescriptors(images[0].features, images[1].features);


#ifdef DEBUG_MATCHES
	// Draw matching features
	Mat matchImageScored;
	Mat img_i = imread(imageFolder + "\\" + images[0].filename, IMREAD_GRAYSCALE);
	Mat img_j = imread(imageFolder + "\\" + images[1].filename, IMREAD_GRAYSCALE);
	hconcat(img_i, img_j, matchImageScored);
	int offset = img_i.cols;
	// Draw the features on the image
	for (unsigned int i = 0; i < matches.size(); ++i)
	{
		Feature f1 = matches[i].first;
		Feature f2 = matches[i].second;
		f2.p.x += offset;

		circle(matchImageScored, f1.p, 2, (255, 255, 0), -1);
		circle(matchImageScored, f2.p, 2, (255, 255, 0), -1);
		line(matchImageScored, f1.p, f2.p, (0, 255, 255), 2, 8, 0);
	}
	// Debug display
	imshow("matches", matchImageScored);
	waitKey(0);
#endif

	if (matches.size() < STEREO_OVERLAP_THRESHOLD)
	{
		cout << matches.size() << " features - not enough overlap between " << images[0].filename << " and " << images[1].filename << endl;
	}
	cout << matches.size() << " features found between " << images[0].filename << " and " << images[1].filename << endl;

	// Compute Fundamental matrix
	Matrix3f fundamentalMatrix;
	if (!FindFundamentalMatrix(matches, fundamentalMatrix))
	{
		cout << "Failed to find fundamental matrix for pair " << images[0].filename << " and " << images[1].filename << endl;
	}
	cout << "Fundamental matrix found for pair " << images[0].filename << " and " << images[1].filename << endl;

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
		
	StereoPair stereo;
	stereo.F = fundamentalMatrix;
	stereo.img1 = images[0];
	stereo.img2 = images[1];
	// Compute essential matrix
	stereo.E = stereo.img2.K.transpose() * fundamentalMatrix * stereo.img1.K;

	for (auto& match : matches)
	{
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
		float d0 = 0;
		float d1 = 0;
		if (!Triangulate(d0, d1, x, xprime, stereo.E))
		{
			match.first.depth = BAD_DEPTH;
			match.second.depth = BAD_DEPTH;
			continue;
		}

		// Is this depth in first frame or second frame?
		match.first.depth = d0;
		match.second.depth = d1;// transform the point in 3D from first camera to second camera

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
				f.depth = match.second.depth;
			}
		}
	}
	

	// Render points

	// Need to have some global structure that holds the points to be rendered
	// Need to enable lighting and shadows

	// For now, just render a small cube at the location of each point in C0
	vector<Vector3f> pointsToDraw;
	for (int i = 0; i < s; ++i)
	{
		for (auto& f : images[i].features)
		{
			Vector3f projectivePoint;
			projectivePoint[0] = f.p.x;
			projectivePoint[1] = f.p.y;
			projectivePoint[2] = 1;
			Vector3f point = images[i].K.inverse()*projectivePoint;
			point = point / point[2];
			point *= f.depth;
			pointsToDraw.push_back(point);
		}
	}
	DrawPoints(pointsToDraw);
	//glutMainLoopEvent();
	glutMainLoop();

	return 0;
}

/* ############################################################################
    OpenGL section below
   ############################################################################ */

/*
	OpenGL helpers for drawing
	I couldn't figure out how to have this in a different file, so it's all here
*/

// Draw a cube to some scale
void DrawCube(const float& scale, const float& r, const float& g, const float& b)
{
	// Top face (y = scale)
	glBegin(GL_QUADS);
		// Define vertices in counter-clockwise (CCW) order with normal pointing out
		glColor3f(r, g, b);
		glVertex3f(scale, scale, -scale);
		glVertex3f(-scale, scale, -scale);
		glVertex3f(-scale, scale, scale);
		glVertex3f(scale, scale, scale);

		// Bottom face (y = -scale)
		glColor3f(scale, 0.5f, 0.0f);
		glVertex3f(scale, -scale, scale);
		glVertex3f(-scale, -scale, scale);
		glVertex3f(-scale, -scale, -scale);
		glVertex3f(scale, -scale, -scale);

		// Front face  (z = scale)
		glColor3f(scale, 0.0f, 0.0f);
		glVertex3f(scale, scale, scale);
		glVertex3f(-scale, scale, scale);
		glVertex3f(-scale, -scale, scale);
		glVertex3f(scale, -scale, scale);

		// Back face (z = -scale)
		glColor3f(scale, scale, 0.0f);
		glVertex3f(scale, -scale, -scale);
		glVertex3f(-scale, -scale, -scale);
		glVertex3f(-scale, scale, -scale);
		glVertex3f(scale, scale, -scale);

		// Left face (x = -scale)
		glColor3f(0.0f, 0.0f, scale);
		glVertex3f(-scale, scale, scale);
		glVertex3f(-scale, scale, -scale);
		glVertex3f(-scale, -scale, -scale);
		glVertex3f(-scale, -scale, scale);

		// Right face (x = scale)
		glColor3f(scale, 0.0f, scale);
		glVertex3f(scale, scale, -scale);
		glVertex3f(scale, scale, scale);
		glVertex3f(scale, -scale, scale);
		glVertex3f(scale, -scale, -scale);
	glEnd();  // End of drawing color-cube
}



void init()
{
	// Three vertexes that define a triangle. 
	GLfloat vertices[][4] = {
		{-0.75, -0.5, 0.0, 1.0},
		{0.75, -0.5, 0.0, 1.0},
		{0.0, 0.75, 0.0, 1.0}
	};

	// Get an unused buffer object name. Required after OpenGL 3.1. 
	glGenBuffers(1, &buffer);

	// If it's the first time the buffer object name is used, create that buffer. 
	glBindBuffer(GL_ARRAY_BUFFER, buffer);

	// Allocate memory for the active buffer object. 
	// 1. Allocate memory on the graphics card for the amount specified by the 2nd parameter.
	// 2. Copy the data referenced by the third parameter (a pointer) from the main memory to the 
	//    memory on the graphics card. 
	// 3. If you want to dynamically load the data, then set the third parameter to be NULL. 
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	// OpenGL vertex shader source code
	const char* vSource = {
		"#version 330\n"
		"in vec4 vPos;"
		"void main() {"
		"	gl_Position = vPos * vec4(1.0f, 1.0f, 1.0f, 1.0f);"
		"}"
	};

	// OpenGL fragment shader source code
	const char* fSource = {
		"#version 330\n"
		"out vec4 fragColor;"
		"void main() {"
		"	fragColor = vec4(0.8, 0.8, 0, 1);"
		"}"
	};

	// Declare shader IDs
	GLuint vShader, fShader;

	// Create empty shader objects
	vShader = glCreateShader(GL_VERTEX_SHADER);
	fShader = glCreateShader(GL_FRAGMENT_SHADER);

	// Attach shader source code the shader objects
	glShaderSource(vShader, 1, &vSource, NULL);
	glShaderSource(fShader, 1, &fSource, NULL);

	// Compile shader objects
	glCompileShader(vShader);
	glCompileShader(fShader);

	// Create an empty shader program object
	program = glCreateProgram();

	// Attach vertex and fragment shaders to the shader program
	glAttachShader(program, vShader);
	glAttachShader(program, fShader);

	// Link the shader program
	glLinkProgram(program);

	// Retrieve the ID of a vertex attribute, i.e. position
	vPos = glGetAttribLocation(program, "vPos");

	// Specify the background color
	glClearColor(0, 0, 0, 1);
}

void reshape(int width, int height)
{
	// Compute aspect ratio of the new window
	if (height == 0) height = 1;                // To prevent divide by 0
	GLfloat aspect = (GLfloat)width / (GLfloat)height;

	// Set the viewport to cover the new window
	glViewport(0, 0, width, height);

	// Set the aspect ratio of the clipping volume to match the viewport
	glMatrixMode(GL_PROJECTION);  // To operate on the Projection matrix
	glLoadIdentity();             // Reset
	// Enable perspective projection with fovy, aspect, zNear and zFar
	gluPerspective(45.0f, aspect, 0.1f, 100.0f);
}

void display()
{
	// Clear the window with the background color
	glClear(GL_COLOR_BUFFER_BIT);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear color and depth buffers
	glMatrixMode(GL_MODELVIEW);     // To operate on model-view matrix

	glLoadIdentity();                 // Reset the model-view matrix
	glTranslatef(0.0f, 0.0f, -6.0f);  // Move into the screen to render the points

	// Render the points as cubes
	DrawCube(0.5f, 0.f, 1.f, 0.f);

	// Render a pyramid consists of 4 triangles
	glLoadIdentity();                  // Reset the model-view matrix

	// Refresh the window
	glutSwapBuffers();
}

void DrawPoints(const vector<Vector3f>& points)
{
	// Clear the window with the background color
	glClear(GL_COLOR_BUFFER_BIT);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear color and depth buffers
	glMatrixMode(GL_MODELVIEW);     // To operate on model-view matrix

	glLoadIdentity();                 // Reset the model-view matrix
	glTranslatef(0.0f, 0.0f, -6.0f);  // Move into the screen to render the points

	// Render the points as cubes
	for (auto& p : points)
	{
		// Probably need to scale these points a wee smidge

		// translate to point
		//p[0], p[1], p[2]
		glTranslatef(p[0], p[1], p[2]);
		// draw tiny cube
		DrawCube(0.5f, 1.f, 1.f, 0.f);
		// Now anti-translate, to come back to the same origin
		glTranslatef(-p[0], -p[1], -p[2]);
	}


	//DrawCube(0.5f, 0.f, 1.f, 0.f);

	// Render a pyramid consists of 4 triangles
	glLoadIdentity();                  // Reset the model-view matrix

	// Refresh the window
	glutSwapBuffers();
}




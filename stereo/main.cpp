#include <stdlib.h>
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
#include <stdlib.h>
#include <GL/glew.h> // This must appear before freeglut.h
#include <GL/freeglut.h>

using namespace std;
using namespace cv;
using namespace Eigen;

#define STEREO_OVERLAP_THRESHOLD 50

#define BUFFER_OFFSET(offset) ((GLvoid *) offset)

GLuint buffer = 0;
GLuint vPos;
GLuint program;

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
	Matrix3f E;
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
	- bring in openGL for visualisation: https://sites.google.com/site/gsucomputergraphics/educational/set-up-opengl

	Question: why does a homography send points to points between two planes, but a fundamental matrix, 
	          still a 3x3, send a point to a line, when it is specified more?

*/
void init();
void reshape(int width, int height);
void display();
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
	glutMainLoop();



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

		// Scale the image to be square along the smaller axis
		// ONLY IF we have no calibration matrices
		if (calibrationMatrices.empty())
		{
			int size = min(img.cols, img.rows);
			resize(img, img, Size(size, size), 0, 0, CV_INTER_LINEAR);
		}

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
			// Now compute the Essential matrix
			matrices[i][j].E = images[j].K.transpose() * fundamentalMatrix * images[i].K;

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
			// So what we need to do next is 
			// 1) store the depth per feature wrt each camera
			// 2) pick a camera, and get all points wrt it via some path of cameras and covisible points
			// 3) render all points wrt one camera




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

			// Not sure we need any of the below. Starting with just triangulating the features that we have
			// For all features that these images share, compute the depth of each feature with respect to the ith
			// camera. This can easily be transformed into the jth coordinate frame, as we have the essential matrix
			// From Lindstrom's paper, copying notation, we have that
			// xEx' = 0
			// and we follow this convention, where m.first is x', and m.second is x
			for (auto& match : matches)
			{
				Point2f xprime = match.first.p;
				Point2f x = match.second.p;
				float d0 = 0;
				float d1 = 0;
				if (!Triangulate(d0, d1, x, xprime, matrices[i][j].E))
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
				for (auto& f : images[i].features)
				{
					if (f == match.first)
					{
						f.depth = match.first.depth;
					}
				}
				for (auto& f : images[j].features)
				{
					if (f == match.second)
					{
						f.depth = match.second.depth;
					}
				}
			}
		}
	}

	// Find covisibility of points to get all points in the one frame
	// for each camera
	//	find the chain of transforms to C0 - shortest path
	// 
	// 

	// Render points

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




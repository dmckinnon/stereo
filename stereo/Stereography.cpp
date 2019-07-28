#include "Stereography.h"
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace Eigen;

/*
	Find the Fundamental Matrix between two images, assuming they have enough overlap, 
	given their matching features. 

	This implements the normalised 8-point algorithm. 
	The algorithm is essentially:
	- normalise all points so that the centre of all points is the origin, and the
	  points are scaled so that the average distance to the origin is root 2
	- get the top 8 points (or more) and do SVD to get the homography between them
	  subject to the epipolar constraint
	- denormalise

	The normalisation reduces numerical error when some values are large and others are small
	and yet they are all compared together and use the same error. This basically translates
	to over-error or under-error for points. 
*/
// Support functions
void GetNormalisationTransformAndNormalisePoints(vector<pair<Feature, Feature>>& matches, Matrix3f& T1, Matrix3f& T2)
{
	// Get centroid of points for each image
	Point2f centroid1(0,0);
	Point2f centroid2(0, 0);
	for (auto& m : matches)
	{
		centroid1 += m.first.p;
		centroid2 += m.second.p;
	}
	centroid1 /= (float)matches.size();
	centroid2 /= (float)matches.size();

	// offset all points by the negative centroid
	for (auto& m : matches)
	{
		m.first.p -= centroid1;
		m.second.p -= centroid2;
	}

	// TODO: confirm that centroid now is origin

	// Find the average distance to the centre
	float avgDist1 = 0;
	float avgDist2 = 0;
	for (auto& m : matches)
	{
		avgDist1 += sqrt(m.first.p.dot(m.first.p));
		avgDist2 += sqrt(m.second.p.dot(m.second.p));
	}
	avgDist1 /= (float)matches.size();
	avgDist2 /= (float)matches.size();

	// Now scale every point by root 2 over this distance
	float scale1 = sqrt(2) / avgDist1;
	float scale2 = sqrt(2) / avgDist2;
	for (auto& m : matches)
	{
		m.first.p *= scale1;
		m.second.p *= scale2;
	}

	T1 << scale1,   0,    centroid1.x*scale1,
		    0,    scale1, centroid1.y*scale1,
		    0,      0,        1;

	T2 << scale2, 0, centroid2.x* scale2,
		0, scale2, centroid2.y* scale2,
		0, 0, 1;
}
// Actual function
bool FindFundamentalMatrix(const vector<pair<Feature, Feature>>& matches, Matrix3f& F)
{
	// create a copy
	vector<pair<Feature, Feature>> pairs;
	for (auto m : matches)
	{
		pairs.push_back(m);
	}

	// Normalise points and get the transforms for denormalisation
	Matrix3f normalise1, normalise2;
	normalise1.setZero();
	normalise2.setZero();
	GetNormalisationTransformAndNormalisePoints(pairs, normalise1, normalise2);

	// Get the top 8? All of them? 
	// Select some subset and form a system of linear equations based on the 
	// epipolar constraint
	// In theory, it shouldn't matter which 8 we pick
	// also in theory, we have a strong matching set of points - why not use all?
	// We should just use the first 8
	MatrixXf Y;
	Y.resize(8, 9);
	// The matrix Y follows the constraint of y' E y = 0 where y' is from the second feature
	// and y is from the first
	for (int i = 0; i < 8;/*pairs.size()*/ ++i)
	{
		Point2f y = pairs[i].first.p;
		Point2f yprime = pairs[i].second.p;
		Y(i, 0) = yprime.x * y.x;
		Y(i, 1) = yprime.x * y.y;
		Y(i, 2) = yprime.x;
		Y(i, 3) = yprime.y * y.x;
		Y(i, 4) = yprime.y * y.y;
		Y(i, 5) = yprime.y;
		Y(i, 6) = y.x;
		Y(i, 7) = y.y;
		Y(i, 8) = 1;
	}

	// Solve with SVD
	BDCSVD<MatrixXf> svd(Y, ComputeThinU | ComputeFullV);
	if (!svd.computeV())
		return false;
	auto & f = svd.matrixV();

	F << f(0, 8), f(1, 8), f(2, 8),
		f(3, 8), f(4, 8), f(5, 8),
		f(6, 8), f(7, 8), f(8, 8);

	// Do I need to scale F by the 2,2 value?

	// Transform the matrix back to the original coordinate system
	F = normalise2.transpose() * F * normalise1;

	return true;
}

/*
	Triangulate using Peter Lindstrom's algorithm
	https://e-reports-ext.llnl.gov/pdf/384387.pdf
	We use algorithm niter1 in Listing 3

	If we want to use this with no scaling, then we must use the Essential matrix.
	Once we have the camera matrix per image, this is easy. 

	To do this with the Fundamental matrix, we replace E with F in the algorithm, but
	for the Euclidean reprojection error to make sense pixels must be square. So we have to scale. 

	First we optimise with Lindstrom's algorithm, and then we use the naive depth computation. 

	I call this optimisation, because it's not really triangulation; all it does
	is improve x and x' for actually getting the depth. The depth-getting algorithm comes next.

*/
// Helpers
void LindstromOptimisation(Vector3f& x, Vector3f& xprime, const Matrix3f E)
{
	MatrixXf S(2, 3);
	S << 1, 0, 0,
		0, 1, 0;

	Matrix3f Etilde = S * E * S.transpose();

	Vector3f n = S * E * xprime;
	Vector3f nprime = S * E.transpose() * x;
	RowVector3f nT = n.transpose();
	RowVector3f nprimeT = nprime.transpose();
	float a = nT * Etilde * nprime;
	float b = 0.5f * (nT*n + nprimeT*nprime)(0);
	float c = x.transpose() * E * xprime;
	float d = sqrt(b*b - a*c);
	float lambda = c / (b + d);
	Vector3f x_delta = lambda * n;
	Vector3f xprime_delta = lambda * nprime;
	n = n - Etilde * xprime_delta;
	nT = n.transpose();
	nprime = nprime - Etilde * x_delta;
	nprimeT = nprime.transpose();
	x_delta = ((x_delta.transpose()*n)(0) / (nT*n)(0)) * n;
	xprime_delta = ((xprime_delta.transpose()*nprime)(0) / (nprimeT*nprime)(0)) * nprime;
	x = x - S.transpose() * x_delta;
	xprime = xprime - S.transpose() * xprime_delta;

}
// Actual Function
float Triangulate(Point2f& x, Point2f& xprime, const Matrix3f E)
{
	// This is invalid. Hopefully before the end we make it valid
	float depth = BAD_DEPTH;

	// Lindstrom's algorithm gives us the optimal points x and xprime
	// So we modify p1 and p2, and then use them to compute depth. 
	Vector3f pointX(x.x, x.y, 1);
	Vector3f pointXPrime(xprime.x, xprime.y, 1);
	LindstromOptimisation(pointX, pointXPrime, E);

	// Now that we have the very best points we can get, we use the naive depth-getter
	// This basically shoots a ray out from each point, and draws a line between the two rays
	// and finds the point on each ray that minimises the length of this line. 
	// Basically the most agreeable point. The depth along each ray, then, is the point depth. 


	return depth;
}

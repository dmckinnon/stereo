#include "Features.h"
#include <iostream>
#include <algorithm>
#include <Eigen/SVD>
#include "Estimation.h"
#include <stdlib.h>
#include <time.h>

using namespace cv;
using namespace std;
using namespace Eigen;

/*
	Find the best homography between the two images. This is returned in H.

	Using a RANSAC approach, pick four random matches. Estimate the homography between
	the images using just these four. Measure the success of this homography by how well
	it predicts the rest of the matches. Refine the homography for the inlier set, 
	and count new number of inliers. Repeat until no more inliers. Measure the error.
	Take the best of these.
	Repeat for another random four.

	We'll do this a maximum number of times, and remember the best.
	If we never find a homography that produces matches below the epsilon, well,
	maybe this image pair just ain't good, yeah?

	NOTE: We can scale and shift the points to have 0 mean and std dev of 1.
	This can be useful if most of your matches are on one part of the image, with a few elsewhere - 
	it can unfairly weight parts of the image. 

	I don't normalise, and I get results that are fine. 
*/
// Support functions
void GetRandomFourIndices(int& i1, int& i2, int& i3, int& i4, int max, const vector<pair<Feature, Feature> >& matches)
{
	i1 = rand() % max;

	do
	{
		i2 = rand() % max;
	} while (i2 == i1);

	do
	{
		i3 = rand() % max;
	} while (i3 == i1 || i3 == i2);

	do
	{
		i4 = rand() % max;
	} while (i4 == i1 || i4 == i2 || i4 == i3);
}
// Normalise points
pair<Matrix3f, Matrix3f> ConvertPoints(const vector<pair<Feature, Feature> >& matches)
{
	// For each point in first and second, collect the mean
	// and compute std deviation
	size_t size = matches.size();
	Point2f firstAvg(0.f, 0.f);
	Point2f secondAvg(0.f, 0.f);
	for (size_t i = 0; i < size; ++i)
	{
		firstAvg += matches[i].first.p;
		secondAvg += matches[i].second.p;
	}
	firstAvg /= (float)size;
	secondAvg /= (float)size;

	// Now compute std deviation
	Point2f firstStdDev(0.f, 0.f);
	Point2f secondStdDev(0.f, 0.f);
	for (unsigned int i = 0; i < size; ++i)
	{
		auto temp = matches[i].first.p - firstAvg;
		firstStdDev += Point2f(temp.x*temp.x, temp.y*temp.y);

		temp = matches[i].second.p - secondAvg;
		secondStdDev += Point2f(temp.x*temp.x, temp.y*temp.y);
	}
	firstStdDev /= (float)size;
	secondStdDev /= (float)size;
	firstStdDev.x = sqrt(firstStdDev.x);
	firstStdDev.y = sqrt(firstStdDev.y);
	secondStdDev.x = sqrt(secondStdDev.x);
	secondStdDev.y = sqrt(secondStdDev.y);

	Matrix3f conversionForSecondPoints;
	conversionForSecondPoints << 1 / secondStdDev.x,             0.f        , -1*secondAvg.x / secondStdDev.x,
		                                  0.f      ,      1 / secondStdDev.y, -1*secondAvg.y / secondStdDev.y,
		                                  0.f      ,             0.f        ,           1.f;
 	Matrix3f conversionForFirstPoints;
	conversionForFirstPoints << 1 / firstStdDev.x,             0.f        , -1*firstAvg.x / firstStdDev.x,
		                                  0.f      ,      1 / firstStdDev.y, -1*firstAvg.y / firstStdDev.y,
		                                  0.f      ,             0.f        ,           1.f;

	return make_pair(conversionForFirstPoints, conversionForSecondPoints);
}
// Actual function
bool FindHomography(Matrix3f& homography, vector<pair<Feature,Feature> > matches)
{
	// Initialise RNG
	srand((unsigned int)time(NULL));

	// Get normalisation matrices, and normalise all points in the matches
	// This distributes the points across a normal distribution, mean 0 std dev 1. 
	// This is to counteract any uneven distribution of points, that might weight a homography
	// towards a certain part of the image. Technically, refinement should fix this. 
	// You can try turning this on and see what effect it has
	/*auto normalisationMatrixPair = ConvertPoints(matches);
	for (unsigned int i = 0; i < matches.size(); ++i)
	{
		auto p1 = matches[i].first.p;
		Vector3f v1(p1.x, p1.y, 1);
		auto v1Prime = normalisationMatrixPair.first * v1;
		matches[i].first.p.x = v1Prime(0);
		matches[i].first.p.y = v1Prime(1);

		auto p2 = matches[i].second.p;
		Vector3f v2(p2.x, p2.y, 1);
		auto v2Prime = normalisationMatrixPair.second * v2;
		matches[i].second.p.x = v2Prime(0);
		matches[i].second.p.y = v2Prime(1);
	}*/

	// RANSAC
	// For a maximum of MAX_RANSAC_ITERATIONS, pick four matches at random
	// Create a homography, perform the tests, refine etc, see how it is
	// If it meets the bar for quality, break. 
	// If not, keep trying
	size_t maxInliers = 0;
	Matrix3f bestH;
	size_t numMatches = matches.size();
	vector<pair<Feature, Feature> > inlierSet;
	for (int k = 0; k < MAX_RANSAC_ITERATIONS; ++k)
	{
		// Pick four random matches by generating four random indices
		// and ensuring they are not equal
		int i1, i2, i3, i4;
		GetRandomFourIndices(i1, i2, i3, i4, (int)numMatches, matches);
		
		// Get the points for those features and generate the homography
		// Since we match from left to right, and the homography goes from right
		// to left, the first in the pair is the feature on the right, and the second on the left
		vector<pair<Point2f, Point2f>> points;
		Matrix3f H;
		points.push_back(make_pair(matches[i1].second.p, matches[i1].first.p));
		points.push_back(make_pair(matches[i2].second.p, matches[i2].first.p));
		points.push_back(make_pair(matches[i3].second.p, matches[i3].first.p));
		points.push_back(make_pair(matches[i4].second.p, matches[i4].first.p));
		if (!GetHomographyFromMatches(points, H))
			continue;
		
		// Test the homography again all matches
		// Normalise homography
		H /= H(2, 2);
		auto set = EvaluateHomography(matches, H);
		if (set.size() > maxInliers)
		{
			maxInliers = inlierSet.size();
			inlierSet = set;
			bestH = H;
		}

		// TODO: add L-M here

		// Not enough inliers. Loop again
	}

	cout << "max inliers: " << maxInliers << endl;

	if (maxInliers != 0)
	{
		cout << "normalised homography: " << endl << bestH << endl;


		// Now bundle adjust on just the inlier set
		cout << "Bundle adjustment" << endl;
		BundleAdjustment(inlierSet, bestH);
		cout << "Refined homography: " << endl << bestH << endl;

		
		// convert H back to regular coords from normalised coords
		homography = /*normalisationMatrixPair.first.inverse() */ bestH /* normalisationMatrixPair.second*/;
		cout << "unnormalised homography: " << endl << homography << endl;
		// renormalise
		homography /= homography(2, 2);
			
		return true;
	}

	// We failed to find anything good
	return false;
}

/*
	Singular Value Decomposition
	This is where we actually construct the homography, for a given set of four matching pairs. 
	This is quite tricky to describe in just text, so I've linked everything I used to figure this out,
	in the README, under https://github.com/dmckinnon/stitch#finding-the-best-transform

	Find the homography for four sets of corresponding points

	How do we estimate the homography?

	Create the Matrix A, which is
	[ -u1  -v1  -1   0    0    0   u1u'1  v1u'1  u'1]
	[  0    0    0  -u1  -v1  -1   u1v'1  v1v'1  v'1] * h = 0
	................................................
	[  0    0    0  -u4  -v4  -1   u4v'4  v4v'4  v'4]
	where x' = Hx and h = [h1 ... h9] as a vector

	Use Singular Value Decomposition to compute A:
	UDV^T = A
	h = V_smallest (column of V corresponding to smallest singular value)
	Then form H out of that.
	Then unnormalise H, using the inverse of the normalisation matrix for the points

	NOTE:
	In the pairs, the first is x, the point from the image we come from
	the second is x prime, the point in the image we are transforming to

	To SVD A, we use Eigen. 
	Really, we could also just do eigenvalue decomposition on AT * A
	Since V's columns are eigenvectors of AT * A
	But whatever
*/
bool GetHomographyFromMatches(const vector<pair<Point2f, Point2f>> points, Matrix3f& H)
{
	// Construct A
	Matrix<float, 8, 9> A;
	A.setZero();
	for (unsigned int i = 0; i < points.size(); ++i)
	{
		auto& p = points[i];

		auto secondPoint = Vector3f(p.second.x, p.second.y, 1.f); // left
		auto firstPoint = Vector3f(p.first.x, p.first.y, 1.f); // right

		// Continue building A
		A(2*i,   0) = -1 * firstPoint(0);
		A(2 * i, 1) = -1 * firstPoint(1);
		A(2 * i, 2) = -1;
		A(2 * i, 6) = firstPoint(0) * secondPoint(0);
		A(2 * i, 7) = firstPoint(1) * secondPoint(0);
		A(2 * i, 8) = secondPoint(0);

		A(2 * i + 1, 3) = -1 * firstPoint(0);
		A(2 * i + 1, 4) = -1 * firstPoint(1);
		A(2 * i + 1, 5) = -1;
		A(2 * i + 1, 6) = firstPoint(0) * secondPoint(1);
		A(2 * i + 1, 7) = firstPoint(1) * secondPoint(1);
		A(2 * i + 1, 8) = secondPoint(1);
	}

	// Get the V matrix of the SVD decomposition
	BDCSVD<MatrixXf> svd(A, ComputeThinU | ComputeFullV);
	if (!svd.computeV())
		return false;
	auto& V = svd.matrixV();

	// Set H to be the column of V corresponding to the smallest singular value
	// which is the last as singular values come well-ordered
	H << V(0, 8), V(1, 8), V(2, 8),
		 V(3, 8), V(4, 8), V(5, 8),
		 V(6, 8), V(7, 8), V(8, 8);

	// Normalise H
	H /= H(2, 2);

	return true;
}

/*
	Evaluate a potential Homography, given the two lists of points. 
	The homography transforms from the second in each pair to the first. 

	We compute what's called the projective error (where we use H to project x into
	the other image and get the difference), and the reprojective error (where we use
	H inverse to project x' into the image for x and get the difference). 
	These can be added to get a good idea of the total error.

	We count the number of inliers, and return the inlier set and TODO: the error
*/
vector<pair<Feature, Feature> > EvaluateHomography(const vector<pair<Feature,Feature> >& matches, const Matrix3f& H)
{
	vector<float> diffs;
	int numInliers = 0;
	float allError = 0;
	vector<pair<Feature, Feature>> inlierSet;
	// Over all matches
	for (unsigned int i = 0; i < matches.size(); ++i)
	{
		// Convert both points to Eigen points
		Vector3f x(matches[i].second.p.x, matches[i].second.p.y, 1);
		Vector3f xprime(matches[i].first.p.x, matches[i].first.p.y, 1);

		Vector3f Hx = H * x;

		// Normalise
		Hx /= Hx(2);

		Vector3f Hxprime = H.inverse() * xprime;
		Hxprime /= Hxprime(2);

		// Use total reprojection error
		// This is L2(x' - Hx) + L2(x - Hinverse x')
		auto projectiveDiff = xprime - Hx;
		auto reprojectiveDiff = x - Hxprime;
		float totalError = projectiveDiff.norm() + reprojectiveDiff.norm();
		if (totalError < POSITIONAL_UNCERTAINTY * RANSAC_INLIER_MULTIPLER)
		{
			numInliers++;
			inlierSet.push_back(matches[i]);
		}
		allError += totalError;
	}

	return inlierSet;
}

/*
	Bundle Adjustment
	See https://github.com/dmckinnon/stitch/blob/master/README.md#finding-the-best-transform, optimsation

	The formula for J comes from Multiple View Geometry, page 146, equ. 5.11

	Here is an explanation and a derivation. This also explains how to get the covariance
	I think. See pages 12, 13, 14
	https://pdfs.semanticscholar.org/66e4/283c28a2a93c4d4674f4213e1e9f67cfc737.pdf

	Ethan Eade on optimisation:
	http://ethaneade.com/optimization.pdf

	We ignore covariance for now

	When the outlier error is large, we use Huber. 
	After this reduces and we want to quickly converge, we switch to Tukey. 
	These have the same function prototypes, so we just implement this as a cost
	function pointer that is swapped out.
*/
// Helper functions
float ErrorInHomography(const vector<pair<Feature, Feature> >& matches, const Matrix3f& H)
{
	float error = 0;
	for (unsigned int i = 0; i < matches.size(); ++i)
	{
		Vector3f x(matches[i].second.p.x, matches[i].second.p.y, 1);
		Vector3f xprime(matches[i].first.p.x, matches[i].first.p.y, 1);

		// Get the error term
		auto projectiveDiff = xprime - H * x;
		error += projectiveDiff.norm();
	}

	return error;
}
// Actual function
void BundleAdjustment(const vector<pair<Feature, Feature> >& matches, Matrix3f& H)
{
	// L-M update parameter
	float lambda =  .001f;
	float prevError = 100000000; // Some massive number so that our first error is always acceptable
	for (int its = 0; its < MAX_BA_ITERATIONS; ++its)
	{
		unsigned int i = 0;

		// These first two loops exist only to compute the standard deviation
		// for the huber and tukey cost functions

		// Get error vector, std dev vector, and Hx vector
		vector<Vector2f> errors;
		float avg = 0;
		float stddev = 0;
		vector<Vector3f> hxVals;
		for (i = 0; i < matches.size(); ++i)
		{
			// As above, first is x, the point on the right,
			// and second is x', the point on the left
			Vector3f x(matches[i].second.p.x, matches[i].second.p.y, 1);
			Vector3f xprime(matches[i].first.p.x, matches[i].first.p.y, 1);

			// Get the error term
			Vector3f Hx = H * x;
			float w = Hx(2);
			Hx /= w;
			Vector3f e = xprime - Hx;
			Vector2f e2(e(0), e(1));

			errors.push_back(e2);
			hxVals.push_back(Hx);

			avg += e2.norm();
		}
		avg /= matches.size();

		// Now compute the std dev
		for (i = 0; i < errors.size(); ++i)
		{
			stddev += pow(errors[i].norm() - avg, 2);
		}
		stddev = sqrt(stddev);


		VectorXf update(9);
		float error_accum = 0;
		MatrixXf JtJ(9, 9);
		JtJ.setZero();
		VectorXf Jte(9);
		Jte.setZero();

		// This loop is the actual refinement
		for (i = 0; i < matches.size(); ++i)
		{
			// As above, first is x, the point on the right,
			// and second is x', the point on the left
			Vector3f x(matches[i].second.p.x, matches[i].second.p.y, 1);
			Vector3f xprime(matches[i].first.p.x, matches[i].first.p.y, 1);

			// Get the error term
			Vector3f Hx = H * x;
			float w = Hx(2);
			Hx /= w;
			Vector3f e = xprime - Hx;
			const Vector2f e2(e(0), e(1));

			float costWeight = 1.f;
			float objectiveValue = 0.f;
			// use a robust cost function, but only if we need to
			/*if (i > 5)
				Huber(errors[i].norm(), stddev, objectiveValue, costWeight);
			else
				Tukey(errors[i].norm(), stddev, objectiveValue, costWeight);*/

			// Build the Jacobian
			MatrixXf J(2, 9);
			J.setZero();
			// We've confirmed by Finite Diff that this Jacobian is correct
			J << x(0), x(1), x(2), 0, 0, 0, -Hx(0)*x(0), -Hx(0)*x(1), -Hx(0),
				0, 0, 0, x(0), x(1), x(2), -Hx(1)*x(0), -Hx(1)*x(1), -Hx(1);
			J /= w;

			// Accumulate
			JtJ += costWeight * J.transpose() * J;
			Jte += costWeight * J.transpose() * e2;

			error_accum += e2.norm();
		}

		// Levenberg-Marquardt update
		for (i = 0; i < JtJ.rows(); ++i)
		{
			JtJ(i, i) += lambda * JtJ(i, i);
		}

		// Compute the update
		update = JtJ.inverse() * Jte;
		Matrix3f updateToH;
		updateToH << update(0), update(1), update(2),
			update(3), update(4), update(5),
			update(6), update(7), update(8);

		float currError = error_accum;

		// Early cutoff if our error is low enough
		if (currError < BA_THRESHOLD)
		{
			break;
		}
		// Update and continue if good enough
		if (currError < prevError)
		{
			lambda /= 10;
			prevError = currError;	
		}
		else
		{
			lambda *= 10;
		}

		H += updateToH;
		H /= H(2, 2);		
	}
	return;
}

/*
	Huber cost function and Jacobian for the optimisation process,
	and Tukey cost function and Jacobian for the same. 
	We use a robust cost function to deal with outliers as our data begin too far
	from the optimal point and I suspect optimisation is getting stuck in a local minimum elsewhere.

	https://onlinelibrary.wiley.com/doi/pdf/10.1002/pamm.201010258
	Ethan Eade: http://ethaneade.com/optimization.pdf
	Introduction to loss functions: https://blog.algorithmia.com/introduction-to-loss-functions/
	Robust estimators: http://users.stat.umn.edu/~sandy/courses/8053/handouts/robust.pdf

	So Tukey would weight too many outliers zero, and not get enough data, so it only works on a
	good inlier set. 
	Huber still weights outliers and can work with it, but is slow to finely converge.

	For the parameters k, smaller values of k produce more resistance to outliers so these can
	be tuned as necessary. 
	Usually a robust measure of spread is used in preference to the standard deviation of
	the residuals. For example, a common approach is to take sigma = MAR/0.6745, where MAR is
	the median absolute residual. This is because std dev is computationally expensive, requiring
	square roots and multiple passes over the data. I'm going to use the true std dev, because here
	I don't care about computation time.
*/
void Huber(const float& e, const float& stddev, float& objectiveValue, float& weight)
{
	float k = HUBER_K * stddev;
	if (abs(e) <= k)
	{
		objectiveValue = 0.5f*e*e;
		weight = 1.f;
	}
	else
	{
		objectiveValue = k * abs(e) - 0.5f*k*k;
		weight = k / abs(e);
	}
}
void Tukey(const float& e, const float& stddev, float& objectiveValue, float& weight)
{
	float k = TUKEY_K * stddev;
	if (abs(e) <= k)
	{
		objectiveValue = (k * k / 6.f) * (1 - pow(1.f - pow(e / k, 2),3));
		weight = pow(1.f - pow(e / k, 2), 2);
	}
	else
	{
		objectiveValue = k * k / 6.f;
		weight = 0;
	}
}

/*
	The purpose of this is to compute the difference between:
	(H + delta_h)(x) - H(x)
	and
	J_H(x)
	
	This is to test whether or not we have the right formulation of the Jacobian.
	This test verifies that we do. 
*/
void FiniteDiff(const Matrix3f& H)
{
	const Vector3f x(1.f, 2.f, 1.f);

	// Let f(h) = Hx
	// Compute f(h+epsilon) and f(h), then divide the difference by epsilon
	Vector3f Hx = H * x;
	float w = Hx(2);
	Hx /= w;
	float e = 0.01f;
	MatrixXf difference(2,9);
	difference.setZero();
	difference(0, 0) = (((H(0,0)+e)*x(0) + H(0,1)*x(1) + H(0,2)*x(2))/w - Hx(0))/e;
	difference(0, 1) = ((H(0, 0)*x(0) + (H(0, 1)+e)*x(1) + H(0, 2)*x(2)) / w - Hx(0)) / e;
	difference(0, 2) = ((H(0, 0)*x(0) + H(0, 1)*x(1) + (H(0, 2)+e)*x(2)) / w - Hx(0)) / e;

	difference(1, 3) = (((H(1, 0) + e)*x(0) + H(1, 1)*x(1) + H(1, 2)*x(2)) / w - Hx(1)) / e;
	difference(1, 4) = ((H(1, 0)*x(0) + (H(1, 1) + e)*x(1) + H(1, 2)*x(2)) / w - Hx(1)) / e;
	difference(1, 5) = ((H(1, 0)*x(0) + H(1, 1)*x(1) + (H(1, 2) + e)*x(2)) / w - Hx(1)) / e;

	float w_e7 = ((H(2,0)+e)*x(0) + H(2,1)*x(1) + H(2,2)*x(2));
	float w_e8 = (H(2, 0)*x(0) + (H(2, 1)+e)*x(1) + H(2, 2)*x(2));
	float w_e9 = (H(2, 0)*x(0) + H(2, 1)*x(1) + (H(2, 2)+e)*x(2));

	float x1 = H(0, 0)*x(0) + H(0, 1)*x(1) + H(0, 2)*x(2);
	float x2 = H(1, 0)*x(0) + H(1, 1)*x(1) + H(1, 2)*x(2);
	difference(0, 6) = (x1 / w_e7 - Hx(0)) / e;
	difference(0, 7) = (x1 / w_e8 - Hx(0)) / e;
	difference(0, 8) = (x1 / w_e9 - Hx(0)) / e;
	difference(1, 6) = (x2 / w_e7 - Hx(1)) / e;
	difference(1, 7) = (x2 / w_e8 - Hx(1)) / e;
	difference(1, 8) = (x2 / w_e9 - Hx(1)) / e;

	// Next, compute the Jacobian using Hartley and Zisserman's method,
	// at x. 
	MatrixXf J(2, 9);
	J.setZero();
	J << x(0), x(1), x(2), 0, 0, 0, -Hx(0)*x(0), -Hx(0)*x(1), -Hx(0)*x(2),
		0, 0, 0, x(0), x(1), x(2), -Hx(1)*x(0), -Hx(1)*x(1), -Hx(1)*x(2);
	J /= w;

	// finally, return the difference between these matrices. The difference should be vanishing
	cout << "J: " << endl << J << endl;
	cout << "Finite difference: " << endl << difference << endl;
	cout << J - difference << endl;
}
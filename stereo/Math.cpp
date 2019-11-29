#include "Math.h"
#include <cmath>

using namespace Eigen;

/*
	Return the skew-symmetric matrix for a vector
*/
Matrix3f SkewSymmetric(_In_ const Vector3f& v)
{
	Matrix3f v_skew;
	v_skew << 0, -v[2], v[1],
		v[2], 0, -v[0],
		-v[1], v[0], 0;
	return v_skew;
}

/*
	The logarithm and exponential for the Special Orthogonal Group.
	See in-depth theory here: http://ethaneade.com/lie.pdf
*/
Eigen::Vector3f SO3_log(_In_ const Eigen::Matrix3f& R)
{
	float trace = R(0, 0) + R(1, 1) + R(2, 2);
	float theta = acosf(0.5f*(trace - 1));
	float coefficient = 0.5;
	if (abs(theta) > 0.0001)
	{
		coefficient = (theta / (2 * sin(theta)));
	}
	Matrix3f log = coefficient * (R - R.transpose());

	// form vector from off-diagonal elements
	// using the convention of 
	// (  0  -z  y )
	// (  z   0 -x )
	// ( -y   x  0 )
	// to form a vector (x, y, z)
	Vector3f log_vector;
	log_vector(2) = log(1,0);
	log_vector(1) = log(0, 2);
	log_vector(0) = log(2, 1);

	return log_vector;
}
Eigen::Matrix3f SO3_exp(_In_ const Eigen::Vector3f& r)
{
	Matrix3f r_skew = SkewSymmetric(r);
	float theta = sqrt(r.transpose()*r);

	Matrix3f I;
	I = I.Identity();

	float sin_term = 0.5;
	float cos_term = 0;
	if (abs(theta) > 0.0001)
	{
		sin_term = (sin(theta) / theta);
		cos_term = ((1 - cos(theta)) / (theta * theta));
	}

	Matrix3f exp = I;
	exp += sin_term * r_skew;
	exp += cos_term * r_skew * r_skew;
	return exp;
}
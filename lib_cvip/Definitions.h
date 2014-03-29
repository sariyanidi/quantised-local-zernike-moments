#ifndef DEFINITONS_H
#define DEFINITONS_H

#define _CRT_SECURE_NO_WARNINGS

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <map>
#include <iomanip>
#include <string>
#include <complex>
#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <omp.h>
#include <numeric>

#ifndef uint
typedef unsigned int uint;
typedef unsigned short ushort;
#endif

namespace cvip 
{
    const double EPS = 0.000001;

    enum pose {POSE_FRONTAL, POSE_LPROFILE, POSE_RPROFILE};

    inline int round(double x) {return (int)std::floor(x+0.5);}
    template <class T> inline T min(T a, T b) {return a < b ? a : b;}
    template <class T> inline T max(T a, T b) {return a > b ? a : b;}
    static const double PI = 3.14159265;
}
#endif

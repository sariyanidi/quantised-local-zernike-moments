#ifndef QLZM_H
#define QLZM_H

#include "lib_cvip/Image.h"

namespace cvip
{

typedef std::vector< std::vector<std::complex<double> > > ZernikeFilters;

enum AFlatLevels {LZMA_ONLY_L1, LZMA_ONLY_L2, LZMA_BOTH_L1L2};
enum AFlatReIm {LZMA_ONLY_REAL, LZMA_ONLY_IMAGINARY, LZMA_BOTH_REALIMAGINARY};

class QLZM
{
    // small-sized LZM filters
    std::vector<cv::Mat> smlFilters;

    // large LZM filters
    std::vector<cv::Mat> lrgFilters;

    void _init();
    void _prepareVnls(int nmax, int w, const std::vector<double> &factorials, int& zmcount, ZernikeFilters& Vnls, std::vector<int>& ns);
    int momentSelector(int nmax, std::vector<int> &selectedMoms);

    std::vector<int> getSelectedMoms1() const { return selectedMoms1; }
    std::vector<int> getSelectedMoms2() const { return selectedMoms2; }


protected:
    AFlatLevels fLevels;
    AFlatReIm fReIm;

    int binCnt;
    int samplingStepSize;

    std::vector<int> selectedMoms1;
    std::vector<int> selectedMoms2;
    std::vector<double> factorials1;
    std::vector<double> factorials2;

    // internal vectors needed in zernike transformation
    std::vector<int> ns1; // n's: the "n" value corresponding to ith moment component
    // ns and ms of the second zernike transformation
    std::vector<int> ns2; // n's: the "n" value corresponding to ith moment component


    // number of zernike moments of the first and second transformation respectively
    int zmcount1;
    int zmcount2;

    int featureSize;

public:
    cvip::ZernikeFilters vnls1;
    cvip::ZernikeFilters vnls2;

    // number of parts in X and Y directions
    size_t numPartsX;
    size_t numPartsY;

    size_t w1;
    size_t w2;

    size_t n1;
    size_t n2;

    size_t getW1() const { return w1; }
    size_t getW2() const { return w2; }

    int activeMoments1;
    int activeMoments2;

    std::vector<int> fullNs; // n's: the "n" value corresponding to ith moment component

    QLZM(size_t _numPartsX=4, size_t _numPartsY=2, size_t w1=7, size_t w2=7, size_t n1=2, size_t n2=2, size_t binCnt=18, cvip::AFlatLevels aFlatLevels = cvip::LZMA_ONLY_L1, cvip::AFlatReIm aFlatReIm = cvip::LZMA_ONLY_IMAGINARY);


    cv::Mat computeHist(const cv::Mat& I, int numPartsX=0, int numPartsY=0) const;
    cv::Mat computePyramidHist(const cv::Mat& I, std::vector<std::pair<int,int> > npxys) const;

    void computeRegionHist(cv::Mat &region, const cv::Mat &gauss, cv::Mat &hist)  const;

    std::vector<cv::Mat> extractCoeffs(const cv::Mat &_input) const;

    std::vector<cv::Mat> extractTransformedImages(const cv::Mat &_input) const;
    std::vector<cv::Mat> extractTransformedFineImages(const cv::Mat &_input) const;
    std::vector<cv::Mat> extractTransformedFineBinaryImages(const cv::Mat &_input) const;

    cv::Mat extractIntPatterns(const cv::Mat& input) const;
    cv::Mat extractImagPatterns(const cv::Mat& input) const;
    cv::Mat extractPatterns(const cv::Mat& input) const;
    cv::Mat extractFinePatterns(const cv::Mat& input) const;

    std::string getUniqueKey() const;

    std::vector<cv::Mat> getSmlFilters() const { return smlFilters; }

};
}

#endif // QLZM_H

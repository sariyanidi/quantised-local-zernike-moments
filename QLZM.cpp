#include "lib_cvip/Image.h"
#include "QLZM.h"

using namespace cvip;

QLZM::QLZM(size_t _numPartsX, size_t _numPartsY, size_t _w1, size_t _w2, size_t _n1, size_t _n2, size_t _binCnt, cvip::AFlatLevels _fLevels, cvip::AFlatReIm _fReIm)
    : numPartsX(_numPartsX), numPartsY(_numPartsY), w1(_w1), w2(_w2), n1(_n1), n2(_n2), binCnt(_binCnt), fLevels(_fLevels), fReIm(_fReIm)
{
    _init();
}

/**
 * Initialize LZM patterns
 *
 * @brief QLZM::init
 */
void QLZM::_init()
{


    activeMoments1 = momentSelector(n1, selectedMoms1);
    activeMoments2 = momentSelector(n2, selectedMoms2);

    for (int i=0; i<=n1; i++)
        factorials1.push_back(factorial(i));

    for (int i=0; i<=n2; i++)
        factorials2.push_back(factorial(i));

    // prepare the zernike filters
    _prepareVnls(n1,w1,factorials1,zmcount1,vnls1,ns1);
    _prepareVnls(n2,w2,factorials2,zmcount2,vnls2,ns2);

    std::vector<int> selectedMoms1 = getSelectedMoms1();
    std::vector<int> selectedMoms2 = getSelectedMoms2();

    // first construct small filters
    for (uint j=0; j<vnls1.size(); ++j)
    {
        if (!selectedMoms1[j])
            continue;

        cv::Mat fre(getW1(),getW1(),CV_64FC1); // real val filter
        cv::Mat fim(getW1(),getW1(),CV_64FC1); // imag val filter

        for (uint i=0; i<vnls1[j].size(); ++i)
        {
            int row = i/getW1();
            int col = i%getW1();

            double rl = std::real(conj(vnls1[j][i]));
            double im = std::imag(conj(vnls1[j][i]));

            fre.at<double>(row,col) = rl;
            fim.at<double>(row,col) = im;
        }

        smlFilters.push_back(fre);
        smlFilters.push_back(fim);
    }

    // then construct large filters
    for (uint j=0; j<vnls2.size(); ++j)
    {
        if (!selectedMoms2[j])
            continue;

        cv::Mat fre(getW2(),getW2(),CV_64FC1); // real val filter
        cv::Mat fim(getW2(),getW2(),CV_64FC1); // imag val filter

        for (uint i=0; i<vnls2[j].size(); ++i)
        {
            int row = i/getW2();
            int col = i%getW2();

            double rl = std::real(vnls2[j][i]);
            double im = std::real(vnls2[j][i]);

            fre.at<double>(row,col) = rl;
            fim.at<double>(row,col) = im;
        }

        lrgFilters.push_back(fre);
        lrgFilters.push_back(fim);
    }
}

std::string QLZM::getUniqueKey() const
{
    std::stringstream ss;
    ss << "npx-" << numPartsX << "npy-" << numPartsY << "w1-" << w1 << "w2-" << w2 << "n1-" << n1 << "n2-" << n2;
    return ss.str();
}



// Function that computes the regional histograms of a given image
// Returns a cv::Mat having 1 row and binCnt*numRegions columns
cv::Mat QLZM::computeHist(const cv::Mat& I, int _numPartsX, int _numPartsY) const
{
    int npx,npy;

    if (!_numPartsX || ! _numPartsY) {
        npx = numPartsX;
        npy = numPartsY;
    } else {
        npx = _numPartsX;
        npy = _numPartsY;
    }

    int numPartsExX = npx-1;
    int numPartsExY = npy-1;

    int numRegions = npx*npy;
    int numRegionsEx = numPartsExX*numPartsExY;

    int rWidth = I.cols/npx;
    int rHeight = I.rows/npy;

    size_t binCnt = pow(2,smlFilters.size());

    cv::Mat hists(1,binCnt*(numRegions+numRegionsEx), CV_64FC1);

    int kernelSize = (rHeight % 2 != 0) ? rHeight : rHeight+1;


    cv::Mat gaussMat = cv::getGaussianKernel(kernelSize,8);
    gaussMat = gaussMat*gaussMat.t();
    cv::Mat gauss(rHeight, rWidth, CV_64FC1);
    cv::resize(gaussMat,gauss, gauss.size());

    // Compute the histograms for the normal grid
    for (int i=0;i<npx;i++)
    {
        for (int j=0;j<npy;j++)
        {
            cv::Mat region = I(cv::Range(j*rHeight, (j+1)*rHeight), cv::Range(i*rWidth, (i+1)*rWidth)).clone();
            cv::Mat regionHist(1, binCnt, CV_64FC1, cv::Scalar(0));
            computeRegionHist(region, gauss, regionHist);

            cv::Mat targetRegion = hists(cv::Rect(binCnt*(j*npx+i),0,binCnt,1));
            regionHist.copyTo(targetRegion);
        }
    }

    // Compute the histograms for the slided grid
    for (int i=0;i<numPartsExX;i++)
    {
        for (int j=0;j<numPartsExY;j++)
        {
            cv::Mat region = I(cv::Range(j*rHeight+rHeight/2, (j+1)*rHeight+rHeight/2), cv::Range(i*rWidth+rWidth/2, (i+1)*rWidth+rWidth/2)).clone();
            cv::Mat regionHist(1,binCnt,CV_64FC1, cv::Scalar(0));
            computeRegionHist(region, gauss, regionHist);

            cv::Mat targetRegion = hists(cv::Rect(binCnt*(numRegions+j*numPartsExX+i),0,binCnt,1));
            regionHist.copyTo(targetRegion);
        }
    }

    return hists;
}



// Function that computes the regional histograms of a given image
// Returns a cv::Mat having 1 row and binCnt*numRegions columns
cv::Mat QLZM::computePyramidHist(const cv::Mat& I, std::vector<std::pair<int,int> > npxys) const
{
    cv::Mat bigMat = computeHist(I,npxys[0].first,npxys[0].second);

    for (size_t i=1; i<npxys.size(); ++i)
    {
        cv::Mat h = computeHist(I,npxys[i].first,npxys[i].second);

        h = h/(cv::norm(h)+1e-10);
        cv::hconcat(bigMat,h,bigMat);

    }

    return bigMat;
}


void QLZM::computeRegionHist(cv::Mat &region, const cv::Mat &gauss, cv::Mat &hist) const
{
    // Construct the histogram
    for (int i=0;i<region.rows;i++)
        for (int j=0;j<region.cols;j++)
            hist.at<double>(0,region.at<uchar>(i,j)) += gauss.at<double>(i,j);

    // Normalize the histogram
    hist = hist/(cv::norm(hist)+1e-10);
}

/**
 * Compute the LZM pattern descriptor of an image
 */
cv::Mat QLZM::extractIntPatterns(const cv::Mat &_input) const
{
    cv::Mat input = _input.clone();

    if (input.channels() != 1) {
        cv::cvtColor(input, input, CV_RGB2GRAY);
    }

    if (input.type() == CV_8U) {
        input.convertTo(input, CV_64F);
    }

    int64 t1 = cv::getTickCount();
    // width of the small filter
    uint sw = getW1();

    uint outWidth = std::floor((double)input.cols/sw);
    uint outHeight = std::floor((double)input.rows/sw);

    cv::Mat out(outHeight, outWidth, CV_16U, cv::Scalar::all(0));
    //std::vector<uchar> goz;
    //! uint idx=0;

    for (uint i=0; i<input.rows-sw+1; i+=sw)
    {
        for (uint j=0; j<input.cols-sw+1; j+=sw)
        {
            // the n-bit output number
            ushort res = 0;

            for (uint k=0; k<smlFilters.size(); ++k)
            {
                double sum(0.);

                for (uint i2=0; i2<sw; ++i2)
                    for (uint j2=0; j2<sw; ++j2)
                        sum += smlFilters[k].at<double>(i2, j2)*input.at<double>(i+i2, j+j2);

                res |= (int)(sum>0) << k;
            }

            out.at<ushort>(i/sw,j/sw) = res;
            //goz.push_back(res);

            //std::cout << (int) res << '\n';
        }
    }

    return out;
}







/**
 * Compute the LZM pattern descriptor of an image
 */
std::vector<cv::Mat> QLZM::extractCoeffs(const cv::Mat &_input) const
{
    cv::Mat input = _input.clone();

    if (input.channels() != 1) {
        cv::cvtColor(input, input, CV_RGB2GRAY);
    }

    if (input.type() != CV_64F) {
        input.convertTo(input, CV_64F);
    }

    cvip::Image::zNormalize(input);

    // width of the small filter
    uint sw = getW1();
    //std::vector<uchar> goz;
    //! uint idx=0;


    std::vector<cv::Mat> outs;
    //std::vector<uchar> goz;
    //! uint idx=0;

    for (uint i=0; i<input.rows-sw+1; i+=sw)
    {
        for (uint j=0; j<input.cols-sw+1; j+=sw)
        {

            cv::Mat out(1, smlFilters.size(), CV_64F, cv::Scalar::all(0));

            for (uint k=0; k<smlFilters.size(); ++k)
            {
                for (uint i2=0; i2<sw; ++i2)
                    for (uint j2=0; j2<sw; ++j2)
                        out.at<double>(0,k) += smlFilters[k].at<double>(i2, j2)*input.at<double>(i+i2, j+j2);
            }

            outs.push_back(out);
            //goz.push_back(res);

            //std::cout << (int) res << '\n';
        }
    }

    return outs;
}



/**
 * Compute the LZM pattern descriptor of an image
 */
std::vector<cv::Mat> QLZM::extractTransformedImages(const cv::Mat &_input) const
{
    cv::Mat input = _input.clone();

    if (input.channels() != 1) {
        cv::cvtColor(input, input, CV_RGB2GRAY);
    }

    if (input.type() != CV_64F) {
        input.convertTo(input, CV_64F);
    }

    cvip::Image::zNormalize(input);

    // width of the small filter
    uint sw = getW1();
    //std::vector<uchar> goz;
    //! uint idx=0;


    std::vector<cv::Mat> outs;
    //std::vector<uchar> goz;
    //! uint idx=0;

    // out width and height
    uint ow = std::floor((double)input.cols/sw);
    uint oh = std::floor((double)input.rows/sw);

    for (uint k=0; k<smlFilters.size(); ++k)
    {
        cv::Mat out(oh,ow, CV_64FC1, cv::Scalar::all(0));
        for (uint i=0; i<input.rows-sw+1; i+=sw)
        {
            for (uint j=0; j<input.cols-sw+1; j+=sw)
            {
                for (uint i2=0; i2<sw; ++i2)
                    for (uint j2=0; j2<sw; ++j2)
                        out.at<double>(i/sw,j/sw) += smlFilters[k].at<double>(i2, j2)*input.at<double>(i+i2, j+j2);
            }
        }
        outs.push_back(out);
    }

    return outs;
}




/**
 * Compute the LZM pattern descriptor of an image
 */
std::vector<cv::Mat> QLZM::extractTransformedFineImages(const cv::Mat &_input) const
{
    cv::Mat input = _input.clone();

    if (input.channels() != 1) {
        cv::cvtColor(input, input, CV_RGB2GRAY);
    }

    if (input.type() != CV_64F) {
        input.convertTo(input, CV_64F);
    }

    cvip::Image::zNormalize(input);

    // width of the small filter
    uint sw = getW1();
    //std::vector<uchar> goz;
    //! uint idx=0;


    std::vector<cv::Mat> outs;
    //std::vector<uchar> goz;
    //! uint idx=0;

    // out width and height
    uint ow = std::floor((double)input.cols);
    uint oh = std::floor((double)input.rows);

    for (uint k=0; k<smlFilters.size(); ++k)
    {
        cv::Mat out(oh-sw+1,ow-sw+1, CV_64FC1, cv::Scalar::all(0));
        for (uint i=0; i<input.rows-sw+1; i++)
        {
            for (uint j=0; j<input.cols-sw+1; j++)
            {
                for (uint i2=0; i2<sw; ++i2)
                    for (uint j2=0; j2<sw; ++j2)
                        out.at<double>(i,j) += smlFilters[k].at<double>(i2, j2)*input.at<double>(i+i2, j+j2);
            }
        }
        outs.push_back(out);
    }

    return outs;
}





/**
 * Compute the LZM pattern descriptor of an image
 */
std::vector<cv::Mat> QLZM::extractTransformedFineBinaryImages(const cv::Mat &_input) const
{
    cv::Mat input = _input.clone();

    if (input.channels() != 1) {
        cv::cvtColor(input, input, CV_RGB2GRAY);
    }

    if (input.type() != CV_64F) {
        input.convertTo(input, CV_64F);
    }

    cvip::Image::zNormalize(input);

    // width of the small filter
    uint sw = getW1();
    //std::vector<uchar> goz;
    //! uint idx=0;


    std::vector<cv::Mat> outs;
    //std::vector<uchar> goz;
    //! uint idx=0;

    // out width and height
    uint ow = std::floor((double)input.cols);
    uint oh = std::floor((double)input.rows);

    for (uint k=0; k<smlFilters.size(); ++k)
    {
        cv::Mat out(oh,ow, CV_64FC1, cv::Scalar::all(0));
        for (uint i=0; i<input.rows-sw+1; i++)
        {
            for (uint j=0; j<input.cols-sw+1; j++)
            {
                for (uint i2=0; i2<sw; ++i2)
                    for (uint j2=0; j2<sw; ++j2)
                        out.at<double>(i,j) += smlFilters[k].at<double>(i2, j2)*input.at<double>(i+i2, j+j2);
            }
        }

        cv::Mat outUchar(oh-sw+1,ow-sw+1, CV_8UC1, cv::Scalar::all(0));
        cv::Mat tmp = cvip::Image::doubleToUchar(out);
        for (uint i=0; i<out.rows-sw+1; i++)
        {
            for (uint j=0; j<out.cols-sw+1; j++)
            {
                double val = tmp.at<uchar>(i,j);
                if (val>128)
                    outUchar.at<uchar>(i,j) = 255;
                else
                    outUchar.at<uchar>(i,j) = 0;
            }
        }

        outs.push_back(outUchar);
    }

    return outs;
}




/**
 * Compute the LZM pattern descriptor of an image
 */
cv::Mat QLZM::extractImagPatterns(const cv::Mat &_input) const
{
    cv::Mat input = _input.clone();

    if (input.channels() != 1) {
        cv::cvtColor(input, input, CV_RGB2GRAY);
    }

    if (input.type() == CV_8U) {
        input.convertTo(input, CV_64F);
    }

    int64 t1 = cv::getTickCount();
    // width of the small filter
    uint sw = getW1();


    uint outWidth = std::floor((double)input.cols/sw);
    uint outHeight = std::floor((double)input.rows/sw);

    cv::Mat out(outHeight, outWidth, CV_8U, cv::Scalar::all(0));
    //std::vector<uchar> goz;
    //! uint idx=0;

    for (uint i=0; i<input.rows-sw+1; i+=sw)
    {
        for (uint j=0; j<input.cols-sw+1; j+=sw)
        {
            // the n-bit output number
            uchar res = 0;

            for (uint k=0; k<smlFilters.size(); k+=2)
            {
                double sum(0.);

                for (uint i2=0; i2<sw; ++i2)
                    for (uint j2=0; j2<sw; ++j2)
                        sum += smlFilters[k].at<double>(i2, j2)*input.at<double>(i+i2, j+j2);

                res |= (uchar)(sum>0) << k;
            }

            out.at<uchar>(i/sw,j/sw) = res;
            //goz.push_back(res);

            //std::cout << (int) res << '\n';
        }
    }

    return out;
}


/**
 * Compute the LZM pattern descriptor of an image
 */
cv::Mat QLZM::extractPatterns(const cv::Mat &_input) const
{
    cv::Mat input = _input.clone();

    if (input.channels() != 1) {
        cv::cvtColor(input, input, CV_RGB2GRAY);
    }

    if (input.type() == CV_8U) {
        input.convertTo(input, CV_64F);
    }

    int64 t1 = cv::getTickCount();
    // width of the small filter
    uint sw = getW1();


    uint outWidth = std::floor((double)input.cols/sw);
    uint outHeight = std::floor((double)input.rows/sw);

    cv::Mat out(outHeight, outWidth, CV_8U, cv::Scalar::all(0));
    //std::vector<uchar> goz;
    //! uint idx=0;

    for (uint i=0; i<input.rows-sw+1; i+=sw)
    {
        for (uint j=0; j<input.cols-sw+1; j+=sw)
        {
            // the n-bit output number
            uchar res = 0;

            for (uint k=0; k<smlFilters.size(); ++k)
            {
                double sum(0.);

                for (uint i2=0; i2<sw; ++i2)
                    for (uint j2=0; j2<sw; ++j2)
                        sum += smlFilters[k].at<double>(i2, j2)*input.at<double>(i+i2, j+j2);

                res |= (uchar)(sum>0) << k;
            }

            out.at<uchar>(i/sw,j/sw) = res;
            //goz.push_back(res);

            //std::cout << (int) res << '\n';
        }
    }

    return out;
}


/**
 * Compute the LZM pattern descriptor of an image
 */
cv::Mat QLZM::extractFinePatterns(const cv::Mat &_input) const
{
    cv::Mat input = _input.clone();

    if (input.channels() != 1) {
        cv::cvtColor(input, input, CV_RGB2GRAY);
    }

    if (input.type() == CV_8U) {
        input.convertTo(input, CV_64F);
    }

    int64 t1 = cv::getTickCount();
    // width of the small filter
    uint sw = getW1();


    cv::Mat out(input.rows-sw, input.cols-sw, CV_8U, cv::Scalar::all(0));
    //std::vector<uchar> goz;
    //! uint idx=0;

    for (uint i=0; i<input.rows-sw; i++)
    {
        for (uint j=0; j<input.cols-sw; j++)
        {
            // the n-bit output number
            uchar res = 0;

            for (uint k=0; k<smlFilters.size(); ++k)
            {
                double sum(0.);

                for (uint i2=0; i2<sw; ++i2)
                    for (uint j2=0; j2<sw; ++j2)
                        sum += smlFilters[k].at<double>(i2, j2)*input.at<double>(i+i2, j+j2);

                res |= (uchar)(sum>0) << k;
            }

            out.at<uchar>(i,j) = res;
            //goz.push_back(res);

            //std::cout << (int) res << '\n';
        }
    }

    return out;

    uint outWidth = std::floor((double)input.cols/sw);
    uint outHeight = std::floor((double)input.rows/sw);

    cv::Mat outout(outHeight, outWidth, CV_8U, cv::Scalar::all(0));
    //std::vector<uchar> goz;
    //! uint idx=0;

    for (uint i=0; i<out.rows-sw; i+=sw)
    {
        for (uint j=0; j<out.cols-sw; j+=sw)
        {
            // the n-bit output number
            uchar res = 0;

            for (uint k=0; k<smlFilters.size(); ++k)
            {
                double sum(0.);

                for (uint i2=0; i2<sw; ++i2)
                    for (uint j2=0; j2<sw; ++j2)
                        sum += smlFilters[k].at<double>(i2, j2)*out.at<double>(i+i2, j+j2);

                res |= (uchar)(sum>0) << k;
            }

            outout.at<uchar>(i/sw,j/sw) = res;
            //goz.push_back(res);

            //std::cout << (int) res << '\n';
        }
    }

    return outout;
    /*

    return outoutput;
    */


    /*    std::cout << (double) (cv::getTickCount()-t1)/cv::getTickFrequency() << std::endl;
    cv::imshow("zsa", out);
    cv::waitKey(0);

    cv::Mat hist = computeHist(out);
    std::vector<double> goz2;
    for (uint i=0; i<hist.cols; ++i)
        goz2.push_back(hist.at<double>(0,i));

    int bidur = 0;
    return computeHist(out);
*/
}


void QLZM::_prepareVnls(int nmax, int w, const std::vector<double> &factorials, int& zmcount, std::vector< std::vector<std::complex<double> > >& Vnls, std::vector<int>& ns)
{
    // diameter of the circle for zernike elements
    double D = (double) w*std::sqrt((double)2);
    int w2 = w*w;

    // construct a lookup table for zernike moments for efficient computation
    for (int _n=0; _n<=nmax; _n++) {
        for (int _m=0; _m<=_n; _m++) {
            if ((_n-_m) % 2 == 0) {
                ns.push_back(_n);
                zmcount++;

                // fill the lookup table
                Vnls.push_back(std::vector<std::complex<double> >(w2, 0.));
                for (int y=0; y<w; ++y) {
                    for (int x=0; x<w; ++x) {
                        double xn = (double)(2*x+1-w)/D;
                        double yn = (double)(2*y+1-w)/D;

                        for (int xm = 0; xm <= (_n-_m)/2; xm++) {

                            // theta must change bw 0,2pi
                            double theta = atan2(yn,xn);
                            if (theta<0)
                                theta = 2*PI+theta;

                            Vnls.back()[w*y+x] +=
                                    (pow(-1.0, (double)xm)) * ( factorials[_n-xm] ) /
                                    (factorials[xm] * (factorials[(_n-2*xm+_m)/2]) *
                                    (factorials[(_n-2*xm-_m)/2]))  *
                                    (pow( sqrt(xn*xn + yn*yn), (_n - 2.*xm)) ) *
                                    4./(D*D)* // Delta_x* Delta_y
                                    std::polar(1., _m*theta);
                        }
                    }
                }
            }
        }
    }
}


int QLZM::momentSelector(int nmax, std::vector<int> &selectedMoms)
{
    int m = 0;
    for (int i=0; i<=nmax; i++) {
        for (int j=0; j<=i; j++) {
            if ((i-j)%2 != 0)
                continue;

            if (j != 0) {
                selectedMoms.push_back(1);
                m++;
            }
            else
                selectedMoms.push_back(0);
        }
    }

    return m;
}










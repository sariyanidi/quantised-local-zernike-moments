#include "lib_cvip/Image.h"
#include "QLZM.h"

/**
 * Written By E. Sariyanidi, Birkan Tunc and Volkan Dagli
 * 
 * If you intend to use this code for research purposes, please cite:
 * E. Sariyanidi, H. Gunes, M. Gokmen, A. Cavallaro 
 * 	  'Local Zernike Moment Representation for Facial Affect Recognition'
 * 	  BMVC'13
 * 
 * This code is licensed under CreativeCommons Non-Commercial license 3.0
 * 	  http://creativecommons.org/licenses/by-nc/3.0/deed.en_GB
**/

int main(int argc, char *argv[])
{
    // initialise feature extractor (here are the parameters passed)
    cvip::QLZM featureExtractor(5,5,7,7,2,2,18,cvip::LZMA_ONLY_L1,cvip::LZMA_ONLY_IMAGINARY);

    std::string imPath("im.png");
    if (argc>1) {
        imPath = argv[1];
    }

    // save features to file? -- yes by default
    bool saveToFile = true;
    if (argc>2) {
        saveToFile = (bool) atoi(argv[2]);
    }

    cv::Mat im = cv::imread(imPath, CV_LOAD_IMAGE_GRAYSCALE);

    // resize to a fixed size -- must be divided to M,N in the paper (_numPartsX,_numPartsY as parameter of the feature extractor -cvip::QLZM- above)
    cv::resize(im,im,cv::Size(100,100));
    im.convertTo(im,CV_64FC1);

    // extract features from overlapping regions (H-QLZM)
    cv::Mat features = featureExtractor.computeHist(featureExtractor.extractFinePatterns(im));

    // extract features from non-overlapping regions (H-NO-QLZM)
    cv::Mat featuresNonOverlapping = featureExtractor.computeHist(featureExtractor.extractPatterns(im));

    if (saveToFile)
    {
        // create file path for features
        size_t pos = imPath.rfind(".");
        std::string featuresPath(imPath);

        // will save them just next to the image file, replacing the image extension with .dat
        featuresPath.replace(featuresPath.begin()+pos+1, featuresPath.end(), "dat");

        cvip::Image::writeToFile(features, featuresPath);

        std::cout << "Saved features to " << featuresPath << std::endl;
    }
    else
    {
        // just print them to the terminal
        for (uint i=0; i<features.cols; ++i)
            std::cout << features.at<double>(0,i) << "\n";
    }

    return 0;
}

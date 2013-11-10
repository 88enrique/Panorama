/**
    Opencv example code: panorama image from multiple images with SIFT feature detection to match
    Enrique Marin
    88enrique@gmail.com
*/

#include <stdio.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>

using namespace std;
using namespace cv;

int main(){

    // Variables
    int nImages = 3;

    // Read image (from left to right)
    Mat src1 = imread("../Images/img3.jpg");
    Mat src2 = imread("../Images/img4.jpg");
    Mat src3 = imread("../Images/img5.jpg");
    resize(src1, src1, Size(src1.cols/4, src1.rows/4));
    resize(src2, src2, Size(src2.cols/4, src2.rows/4));
    resize(src3, src3, Size(src3.cols/4, src3.rows/4));
    vector<Mat> imgs;
    imgs.push_back(src1);
    imgs.push_back(src2);
    imgs.push_back(src3);
    Mat result;

    // SIFT feature detector and feature extractor
    SiftFeatureDetector detector( 0.05, 5.0 );
    SiftDescriptorExtractor extractor( 3.0 );

    Mat img1 = imgs.at(0);
    int n = 1;
    while (n<nImages){
        Mat img2 = imgs.at(n);

        // Feature detection
        vector<KeyPoint> keypoints1;
        vector<KeyPoint> keypoints2;
        detector.detect(img1, keypoints1);
        detector.detect(img2, keypoints2);

        // Feature display and print
        Mat features1, features2;
        drawKeypoints(img1,keypoints1,features1,Scalar(255, 0, 0),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        drawKeypoints(img2,keypoints2,features2,Scalar(0, 255, 0),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        printf("Keypoint1=%d \n", (int)keypoints1.size());
        printf("Keypoint2=%d \n", (int)keypoints2.size());

        // Feature descriptor computation
        Mat descriptors1, descriptors2;
        extractor.compute(img1, keypoints1, descriptors1 );
        extractor.compute(img2, keypoints2, descriptors2 );

        // Matching descriptors in two images
        FlannBasedMatcher matcher;
        vector<DMatch> matches;
        matcher.match(descriptors1, descriptors2, matches);
        printf("Matching: %d\n", matches.size());

        // Calculation of distances between keypoints
        double max_dist = 0; double min_dist = 100;
        for( int i = 0; i < descriptors1.rows; i++ ){
            double dist = matches[i].distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        // Store matches between good distance
        vector< DMatch > good_matches;
        for( int i = 0; i < descriptors1.rows; i++ ){
            if( matches[i].distance < 3*min_dist ){
                good_matches.push_back( matches[i]);
            }
        }

        // Draw good matches
        Mat img_matches;
        drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        // Show selected matches
        //imshow("Selected Matches", img_matches);

        // Compute Homography to join images
        vector<Point2f> image;
        vector<Point2f> world;
        for (int i=0; i<good_matches.size(); i++){
            image.push_back(keypoints1.at(good_matches[i].queryIdx).pt);
            world.push_back(keypoints2.at(good_matches[i].trainIdx).pt);
        }
        Mat H = findHomography(world, image, CV_RANSAC);

        // Use the Homography Matrix to warp the images
        warpPerspective(img2,result,H,cv::Size(img2.cols+img1.cols,img2.rows));

        // Create ROI in result
        Mat half(result,cv::Rect(0,0,img1.cols,img1.rows));

        // Copy first image in result ROI
        img1.copyTo(half);

        // Get the end of the image (on the right) (the border from where the next image is copied)
        int max = 0;
        Mat threshImage;
        threshold(result, threshImage, 1, 255, CV_THRESH_BINARY);
        flip(threshImage, threshImage, 1);
        for (int i=0; i<threshImage.rows; i++){
            for(int j=0; j<threshImage.cols; j++){
                if (threshImage.at<uchar>(i,j) > 0){
                    if (j > max) max = j;
                    break;
                }
            }
        }

        // Copy the image without right black area
        Rect myRoi(0, 0, result.cols-max/3, result.rows);
        Mat(result, myRoi).copyTo(result);

        // Iterate to the next image
        n++;

        // Copy result image into img1 for next iteration
        img1 = result.clone();

        printf("\n");
    }

    // Show result image
    imshow("Panorama", result);
    for (int i=0; i<imgs.size(); i++){
        char img_name[10];
        sprintf(img_name, "Image_%d", i);
        imshow(img_name, imgs.at(i));
    }

    cvWaitKey(0);

    // Release memory
    result.release();
    src1.release();
    src2.release();
    src3.release();

    return 0;
}

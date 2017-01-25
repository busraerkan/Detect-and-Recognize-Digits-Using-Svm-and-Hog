#include < opencv2\opencv.hpp>    
#include < string>    
#include < stdio.h>    
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "opencv2/opencv.hpp"
#include <stdio.h>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv/cv.h>
#include <sys/timeb.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <limits>

#define TRAIN false
#define CENTRAL_CROP false
#define PosSamNo0 380
#define NegSamNo0 1860

using namespace std;
using namespace cv;

void SvmLoad();
void originalResimOkuma();

void resmiBol(int homeCoorX1, int homeCoorY1, int homeCoorX2, int homeCoorY2, int visitorsCoorX1, int visitorsCoorY1,
	int visitorsCoorX2, int visitorsCoorY2, int homeScoreCoorX1, int homeScoreCoorY1, int homeScoreCoorX2, int homeScoreCoorY2,
	int homeScoreCoorX3, int homeScoreCoorY3, int visitorsScoreCoorX1, int visitorsScoreCoorY1, int visitorsScoreCoorX2,
	int visitorsScoreCoorY2, int visitorsScoreCoorX3, int visitorsScoreCoorY3, int timeCoorX1, int timeCoorY1, int timeCoorX2,
	int timeCoorY2, int timeCoorX3, int timeCoorY3, int timeCoorX4, int timeCoorY4, int width, int height);

void createSubImages(Mat scoreboardImage, Point homeCoor1, Point homeCoor2, Point visitorsCoor1,
	Point visitorsCoor2, Point homeScoreCoor1, Point homeScoreCoor2,
	Point homeScoreCoor3, Point visitorsScoreCoor1, Point visitorsScoreCoor2,
	Point visitorsScoreCoor3, Point timeCoor1, Point timeCoor2, Point timeCoor3,
	Point timeCoor4, int width, int height);

void whichDigit(Mat scoreboardImage, Point p1);
int findMaxScore(vector<float> scores);
int registered = 0;

Mat createScoreboardImage(Point point1, Point point2);
void showDigitResults(Vector<Point> result);

void detectDigit(Mat scoreboardImage);

Point findMaxScoreLocation(Mat subImage);
Vector<Point> findMaxLocations();
int findMaxValue(int sinir, int index, Vector<float> maxScoresValues);
Vector<Point> findGroups(Vector<float> maxScoresValues, Vector<Point> maxScoresCoor);

static void onMouse(int event, int x, int y, int, void*);

Point2f roi4point[4] = { 0, };
bool oksign = false;
int roiIndex = 0;

Mat src;
Mat originals[22];
int DescriptorDim0;
Mat digitMat;
Mat homeR1, homeR2, visitorsR1, visitorsR2, homeScoreR1, homeScoreR2;
Mat homeScoreR3, visitorsScoreR1, visitorsScoreR2, visitorsScoreR3;
Mat	timeR1, timeR2, timeR3, timeR4;
Vector<Point> maxScoresCoorAll;
Vector<float> maxScoresValuesAll;
CvSVM svm0, svm1, svm2, svm3, svm4, svm5, svm6, svm7, svm8, svm9, svmDigit;
HOGDescriptor hog0(Size(64, 32), Size(8, 8), Size(8, 8), Size(4, 4), 9);
HOGDescriptor hog1(Size(64, 32), Size(8, 8), Size(8, 8), Size(4, 4), 9);
HOGDescriptor hog2(Size(64, 32), Size(8, 8), Size(8, 8), Size(4, 4), 9);
HOGDescriptor hog3(Size(64, 32), Size(8, 8), Size(8, 8), Size(4, 4), 9);
HOGDescriptor hog4(Size(64, 32), Size(8, 8), Size(8, 8), Size(4, 4), 9);
HOGDescriptor hog5(Size(64, 32), Size(8, 8), Size(8, 8), Size(4, 4), 9);
HOGDescriptor hog6(Size(64, 32), Size(8, 8), Size(8, 8), Size(4, 4), 9);
HOGDescriptor hog7(Size(64, 32), Size(8, 8), Size(8, 8), Size(4, 4), 9);
HOGDescriptor hog8(Size(64, 32), Size(8, 8), Size(8, 8), Size(4, 4), 9);
HOGDescriptor hog9(Size(64, 32), Size(8, 8), Size(8, 8), Size(4, 4), 9);
HOGDescriptor hogDigit(Size(64, 32), Size(8, 8), Size(8, 8), Size(4, 4), 9);

Point point1, point2, point3, point4;


void mouseHandler(int event, int x, int y, int flags, void* param)
{
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		//4 point select  
		if (roiIndex >= 4)
		{
			roiIndex = 0;
			for (int i = 0; i< 4; ++i)
				roi4point[i].x = roi4point[i].y = 0;
		}
		cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
		roi4point[roiIndex].x = x;
		roi4point[roiIndex].y = y;
		point3 = Point(x, y);
		roiIndex++;
	}

	if (event == CV_EVENT_LBUTTONUP)
	{
		//set point. 
		if (roiIndex == 4)
		{
			oksign = true;
			printf("Warping Start!!!\n");
		}
	}
}

int main()
{

	SvmLoad(); // load svm
	//src = imread("C:/Users/user/Documents/Visual Studio 2013/Projects/19agustos/build/Resize/0/191.png", 1);
	//src = imread("C:/Users/user/Desktop/cam_video/7.png", 1);
	originalResimOkuma();
	VideoCapture cap("C:/Users/user/Desktop/bitirme video test.mp4"); // open video
	if (!cap.isOpened())  // check if we succeeded
		return -1;
	else
		//cout << "video acildi  " << endl;
		cap >> src;
	//resize(src, src, Size(2120, 400), 0.0, 0.0, 1); // okunan resmi 2120x400 olarak resize etme
	resize(src, src, Size(3500, 550), 0.0, 0.0, 1); // okunan resmi 2120x400 olarak resize etme
	String winname = "Skor Ekraný";
	namedWindow(winname, WINDOW_NORMAL);
	imshow(winname, src);
	cvSetMouseCallback("Skor Ekraný", mouseHandler, NULL);
	waitKey();
	destroyWindow("Skor Ekraný");
	//resize(originals[10], originals[10], Size(4000, 800), 0.0, 0.0, 1);
	Mat scoreboardImage = createScoreboardImage(roi4point[0], roi4point[3]);
	Rect scoreboardRect(roi4point[0], roi4point[3]);
	originals[10] = originals[10](scoreboardRect);

	imwrite("scoreboard.png", scoreboardImage);
	detectDigit(scoreboardImage);
	Vector<Point> maxLocations = findMaxLocations();
	showDigitResults(maxLocations);

	for (int i = 0; i < maxLocations.size(); ++i) {
		whichDigit(scoreboardImage, maxLocations[i]);
	}
	for (;;) {
		cap >> src;
		if (src.empty())
			return 0;
		resize(src, src, Size(3500, 550), 0.0, 0.0, 1); // okunan resmi 2120x400 olarak resize etme
		scoreboardImage = createScoreboardImage(roi4point[0], roi4point[3]);
		imwrite("scoreboard.png", scoreboardImage);
		for (int i = 0; i < maxLocations.size(); ++i) {
			whichDigit(scoreboardImage, maxLocations[i]);
		}
		//cout << "********************" << endl;
	}

	//resize(src, src, Size(2120, 400), 0.0, 0.0, 1); // okunan resmi 2120x400 olarak resize etme
	//imwrite("cap.png", src);
	//Mat gray;

	//cvtColor(scoreboardImage, gray, CV_RGB2GRAY);


	//detectDigit(scoreboardImage);

	//imwrite("src.png", src);
	waitKey();
	system("pause");
}

Vector<Point> findMaxLocations() {
	float max = 0;
	int maxX, maxY, temp;
	Vector<Point> maxScoresCoorEnd;
	while (true) {
		for (int s = 0; s < maxScoresValuesAll.size(); s++){
			if (max < abs(maxScoresValuesAll[s])){
				max = abs(maxScoresValuesAll[s]);
				//maxScoresValuesAll[s] = 0;
				temp = s;
			}
		}
		if (maxScoresValuesAll[temp] != 0){
			maxScoresCoorEnd.push_back(maxScoresCoorAll[temp]);
			maxX = maxScoresCoorAll[temp].x;
			maxY = maxScoresCoorAll[temp].y;
		}
		for (int m = 0; m < maxScoresCoorAll.size(); m++){
			if (maxScoresCoorAll[m].x > maxX - 16 && maxScoresCoorAll[m].x<maxX + 15
				&& maxScoresCoorAll[m].y>maxY - 16 && maxScoresCoorAll[m].y < maxY + 15){
				maxScoresValuesAll[m] = 0;
			}
		}
		if (max == 0)
			break;
		max = 0;
	}
	//cout << maxScoresCoorEnd.size();
	return maxScoresCoorEnd;

}

void detectDigit(Mat scoreboardImage) {
	Mat subImage; /*hog hesaplanacak image*/
	float maxScore = 0;
	int xMax = 0, yMax = 0;
	vector<float> descriptors, res;
	for (int x = 0; x < scoreboardImage.cols - 64; x = x + 3) {
		for (int y = 0; y < scoreboardImage.rows - 32; y = y + 3) {
			Rect subImageRect(x, y, 64, 32);
			subImage = scoreboardImage(subImageRect);
			hogDigit.compute(subImage, descriptors, Size(8, 8));
			cv::Mat1f matrixT(cv::Mat1f(descriptors).t());
			cv::Mat1f newMatrix;
			newMatrix.push_back(matrixT);
			float fDistance = svmDigit.predict(newMatrix, true);
			if (fDistance <= -0.7) {
				if (fDistance < maxScore){
					xMax = x;	yMax = y;
					maxScore = fDistance;
					//cout << "x: " << x << " y: " << y << " ";
					//cout << fDistance << endl;
					maxScoresCoorAll.push_back(Point(xMax, yMax));
					maxScoresValuesAll.push_back(fDistance);
				}
			}
			if ((x % 64 == 0) || (y % 32 == 0)) {
				maxScore = 0;
			}
		}
	}
	cout << "Digit locations detected." << endl;
}

void whichDigit(Mat scoreboardImage, Point p1) {
	vector<float> scores;
	vector<float> descriptors, res;
	Rect digitRect(p1.x, p1.y, 64, 32);
	digitMat = scoreboardImage(digitRect);
	hog0.compute(digitMat, descriptors, Size(8, 8));
	cv::Mat1f matrixT(cv::Mat1f(descriptors).t());
	cv::Mat1f newMatrix;
	newMatrix.push_back(matrixT);
	float fDistance = svm0.predict(newMatrix, true);
	scores.push_back(fDistance);

	hog1.compute(digitMat, descriptors, Size(8, 8));
	cv::Mat1f matrixT1(cv::Mat1f(descriptors).t());
	cv::Mat1f newMatrix1;
	newMatrix1.push_back(matrixT1);
	fDistance = svm1.predict(newMatrix, true);
	scores.push_back(fDistance);

	hog2.compute(digitMat, descriptors, Size(8, 8));
	cv::Mat1f matrixT2(cv::Mat1f(descriptors).t());
	cv::Mat1f newMatrix2;
	newMatrix2.push_back(matrixT2);
	fDistance = svm2.predict(newMatrix, true);
	scores.push_back(fDistance);

	hog3.compute(digitMat, descriptors, Size(8, 8));
	cv::Mat1f matrixT3(cv::Mat1f(descriptors).t());
	cv::Mat1f newMatrix3;
	newMatrix3.push_back(matrixT3);
	fDistance = svm3.predict(newMatrix, true);
	scores.push_back(fDistance);

	hog4.compute(digitMat, descriptors, Size(8, 8));
	cv::Mat1f matrixT4(cv::Mat1f(descriptors).t());
	cv::Mat1f newMatrix4;
	newMatrix4.push_back(matrixT4);
	fDistance = svm4.predict(newMatrix, true);
	scores.push_back(fDistance);

	hog5.compute(digitMat, descriptors, Size(8, 8));
	cv::Mat1f matrixT5(cv::Mat1f(descriptors).t());
	cv::Mat1f newMatrix5;
	newMatrix5.push_back(matrixT5);
	fDistance = svm5.predict(newMatrix, true);
	scores.push_back(fDistance);

	hog6.compute(digitMat, descriptors, Size(8, 8));
	cv::Mat1f matrixT6(cv::Mat1f(descriptors).t());
	cv::Mat1f newMatrix6;
	newMatrix6.push_back(matrixT6);
	fDistance = svm6.predict(newMatrix, true);
	scores.push_back(fDistance);

	hog7.compute(digitMat, descriptors, Size(8, 8));
	cv::Mat1f matrixT7(cv::Mat1f(descriptors).t());
	cv::Mat1f newMatrix7;
	newMatrix7.push_back(matrixT7);
	fDistance = svm7.predict(newMatrix, true);
	scores.push_back(fDistance);

	hog8.compute(digitMat, descriptors, Size(8, 8));
	cv::Mat1f matrixT8(cv::Mat1f(descriptors).t());
	cv::Mat1f newMatrix8;
	newMatrix8.push_back(matrixT8);
	fDistance = svm8.predict(newMatrix, true);
	scores.push_back(fDistance);

	hog9.compute(digitMat, descriptors, Size(8, 8));
	cv::Mat1f matrixT9(cv::Mat1f(descriptors).t());
	cv::Mat1f newMatrix9;
	newMatrix9.push_back(matrixT9);
	fDistance = svm9.predict(newMatrix, true);
	scores.push_back(fDistance);

	int max = findMaxScore(scores);
	originals[max].copyTo(originals[10](cv::Rect(p1.x, p1.y, originals[max].cols, originals[max].rows)));
	char imgName[100];
	sprintf_s(imgName, "Scores/%d.png", registered++);
	imwrite(imgName, originals[10]);
	//imwrite("skor ekraný.png", originals[10]);
	//imshow("skor ekraný", originals[10]);
	//cout << p1 << " " << max << endl;

}

int findMaxScore(vector<float> scores){
	int maxIndex = 0;
	for (int i = 1; i < scores.size(); ++i){
		if (scores[maxIndex] > scores[i]){
			maxIndex = i;
		}
	}
	return maxIndex;
}

void originalResimOkuma(){
	originals[0] = imread("../build/Original/0.png");	  // 0
	resize(originals[0], originals[0], Size(64, 32), 0.0, 0.0, 1);
	originals[1] = imread("../build/Original/1.png");	  // 1
	resize(originals[1], originals[1], Size(64, 32), 0.0, 0.0, 1);
	originals[2] = imread("../build/Original/2.png");	  // 2
	resize(originals[2], originals[2], Size(64, 32), 0.0, 0.0, 1);
	originals[3] = imread("../build/Original/3.png");	  // 3
	resize(originals[3], originals[3], Size(64, 32), 0.0, 0.0, 1);
	originals[4] = imread("../build/Original/4.png");	  // 4
	resize(originals[4], originals[4], Size(64, 32), 0.0, 0.0, 1);
	originals[5] = imread("../build/Original/5.png");	  // 5
	resize(originals[5], originals[5], Size(64, 32), 0.0, 0.0, 1);
	originals[6] = imread("../build/Original/6.png");	  // 6
	resize(originals[6], originals[6], Size(64, 32), 0.0, 0.0, 1);
	originals[7] = imread("../build/Original/7.png");	  // 7
	resize(originals[7], originals[7], Size(64, 32), 0.0, 0.0, 1);
	originals[8] = imread("../build/Original/8.png");	  // 8
	resize(originals[8], originals[8], Size(64, 32), 0.0, 0.0, 1);
	originals[9] = imread("../build/Original/9.png");	  // 9
	resize(originals[9], originals[9], Size(64, 32), 0.0, 0.0, 1);
	originals[10] = imread("../build/Original/black.png"); // black
	originals[11] = imread("../build/Original/nokta.png"); // nokta black
	originals[12] = imread("../build/Original/0b.png");	  // 0b
	originals[13] = imread("../build/Original/1b.png");	  // 1b
	originals[14] = imread("../build/Original/2b.png");	  // 2b
	originals[15] = imread("../build/Original/3b.png");	  // 3b
	originals[16] = imread("../build/Original/4b.png");	  // 4b
	originals[17] = imread("../build/Original/5b.png");	  // 5b
	originals[18] = imread("../build/Original/6b.png");	  // 6b
	originals[19] = imread("../build/Original/7b.png");	  // 7b
	originals[20] = imread("../build/Original/8b.png");	  // 8b
	originals[21] = imread("../build/Original/9b.png");	  // 9b
}
void showDigitResults(Vector<Point> result){
	for (int i = 0; i < result.size(); ++i){
		//cout << result[i] << endl;
	}
	//cout << result.size() << endl;
}
Mat createScoreboardImage(Point point1, Point point2) {
	Rect scoreboardRect(point1, point2);
	Mat scoreboardImage = src(scoreboardRect);
	return scoreboardImage;
}
void SvmLoad() {
	svm0.load("SVM0.xml");
	//cout << "svm0 yuklendi" << endl;
	svm1.load("SVM1.xml");
	//cout << "svm1 yuklendi" << endl;
	svm2.load("SVM2.xml");
	//cout << "svm2 yuklendi" << endl;
	svm3.load("SVM3.xml");
	//cout << "svm3 yuklendi" << endl;
	svm4.load("SVM4.xml");
	//cout << "svm4 yuklendi" << endl;
	svm5.load("SVM5.xml");
	//cout << "svm5 yuklendi" << endl;
	svm6.load("SVM6.xml");
	//cout << "svm6 yuklendi" << endl;
	svm7.load("SVM7.xml");
	//cout << "svm7 yuklendi" << endl;
	svm8.load("SVM8.xml");
	//cout << "svm8 yuklendi" << endl;
	svm9.load("SVM9.xml");
	//cout << "svm9 yuklendi" << endl;
	svmDigit.load("SVMDigit.xml");
	cout << "svmDigit yuklendi" << endl;
}
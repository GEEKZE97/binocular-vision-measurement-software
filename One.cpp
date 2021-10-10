#include <opencv2\opencv.hpp>
#include <fstream>
#include <iostream>
#include <utility>
#include <cmath>
#include <fstream>  
#include <sstream>
#include <algorithm>
#include <numeric>
#include <string>

int LEVEL_L = 250;                     //二值化阈值
double METRIC_L = 0.90;                //圆度筛选值
double THE_AREA_MIN = 7000;            //轮廓面积最小值
double THE_AREA_MAX = 25000;           //轮廓面积最大值
double THE_DIZTANCE = 10;              //极线约束最大值

using namespace std;
using namespace cv;

///////////////////////////提点
Mat calLeft(3, 3, CV_64F);
Mat calRight(3, 3, CV_64F);
Mat calLeftRT = (Mat_<double>(3, 4) << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0);
Mat calRightRT(3, 4, CV_64F);
Mat cal1;
Mat cal2;
Mat matEssential;
Mat matFundamental;

////////////////////////////重建
Mat R(3, 3, CV_64F);
Mat T(3, 1, CV_64F);
pair<double, double> fcLeft;
vector<double> kcLeft(5, 0);
Point2d ccLeft;
double alpha_cLeft = 0;
pair<double, double> fcRight;
vector<double> kcRight(5, 0);
Point2d ccRight;
double alpha_cRight = 0;
/////////////////////////////函数
Mat convertTo3Channels(const Mat& binImg)//将单通道图片转换成三通道
{
	Mat three_channel = Mat::zeros(binImg.rows, binImg.cols, CV_8UC3);
	vector<Mat> channels;
	for (int i = 0; i < 3; i++)
	{
		channels.push_back(binImg);
	}
	merge(channels, three_channel);
	return three_channel;
}
void normalize_pixel(vector<Point2d>& x_kk, pair<double, double> fc, Point2d cc, vector<double> kc, double alpha_c) {
	if (x_kk.empty())
		return;
	for (int i = 0; i < x_kk.size(); ++i) {
		x_kk[i].x = (x_kk[i].x - cc.x) / fc.first;
		x_kk[i].y = (x_kk[i].y - cc.y) / fc.second;
	}
	if (alpha_c != 0) {
		for (int i = 0; i < x_kk.size(); ++i) {
			x_kk[i].x = x_kk[i].x - alpha_c * x_kk[i].y;
		}
	}
	// compdistortionoulu
	vector<Point2d> x_d = x_kk;
	for (int i = 0; i < 20; ++i) {
		vector<double> r_2;
		for (int i = 0; i < x_kk.size(); ++i) {
			r_2.push_back(x_kk[i].x * x_kk[i].x + x_kk[i].y * x_kk[i].y);
		}
		vector<double> k_radial;
		for (int i = 0; i < r_2.size(); ++i) {
			k_radial.push_back(1 + kc[0] * r_2[i] + kc[1] * r_2[i] * r_2[i] + kc[4] * r_2[i] * r_2[i] * r_2[i]);
		}
		vector<Point2d> delta_x;
		for (int i = 0; i < r_2.size(); ++i) {
			double temp0 = 2 * kc[2] * x_kk[i].x * x_kk[i].y + kc[3] * (r_2[i] + 2 * x_kk[i].x * x_kk[i].x);
			double temp1 = kc[2] * (r_2[i] + 2 * x_kk[i].y * x_kk[i].y) + 2 * kc[3] * x_kk[i].x * x_kk[i].y;
			delta_x.push_back(Point2d(temp0, temp1));
		}
		for (int i = 0; i < x_kk.size(); ++i) {
			x_kk[i].x = (x_d[i].x - delta_x[i].x) / k_radial[i];
			x_kk[i].y = (x_d[i].y - delta_x[i].y) / k_radial[i];
		}
		int khi = 1;
	}

	return;
}
Mat rebuild(vector<Point2d> DD, vector<Point2d> BB) {
	if (DD.size() != BB.size() || DD.size() == 0) {
		Mat Empty;
		return Empty;
	}
	vector<double> DD3(DD.size(), 1);
	vector<double> BB3(BB.size(), 1);
	normalize_pixel(DD, fcLeft, ccLeft, kcLeft, alpha_cLeft);
	normalize_pixel(BB, fcRight, ccRight, kcRight, alpha_cRight);
	Mat DDleft(3, DD.size(), CV_64F);
	Mat BBright(3, BB.size(), CV_64F);
	for (int i = 0; i < DD.size(); ++i) {
		DDleft.at<double>(0, i) = DD[i].x;
		DDleft.at<double>(1, i) = DD[i].y;
		DDleft.at<double>(2, i) = 1.0;
		BBright.at<double>(0, i) = BB[i].x;
		BBright.at<double>(1, i) = BB[i].y;
		BBright.at<double>(2, i) = 1.0;
	}
	Mat u = R * DDleft;
	//cout << u << endl;
	Mat n_xt2temp = DDleft.mul(DDleft);
	Mat n_xt2(1, DD.size(), CV_64F);
	for (int i = 0; i < DD.size(); ++i) {
		n_xt2.at<double>(0, i) = n_xt2temp.at<double>(0, i) + n_xt2temp.at<double>(1, i) + n_xt2temp.at<double>(2, i);
	}
	//cout << n_xt2;
	Mat n_xtt2temp = BBright.mul(BBright);
	Mat n_xtt2(1, BB.size(), CV_64F);
	for (int i = 0; i < BB.size(); ++i) {
		n_xtt2.at<double>(0, i) = n_xtt2temp.at<double>(0, i) + n_xtt2temp.at<double>(1, i) + n_xtt2temp.at<double>(2, i);
	}
	//cout << n_xtt2 << endl;

	Mat doT = u.mul(BBright);
	Mat doT2(1, DD.size(), CV_64F);
	for (int i = 0; i < DD.size(); ++i) {
		doT2.at<double>(0, i) = (doT.at<double>(0, i) + doT.at<double>(1, i) + doT.at<double>(2, i)) *
			(doT.at<double>(0, i) + doT.at<double>(1, i) + doT.at<double>(2, i));
	}
	Mat DDfinal = n_xt2.mul(n_xtt2) - doT2;
	//cout << DDfinal << endl;
	Mat dot_uT(1, DD.size(), CV_64F);
	for (int i = 0; i < DD.size(); ++i) {
		dot_uT.at<double>(0, i) =
			u.at<double>(0, i) * T.at<double>(0, 0) + u.at<double>(1, i) * T.at<double>(1, 0) + u.at<double>(2, i) * T.at<double>(2, 0);
	}
	//cout << dot_uT << endl;
	Mat dot_xttT(1, DD.size(), CV_64F);
	for (int i = 0; i < DD.size(); ++i) {
		dot_xttT.at<double>(0, i) =
			BBright.at<double>(0, i) * T.at<double>(0, 0) + BBright.at<double>(1, i) * T.at<double>(1, 0) + BBright.at<double>(2, i) * T.at<double>(2, 0);
	}
	//cout << dot_xttT << endl;
	Mat dot_xttu(1, DD.size(), CV_64F);
	for (int i = 0; i < DD.size(); ++i) {
		dot_xttu.at<double>(0, i) =
			u.at<double>(0, i) * BBright.at<double>(0, i) + u.at<double>(1, i) * BBright.at<double>(1, i) + u.at<double>(2, i) * BBright.at<double>(2, i);
	}
	//cout << dot_xttu << endl;
	Mat NN1 = dot_xttu.mul(dot_xttT) - n_xtt2.mul(dot_uT);
	//cout << NN1 << endl;
	Mat NN2 = n_xt2.mul(dot_xttT) - dot_uT.mul(dot_xttu);
	//cout << NN2 << endl;
	Mat Zt(1, DD.size(), CV_64F);
	for (int i = 0; i < DD.size(); ++i) {
		Zt.at<double>(0, i) =
			NN1.at<double>(0, i) / DDfinal.at<double>(0, i);
	}
	Mat Ztt(1, DD.size(), CV_64F);
	for (int i = 0; i < DD.size(); ++i) {
		Ztt.at<double>(0, i) =
			NN2.at<double>(0, i) / DDfinal.at<double>(0, i);
	}
	//cout << Zt << endl;
	//cout << Ztt << endl;
	Mat X1(3, DD.size(), CV_64F);
	for (int i = 0; i < DD.size(); ++i) {
		X1.at<double>(0, i) =
			DDleft.at<double>(0, i) * Zt.at<double>(0, i);
		X1.at<double>(1, i) =
			DDleft.at<double>(1, i) * Zt.at<double>(0, i);
		X1.at<double>(2, i) =
			DDleft.at<double>(2, i) * Zt.at<double>(0, i);
	}
	//cout << X1 << endl;
	Mat X2temp(3, DD.size(), CV_64F);
	for (int i = 0; i < DD.size(); ++i) {
		X2temp.at<double>(0, i) =
			BBright.at<double>(0, i) * Ztt.at<double>(0, i);
		X2temp.at<double>(1, i) =
			BBright.at<double>(1, i) * Ztt.at<double>(0, i);
		X2temp.at<double>(2, i) =
			BBright.at<double>(2, i) * Ztt.at<double>(0, i);
	}
	//cout << X2temp << endl;
	//cout << R.t() << endl;
	Mat X2 = X2temp;
	//cout << X2 << endl;
	for (int i = 0; i < DD.size(); ++i) {
		X2.at<double>(0, i) = X2.at<double>(0, i) - T.at<double>(0, 0);
		X2.at<double>(1, i) = X2.at<double>(1, i) - T.at<double>(1, 0);
		X2.at<double>(2, i) = X2.at<double>(2, i) - T.at<double>(2, 0);
	}
	X2 = R.t() * X2;
	//cout << X2 << endl;
	Mat XL = 0.5 * (X1 + X2);
	//cout << XL << endl;
	Mat XR = R * XL;
	for (int i = 0; i < DD.size(); ++i) {
		XR.at<double>(0, i) = XR.at<double>(0, i) + T.at<double>(0, 0);
		XR.at<double>(1, i) = XR.at<double>(1, i) + T.at<double>(1, 0);
		XR.at<double>(2, i) = XR.at<double>(2, i) + T.at<double>(2, 0);
	}
	return XR;
}
vector<Point2d> getCenterPoints(const Mat& image, string LorR)
{
	Mat imageBinary;
	threshold(image, imageBinary, LEVEL_L, 255, THRESH_BINARY);

	vector<vector<Point>>* contours = new vector<vector<Point>>;
	vector<Vec4i> hierarchy;
	findContours(imageBinary, *contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);//提取外轮廓

	Mat colImage = convertTo3Channels(imageBinary);
	drawContours(colImage, *contours, -1, Scalar(0, 0, 255), 10, 8);
	imwrite(string("Pic/AllContours") + LorR + ".bmp", colImage);
	vector<vector<Point>> count;//记录满足要求的轮廓
	for (auto beg = contours->begin(), end = contours->end(); beg != end; ++beg)
	{
		double area = contourArea(*beg);//轮廓面积
		double l = arcLength(*beg, true);//轮廓周长
		double metric;
		metric = 4 * 3.1415926 * area / (l * l);//轮廓圆度
		if (area <= THE_AREA_MIN || area >= THE_AREA_MAX || metric < METRIC_L)
			continue;
		else {
			count.push_back(*beg);
		}
	}
	colImage = convertTo3Channels(imageBinary);
	drawContours(colImage, count, -1, Scalar(0, 0, 255), 10, 8);
	imwrite(string("Pic/Contours") + LorR + ".bmp", colImage);
	delete contours;//释放空间
	vector<Point2d> final;//提取的中心点
	Point2d finalcenter(0, 0);
	for (int i = 0; i < count.size(); ++i)
	{
		auto tey = fitEllipse(count[i]);
		final.push_back(Point2d(double(tey.center.x), double(tey.center.y)));

	}
	return final;
}
double getDist_P2L(Point2d powerPoint, Vec<double, 3> floatingGato)
{
	//求直线方程
	double A = floatingGato[0];
	double B = floatingGato[1];
	double C = floatingGato[2];
	//代入点到直线距离公式
	double distance = 0;
	distance = ((double)abs(A * powerPoint.x + B * powerPoint.y + C)) / ((double)sqrtf(A * A + B * B));
	return distance;
}
void match(vector<Point2d>& pointGroupA, vector<Point2d>& pointGroupB, const Mat& matFundamental)//单极线匹配
{
	if (pointGroupA.empty() || pointGroupB.empty()) {
		pointGroupA.clear();
		pointGroupB.clear();
		return;
	}
	sort(pointGroupA.begin(), pointGroupA.end(), [](Point2d a, Point2d b) {return a.x < b.x; });
	sort(pointGroupB.begin(), pointGroupB.end(), [](Point2d a, Point2d b) {return a.x < b.x; });
	vector<Vec<double, 3>> epilinez, epilinez2;
	computeCorrespondEpilines(pointGroupA, 1, matFundamental, epilinez);
	computeCorrespondEpilines(pointGroupB, 2, matFundamental, epilinez2);
	vector<Point2d> atrazMatchA, atrazMatchB;
	for (int i = 0; i < pointGroupA.size(); ++i)
	{
		for (int j = 0; j < epilinez2.size(); ++j)
		{
			double diz = getDist_P2L(pointGroupA[i], epilinez2[j]);
			if (diz < THE_DIZTANCE)
			{
				atrazMatchA.push_back(pointGroupA[i]);
				atrazMatchB.push_back(pointGroupB[j]);
				epilinez2[j][0] = 1;
				epilinez2[j][1] = 1;
				epilinez2[j][2] = 1;
				break;
			}
		}
	}
	pointGroupA = atrazMatchA;
	pointGroupB = atrazMatchB;
}
void mergeImage(Mat& dst, vector<Mat>& images)//图像拼接，需要输入三通道图片
{
	int imgCount = (int)images.size();
	int rows = 6004;//将每个图片缩小为指定大小
	int cols = 7904;
	for (int i = 0; i < imgCount; i++)
	{
		resize(images[i], images[i], Size(cols, rows)); //注意区别：Size函数的两个参数分别为：宽和高，宽对应cols，高对应rows
	}
	dst.create(rows * imgCount / 2, cols * 2, CV_8UC3);//创建新图片的尺寸，高：rows * imgCount/2，宽：cols * 2
	for (int i = 0; i < imgCount; i++)
	{
		images[i].copyTo(dst(Rect((i % 2) * cols, (i / 2) * rows, images[0].cols, images[0].rows)));
	}
}
Mat exhibit(Mat ImageLeft, Mat ImageRight, vector<Point2d> pointGroupA, vector<Point2d> pointGroupB)
{
	vector<Mat> ImagesColorVector{ convertTo3Channels(ImageLeft) ,convertTo3Channels(ImageRight) };
	Mat ImagesColor;
	mergeImage(ImagesColor, ImagesColorVector);
	//Mat imageWhite = Mat(6004, 400, CV_8UC3, Scalar(255, 255, 255));//如果加白块
	vector<string> text{ "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W" };
	Mat littleP;
	resize(ImagesColor, littleP, Size(ImagesColor.cols / 4, ImagesColor.rows / 4), 0, 0, INTER_LINEAR);
	for (auto& point : pointGroupB)
	{
		point.x += 7904;
	}
	for (int i = 0; i < pointGroupA.size(); ++i)
	{
		RNG& rng = theRNG();
		Scalar certezaColor2 = Scalar(0, 0, 255);
		Scalar certezaColor = Scalar(255, 255, 255);
		Scalar RandomColor = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		circle(littleP, pointGroupA[i] / 4, 10, certezaColor2, 3, 8, 0);
		circle(littleP, pointGroupB[i] / 4, 10, certezaColor2, 3, 8, 0);
		line(littleP, pointGroupA[i] / 4, pointGroupB[i] / 4, RandomColor, 3, 8);
		putText(littleP, text[i], Point2d(pointGroupA[i].x / 4, pointGroupA[i].y / 4 + 100), cv::FONT_HERSHEY_PLAIN, 4, RandomColor, 4, 8, false);
		putText(littleP, text[i], Point2d(pointGroupB[i].x / 4, pointGroupB[i].y / 4 + 100), cv::FONT_HERSHEY_PLAIN, 4, RandomColor, 4, 8, false);
	}
	return littleP;
}
void initailAllMat(string fileName) {
	cout << "参数值" << endl;
	ifstream inFile(fileName, ios::in);
	vector<double> tempVec;
	string tempZtr;
	while (getline(inFile, tempZtr)) {
		tempVec.push_back(stod(tempZtr));
	}
	if (tempVec.size() != 50) {
		cout << "file error" << endl;
		return;
	}
	calLeft.at<double>(0, 0) = tempVec[0];
	calLeft.at<double>(0, 1) = tempVec[1];
	calLeft.at<double>(0, 2) = tempVec[2];
	calLeft.at<double>(1, 0) = tempVec[3];
	calLeft.at<double>(1, 1) = tempVec[4];
	calLeft.at<double>(1, 2) = tempVec[5];
	calLeft.at<double>(2, 0) = tempVec[6];
	calLeft.at<double>(2, 1) = tempVec[7];
	calLeft.at<double>(2, 2) = tempVec[8];
	cout << "calLeft" << endl;
	cout << calLeft << endl;
	calRight.at<double>(0, 0) = tempVec[9];
	calRight.at<double>(0, 1) = tempVec[10];
	calRight.at<double>(0, 2) = tempVec[11];
	calRight.at<double>(1, 0) = tempVec[12];
	calRight.at<double>(1, 1) = tempVec[13];
	calRight.at<double>(1, 2) = tempVec[14];
	calRight.at<double>(2, 0) = tempVec[15];
	calRight.at<double>(2, 1) = tempVec[16];
	calRight.at<double>(2, 2) = tempVec[17];
	cout << "calRight" << endl;
	cout << calRight << endl;
	R.at<double>(0, 0) = tempVec[18];
	R.at<double>(0, 1) = tempVec[19];
	R.at<double>(0, 2) = tempVec[20];
	R.at<double>(1, 0) = tempVec[21];
	R.at<double>(1, 1) = tempVec[22];
	R.at<double>(1, 2) = tempVec[23];
	R.at<double>(2, 0) = tempVec[24];
	R.at<double>(2, 1) = tempVec[25];
	R.at<double>(2, 2) = tempVec[26];
	cout << "R" << endl;
	cout << R << endl;
	T.at<double>(0, 0) = tempVec[27];
	T.at<double>(1, 0) = tempVec[28];
	T.at<double>(2, 0) = tempVec[29];
	cout << "T" << endl;
	cout << T << endl;
	calRightRT.at<double>(0, 0) = tempVec[18];
	calRightRT.at<double>(0, 1) = tempVec[19];
	calRightRT.at<double>(0, 2) = tempVec[20];
	calRightRT.at<double>(0, 3) = tempVec[27];
	calRightRT.at<double>(1, 0) = tempVec[21];
	calRightRT.at<double>(1, 1) = tempVec[22];
	calRightRT.at<double>(1, 2) = tempVec[23];
	calRightRT.at<double>(1, 3) = tempVec[28];
	calRightRT.at<double>(2, 0) = tempVec[24];
	calRightRT.at<double>(2, 1) = tempVec[25];
	calRightRT.at<double>(2, 2) = tempVec[26];
	calRightRT.at<double>(2, 3) = tempVec[29];
	cout << "calRightRT" << endl;
	cout << calRightRT << endl;
	///
	cal1 = calLeft * calLeftRT;
	cal2 = calRight * calRightRT;
	matEssential = (Mat_<double>(3, 3) << 0, -calRightRT.at<double>(2, 3), calRightRT.at<double>(1, 3), calRightRT.at<double>(2, 3), 0, -calRightRT.at<double>(0, 3),
		-calRightRT.at<double>(1, 3), calRightRT.at<double>(0, 3), 0) * (Mat_<double>(3, 3) << calRightRT.at<double>(0, 0), calRightRT.at<double>(0, 1), calRightRT.at<double>(0, 2),
			calRightRT.at<double>(1, 0), calRightRT.at<double>(1, 1), calRightRT.at<double>(1, 2), calRightRT.at<double>(2, 0), calRightRT.at<double>(2, 1), calRightRT.at<double>(2, 2));
	matFundamental = (calRight.inv().t()) * matEssential * (calLeft.inv());
	///
	fcLeft = make_pair(tempVec[30], tempVec[31]);
	cout << "fcLeft" << endl;
	cout << fcLeft.first << " " << fcLeft.second << endl;
	kcLeft[0] = tempVec[32];
	kcLeft[1] = tempVec[33];
	kcLeft[2] = tempVec[34];
	kcLeft[3] = tempVec[35];
	kcLeft[4] = tempVec[36];
	cout << "kcLeft" << endl;
	cout << kcLeft[0] << " " << kcLeft[1] << " " << kcLeft[2] << " " << kcLeft[3] << " " << kcLeft[4] << " " << endl;
	ccLeft = Point2d(tempVec[37], tempVec[38]);
	cout << "ccLeft" << endl;
	cout << ccLeft << endl;
	alpha_cLeft = tempVec[39];
	cout << "alpha_cLeft" << endl;
	cout << alpha_cLeft << endl;
	fcRight = make_pair(tempVec[40], tempVec[41]);
	cout << "fcRight" << endl;
	cout << fcRight.first << " " << fcRight.second << endl;
	kcRight[0] = tempVec[42];
	kcRight[1] = tempVec[43];
	kcRight[2] = tempVec[44];
	kcRight[3] = tempVec[45];
	kcRight[4] = tempVec[46];
	cout << "kcRight" << endl;
	cout << kcRight[0] << " " << kcRight[1] << " " << kcRight[2] << " " << kcRight[3] << " " << kcRight[4] << " " << endl;
	ccRight = Point2d(tempVec[47], tempVec[48]);
	cout << "ccRight" << endl;
	cout << ccRight << endl;
	alpha_cRight = tempVec[49];
	cout << "alpha_cRight" << endl;
	cout << alpha_cRight << endl;
}
void getConfig(const string& fileName, string& addrImageLeft, string& addrImageRight, string& initailAllMatAddr, string& RandTMatAddrC2B, string& RandTMatAddrB2B) {
	ifstream inFile(fileName, ios::in);
	vector<string> tempVec;
	string tempStr;
	while (getline(inFile, tempStr)) {
		tempVec.push_back(tempStr);
	}
	LEVEL_L = stoi(tempVec[1]);
	METRIC_L = stod(tempVec[3]);
	THE_AREA_MIN = stod(tempVec[5]);
	THE_AREA_MAX = stod(tempVec[7]);
	THE_DIZTANCE = stod(tempVec[9]);
	addrImageLeft = tempVec[11];
	addrImageRight = tempVec[13];
	initailAllMatAddr = tempVec[15];
	RandTMatAddrC2B = tempVec[17];
	RandTMatAddrB2B = tempVec[19];
	return;
}
vector<Point3d> vec2PointVec(Mat XR) {
	int cols = XR.cols;
	vector<Point3d> tempVec;
	for (int i = 0; i < cols; ++i) {
		tempVec.push_back(Point3d(XR.at<double>(0, i), XR.at<double>(1, i), XR.at<double>(2, i)));
	}
	return tempVec;
}
Mat PointVec2Mat(vector<Point3d> vecPoints) {
	Mat XR(3, vecPoints.size(), CV_64F);
	for (int i = 0; i < vecPoints.size(); ++i) {
		XR.at<double>(0, i) = vecPoints[i].x;
		XR.at<double>(1, i) = vecPoints[i].y;
		XR.at<double>(2, i) = vecPoints[i].z;
	}
	return XR;
}
vector<Point3d> file2RandTMat(const string& fileName, Mat XR) {
	ifstream inFile(fileName, ios::in);
	vector<double> tempVec;
	string tempStr;
	while(getline(inFile, tempStr)) {
		tempVec.push_back(stod(tempStr));
	}
	Mat R(3, 3, CV_64F);
	Mat T(3, 1, CV_64F);
	R.at<double>(0, 0) = tempVec[0];
	R.at<double>(0, 1) = tempVec[1];
	R.at<double>(0, 2) = tempVec[2];
	R.at<double>(1, 0) = tempVec[3];
	R.at<double>(1, 1) = tempVec[4];
	R.at<double>(1, 2) = tempVec[5];
	R.at<double>(2, 0) = tempVec[6];
	R.at<double>(2, 1) = tempVec[7];
	R.at<double>(2, 2) = tempVec[8];
	T.at<double>(0, 0) = tempVec[9];
	T.at<double>(1, 0) = tempVec[10];
	T.at<double>(2, 0) = tempVec[11];
	vector<Point3d> vecPoints = vec2PointVec(R * XR);
	for (auto& point : vecPoints) {
		point.x += T.at<double>(0, 0);
		point.y += T.at<double>(1, 0);
		point.z += T.at<double>(2, 0);
	}
	return vecPoints;
}
void Log01(vector<Point3d> vecPoints) {
	string log;
	for (auto& point : vecPoints) {
		log += to_string(point.x) + ',';
		log += to_string(point.y) + ',';
		log += to_string(point.z) + '\n';
	}
	ofstream log01;
	log01.open("Log/Log01.txt");
	log01 << log;
	log01.close();
	return;
}
void Log02(vector<Point3d> vecPoints) {
	string log = ("{\"From\":\"Camera\",\"MsgType\":1,\"Counts\":");
	log = log + to_string(vecPoints.size());
	log = log + ",\"Pos\":[";
	for (auto& point : vecPoints) {
		log += to_string(point.x) + ',';
		log += to_string(point.y) + ',';
		log += to_string(point.z) + ',';
	}
	log.pop_back();
	log += "]}";
	ofstream log02;
	log02.open("Log/Log02.txt");
	log02 << log;
	log02.close();
	return;
}
int main(int argc, char* argv[]) {
	
	string addrImageLeft;
	string addrImageRight;
	string initailAllMatAddr;
	string RandTMatAddrC2B;
	string RandTMatAddrB2B;
/*	getConfig("PicConfig.csv", addrImageLeft, addrImageRight, initailAllMatAddr, RandTMatAddrC2B, RandTMatAddrB2B);
	if (argc == 3) {
		addrImageLeft = argv[1];
		addrImageRight = argv[2];
	}

	initailAllMat(initailAllMatAddr);


	if (addrImageLeft.empty() || addrImageRight.empty())
		return 0;
*/
	Mat ImageLeft = imread("F:/程序/Halo/Halo/Pic/Left.bmp", CV_8U);
	Mat ImageRight = imread("F:/程序/Halo/Halo/Pic/Left.bmp", CV_8U);
	auto pointsImageLeft = getCenterPoints(ImageLeft, "Left");
	auto pointsImageRight = getCenterPoints(ImageRight, "Right");
	match(pointsImageLeft, pointsImageRight, matFundamental);
	Mat finalshow = exhibit(ImageLeft, ImageRight, pointsImageLeft, pointsImageRight);//返回图片，将其显示在主界面上
	imwrite("Pic/LandR.bmp", finalshow);
	if (pointsImageLeft.empty() || pointsImageRight.empty()) {
		Log01(vector<Point3d>(0));
		Log02(vector<Point3d>(0));
		return 0;
	}
	Mat XR = rebuild(pointsImageLeft, pointsImageRight);
	vector<Point3d> vecPoints = file2RandTMat(RandTMatAddrC2B, XR);
	XR = PointVec2Mat(vecPoints);
	vecPoints = file2RandTMat(RandTMatAddrB2B, XR);
	Log01(vecPoints);
	Log02(vecPoints);
	return 0;
}
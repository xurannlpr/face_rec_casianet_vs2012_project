#include "face_align.h"
#include "stdafx.h"
#include "intraface/FaceAlignment.h"
#include "intraface/XXDescriptor.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/video/tracking.hpp"
int face_align(cv::Mat face_img,cv::Mat* aligned_img, cv::Rect face_box)
{
	vector<Point2f> face_landmark;
	vector<Point2f> dst_landmark;
	bool bbox = true;
	cv::Mat X;
	
	INTRAFACE::FaceAlignment* fa;
	char* detectionModel = "models/DetectionModel-v1.5.bin";
	char* trackingModel = "models/TrackingModel-v1.10.bin";
	INTRAFACE::XXDescriptor xxd(4);
	int * pResults = NULL; 
	int num_faces=0;

	fa = new INTRAFACE::FaceAlignment(detectionModel, trackingModel, &xxd);
	if (!fa->Initialized()) {
		cerr << "FaceAlignment cannot be initialized." << endl;
		return 0;
	}
	cout<<"here"<<endl;
	
	float score=0;
	float notFace = 0.5;
	vector<Point2f> landmark_point;
	if(fa->Detect(face_img,face_box,X,score) == INTRAFACE::IF_OK)
	{
		if (score>notFace)
		{
			for (int i=0;i<X.cols;i++)
			{
				landmark_point.push_back(Point2f(X.at<float>(0,i),X.at<float>(1,i)));
			}
		}
	}
	Point2f left_eye(0,0);
	Point2f right_eye(0,0);
	Point2f mouth(0,0);

	for (int i=0;i<6;i++)
	{
		left_eye.x += landmark_point[i+19].x;
		left_eye.y += landmark_point[i+19].y;
		right_eye.x += landmark_point[i+25].x;
		right_eye.y += landmark_point[i+25].y;
		mouth.x += landmark_point[i+43].x;
		mouth.y += landmark_point[i+43].y;
	}
	left_eye.x=left_eye.x/6.0;
	left_eye.y=left_eye.y/6.0;
	right_eye.x=right_eye.x/6.0;
	right_eye.y=right_eye.y/6.0;
	mouth.x=mouth.x/6.0;
	mouth.y=mouth.y/6.0;

	face_landmark.push_back(left_eye);
	face_landmark.push_back(right_eye);
	face_landmark.push_back(mouth);
	//需要对齐到的人眼与嘴巴中心坐标点;
	dst_landmark.push_back(Point2f(40.0,40.0));
	dst_landmark.push_back(Point2f(90.0,40.0));
	dst_landmark.push_back(Point2f(64.0,88.0));


	cv::Mat trans_mar = estimateRigidTransform(face_landmark, dst_landmark,false);
	if (trans_mar.cols == 0 ||trans_mar.rows == 0)
	{
		return 0;
	}
	cv::Mat transformed_img_mat(128,128,CV_8UC1);
	cv::warpAffine(face_img,transformed_img_mat,trans_mar,cvSize(128,128),INTER_LINEAR,BORDER_CONSTANT,125);

	for (int i=0;i<aligned_img->rows;i++)
	{
		uchar* pdata = aligned_img->ptr<uchar>(i);
		for (int j=0;j<aligned_img->cols;j++)
		{
			//aligned_img.at<uchar>(i,j) = transformed_img_mat.at<uchar>(i,j);
			pdata[j] = transformed_img_mat.at<uchar>(i,j);
		}
	}

	trans_mar.~Mat();
	transformed_img_mat.~Mat();
	X.~Mat();
	return 1;
}
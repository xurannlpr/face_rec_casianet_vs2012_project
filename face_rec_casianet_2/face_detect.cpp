#include "face_detect.h"
#include "stdafx.h"
#include "facedetect-dll.h"
#pragma comment(lib,"C:/Users/xuran/Documents/Visual Studio 2012/Projects/face_rec_casianet_cnnlib/face_rec_casianet_cnnlib/libfacedetect.lib")
cv::Rect face_detect(Mat& input_img)
{
	Mat gray_img(input_img.rows,input_img.cols,CV_8UC1);
	int num_faces;
	int * pResults = NULL;
	cv::Rect face_box;
	face_box.x=face_box.y=face_box.height=face_box.width=0;
	if(input_img.channels()>1)
	{
		cvCvtColor(&input_img,&gray_img,CV_BGR2GRAY);
	}
	else
	{
		gray_img = input_img;
	}
	pResults = facedetect_multiview((unsigned char*)(gray_img.ptr(0)), gray_img.cols, gray_img.rows, gray_img.step, 1.2f, 4, 24);
	num_faces = (pResults?*pResults:0);
	if(num_faces == 0)
	{
		return face_box;
	}
	else
	{
		vector<int> neighbors(num_faces);
		vector<int> xs(num_faces);
		vector<int> ys(num_faces);
		vector<int> ws(num_faces);
		vector<int> hs(num_faces);
		for (int i=0;i<num_faces;i++)
		{
			short * p = ((short*)(pResults+1))+6*i;
			xs[i] = p[0];
			ys[i] = p[1];
			ws[i]= p[2];
			hs[i] = p[3];
			neighbors[i]=p[4];
		}
		vector<int>::const_iterator it_max=max_element(neighbors.begin(),neighbors.end());
		int ix_max=it_max-neighbors.begin();
		face_box.x = xs[ix_max];
		face_box.y = ys[ix_max];
		face_box.height = hs[ix_max];
		face_box.width = ws[ix_max];
	}
	return face_box;
}
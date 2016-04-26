#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv2/opencv.hpp> 

#include "cnn.h"
#include "cnnFace.h"

using namespace std;
using namespace cv;

int cnnFace::cnnFaceInit() {

	int ret;  
	ret = _cnnFaceNet.LoadFromFile(_modelPath);
	if (ret != 0) {
		cout << "[Error] Loading model is error!\n";
	}
	return ret;
}

int cnnFace::getFeature(Mat &faceImg, float* feat) {
	Blob * blob;
	int w = faceImg.cols;
	int h = faceImg.rows;
	int c = faceImg.channels();
	int cnt = w * h * c;

	float* data = (float *)malloc(cnt * sizeof(float));
	//for (int i = 0; i < w;i++)
	//{
	//	for (int j = 0; j < h;j++)
	//	{
	//		data[i*h + j] = static_cast<float>(faceImg.at<uchar>(j, i))*0.00390625;
	//	}
	//	
	//}
	
	for (int i = 0; i < cnt; i++) {
		data[i] = static_cast<float>(faceImg.data[i])*0.00390625;
	}

	int ret = _cnnFaceNet.TakeInput(data, h, w, c);

	if ( ret != 0) {
		cout << "[Error "<< ret <<" ] CNN input error for the image!\n";
		return -1;
	}
	_cnnFaceNet.Forward();
	blob = _cnnFaceNet.get_blob(_layerIdx);

	for (int i =0; i < 320; i++) {
		float ave = 0;
		for (int j = 0; j < 8 * 8;j++)
		{
			ave += blob->data[i*64 + j];
		}
		ave = ave / 64;////手动pooling，运行不用网络的时候需要根据提取特征的维度对此处进行修改;
		
		feat[i] = ave;
	}

	free(data);
	data = NULL;
	return 0;
}


int cnnFace::getFeature(float *faceImg, float* feat, int w, int h, int c) {
	Blob * blob;

	//float* data = (float *)malloc(cnt * sizeof(float));
	/*for (int i = 0; i < cnt; i++) {
		data[i] = static_cast<float>(faceImg.data[i]) / 255.0;
	}*/

	int ret = _cnnFaceNet.TakeInput(faceImg, h, w, c);

	if ( ret != 0) {
		cout << "[Error "<< ret <<" ] CNN input error for the image!\n";
		return -1;
	}
	_cnnFaceNet.Forward();
	blob = _cnnFaceNet.get_blob(_layerIdx);

	for (int i =0; i < _len; i++) {
		feat[i] = blob->data[i];
	}
	return 0;
}

float cnnFace::getScore(float* feat1, float* feat2) {
	float dotProduct = 0.0;
	float norm1 = 0.0;
	float norm2 = 0.0;

	for (int i = 0; i < _len; i++) {
		dotProduct += feat1[i] * feat2[i];
		norm1 += feat1[i] * feat1[i];
		norm2 += feat2[i] * feat2[i];
	}
	return dotProduct / sqrt(norm1) / sqrt(norm2);
}

/*int cnnFace::faceVerification(Mat &faceData1, Mat &faceData2) {

	Blob * blob;
	// 1st image processing
	int w1 = faceData1.rows;
	int h1 = faceData1.cols;
	int c1 = faceData1.channels();
	int cnt1 = w1 * h1 * c1;
	float* data1 = (float *)malloc(cnt1 * sizeof(float));

	for (int i = 0; i < cnt1; i++) {
		data1[i] = static_cast<float>(faceData1.data[i]) / 255.0;
	}

	if ( _cnnFaceNet.TakeInput(data1, h1, w1, c1) != 0) {
		cout << "[Error] Cnn input error for the first image!\n";
		return -1;
	}
	_cnnFaceNet.Forward();
	blob = _cnnFaceNet.get_blob(_layerIdx);

	int len1 = blob->count;
	float norm1 = 0.0;
	float* feat1 = (float *)malloc(len1 * sizeof(float));
	for (int i = 0; i < len1; i++) {
		feat1[i] = blob->data[i];
		norm1 += feat1[i] * feat1[i];
	}

	// 2nd image processing
	int w2 = faceData2.rows;
	int h2 = faceData2.cols;
	int c2 = faceData2.channels();
	int cnt2 = w2 * h2 * c2;
	float* data2 = (float *)malloc(cnt2 * sizeof(float));

	for (int i = 0; i < cnt2; i++) {
		data2[i] = static_cast<float>(faceData2.data[i]) / 255.0;
	}

	if ( _cnnFaceNet.TakeInput(data2, h2, w2, c2) != 0) {
		cout << "[Error] Cnn input error for the second image!\n";
		return -1;
	}
	_cnnFaceNet.Forward();
	blob = _cnnFaceNet.get_blob(_layerIdx);

	int len2 = blob->count;
	float norm2 = 0.0;
	float* feat2 = (float *)malloc(len2 * sizeof(float));
	for (int i = 0; i < len2; i++) {
		feat2[i] = blob->data[i];
		norm2 += feat2[i] * feat2[i];
	}

	// verification
	if (len1 != len2) {
		cout << "[Error] The dimensions of two features are not matching!\n";
		return -2;
	}

	// compute cosine similarity
	float dotProduct = 0.0;
	for (int i = 0; i < len1; i++) {
		dotProduct += feat1[i] * feat2[i];
	}
	_score = dotProduct / sqrt(norm1 * norm2);

	return 0;
}*/
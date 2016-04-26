#ifndef _CNN_FACE_H
#define _CNN_FACE_H

#define _DLL_COMPILE

#ifdef _DLL_COMPILE
#define testAPI extern "C" _declspec(dllexport)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp> 

#include "cnn.h"

using namespace std;
using namespace cv;

class cnnFace{
public:
	cnnFace(const char* modelPath, const int layerIdx, const int len):
		_modelPath(modelPath),
	    _layerIdx(layerIdx), 
	    _len(len){};

	~cnnFace() {
		_cnnFaceNet.~Net();
	};
	
	int cnnFaceInit();
	int getFeature(Mat &faceImg, float* feat);
	int getFeature(float *faceImg, float* feat, int w, int h, int c);
	float getScore(float* feat1, float* feat2);
	//float faceVerification(Mat &faceData1, Mat &faceData2, char* modelPath);
	
private:
	const char* _modelPath;
	const int _layerIdx;
	
	int _len;
	Net _cnnFaceNet;
};
#endif
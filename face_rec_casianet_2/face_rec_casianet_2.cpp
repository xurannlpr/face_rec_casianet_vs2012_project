// face_rec_casianet_2.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <ctime>
#include "cnnFace.h"
#include "face_align.h"
#include "face_detect.h"


int _tmain(int argc, _TCHAR* argv[])
{
	IplImage* input_img = cvLoadImage("Carol_Niedermayer_0001.jpg",0);
	Mat input_mat=cvarrToMat(input_img,true);
	if (input_mat.cols == 0)
	{
		cout<<"can not load image"<<endl;
		return 1;
	}
	Rect face_box = face_detect(input_mat);
	cout<<"here"<<endl;
	Mat aligned_face(128,128,CV_8UC1);
	int flag = face_align(input_mat,&aligned_face,face_box);
	if(flag == 0)
	{
		cout<<"can not align the face image"<<endl;
		return 1;
	}
	imwrite("aligned_face.bmp",aligned_face);
	char* modelPath = "casia_144_ve_id_model_iter_1210000.bin";
	const int layer_index =14;
	const int len = 320*8*8;
	float* feature = (float*)malloc(320*sizeof(float));
	clock_t start, end;
	double time;
	start = clock();
	cnnFace cnn(modelPath,layer_index,len);
	if (cnn.cnnFaceInit() != 0) {
		return 1;
	}
	cnn.getFeature(aligned_face, feature);
	//cnn.getFeature(imgFace2, feat2);

	//float score = cnn.getScore(feat1, feat2);

	end = clock();
	time = (double)(end -  start) / CLOCKS_PER_SEC;

	cout << "The score is " << "\nTime is " << time << endl;
	cnn.~cnnFace();
	free(feature);
	return 0;
}


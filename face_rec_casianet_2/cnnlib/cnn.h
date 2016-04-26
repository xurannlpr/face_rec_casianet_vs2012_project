#ifndef CNN_H
#define CNN_H

#include <cstdio>
#include <assert.h>
#include <string.h>


#include <vector>

using std::vector;

typedef float Dtype;

//typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;


enum LAYER_TYPE { LAYER_TYPE_CONV = 4, 
				  LAYER_TYPE_POOLING = 17,
				  LAYER_TYPE_INNER_PRODUCT = 14,
				  LAYER_TYPE_RELU = 18,
				  LAYER_TYPE_SOFTMAX = 20,
				  LAYER_TYPE_SPLIT = 22,
				  LAYER_TYPE_CONCAT = 3,
				  LAYER_TYPE_FLATTEN = 8,
				  LAYER_TYPE_SLICE = 33,
				  LAYER_TYPE_ELTWISE = 25 };



class BinStream {
public:
	explicit BinStream(): data(NULL), offset(0), len(0) {}
	~BinStream() {
		if (data) {
			delete[] data;
		}
	}
	int Read(void* dst, int read_size);
	int Load(const char* fn);
	unsigned char* data;
	int offset;
	int len;
};

class Blob {
public:
	Blob() : data(NULL), num(0), channels(0), height(0), width(0), count(0), own_data_(true) {
	}
	Blob(int n, int c, int h, int w) : num(n), channels(c), height(h), width(w), count(0) {
		count  = num * channels * height * width;
		// check!!
		data = new Dtype[count];
		own_data_ = true;
	}

	~Blob() {
		if ((data != NULL) && (own_data_)) {
			delete[] data;
		}
	}

	int Reshape(int n, int c, int h, int w) {

		num = n;
		channels = c;
		height = h;
		width = w;
		int new_count  = num * channels * height * width;
		
		if (own_data_) {
			if (data != NULL) {
				delete[] data;
			}
			//check!!
			data = new Dtype[new_count];
			own_data_ = true;
		} else if (new_count > count) {
			return -1;
		}
		count = new_count;

		return 0;
	}

	int ReshapeLike(const Blob& bl) {
		Reshape(bl.num, bl.channels, bl.height, bl.width);
		return 0;
	}

	int ShareData(const Blob& other) {
		if (count != other.count) {
			return -1;
		}
		if (own_data_) {
			delete[] data;
		}
		data = other.data;
		own_data_ = false;
		return 0;
	}

	inline int offset(const int n, const int c = 0, const int h = 0,
		const int w = 0) const {
			assert( n >= 0 );
			assert( n <= num);
			assert( channels >= 0);
			assert( c <= channels );
			assert( height > 0 );
			assert( h <= height);
			assert( width >0 );
			assert( w <= width);

			return ((n * channels + c) * height + h) * width + w;
	}

	int ParseFromBin(BinStream& bs) {
		
		bs.Read(&num, sizeof(int));
		bs.Read(&channels, sizeof(int));
		bs.Read(&height, sizeof(int));
		bs.Read(&width, sizeof(int));

		count  = num * channels * height * width; 
		Reshape(num, channels, height, width);

		int rtn = bs.Read(data, sizeof(Dtype) * count);

		if (rtn != 0) {
			return -1;
		}

		return 0;
	} 


	int num, channels, height, width;
	int count;
	Dtype* data;
	bool own_data_;
};

class Layer {
public:
	virtual ~Layer() { }
	virtual void Forward(vector<Blob*>& blobs, const vector<int>& bottom, const vector<int>& top) = 0;
	virtual int ParseFromBin(BinStream& bs) = 0;
	virtual int SetUp(vector<Blob*>& blobs, const vector<int>& bottom, const vector<int>& top) = 0;
private:

};

class Net {
public:
	~Net();
	explicit Net() { }

	int LoadFromFile(const char* fn);
	int ParseFromBin(BinStream& bs);
	void ForwardFromTo(int start, int end);
	
	void Forward() {
		ForwardFromTo(0, layers.size() - 1);
	}

	int TakeInput(const float* img_input, int height, int width, int channel);
	
	Blob* get_blob(int index) {
		if ((index >= 0) && (index < blobs.size())) {
			return blobs[index];
		} else {
			return 0;
		}
	}

	Blob* get_output_blob() { 
		if (blobs.size() > 0) {
			return blobs[blobs.size() - 1]; 
		} else {
			return 0;
		}
	}

	int get_blob_size() { return blobs.size();}

private:
	Layer* get_layer(int layer_type);

	int input_channels;
	int input_height;
	int input_width;
	int num_blobs;
	int num_layers;
	vector<Blob*> blobs;
	vector<Layer*> layers;
	vector<vector<int> > bottom_id_vecs;
	vector<vector<int> > top_id_vecs;

};

class ConvLayer: public Layer {
public:
	explicit ConvLayer() { }
	virtual void Forward(vector<Blob*>& blobs, const vector<int>& bottom, const vector<int>& top);
	virtual int ParseFromBin(BinStream& bs);
	virtual int SetUp(vector<Blob*>& blobs, const vector<int>& bottom, const vector<int>& top);

public:
	Blob weight, bias;
	int count_;

protected:
	int kernel_size_;
	int stride_;
	int num_;
	int channels_;
	int pad_;
	int height_;
	int width_;
	int num_output_;
	int group_;
	//Blob<Dtype> col_buffer_;
	//shared_ptr<SyncedMemory> bias_multiplier_;
	//bool bias_term_;
	int M_;
	int K_;
	int N_;
	Blob col_buffer;
	Blob bias_multiplier_;
};

class PoolingLayer: public Layer
{
public:
	explicit PoolingLayer() { }
	virtual void Forward(vector<Blob*>& blobs, const vector<int>& bottom, const vector<int>& top);
	virtual int ParseFromBin(BinStream& bs);
	virtual int SetUp(vector<Blob*>& blobs, const vector<int>& bottom, const vector<int>& top);

protected:
	int kernel_size_;
	int stride_;
	int pad_;
	int channels_;
	int height_;
	int width_;
	int pooled_height_;
	int pooled_width_;
};


class InnerProductLayer: public Layer {
public:
	explicit InnerProductLayer() { }
	virtual void Forward(vector<Blob*>& blobs, const vector<int>& bottom, const vector<int>& top);
	virtual int ParseFromBin(BinStream& bs);
	virtual int SetUp(vector<Blob*>& blobs, const vector<int>& bottom, const vector<int>& top);

	Blob weight, bias;

protected:
	int num_output_;
	int M_;
	int K_;
	int N_;
	//bool bias_term_;
	//shared_ptr<SyncedMemory> bias_multiplier_;
	Blob bias_multiplier_;
};

class RELULayer: public Layer {
public:
	explicit RELULayer() { }
	virtual void Forward(vector<Blob*>& blobs, const vector<int>& bottom, const vector<int>& top);
	virtual int ParseFromBin(BinStream& bs);
	virtual int SetUp(vector<Blob*>& blobs, const vector<int>& bottom, const vector<int>& top);

protected:

};

class SoftmaxLayer: public Layer
{
public:
	explicit SoftmaxLayer() { }
	virtual void Forward(vector<Blob*>& blobs, const vector<int>& bottom, const vector<int>& top);
	virtual int ParseFromBin(BinStream& bs);
	virtual int SetUp(vector<Blob*>& blobs, const vector<int>& bottom, const vector<int>& top);

protected:
	Blob sum_multiplier_;
	Blob scale_;

};

class FlattenLayer: public Layer {
public:
	explicit FlattenLayer() { }
	virtual void Forward(vector<Blob*>& blobs, const vector<int>& bottom, const vector<int>& top);
	virtual int ParseFromBin(BinStream& bs);
	virtual int SetUp(vector<Blob*>& blobs, const vector<int>& bottom, const vector<int>& top);

};

class SplitLayer: public Layer {
public:
	explicit SplitLayer() { }
	virtual void Forward(vector<Blob*>& blobs, const vector<int>& bottom, const vector<int>& top);
	virtual int ParseFromBin(BinStream& bs);
	virtual int SetUp(vector<Blob*>& blobs, const vector<int>& bottom, const vector<int>& top);

};

class ConcatLayer: public Layer {
public:
public:
	explicit ConcatLayer() { }
	virtual void Forward(vector<Blob*>& blobs, const vector<int>& bottom, const vector<int>& top);
	virtual int ParseFromBin(BinStream& bs);
	virtual int SetUp(vector<Blob*>& blobs, const vector<int>& bottom, const vector<int>& top);

};

class SliceLayer: public Layer {
public:
	explicit SliceLayer() { }
	virtual void Forward(vector<Blob*>& blobs, const vector<int>& bottom, const vector<int>& top);
	virtual int ParseFromBin(BinStream& bs);
	virtual int SetUp(vector<Blob*>& blobs, const vector<int>& bottom, const vector<int>& top);
protected:
	int slice_dim_;

	int count_;
	int num_;
	int channels_;
	int height_;
	int width_;
};

class EltWiseLayer: public Layer {
public:
	explicit EltWiseLayer() { }
	virtual void Forward(vector<Blob*>& blobs, const vector<int>& bottom, const vector<int>& top);
	virtual int ParseFromBin(BinStream& bs);
	virtual int SetUp(vector<Blob*>& blobs, const vector<int>& bottom, const vector<int>& top);
protected:
	int operation_;
};


#endif
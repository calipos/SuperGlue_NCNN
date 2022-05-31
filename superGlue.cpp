#include <fstream>
#include <map>
#include <string>
#include <tuple>
#include <iostream>
#include <memory>
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
#include "ncnn/c_api.h"
#include "ncnn/mat.h" 
#include "ncnn/layer.h"
#include "ncnn/layer_type.h"
#include "ncnn/paramdict.h"
#include "ncnn/net.h"
#define NUM_THREAD   (2); 
#define descriptor_dim 256
#define head (4)
#define dim (64)
#define dim_sqrt (8)
#define dim_sqrt_inv (0.125)
#define descriptor_dim_sqrt_inv (0.0625) 
std::vector<ncnn::Mat>superglueAttentionWeights; 
std::vector<ncnn::Layer*>attentionNets;
 
ncnn::Mat from_float(const cv::Mat& data)
{
	CHECK(data.channels() == 1);
	ncnn::Mat ret;
	ret.create(data.rows, data.cols);
	for (int r = 0; r < data.rows; r++)
	{
		for (int c = 0; c < data.cols; c++)
		{
			ret.row(c)[r] = data.ptr<float>(r)[c];
		}
	}
	return ret;
}
ncnn::Mat from_float2(const cv::Mat& data)
{
	CHECK(data.channels() == 1);
	ncnn::Mat ret;
	ret.create(data.cols,data.rows,1);
	for (int r = 0; r < data.rows; r++)
	{
		for (int c = 0; c < data.cols; c++)
		{
			ret.row(r)[c] = data.ptr<float>(r)[c];
		}
	}
	return ret;
}
cv::Mat normalize_keypoints(const cv::Mat& keypoint, const int& imgHeight, const int& imgWidth)
{
	cv::Mat normaledKp;
	keypoint.copyTo(normaledKp);
	normaledKp.convertTo(normaledKp, CV_32FC1);
	float centerX = imgWidth * 0.5;
	float centerY = imgHeight * 0.5;
	float scaling = 0.7 * (std::max)(imgWidth, imgHeight);
	for (int i = 0; i < normaledKp.rows; i++)
	{
		normaledKp.ptr<float>(i)[0] = (normaledKp.ptr<float>(i)[0] - centerX) / scaling;
		normaledKp.ptr<float>(i)[1] = (normaledKp.ptr<float>(i)[1] - centerY) / scaling;
	}
	return normaledKp;
}
void* getSuperGlueNet(const std::string& weightPath)
{
	ncnn::Option opt;
	opt.num_threads = NUM_THREAD;
	attentionNets.resize(7+18*6+1+1+5);
	attentionNets[0] = ncnn::create_layer("Gemm");
	attentionNets[1] = ncnn::create_layer("Permute");
	attentionNets[2] = ncnn::create_layer("Softmax"); 
	attentionNets[3] = ncnn::create_layer("Gemm");
	attentionNets[4] = ncnn::create_layer("Permute");
	attentionNets[5] = ncnn::create_layer("Concat");
	attentionNets[6] = ncnn::create_layer("Eltwise"); 
	for (int i = 0; i < 18*6+1; i++)
	{
		attentionNets[7+i] = ncnn::create_layer("Convolution1D");
	}
	attentionNets[7 + 18 * 6 + 1] = ncnn::create_layer("Gemm");
	attentionNets[7 + 18 * 6 + 2] = ncnn::create_layer("Convolution1D");
	attentionNets[7 + 18 * 6 + 3] = ncnn::create_layer("Convolution1D");
	attentionNets[7 + 18 * 6 + 4] = ncnn::create_layer("Convolution1D");
	attentionNets[7 + 18 * 6 + 5] = ncnn::create_layer("Convolution1D");
	attentionNets[7 + 18 * 6 + 6] = ncnn::create_layer("Convolution1D");
	std::vector<float>values;	 
	{
		std::fstream fin(weightPath, std::ios::in | std::ios::binary);
		int num = -1;
		fin.read((char*)&num, sizeof(int));
		CHECK(num > 1);
		values.reserve(num);
		float temp;
		for (int i = 0; i < num; i++)
		{
			fin.read((char*)&temp, sizeof(float));
			values.emplace_back(temp);
		}
	} 
	superglueAttentionWeights.resize(2 * 18 * 6 + 2 + 10);
	int weightPos = 0;
	for (int m = 0; m < 18; m++)
	{ 
		std::vector<int>eachLayerOutput = { descriptor_dim ,descriptor_dim ,descriptor_dim ,descriptor_dim ,descriptor_dim * 2,descriptor_dim };
		std::vector<int>eachLayerInput = { descriptor_dim ,descriptor_dim ,descriptor_dim ,descriptor_dim ,descriptor_dim * 2,descriptor_dim * 2 };
		std::vector<int>eachLayerReluFlag = { 0,0,0,0,1,0};
		for (int i = 0; i < 6; i++)
		{
			int layerId = 6 * m + i + 7;
			int weightId = 2 * (6 * m + i);
			int outputNum = eachLayerOutput[i];
			int inputNum = eachLayerInput[i];
			ncnn::ParamDict pd;
			pd.set(0, outputNum);
			pd.set(1, 1);
			pd.set(5, 1);
			pd.set(6, outputNum * inputNum);
			pd.set(9, eachLayerReluFlag[i]);
			attentionNets[layerId]->load_param(pd);
			superglueAttentionWeights[weightId].create(outputNum * inputNum);
			superglueAttentionWeights[weightId + 1].create(outputNum);
			memcpy(superglueAttentionWeights[weightId].data, &(values[weightPos]), sizeof(float) * outputNum * inputNum);
			weightPos += outputNum * inputNum;
			memcpy(superglueAttentionWeights[weightId + 1].data, &(values[weightPos]), sizeof(float) * outputNum);
			weightPos += outputNum;
			attentionNets[layerId]->load_model(ncnn::ModelBinFromMatArray(&(superglueAttentionWeights[weightId])));
			attentionNets[layerId]->create_pipeline(opt);
		}		
	}
	{ 
		int layerId = 7 + 18 * 6;
		int weightId = 2 * 18 * 6;
		int outputNum = descriptor_dim;
		int inputNum = descriptor_dim;
		ncnn::ParamDict pd;
		pd.set(0, outputNum);
		pd.set(1, 1);
		pd.set(5, 1);
		pd.set(6, outputNum * inputNum);
		pd.set(9, 0);
		attentionNets[layerId]->load_param(pd);
		superglueAttentionWeights[weightId].create(outputNum * inputNum);
		superglueAttentionWeights[weightId + 1].create(outputNum);
		memcpy(superglueAttentionWeights[weightId].data, &(values[weightPos]), sizeof(float) * outputNum * inputNum);
		weightPos += outputNum * inputNum;
		memcpy(superglueAttentionWeights[weightId + 1].data, &(values[weightPos]), sizeof(float) * outputNum);
		weightPos += outputNum;
		attentionNets[layerId]->load_model(ncnn::ModelBinFromMatArray(&(superglueAttentionWeights[weightId])));
		attentionNets[layerId]->create_pipeline(opt);
	}
	std::vector<int>outputs = { 32, 64, 128, 256, 256 }; 
	for (int i = 0; i < 5; i++)
	{
		int layerId = 7 + 18 * 6 + 2 + i;
		int weightId = 2 * 18 * 6 + 2 + 2 * i;
		int outputNum = outputs[i];
		int inputNum = 3;
		if (i != 0)
		{
			inputNum = outputs[i - 1];
		}
		ncnn::ParamDict pd;
		pd.set(0, outputNum);
		pd.set(1, 1);
		pd.set(2, 0);
		pd.set(5, 1);
		pd.set(6, outputNum * inputNum);
		if (i != 4)pd.set(9, 1);
		else pd.set(9, 0);
		attentionNets[layerId]->load_param(pd);
		superglueAttentionWeights[weightId].create(outputNum * inputNum);
		superglueAttentionWeights[weightId + 1].create(outputNum);
		memcpy(superglueAttentionWeights[weightId].data, &(values[weightPos]), sizeof(float) * outputNum * inputNum);
		weightPos += outputNum * inputNum;
		memcpy(superglueAttentionWeights[weightId + 1].data, &(values[weightPos]), sizeof(float) * outputNum);
		weightPos += outputNum;
		attentionNets[layerId]->load_model(ncnn::ModelBinFromMatArray(&(superglueAttentionWeights[weightId])));
		attentionNets[layerId]->create_pipeline(opt);
	}

	{
		ncnn::ParamDict pd;
		pd.set(0, (float)dim_sqrt_inv);
		pd.set(2, 0);
		pd.set(3, 1);
		attentionNets[0]->load_param(pd);
		attentionNets[0]->create_pipeline(opt);
	} 
	{
		ncnn::ParamDict pd; 
		pd.set(0, 3); 
		attentionNets[1]->load_param(pd);
		attentionNets[1]->create_pipeline(opt);
	}
	{
		ncnn::ParamDict pd;
		pd.set(0, 2);
		pd.set(1, 1);
		attentionNets[2]->load_param(pd);
		attentionNets[2]->create_pipeline(opt);
	}	
	{
		ncnn::ParamDict pd;
		pd.set(2, 0);
		pd.set(3, 0);
		attentionNets[3]->load_param(pd);
		attentionNets[3]->create_pipeline(opt);
	}
	{
		ncnn::ParamDict pd;
		pd.set(0, 4);
		attentionNets[4]->load_param(pd);
		attentionNets[4]->create_pipeline(opt);
	}
	{
		ncnn::ParamDict pd;
		pd.set(0, 0);
		attentionNets[5]->load_param(pd);
		attentionNets[5]->create_pipeline(opt);
	}
	{
		ncnn::ParamDict pd;
		pd.set(0, 1);
		pd.set(1, ncnn::Mat());
		attentionNets[6]->load_param(pd);
		attentionNets[6]->create_pipeline(opt);
	}
	{
		ncnn::ParamDict pd;
		pd.set(0, (float)descriptor_dim_sqrt_inv);
		pd.set(2, 1);
		pd.set(3, 0);
		attentionNets[7 + 18 * 6 + 1]->load_param(pd);
		attentionNets[7 + 18 * 6 + 1]->create_pipeline(opt);
	}


	return (void*)attentionNets[0];
}

ncnn::Mat attentionProc(void* nets, const int&iter,const ncnn::Mat& inputQuery, const ncnn::Mat& inputKey, const ncnn::Mat& inputValue)
{
	ncnn::Option opt;
	opt.num_threads = NUM_THREAD;
	CHECK((void*)attentionNets[0] == nets);
	ncnn::Mat query0, key0, value0;
	attentionNets[6*iter + 7]->forward(inputQuery, query0, opt);
	attentionNets[6*iter + 8]->forward(inputKey, key0, opt);
	attentionNets[6*iter + 9]->forward(inputValue, value0, opt);
	
	ncnn::Mat attPermute;
	{ 
		ncnn::Layer* reshapeNet0 = ncnn::create_layer("Reshape");
		ncnn::Layer* reshapeNet1 = ncnn::create_layer("Reshape");
		ncnn::Layer* reshapeNet2 = ncnn::create_layer("Reshape");
		{
			ncnn::ParamDict pd0;
			pd0.set(2, 64);
			pd0.set(1, 4);
			pd0.set(0, query0.w);
			ncnn::ParamDict pd1;
			pd1.set(2, 64);
			pd1.set(1, 4);
			pd1.set(0, key0.w);
			ncnn::ParamDict pd2;
			pd2.set(2, 64);
			pd2.set(1, 4);
			pd2.set(0, value0.w);
			reshapeNet0->load_param(pd0);
			reshapeNet0->create_pipeline(opt);
			reshapeNet1->load_param(pd1);
			reshapeNet1->create_pipeline(opt);
			reshapeNet2->load_param(pd2);
			reshapeNet2->create_pipeline(opt);
		}
		ncnn::Mat queryReshape, keyReshape, valueReshape; 
		ncnn::Mat queryPermute, keyPermute, valuePermute;
		opt.use_packing_layout = false;
		reshapeNet0->forward(query0, queryReshape, opt);
		reshapeNet1->forward(key0, keyReshape, opt);
		reshapeNet2->forward(value0, valueReshape, opt);
		attentionNets[1]->forward(queryReshape, queryPermute, opt);
		attentionNets[1]->forward(keyReshape, keyPermute, opt);
		attentionNets[1]->forward(valueReshape, valuePermute, opt);

		ncnn::Mat qk(keyPermute.h, queryPermute.h, queryPermute.c);
		int channelTotal = queryPermute.h * keyPermute.h;
		for (size_t i = 0; i < queryPermute.c; i++)
		{
			std::vector<ncnn::Mat> out(1);
			attentionNets[0]->forward(std::vector<ncnn::Mat>{queryPermute.channel(i), keyPermute.channel(i)},
				out, opt);
			memcpy((float*)qk.channel(i).data, out[0].data, sizeof(float) * channelTotal);
		}
 
		ncnn::Layer* reshapeNet3 = ncnn::create_layer("Reshape");
		{ 
			ncnn::ParamDict pd3;
			pd3.set(2, -233);
			pd3.set(1, qk.c* qk.h);
			pd3.set(0, qk.w);
			reshapeNet3->load_param(pd3);
			reshapeNet3->create_pipeline(opt); 
		}
		ncnn::Mat qkReshape;
		reshapeNet3->forward(qk, qkReshape, opt);
		ncnn::Mat qkSoftmax;
		attentionNets[2]->forward(qk, qkSoftmax, opt);

		ncnn::Mat att(dim, qkSoftmax.h,  head);
		channelTotal = att.h * att.w;
		for (size_t i = 0; i < qkSoftmax.c; i++)
		{
			std::vector<ncnn::Mat> out(1);
			attentionNets[3]->forward(std::vector<ncnn::Mat>{  qkSoftmax.channel(i), valuePermute.channel(i)},
				out, opt);
			memcpy((float*)att.channel(i).data, out[0].data, sizeof(float) * channelTotal);
		}
		ncnn::Layer* reshapeNet4 = ncnn::create_layer("Reshape");
		{
			ncnn::ParamDict pd4;
			pd4.set(1, att.w);
			pd4.set(0, att.h);
			pd4.set(2, att.c);
			reshapeNet4->load_param(pd4);
			reshapeNet4->create_pipeline(opt);
		}
		//reshapeNet4->forward(att, att, opt);
		attentionNets[4]->forward(att, attPermute, opt);
		ncnn::Layer* reshapeNet5 = ncnn::create_layer("Reshape");
		{
			ncnn::ParamDict pd5;
			pd5.set(1, attPermute.h* attPermute.c);
			pd5.set(0, attPermute.w);
			pd5.set(2, 1); 
			reshapeNet5->load_param(pd5);
			reshapeNet5->create_pipeline(opt);
		} 
		reshapeNet5->forward(attPermute, attPermute, opt);
		reshapeNet0->destroy_pipeline(opt);
		reshapeNet1->destroy_pipeline(opt);
		reshapeNet2->destroy_pipeline(opt);
		reshapeNet3->destroy_pipeline(opt);
		reshapeNet4->destroy_pipeline(opt);
		reshapeNet5->destroy_pipeline(opt);
		delete reshapeNet0;
		delete reshapeNet1;
		delete reshapeNet2;
		delete reshapeNet3;
		delete reshapeNet4;
		delete reshapeNet5;
	}
	ncnn::Mat message;
	attentionNets[6*iter + 10]->forward(attPermute, message, opt);
	std::vector<ncnn::Mat> delta(1);
	attentionNets[5]->forward(std::vector<ncnn::Mat>{inputQuery, message}, delta, opt);
	ncnn::Mat deltaIn,deltaOut;
	deltaIn = delta[0];
	attentionNets[6*iter + 11]->forward(deltaIn, deltaOut, opt);
	deltaIn = deltaOut.clone();
	attentionNets[6*iter + 12]->forward(deltaIn, deltaOut, opt);
	return deltaOut;
}

cv::Mat superGlue(void* net, const int& imgHeight, const int& imgWidth,
	const cv::Mat& kps0, cv::Mat& descripCv0, const cv::Mat& scores0,
	const cv::Mat& kps1, cv::Mat& descripCv1, const cv::Mat& scores1)
{
	CHECK((void*)attentionNets[0] == net);
	ncnn::Option opt;
	opt.num_threads = NUM_THREAD;
    opt.use_packing_layout = false;
	cv::Mat kp0Encode, kp1Encode;
	{
		ncnn::Mat input(kps0.rows, 3);
		cv::Mat imputcv(kps0.rows, 3, CV_32FC1);
		kps0.copyTo(imputcv(cv::Rect(0, 0, 2, kps0.rows)));
		scores0.copyTo(imputcv(cv::Rect(2, 0, 1, kps0.rows)));
		imputcv = imputcv.t();
		memcpy(input.data, imputcv.data, 3 * kps0.rows * sizeof(float));
		ncnn::Mat out;
		for (int i = 0; i < 5; i++)
		{
			attentionNets[7 + 18 * 6 + 2 + i]->forward(input, out, opt);
			input = out.clone();
		}
		kp0Encode = cv::Mat::zeros(out.h, out.w, CV_32FC1);
		memcpy(kp0Encode.data, out.data, out.h * out.w * sizeof(float));
	}
	{
		ncnn::Mat input(kps1.rows, 3);
		cv::Mat imputcv(kps1.rows, 3, CV_32FC1);
		kps1.copyTo(imputcv(cv::Rect(0, 0, 2, kps1.rows)));
		scores1.copyTo(imputcv(cv::Rect(2, 0, 1, kps1.rows)));
		imputcv = imputcv.t();
		memcpy(input.data, imputcv.data, 3 * kps1.rows * sizeof(float));
		ncnn::Mat out;
		for (int i = 0; i < 5; i++)
		{
			attentionNets[7 + 18 * 6 + 2 + i]->forward(input, out, opt);
			input = out.clone();
		}
		kp1Encode = cv::Mat::zeros(out.h, out.w, CV_32FC1);
		memcpy(kp1Encode.data, out.data, out.h * out.w * sizeof(float));
	}
	descripCv0.convertTo(descripCv0, CV_32FC1);
	descripCv1.convertTo(descripCv1, CV_32FC1);
	descripCv0 = descripCv0 + kp0Encode.t();
	descripCv1 = descripCv1 + kp1Encode.t(); 

	ncnn::Mat descrip0 = from_float(descripCv0);
	ncnn::Mat descrip1 = from_float(descripCv1);
	ncnn::Mat delta0, delta1;
	std::vector<ncnn::Mat> newDescriptor0(1), newDescriptor1(1);
	for (size_t i = 0; i < 9; i++)
	{
		{
			delta0 = attentionProc(net, 2 * i, descrip0, descrip0, descrip0);
			delta1 = attentionProc(net, 2 * i, descrip1, descrip1, descrip1);
			attentionNets[6]->forward({ descrip0 ,delta0 }, newDescriptor0, opt);
			attentionNets[6]->forward({ descrip1 ,delta1 }, newDescriptor1, opt);
			descrip0 = newDescriptor0[0];
			descrip1 = newDescriptor1[0];
		}
		{
			delta0 = attentionProc(net, 2 * i + 1, descrip0, descrip1, descrip1);
			delta1 = attentionProc(net, 2 * i + 1, descrip1, descrip0, descrip0);
			attentionNets[6]->forward({ descrip0 ,delta0 }, newDescriptor0, opt);
			attentionNets[6]->forward({ descrip1 ,delta1 }, newDescriptor1, opt);
			descrip0 = newDescriptor0[0];
			descrip1 = newDescriptor1[0];
		}
	}
	ncnn::Mat finalDescrip0, finalDescrip1;
	attentionNets[7 + 18 * 6]->forward(descrip0, finalDescrip0, opt);
	attentionNets[7 + 18 * 6]->forward(descrip1, finalDescrip1, opt);

	std::vector<ncnn::Mat> score(1);
	attentionNets[7 + 18 * 6 + 1]->forward(std::vector<ncnn::Mat>{finalDescrip0, finalDescrip1},
		score, opt);
	cv::Mat scoreRet(score[0].h, score[0].w, CV_32FC1);
	for (int r = 0; r < score[0].h; r++)
	{
		for (int c = 0; c < score[0].w; c++)
		{
			scoreRet.ptr<float>(r)[c] = score[0].row(r)[c];
		}
	}
	return scoreRet;
}
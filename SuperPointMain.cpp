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
#define descriptor_dim 256
#define nms_radius (4)
#define img_h (480)
#define img_w (640)
#define net_h (60)
#define net_w (80)
#define channelTotal (4800)
#define NUM_THREAD   (2);

#include <stdio.h>
#include <stdlib.h>

std::vector<float>superPointWeightValues;
void scoreResort(const float* data, const int& n, const int& c, const int& h, const int& w,std::vector<cv::Mat>&scoreMat)
{
	int s = (h * w);
	scoreMat.resize(n);
	for (int batchId = 0; batchId < n; batchId++)
	{
		const float* theBatchdata = data + batchId * c * h * w;
		scoreMat[batchId] = cv::Mat(h * 8, w * 8, CV_32FC1);
		const float* head = theBatchdata;
		for (int row = 0; row < h ; row++)
		{
			for (int col = 0; col < w ; col++)
			{
				for (int i = 0; i < 8; i++)
				{
					for (int j = 0; j < 8; j++)
					{
						scoreMat[batchId].ptr<float>(8 * row + i)[8 * col + j] = head[8 * i + j];
					}
				}
				head += 64;
			}
		}
	}
	return;
}
cv::Mat quel(const cv::Mat& src1, const cv::Mat& src2)
{
	cv::Mat ret = cv::Mat::zeros(src1.size(), src1.type());
	for (int i = 0; i < src1.rows; i++)
	{
		for (int j = 0; j < src1.cols; j++)
		{
			if (src1.ptr<float>(i)[j] == src2.ptr<float>(i)[j])
			{
				ret.ptr<float>(i)[j] = 1.;
			}
		}
	}
	return ret;
}
cv::Mat where(const cv::Mat& condition, const cv::Mat& src1, const cv::Mat& src2)
{
	cv::Mat ret = cv::Mat::zeros(src1.size(), src1.type());
	for (int i = 0; i < src1.rows; i++)
	{
		for (int j = 0; j < src1.cols; j++)
		{
			if (condition.ptr<float>(i)[j] > 0)
			{
				ret.ptr<float>(i)[j] = src1.ptr<float>(i)[j];
			}
			else
			{
				ret.ptr<float>(i)[j] = src2.ptr<float>(i)[j];
			}
		}
	}
	return ret;
}
void scoreNMS(std::vector<cv::Mat>& scoreMats)
{
	CHECK(nms_radius >= 0);
	ncnn::Option opt;
	opt.num_threads = NUM_THREAD;
	ncnn::Layer* op = ncnn::create_layer("Pooling");
	ncnn::ParamDict pd;
	{
		pd.set(0, 0);
		pd.set(1, 2 * nms_radius + 1);
		pd.set(2, 1);
		pd.set(3, nms_radius);
		op->load_param(pd);
		op->create_pipeline(opt);
	}
	const auto& maxPool = [&op,&opt](const cv::Mat&in)->cv::Mat
	{
		CHECK(in.channels() == 1);
		ncnn::Mat input(in.cols, in.rows, 1);
		memcpy(input.data, in.data, in.rows * in.cols * sizeof(float));
		ncnn::Mat out;
		op->forward(input, out, opt);
		cv::Mat outMat = cv::Mat::zeros(out.h, out.w, CV_32FC1);
		memcpy(outMat.data, out.data, in.rows * in.cols * sizeof(float));
		return outMat;
	};
	 
	for (auto& scores : scoreMats)
	{ 
		cv::Mat max_mask = quel(scores, maxPool(scores));
		for (size_t i = 0; i < 2; i++)
		{
			cv::Mat supp_mask = maxPool(max_mask);
			cv::Mat supp_scores = where(supp_mask, cv::Mat::zeros(supp_mask.size(), supp_mask.type()), scores);
			cv::Mat new_max_mask = quel(supp_scores, maxPool(supp_scores));
			cv::Mat new_max_mask2 = new_max_mask & (~supp_mask);
			max_mask = max_mask | (new_max_mask2);
		}
		cv::Mat scores0 = where(max_mask, scores, cv::Mat::zeros(scores.size(), scores.type()));
		scores0.copyTo(scores);
	} 
	op->destroy_pipeline(opt);
	delete op;
}
std::vector< std::vector<std::tuple<float, cv::Point, cv::Mat>>>pickKeyPointsAndDescp(
	const std::vector<cv::Mat>& scoreMats,const float*descp,
	const float& keypointThreshold=0.05,const int&borderSize=4)
{
	std::vector< std::vector<std::tuple<float, cv::Point, cv::Mat>>> ret(scoreMats.size());
	CHECK(borderSize >= 0);
	int xStart = borderSize;
	int yStart = borderSize;
	int xEnd = scoreMats[0].cols - borderSize;
	int yEnd = scoreMats[0].rows - borderSize; 
	for (size_t id = 0; id < scoreMats.size(); id++)
	{
		for (int i = 0; i < scoreMats[0].rows; i++)
		{
			for (int j = 0; j < scoreMats[0].cols; j++)
			{
				if (scoreMats[id].ptr<float>(i)[j]> keypointThreshold)
				{
					cv::Point2d anchor = cv::Point2d(j, i) - cv::Point2d(8 / 2 - .5, 8 / 2 - .5);
					anchor.x /= (img_w - 8 / 2 - 0.5);
					anchor.y /= (img_h - 8 / 2 - 0.5);
					anchor.x *= (net_w - 1);
					anchor.y *= (net_h - 1);
					int x0 = static_cast<int>(anchor.x);
					int x1 = x0 == net_w - 1 ? x0 : x0 + 1;
					int y0 = static_cast<int>(anchor.y);
					int y1 = y0 == net_h - 1 ? y0 : y0 + 1;
					double alpha = (anchor.x - x0);
					double beta =  (anchor.y - y0);
					
					cv::Mat newFeature(descriptor_dim, 1, CV_64FC1);
					cv::Mat newFeature_sqr(descriptor_dim, 1, CV_64FC1);
					for (int k = 0; k < descriptor_dim; k++)
					{
						double d1 = descp[channelTotal * k + net_w * y0 + x0];
						double d2 = descp[channelTotal * k + net_w * y0 + x1];
						double d3 = descp[channelTotal * k + net_w * y1 + x0];
						double d4 = descp[channelTotal * k + net_w * y1 + x1];
						double a = (1 - alpha) * d1 + alpha * d2;
						double b = (1 - alpha) * d3 + alpha * d4;
						newFeature.ptr<double>(k)[0] = (1 - beta) * a + beta * b;
						newFeature_sqr.ptr<double>(k)[0] = newFeature.ptr<double>(k)[0]* newFeature.ptr<double>(k)[0];
					}
					cv::reduce(newFeature_sqr, newFeature_sqr,0,cv::REDUCE_SUM);
					newFeature_sqr.ptr<double>(0)[0] = sqrt(newFeature_sqr.ptr<double>(0)[0]);
					for (int k = 0; k < descriptor_dim; k++)
					{
						newFeature.ptr<double>(k)[0] /= newFeature_sqr.ptr<double>(0)[0];
					}
					ret[id].emplace_back(std::make_tuple(scoreMats[id].ptr<float>(i)[j], cv::Point(j, i), newFeature));
				}
			}
		}
	}
	return ret;
}
std::vector<std::tuple<cv::Mat, cv::Mat, cv::Mat>>  convt(const std::vector< std::vector<std::tuple<float, cv::Point, cv::Mat>>>& features)
{
	std::vector<std::tuple<cv::Mat, cv::Mat, cv::Mat>> ret(features.size());
	for (int i = 0; i < features.size(); i++)
	{
		cv::Mat scores(features[i].size(), 1, CV_32FC1);
		cv::Mat points(features[i].size(), 2, CV_32SC1);
		cv::Mat featureMat(features[i].size(), descriptor_dim, CV_64FC1);
		for (int j = 0; j < features[i].size(); j++)
		{
			scores.ptr<float>(j)[0] = std::get<0>(features[i][j]);
			points.ptr<int>(j)[0] = std::get<1>(features[i][j]).x;
			points.ptr<int>(j)[1] = std::get<1>(features[i][j]).y;
			for (int k = 0; k < descriptor_dim; k++)
			{
				featureMat.ptr<double>(j)[k] = std::get<2>(features[i][j]).ptr<double>(k)[0];
			}
		}
		ret[i] = std::make_tuple(points, featureMat, scores);
	}
	return ret;
}
extern cv::Mat normalize_keypoints(const cv::Mat& keypoint, const int& imgHeight, const int& imgWidth);
extern ncnn::Mat from_float2(const cv::Mat& data);
std::string getSuperPointNet(const int& batch, const int& imgHeight, const int& imgWidth)
{
	CHECK(batch==1 && imgHeight % 8 == 0 && imgWidth % 8 == 0);
	std::string batchStr = std::to_string(batch);
	std::string imgHeightStr = std::to_string(imgHeight);
	std::string imgWidthStr = std::to_string(imgWidth);
	std::string imgHeightStr2 = std::to_string(imgHeight / 2);
	std::string imgWidthStr2 = std::to_string(imgWidth / 2);
	std::string imgHeightStr3 = std::to_string(imgHeight / 4);
	std::string imgWidthStr3 = std::to_string(imgWidth / 4);
	std::string imgHeightStr4 = std::to_string(imgHeight / 8);
	std::string imgWidthStr4 = std::to_string(imgWidth / 8);
	std::string paramStr ="7767517\n""21 23\n";
	paramStr += ("Input data 0 1 data -23330=4," + batchStr + "," + imgWidthStr + "," + imgHeightStr + ",1 0=" + imgWidthStr + " 1=" + imgHeightStr + " 2=1\n");
	paramStr += ("Convolution conv1a 1 1 data conv1a_relu -23330=4," + batchStr + "," + imgWidthStr + "," + imgHeightStr + ",64 0=64 1=3 3=1 4=1 5=1 6=576 9=1\n");
	paramStr += ("Convolution conv1b 1 1 conv1a_relu conv1b_relu -23330=4," + batchStr + "," + imgWidthStr + "," + imgHeightStr + ",64 0=64 1=3 3=1 4=1 5=1 6=36864 9=1\n");
	paramStr += ("Pooling pool1 1 1 conv1b_relu pool1 -23330=4," + batchStr + "," + imgWidthStr2 + "," + imgHeightStr2 + ",64 0=0 1=2 2=2\n");
	paramStr += ("Convolution conv2a 1 1 pool1 conv2a_relu -23330=4," + batchStr + "," + imgWidthStr2 + "," + imgHeightStr2 + ",64 0=64 1=3 3=1 4=1 5=1 6=36864 9=1\n");
	paramStr += ("Convolution conv2b 1 1 conv2a_relu conv2b_relu -23330=4," + batchStr + "," + imgWidthStr2 + "," + imgHeightStr2 + ",64 0=64 1=3 3=1 4=1 5=1 6=36864 9=1\n");
	paramStr += ("Pooling pool2 1 1 conv2b_relu pool2 -23330=4," + batchStr + "," + imgWidthStr3 + "," + imgHeightStr3 + ",64 0=0 1=2 2=2\n");
	paramStr += ("Convolution conv3a 1 1 pool2 conv3a_relu -23330=4," + batchStr + "," + imgWidthStr3 + "," + imgHeightStr3 + ",128 0=128 1=3 3=1 4=1 5=1 6=73728 9=1\n");
	paramStr += ("Convolution conv3b 1 1 conv3a_relu conv3b_relu -23330=4," + batchStr + "," + imgWidthStr3 + "," + imgHeightStr3 + ",128 0=128 1=3 3=1 4=1 5=1 6=147456 9=1\n");
	paramStr += ("Pooling pool3 1 1 conv3b_relu pool3 -23330=4," + batchStr + "," + imgWidthStr4 + "," + imgHeightStr4 + ",128 0=0 1=2 2=2\n");
	paramStr += ("Convolution conv4a 1 1 pool3 conv4a_relu -23330=4," + batchStr + "," + imgWidthStr4 + "," + imgHeightStr4 + ",128 0=128 1=3 3=1 4=1 5=1 6=147456 9=1\n");
	paramStr += ("Convolution conv4b 1 1 conv4a_relu conv4b_relu -23330=4," + batchStr + "," + imgWidthStr4 + "," + imgHeightStr4 + ",128 0=128 1=3 3=1 4=1 5=1 6=147456 9=1\n");
	paramStr += ("Split term 1 2 conv4b_relu conv4b_relu_0 conv4b_relu_1 -23330=8," + batchStr + "," + imgWidthStr4 + "," + imgHeightStr4 + ",128," + batchStr + "," + imgWidthStr4 + "," + imgHeightStr4 + ",128\n");
	paramStr += ("Convolution convPa 1 1 conv4b_relu_0 convPa_relu -23330=4," + batchStr + "," + imgWidthStr4 + "," + imgHeightStr4 + ",256 0=256 1=3 3=1 4=1 5=1 6=294912 9=1\n");
	paramStr += ("Convolution convPb 1 1 convPa_relu convPb -23330=4," + batchStr + "," + imgWidthStr4 + "," + imgHeightStr4 + ",65 0=65 1=1 3=1 4=0 5=1 6=16640 9=0\n");
	paramStr += ("Softmax prob 1 1 convPb score -23330=4," + batchStr + "," + imgWidthStr4 + "," + imgHeightStr4 + ",65 0=0 1=1\n");
	paramStr += ("Slice slice 1 2 score score0 score1 -23330=8," + batchStr + "," + imgWidthStr4 + "," + imgHeightStr4 + ",64,1,80,60,1 -23300=2,64,-233\n");
	paramStr += ("Permute scorePermute 1 1 score0 scorePermute -23330=4," + batchStr + ",64," + imgWidthStr4 + "," + imgHeightStr4 + " 0=3\n");
	paramStr += ("Convolution convDa 1 1 conv4b_relu_1 convDa_relu -23330=4," + batchStr + "," + imgWidthStr4 + "," + imgHeightStr4 + ",256 0=256 1=3 3=1 4=1 5=1 6=294912 9=1\n");
	paramStr += ("Convolution convDb 1 1 convDa_relu convDb -23330=4," + batchStr + "," + imgWidthStr4 + "," + imgHeightStr4 + ",256 0=256 1=1 3=1 4=0 5=1 6=65536 9=0\n");
	paramStr += ("Normalize descpNorm 1 1 convDb descpNorm -23330=4," + batchStr + "," + imgWidthStr4 + "," + imgHeightStr4 + ",256 1=1 3=1 4=1 9=1\n");
	return paramStr;
}
std::shared_ptr<ncnn::Net> getSuperPointNet(const int& imgHeight, const int& imgWidth,const std::string&weightPath)
{ 
	std::string paramStr = getSuperPointNet(1, imgHeight, imgWidth);	
	{
		std::ifstream inFile(weightPath, std::ios::in | std::ios::binary); //二进制读方式打开
		CHECK(inFile);
		int weightCnt = -1;
		inFile.read((char*)&weightCnt, sizeof(int));
		CHECK(weightCnt>1);
		superPointWeightValues.resize(weightCnt);
		for (int i = 0; i < weightCnt; i++)
		{
			inFile.read((char*)&(superPointWeightValues[i]), sizeof(float));
		} 
	}
	std::shared_ptr<ncnn::Net> pnet(new ncnn::Net()); 
	pnet->opt.num_threads = NUM_THREAD;
	pnet->load_param_mem(paramStr.c_str());
	pnet->load_model((const unsigned char*)&superPointWeightValues[0]);
	return pnet;
}
 
int superpoint_show()
{
	int imgHeight = img_h;
	int imgWidth = img_w;
	std::shared_ptr<ncnn::Net> net = getSuperPointNet(imgHeight, imgWidth,"superpointWeight.dat");
	ncnn::Mat input(imgWidth, imgHeight, 1);
	cv::VideoCapture capture;
	capture.open(0);
	cv::Mat imgtemp;
	for (size_t i = 0;  ; i++)
	{
		ncnn::Extractor ex = net->create_extractor();
		capture.read(imgtemp);
		cv::Mat img ;
		cv::resize(imgtemp, img, cv::Size(imgWidth, imgHeight)); 
		cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
		img.convertTo(img,CV_32FC1);
		img /= 255.;
		memcpy(input.data, img.data, imgWidth * imgHeight * sizeof(float));
		ex.input("data", input);
		ncnn::Mat o0, o1;
		ex.extract("scorePermute", o0);
		ex.extract("descpNorm", o1);
		std::vector<cv::Mat>scoreMats;
		scoreResort((const float*)o0.data, input.c, 64, imgHeight / 8, imgWidth / 8, scoreMats);
		scoreNMS(scoreMats);
		std::vector< std::vector<std::tuple<float, cv::Point, cv::Mat>>> keyPoints
			= pickKeyPointsAndDescp(scoreMats, (const float*)o1.data);
		{
			cv::resize(imgtemp, imgtemp, cv::Size(imgWidth, imgHeight));
			for (int j = 0; j < keyPoints[0].size(); j++)
			{
				cv::circle(imgtemp, std::get<1>(keyPoints[0][j]), 3, cv::Scalar(0, 0, 255), -1);
			}
		}
		LOG(INFO);
		cv::imshow("123", imgtemp); cv::waitKey(10);
	}
	return 0;
}

extern void* getSuperGlueNet(const std::string& weightPath);

extern cv::Mat superGlue(void* net, const int& imgHeight, const int& imgWidth,
	const cv::Mat& kps0, cv::Mat& descripCv0, const cv::Mat& scores0,
	const cv::Mat& kps1, cv::Mat& descripCv1, const cv::Mat& scores1);
extern std::vector<std::tuple<float, int, int>> logOptimalTransport(const cv::Mat& scores, const float& bin_score = 2.3457, const int& iter = 20, const float& threshold = 0.2);
void drawMatch(cv::Mat&img, const std::vector<std::tuple<float, int, int>>&match, const cv::Mat& kp0, const cv::Mat& kp1,const float&thre=0.2)
{
	const auto& getColor = [](const float& a)->cv::Vec3b
	{
		float b = min(max(0, a), 1.);
		int s = 255 * b;
		if (s < 32)
		{
			return cv::Vec3b(128 + 4 * s, 0, 0);
		}
		else if (s == 32)
		{
			return cv::Vec3b(255, 0, 0);
		}
		else if (s < 96)
		{
			return cv::Vec3b(255, 4 * s - 128, 0);
		}
		else if (s == 96)
		{
			return cv::Vec3b(254, 255, 2);
		}
		else if (s < 159)
		{
			return cv::Vec3b(-4 * s + 638, 255, -382 + 4 * s);
		}
		else if (s == 159)
		{
			return cv::Vec3b(1, 255, 254);
		}
		else if (s < 224)
		{
			return cv::Vec3b(0, -4 * s + 892, 255);
		}
		else return cv::Vec3b(0, 0, 1148 - 4 * s);
	};
	 
	for (int i = 0; i < match.size(); i++)
	{
		float s = std::get<0>(match[i]);
		if (s > thre)
		{
			int i0 = std::get<1>(match[i]);
			int i1 = std::get<2>(match[i]);
			cv::Point p0(kp0.ptr<int>(i0)[0], kp0.ptr<int>(i0)[1]);
			cv::Point p1(kp1.ptr<int>(i1)[0] + img.cols / 2, kp1.ptr<int>(i1)[1]);
			cv::line(img, p0, p1, cv::Scalar(getColor(s)));
		}
	}
	return;
}
int main_superpoint_show(int argc, char** argv)
{
	superpoint_show();
	return 0;
}

int main()
{ 
	bool isIndoor = true;
	std::shared_ptr<ncnn::Net> net = getSuperPointNet(img_h, img_w, "superpointWeight.dat");
	void* attentionNets = getSuperGlueNet(isIndoor ? "superGlueWeightIndoor.dat" : "superGlueWeightOutdoor.dat");
	CHECK(attentionNets);
	
	std::vector<std::pair<std::string, std::string>> imgPairs = { {"1.jpg","2.jpg"}};
	for (const auto& d : imgPairs)
	{

		cv::Mat img0 = cv::imread(d.first);
		cv::Mat img1 = cv::imread(d.second);
		CHECK(!img0.empty() && !img1.empty());
		cv::resize(img0, img0, cv::Size(img_w, img_h));
		cv::resize(img1, img1, cv::Size(img_w, img_h));
		cv::Mat total(img0.rows, img0.cols * 2, CV_8UC3);;
		img0.copyTo(total(cv::Rect(0, 0, img0.cols, img0.rows)));
		img1.copyTo(total(cv::Rect(img0.cols, 0, img0.cols, img0.rows)));
		cv::cvtColor(img0, img0, cv::COLOR_BGR2GRAY);
		cv::cvtColor(img1, img1, cv::COLOR_BGR2GRAY);
		img0.convertTo(img0, CV_32FC1);
		img1.convertTo(img1, CV_32FC1);
		img0 /= 255.;
		img1 /= 255.;
		std::vector<std::tuple< cv::Mat, cv::Mat, cv::Mat>> keyPoints0;
		std::vector<std::tuple< cv::Mat, cv::Mat, cv::Mat>>  keyPoints1;
		{
			ncnn::Extractor ex = net->create_extractor();
			ncnn::Mat input = from_float2(img0);
			ex.input("data", input);
			ncnn::Mat o0, o1;
			ex.extract("scorePermute", o0);
			ex.extract("descpNorm", o1);
			std::vector<cv::Mat>scoreMats;
			scoreResort((const float*)o0.data, input.c, 64, img_h / 8, img_w / 8, scoreMats);
			scoreNMS(scoreMats);
			keyPoints0 = convt(pickKeyPointsAndDescp(scoreMats, (const float*)o1.data));
		}
		{
			ncnn::Extractor ex = net->create_extractor();
			ncnn::Mat input = from_float2(img1);
			ex.input("data", input);
			ncnn::Mat o0, o1;
			ex.extract("scorePermute", o0);
			ex.extract("descpNorm", o1);
			std::vector<cv::Mat>scoreMats;
			scoreResort((const float*)o0.data, input.c, 64, img_h / 8, img_w / 8, scoreMats);
			scoreNMS(scoreMats);
			keyPoints1 = convt(pickKeyPointsAndDescp(scoreMats, (const float*)o1.data));
		}
		if (std::get<0>(keyPoints0[0]).rows<1|| std::get<0>(keyPoints1[0]).rows<1)
		{
			continue;
		}
		cv::Mat d0 = std::get<1>(keyPoints0[0]);
		cv::Mat d1 = std::get<1>(keyPoints1[0]);
		cv::Mat normalKp0 =normalize_keypoints(std::get<0>(keyPoints0[0]), img_h, img_w);
		cv::Mat normalKp1 =normalize_keypoints(std::get<0>(keyPoints1[0]), img_h, img_w);
		cv::Mat scores = superGlue(attentionNets, img_h, img_w,
			normalKp0, std::get<1>(keyPoints0[0]), std::get<2>(keyPoints0[0]),
			normalKp1, std::get<1>(keyPoints1[0]), std::get<2>(keyPoints1[0]));
		float indoor_bin = 2.3457;
		float outdoor_bin = 4.4124;
		std::vector<std::tuple<float, int, int>> pairs=logOptimalTransport(scores, isIndoor ? indoor_bin: outdoor_bin);
		drawMatch(total, pairs, std::get<0>(keyPoints0[0]), std::get<0>(keyPoints1[0]),0.8); 
		cv::imwrite("123.jpg", total);
	}
	return 0;
}

 
#include <fstream>
#include <map>
#include <string>
#include <tuple>
#include <iostream>
#include <memory>
#include "opencv2/opencv.hpp"  
#include "glog/logging.h"
cv::Mat plus_h(const cv::Mat& Z, const cv::Mat& u)
{
	cv::Mat ret(Z.size(),Z.type());
	CHECK(Z.cols == u.rows);
	for (int r = 0; r < Z.rows; r++)
	{
		for (int c = 0; c < Z.cols; c++)
		{
			ret.ptr<float>(r)[c] = Z.ptr<float>(r)[c]+ u.ptr<float>(c)[0];
		}
	}
	return ret;
}
cv::Mat plus_v(const cv::Mat& Z, const cv::Mat& u)
{
	cv::Mat ret(Z.size(), Z.type());
	CHECK(Z.rows == u.rows);
	for (int r = 0; r < Z.rows; r++)
	{
		for (int c = 0; c < Z.cols; c++)
		{
			ret.ptr<float>(r)[c] = Z.ptr<float>(r)[c]+ u.ptr<float>(r)[0];
		}
	}
	return ret;
}
cv::Mat logsumexp(const cv::Mat& Z, const int& m)
{
	CHECK(m == 0 || m == 1);
	cv::Mat ret(Z.size(),Z.type());
	for (int i = 0; i < Z.rows; i++)
	{ 
		for (int j = 0; j < Z.cols; j++)
		{
			ret.ptr<float>(i)[j] = exp(Z.ptr<float>(i)[j]);
		}
	}
	cv::reduce(ret, ret, m, cv::REDUCE_SUM);
	for (int i = 0; i < ret.rows; i++)
	{
		for (int j = 0; j < ret.cols; j++)
		{
			ret.ptr<float>(i)[j] = log(ret.ptr<float>(i)[j]);
		}
	}
	if (m==0)
	{
		return ret.t();
	}
	return ret;
}
std::vector<std::tuple<float, int, int>> max_hv(const cv::Mat& z,const float&threshold)
{
	std::vector<std::tuple<float, int, int>>matches;
	matches.reserve(z.rows);
	cv::Mat max_v_value = cv::Mat::ones(1, z.cols, z.type()) * -std::numeric_limits<float>::max();
	cv::Mat max_h_value = cv::Mat::ones(z.rows, 1, z.type()) * -std::numeric_limits<float>::max();
	cv::Mat max_v_index = cv::Mat::ones(1, z.cols, CV_32SC1) * -1;
	cv::Mat max_h_index = cv::Mat::ones(z.rows, 1, CV_32SC1) * -1;
	for (int r = 0; r < z.rows; r++)
	{
		for (int c = 0; c < z.cols; c++)
		{
			if (z.ptr<float>(r)[c]> max_h_value.ptr<float>(r)[0])
			{
				max_h_value.ptr<float>(r)[0] = z.ptr<float>(r)[c];
				max_h_index.ptr<int>(r)[0] = c;
			}
		}
	}
	for (int c = 0; c < z.cols; c++)
	{
		for (int r = 0; r < z.rows; r++)
		{
			if (z.ptr<float>(r)[c] > max_v_value.ptr<float>(0)[c])
			{
				max_v_value.ptr<float>(0)[c] = z.ptr<float>(r)[c];
				max_v_index.ptr<int>(0)[c] = r;
			}
		}
	}
	for (int i = 0; i < max_h_index.rows; i++) max_h_value.ptr<float>(i)[0] = exp(max_h_value.ptr<float>(i)[0]);
	for (int i = 0; i < max_v_index.cols; i++) max_v_value.ptr<float>(0)[i] = exp(max_v_value.ptr<float>(0)[i]);
	for (int i = 0; i < max_h_index.rows; i++)
	{		 
			if (max_v_index.ptr<int>(0)[max_h_index.ptr<int>(i)[0]]==i 
				&& max_h_value.ptr<float>(i)[0] > threshold
				&& max_v_value.ptr<float>(0)[max_h_index.ptr<int>(i)[0]] > threshold)
			{
				matches.emplace_back(std::make_tuple( max_h_value.ptr<float>(i)[0], max_v_index.ptr<int>(0)[max_h_index.ptr<int>(i)[0]], max_h_index.ptr<int>(i)[0]));
			}
		
	}
	return matches;
}
std::vector<std::tuple<float, int, int>> logOptimalTransport(const cv::Mat& scores, const float& bin_score = 2.3457, const int& iter = 20, const float& threshold=0.2)
{
	cv::Mat Z(scores.rows + 1, scores.cols + 1, CV_32FC1);
	Z.setTo(bin_score);
	float n = -log(scores.rows + scores.cols);
	scores.copyTo(Z(cv::Rect(0, 0, scores.cols, scores.rows))); 
	cv::Mat log_u = cv::Mat::zeros(Z.rows, 1, CV_32FC1);
	cv::Mat log_v = cv::Mat::zeros(Z.cols, 1, CV_32FC1);
	log_u.setTo(n);
	log_v.setTo(n);
	log_u.ptr<float>(log_u.rows - 1)[0] = log(Z.cols-1) + n;
	log_v.ptr<float>(log_v.rows - 1)[0] = log(Z.rows-1) + n;
	cv::Mat u = cv::Mat::zeros(Z.rows, 1, CV_32FC1);
	cv::Mat v = cv::Mat::zeros(Z.cols, 1, CV_32FC1); 
	for (int i = 0; i < iter; i++)
	{
		u = log_u - logsumexp(plus_h(Z, v), 1);
		v = log_v - logsumexp(plus_v(Z, u), 0);
	}
	Z= plus_h(plus_v(Z, u), v) - n;
	Z = Z(cv::Rect(0, 0, scores.cols, scores.rows));
	return  max_hv(Z, threshold);
}
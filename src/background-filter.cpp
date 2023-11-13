#include <obs-module.h>
#include <media-io/video-scaler.h>

#if defined(__APPLE__)
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
#else
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#endif
#ifdef WITH_CUDA
#include <cuda_provider_factory.h>
#endif
#ifdef _WIN32
#ifndef WITH_CUDA
#include <dml_provider_factory.h>
#endif
#include <wchar.h>
#endif

#include <opencv2/imgproc.hpp>
//#include <opencv2/ximgproc/edge_filter.hpp>

#include <numeric>
#include <memory>
#include <exception>
#include <fstream>

#include "plugin-macros.generated.h"
#include "Model.h"

/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *
 *
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */

//#include "precomp.hpp"
#include <climits>
#include <iostream>
using namespace std;

#ifdef _MSC_VER
#   pragma warning(disable: 4512)
#endif

namespace cv
{
namespace ximgproc
{

typedef Vec<float, 1> Vec1f;
typedef Vec<uchar, 1> Vec1b;

#ifndef SQR
#define SQR(a) ((a)*(a))
#endif

void jointBilateralFilter_32f(Mat& joint, Mat& src, Mat& dst, int radius, double sigmaColor, double sigmaSpace, int borderType);

void jointBilateralFilter_8u(Mat& joint, Mat& src, Mat& dst, int radius, double sigmaColor, double sigmaSpace, int borderType);

template<typename JointVec, typename SrcVec>
class JointBilateralFilter_32f : public ParallelLoopBody
{
    Mat &joint, &src;
    Mat &dst;
    int radius, maxk;
    float scaleIndex;
    int *spaceOfs;
    float *spaceWeights, *expLUT;

public:

    JointBilateralFilter_32f(Mat& joint_, Mat& src_, Mat& dst_, int radius_,
        int maxk_, float scaleIndex_, int *spaceOfs_, float *spaceWeights_, float *expLUT_)
        :
        joint(joint_), src(src_), dst(dst_), radius(radius_), maxk(maxk_),
        scaleIndex(scaleIndex_), spaceOfs(spaceOfs_), spaceWeights(spaceWeights_), expLUT(expLUT_)
    {
        CV_DbgAssert(joint.type() == traits::Type<JointVec>::value && src.type() == dst.type() && src.type() == traits::Type<SrcVec>::value);
        CV_DbgAssert(joint.rows == src.rows && src.rows == dst.rows + 2*radius);
        CV_DbgAssert(joint.cols == src.cols && src.cols == dst.cols + 2*radius);
    }

    void operator () (const Range& range) const CV_OVERRIDE
    {
        for (int i = radius + range.start; i < radius + range.end; i++)
        {
            for (int j = radius; j < src.cols - radius; j++)
            {
                JointVec *jointCenterPixPtr = joint.ptr<JointVec>(i) + j;
                SrcVec *srcCenterPixPtr = src.ptr<SrcVec>(i) + j;

                JointVec jointPix0 = *jointCenterPixPtr;
                SrcVec srcSum = SrcVec::all(0.0f);
                float wSum = 0.0f;

                for (int k = 0; k < maxk; k++)
                {
                    float *jointPix = reinterpret_cast<float*>(jointCenterPixPtr + spaceOfs[k]);
                    float alpha = 0.0f;

                    for (int cn = 0; cn < JointVec::channels; cn++)
                        alpha += std::abs(jointPix0[cn] - jointPix[cn]);
                    alpha *= scaleIndex;
                    int idx = (int)(alpha);
                    alpha -= idx;
                    float weight = spaceWeights[k] * (expLUT[idx] + alpha*(expLUT[idx + 1] - expLUT[idx]));

                    float *srcPix = reinterpret_cast<float*>(srcCenterPixPtr + spaceOfs[k]);
                    for (int cn = 0; cn < SrcVec::channels; cn++)
                        srcSum[cn] += weight*srcPix[cn];
                    wSum += weight;
                }

                dst.at<SrcVec>(i - radius, j - radius) = srcSum / wSum;
            }
        }
    }
};

void jointBilateralFilter_32f(Mat& joint, Mat& src, Mat& dst, int radius, double sigmaColor, double sigmaSpace, int borderType)
{
    CV_DbgAssert(joint.depth() == CV_32F && src.depth() == CV_32F);

    int d = 2*radius + 1;
    int jCn = joint.channels();
    const int kExpNumBinsPerChannel = 1 << 12;
    double minValJoint, maxValJoint;

    minMaxLoc(joint, &minValJoint, &maxValJoint);
    if (abs(maxValJoint - minValJoint) < FLT_EPSILON)
    {
        //TODO: make circle pattern instead of square
        GaussianBlur(src, dst, Size(d, d), sigmaSpace, 0, borderType);
        return;
    }
    float colorRange = (float)(maxValJoint - minValJoint) * jCn;
    colorRange = std::max(0.01f, colorRange);

    int kExpNumBins = kExpNumBinsPerChannel * jCn;
    vector<float> expLUTv(kExpNumBins + 2);
    float *expLUT = &expLUTv[0];
    float scaleIndex = kExpNumBins/colorRange;

    double gaussColorCoeff = -0.5 / (sigmaColor*sigmaColor);
    double gaussSpaceCoeff = -0.5 / (sigmaSpace*sigmaSpace);

    for (int i = 0; i < kExpNumBins + 2; i++)
    {
        double val = i / scaleIndex;
        expLUT[i] = (float) std::exp(val * val * gaussColorCoeff);
    }

    Mat jointTemp, srcTemp;
    copyMakeBorder(joint, jointTemp, radius, radius, radius, radius, borderType);
    copyMakeBorder(src, srcTemp, radius, radius, radius, radius, borderType);
    size_t srcElemStep = srcTemp.step / srcTemp.elemSize();
    size_t jElemStep = jointTemp.step / jointTemp.elemSize();
    CV_Assert(srcElemStep == jElemStep);

    vector<float> spaceWeightsv(d*d);
    vector<int> spaceOfsJointv(d*d);
    float *spaceWeights = &spaceWeightsv[0];
    int *spaceOfsJoint = &spaceOfsJointv[0];

    int maxk = 0;
    for (int i = -radius; i <= radius; i++)
    {
        for (int j = -radius; j <= radius; j++)
        {
            double r2 = i*i + j*j;
            if (r2 > SQR(radius))
                continue;

            spaceWeights[maxk] = (float) std::exp(r2 * gaussSpaceCoeff);
            spaceOfsJoint[maxk] = (int) (i*jElemStep + j);
            maxk++;
        }
    }

    Range range(0, joint.rows);
    if (joint.type() == CV_32FC1)
    {
        if (src.type() == CV_32FC1)
        {
            parallel_for_(range, JointBilateralFilter_32f<Vec1f, Vec1f>(jointTemp, srcTemp, dst, radius, maxk, scaleIndex, spaceOfsJoint, spaceWeights, expLUT));
        }
        if (src.type() == CV_32FC3)
        {
            parallel_for_(range, JointBilateralFilter_32f<Vec1f, Vec3f>(jointTemp, srcTemp, dst, radius, maxk, scaleIndex, spaceOfsJoint, spaceWeights, expLUT));
        }
    }

    if (joint.type() == CV_32FC3)
    {
        if (src.type() == CV_32FC1)
        {
            parallel_for_(range, JointBilateralFilter_32f<Vec3f, Vec1f>(jointTemp, srcTemp, dst, radius, maxk, scaleIndex, spaceOfsJoint, spaceWeights, expLUT));
        }
        if (src.type() == CV_32FC3)
        {
            parallel_for_(range, JointBilateralFilter_32f<Vec3f, Vec3f>(jointTemp, srcTemp, dst, radius, maxk, scaleIndex, spaceOfsJoint, spaceWeights, expLUT));
        }
    }
}

template<typename JointVec, typename SrcVec>
class JointBilateralFilter_8u : public ParallelLoopBody
{
    Mat &joint, &src;
    Mat &dst;
    int radius, maxk;
    float scaleIndex;
    int *spaceOfs;
    float *spaceWeights, *expLUT;

public:

    JointBilateralFilter_8u(Mat& joint_, Mat& src_, Mat& dst_, int radius_,
        int maxk_, int *spaceOfs_, float *spaceWeights_, float *expLUT_)
        :
        joint(joint_), src(src_), dst(dst_), radius(radius_), maxk(maxk_),
        spaceOfs(spaceOfs_), spaceWeights(spaceWeights_), expLUT(expLUT_)
    {
        CV_DbgAssert(joint.type() == traits::Type<JointVec>::value && src.type() == dst.type() && src.type() == traits::Type<SrcVec>::value);
        CV_DbgAssert(joint.rows == src.rows && src.rows == dst.rows + 2 * radius);
        CV_DbgAssert(joint.cols == src.cols && src.cols == dst.cols + 2 * radius);
    }

    void operator () (const Range& range) const CV_OVERRIDE
    {
        typedef Vec<int, JointVec::channels> JointVeci;
        typedef Vec<float, SrcVec::channels> SrcVecf;

        for (int i = radius + range.start; i < radius + range.end; i++)
        {
            for (int j = radius; j < src.cols - radius; j++)
            {
                JointVec *jointCenterPixPtr = joint.ptr<JointVec>(i) + j;
                SrcVec *srcCenterPixPtr = src.ptr<SrcVec>(i) + j;

                JointVeci jointPix0 = JointVeci(*jointCenterPixPtr);
                SrcVecf srcSum = SrcVecf::all(0.0f);
                float wSum = 0.0f;

                for (int k = 0; k < maxk; k++)
                {
                    uchar *jointPix = reinterpret_cast<uchar*>(jointCenterPixPtr + spaceOfs[k]);
                    int alpha = 0;
                    for (int cn = 0; cn < JointVec::channels; cn++)
                        alpha += std::abs(jointPix0[cn] - (int)jointPix[cn]);

                    float weight = spaceWeights[k] * expLUT[alpha];

                    uchar *srcPix = reinterpret_cast<uchar*>(srcCenterPixPtr + spaceOfs[k]);
                    for (int cn = 0; cn < SrcVec::channels; cn++)
                        srcSum[cn] += weight*srcPix[cn];
                    wSum += weight;
                }

                dst.at<SrcVec>(i - radius, j - radius) = SrcVec(srcSum / wSum);
            }
        }
    }
};

void jointBilateralFilter_8u(Mat& joint, Mat& src, Mat& dst, int radius, double sigmaColor, double sigmaSpace, int borderType)
{
    CV_DbgAssert(joint.depth() == CV_8U && src.depth() == CV_8U);

    int d = 2 * radius + 1;
    int jCn = joint.channels();

    double gaussColorCoeff = -0.5 / (sigmaColor*sigmaColor);
    double gaussSpaceCoeff = -0.5 / (sigmaSpace*sigmaSpace);

    vector<float> expLUTv(jCn*256);
    float *expLUT = &expLUTv[0];

    for (int i = 0; i < (int)expLUTv.size(); i++)
    {
        expLUT[i] = (float)std::exp(i * i * gaussColorCoeff);
    }

    Mat jointTemp, srcTemp;
    copyMakeBorder(joint, jointTemp, radius, radius, radius, radius, borderType);
    copyMakeBorder(src, srcTemp, radius, radius, radius, radius, borderType);
    size_t srcElemStep = srcTemp.step / srcTemp.elemSize();
    size_t jElemStep = jointTemp.step / jointTemp.elemSize();
    CV_Assert(srcElemStep == jElemStep);

    vector<float> spaceWeightsv(d*d);
    vector<int> spaceOfsJointv(d*d);
    float *spaceWeights = &spaceWeightsv[0];
    int *spaceOfsJoint = &spaceOfsJointv[0];

    int maxk = 0;
    for (int i = -radius; i <= radius; i++)
    {
        for (int j = -radius; j <= radius; j++)
        {
            double r2 = i*i + j*j;
            if (r2 > SQR(radius))
                continue;

            spaceWeights[maxk] = (float)std::exp(r2 * gaussSpaceCoeff);
            spaceOfsJoint[maxk] = (int)(i*jElemStep + j);
            maxk++;
        }
    }

    Range range(0, src.rows);
    if (joint.type() == CV_8UC1)
    {
        if (src.type() == CV_8UC1)
        {
            parallel_for_(range, JointBilateralFilter_8u<Vec1b, Vec1b>(jointTemp, srcTemp, dst, radius, maxk, spaceOfsJoint, spaceWeights, expLUT));
        }
        if (src.type() == CV_8UC3)
        {
            parallel_for_(range, JointBilateralFilter_8u<Vec1b, Vec3b>(jointTemp, srcTemp, dst, radius, maxk, spaceOfsJoint, spaceWeights, expLUT));
        }
    }

    if (joint.type() == CV_8UC3)
    {
        if (src.type() == CV_8UC1)
        {
            parallel_for_(range, JointBilateralFilter_8u<Vec3b, Vec1b>(jointTemp, srcTemp, dst, radius, maxk, spaceOfsJoint, spaceWeights, expLUT));
        }
        if (src.type() == CV_8UC3)
        {
            parallel_for_(range, JointBilateralFilter_8u<Vec3b, Vec3b>(jointTemp, srcTemp, dst, radius, maxk, spaceOfsJoint, spaceWeights, expLUT));
        }
    }
}

void jointBilateralFilter(InputArray joint_, InputArray src_, OutputArray dst_, int d, double sigmaColor, double sigmaSpace, int borderType)
{
    CV_Assert(!src_.empty());

    if (joint_.empty())
    {
        bilateralFilter(src_, dst_, d, sigmaColor, sigmaSpace, borderType);
        return;
    }

    Mat src = src_.getMat();
    Mat joint = joint_.getMat();

    if (src.data == joint.data)
    {
        bilateralFilter(src_, dst_, d, sigmaColor, sigmaSpace, borderType);
        return;
    }

    CV_Assert(src.size() == joint.size());
    CV_Assert(src.depth() == joint.depth() && (src.depth() == CV_8U || src.depth() == CV_32F) );

    if (sigmaColor <= 0)
        sigmaColor = 1;
    if (sigmaSpace <= 0)
        sigmaSpace = 1;

    int radius;
    if (d <= 0)
        radius = cvRound(sigmaSpace*1.5);
    else
        radius = d / 2;
    radius = std::max(radius, 1);

    dst_.create(src.size(), src.type());
    Mat dst = dst_.getMat();

    if (dst.data == joint.data)
        joint = joint.clone();
    if (dst.data == src.data)
        src = src.clone();

    int jointCnNum = joint.channels();
    int srcCnNum = src.channels();

    if ( (srcCnNum == 1 || srcCnNum == 3) && (jointCnNum == 1 || jointCnNum == 3) )
    {
        if (joint.depth() == CV_8U)
        {
            jointBilateralFilter_8u(joint, src, dst, radius, sigmaColor, sigmaSpace, borderType);
        }
        else
        {
            jointBilateralFilter_32f(joint, src, dst, radius, sigmaColor, sigmaSpace, borderType);
        }
    }
    else
    {
        CV_Error(Error::BadNumChannels, "Unsupported number of channels");
    }
}

}
}

const char* MODEL_SINET = "SINet_Softmax_simple.onnx";
const char* MODEL_MODNET = "modnet_simple.onnx";
const char* MODEL_MEDIAPIPE = "mediapipe.onnx";
const char* MODEL_SELFIE = "selfie_segmentation.onnx";
const char* MODEL_RVM = "rvm_mobilenetv3_fp32.onnx";

const char* USEGPU_CPU = "cpu";
const char* USEGPU_DML = "dml";
const char* USEGPU_CUDA = "cuda";

struct background_removal_filter {
	std::unique_ptr<Ort::Session> session;
	std::unique_ptr<Ort::Env> env;
	std::vector<const char*> inputNames;
	std::vector<const char*> outputNames;
	std::vector<Ort::Value> inputTensor;
	std::vector<Ort::Value> outputTensor;
	std::vector<std::vector<int64_t> > inputDims;
	std::vector<std::vector<int64_t> > outputDims;
	std::vector<std::vector<float> > outputTensorValues;
	std::vector<std::vector<float> > inputTensorValues;
	Ort::MemoryInfo memoryInfo;
	float threshold = 0.5f;
	cv::Scalar backgroundColor{0, 0, 0};
	float contourFilter = 0.05f;
	float smoothContour = 0.5f;
	float feather = 0.0f;
	std::string useGPU;
	std::string modelSelection;
	std::unique_ptr<Model> model;

	// Use the media-io converter to both scale and convert the colorspace
	video_scaler_t* scalerToBGR;
	video_scaler_t* scalerFromBGR;

	cv::Mat backgroundMask;
	int maskEveryXFrames = 1;
	int maskEveryXFramesCount = 0;


#if _WIN32
	const wchar_t* modelFilepath = nullptr;
#else
	const char* modelFilepath = nullptr;
#endif
};


static const char *filter_getname(void *unused)
{
	UNUSED_PARAMETER(unused);
	return "Background Removal";
}


/**                   PROPERTIES                     */

static obs_properties_t *filter_properties(void *data)
{
	obs_properties_t *props = obs_properties_create();

	obs_property_t *p_threshold = obs_properties_add_float_slider(
		props,
		"threshold",
            obs_module_text("Threshold"),
		0.0,
		1.0,
		0.025);

	obs_property_t *p_contour_filter = obs_properties_add_float_slider(
		props,
		"contour_filter",
		obs_module_text("Contour Filter (% of image)"),
		0.0,
		1.0,
		0.025);

	obs_property_t *p_smooth_contour = obs_properties_add_float_slider(
		props,
		"smooth_contour",
		obs_module_text("Smooth silhouette"),
		0.0,
		1.0,
		0.05);

	obs_property_t *p_feather = obs_properties_add_float_slider(
		props,
		"feather",
		obs_module_text("Feather blend silhouette"),
		0.0,
		1.0,
		0.05);

	obs_property_t *p_color = obs_properties_add_color(
		props,
		"replaceColor",
		obs_module_text("Background color"));

	obs_property_t *p_use_gpu = obs_properties_add_list(
		props,
		"useGPU",
		obs_module_text("Inference device"),
		OBS_COMBO_TYPE_LIST,
		OBS_COMBO_FORMAT_STRING);

	obs_property_list_add_string(p_use_gpu, obs_module_text("CPU"), USEGPU_CPU);
#ifdef WITH_CUDA
	obs_property_list_add_string(p_use_gpu, obs_module_text("GPU - CUDA"), USEGPU_CUDA);
#elif _WIN32
	obs_property_list_add_string(p_use_gpu, obs_module_text("GPU - DirectML"), USEGPU_DML);
#endif

	obs_property_t *p_model_select = obs_properties_add_list(
		props,
		"model_select",
		obs_module_text("Segmentation model"),
		OBS_COMBO_TYPE_LIST,
		OBS_COMBO_FORMAT_STRING);

	obs_property_list_add_string(p_model_select, obs_module_text("SINet"), MODEL_SINET);
	obs_property_list_add_string(p_model_select, obs_module_text("MODNet"), MODEL_MODNET);
	obs_property_list_add_string(p_model_select, obs_module_text("MediaPipe"), MODEL_MEDIAPIPE);
	obs_property_list_add_string(p_model_select, obs_module_text("Selfie Segmentation"), MODEL_SELFIE);
	obs_property_list_add_string(p_model_select, obs_module_text("Robust Video Matting"), MODEL_RVM);

	obs_property_t *p_mask_every_x_frames = obs_properties_add_int(
		props,
		"mask_every_x_frames",
		obs_module_text("Calculate mask every X frame"),
		1,
		300,
		1);

	UNUSED_PARAMETER(data);
	return props;
}

static void filter_defaults(obs_data_t *settings) {
	obs_data_set_default_double(settings, "threshold", 0.5);
	obs_data_set_default_double(settings, "contour_filter", 0.05);
	obs_data_set_default_double(settings, "smooth_contour", 0.5);
	obs_data_set_default_double(settings, "feather", 0.0);
	obs_data_set_default_int(settings, "replaceColor", 0x000000);
	obs_data_set_default_string(settings, "useGPU", USEGPU_CPU);
	obs_data_set_default_string(settings, "model_select", MODEL_MEDIAPIPE);
	obs_data_set_default_int(settings, "mask_every_x_frames", 1);
}

static void createOrtSession(struct background_removal_filter *tf) {

	Ort::SessionOptions sessionOptions;

	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	if (tf->useGPU != USEGPU_CPU) {
		sessionOptions.DisableMemPattern();
		sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
	}

	char* modelFilepath_rawPtr = obs_module_file(tf->modelSelection.c_str());

	if (modelFilepath_rawPtr == nullptr) {
		blog(LOG_ERROR, "Unable to get model filename %s from plugin.", tf->modelSelection.c_str());
		return;
	}

	std::string modelFilepath_s(modelFilepath_rawPtr);
	bfree(modelFilepath_rawPtr);

#if _WIN32
	std::wstring modelFilepath_ws(modelFilepath_s.size(), L' ');
	std::copy(modelFilepath_s.begin(), modelFilepath_s.end(), modelFilepath_ws.begin());
	tf->modelFilepath = modelFilepath_ws.c_str();
#else
	tf->modelFilepath = modelFilepath_s.c_str();
#endif

	try {
#ifdef WITH_CUDA
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0));
#elif _WIN32
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(sessionOptions, 0));
#endif
		tf->session.reset(new Ort::Session(*tf->env, tf->modelFilepath, sessionOptions));
	} catch (const std::exception& e) {
		blog(LOG_ERROR, "%s", e.what());
		return;
	}

	Ort::AllocatorWithDefaultOptions allocator;

  tf->model->populateInputOutputNames(tf->session, tf->inputNames, tf->outputNames);

  if (!tf->model->populateInputOutputShapes(tf->session, tf->inputDims, tf->outputDims)) {
    blog(LOG_ERROR, "Unable to get model input and output shapes");
    return;
  }

  for (size_t i = 0; i < tf->inputNames.size(); i++) {
    blog(LOG_INFO, "Model %s input %d: name %s shape (%d dim) %d x %d x %d x %d",
      tf->modelSelection.c_str(), (int)i,
      tf->inputNames[i],
      (int)tf->inputDims[i].size(),
      (int)tf->inputDims[i][0],
      ((int)tf->inputDims[i].size() > 1) ? (int)tf->inputDims[i][1] : 0,
      ((int)tf->inputDims[i].size() > 2) ? (int)tf->inputDims[i][2] : 0,
      ((int)tf->inputDims[i].size() > 3) ? (int)tf->inputDims[i][3] : 0
    );
  }
  for (size_t i = 0; i < tf->outputNames.size(); i++) {
    blog(LOG_INFO, "Model %s output %d: name %s shape (%d dim) %d x %d x %d x %d",
      tf->modelSelection.c_str(), (int)i,
      tf->outputNames[i],
      (int)tf->outputDims[i].size(),
      (int)tf->outputDims[i][0],
      ((int)tf->outputDims[i].size() > 1) ? (int)tf->outputDims[i][1] : 0,
      ((int)tf->outputDims[i].size() > 2) ? (int)tf->outputDims[i][2] : 0,
      ((int)tf->outputDims[i].size() > 3) ? (int)tf->outputDims[i][3] : 0);
  }

	// Allocate buffers
  tf->model->allocateTensorBuffers(
    tf->inputDims,
    tf->outputDims,
    tf->outputTensorValues,
  	tf->inputTensorValues,
    tf->inputTensor,
    tf->outputTensor
  );
}


static void destroyScalers(struct background_removal_filter *tf) {
	blog(LOG_INFO, "Destroy scalers.");
	if (tf->scalerToBGR != nullptr) {
		video_scaler_destroy(tf->scalerToBGR);
		tf->scalerToBGR = nullptr;
	}
	if (tf->scalerFromBGR != nullptr) {
		video_scaler_destroy(tf->scalerFromBGR);
		tf->scalerFromBGR = nullptr;
	}
}


static void filter_update(void *data, obs_data_t *settings)
{
	struct background_removal_filter *tf = reinterpret_cast<background_removal_filter *>(data);
	tf->threshold = (float)obs_data_get_double(settings, "threshold");

	uint64_t color = obs_data_get_int(settings, "replaceColor");
	tf->backgroundColor.val[0] = (double)((color >> 16) & 0x0000ff);
	tf->backgroundColor.val[1] = (double)((color >> 8) & 0x0000ff);
	tf->backgroundColor.val[2] = (double)(color & 0x0000ff);

	tf->contourFilter         = (float)obs_data_get_double(settings, "contour_filter");
	tf->smoothContour         = (float)obs_data_get_double(settings, "smooth_contour");
	tf->feather               = (float)obs_data_get_double(settings, "feather");
	tf->maskEveryXFrames      = (int)obs_data_get_int(settings, "mask_every_x_frames");
	tf->maskEveryXFramesCount = (int)(0);


	const std::string newUseGpu = obs_data_get_string(settings, "useGPU");
	const std::string newModel = obs_data_get_string(settings, "model_select");

	if (tf->modelSelection.empty() ||
        tf->modelSelection != newModel ||
        tf->useGPU != newUseGpu)
	{
		// Re-initialize model if it's not already the selected one or switching inference device
		tf->modelSelection = newModel;
		tf->useGPU = newUseGpu;
		destroyScalers(tf);

		if (tf->modelSelection == MODEL_SINET) {
			tf->model.reset(new ModelSINET);
		}
		if (tf->modelSelection == MODEL_MODNET) {
			tf->model.reset(new ModelMODNET);
		}
		if (tf->modelSelection == MODEL_SELFIE) {
			tf->model.reset(new ModelSelfie);
		}
		if (tf->modelSelection == MODEL_MEDIAPIPE) {
			tf->model.reset(new ModelMediaPipe);
		}
		if (tf->modelSelection == MODEL_RVM) {
			tf->model.reset(new ModelRVM);
		}

		createOrtSession(tf);
	}
}


/**                   FILTER CORE                     */

static void *filter_create(obs_data_t *settings, obs_source_t *source)
{
	struct background_removal_filter *tf = reinterpret_cast<background_removal_filter *>(bzalloc(sizeof(struct background_removal_filter)));

	std::string instanceName{"background-removal-inference"};
	tf->env.reset(new Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, instanceName.c_str()));

    tf->modelSelection = MODEL_MEDIAPIPE;
	filter_update(tf, settings);

	return tf;
}


static void initializeScalers(
	cv::Size frameSize,
	enum video_format frameFormat,
	struct background_removal_filter *tf
) {

	struct video_scale_info dst{
		VIDEO_FORMAT_BGR3,
		(uint32_t)frameSize.width,
		(uint32_t)frameSize.height,
		VIDEO_RANGE_DEFAULT,
		VIDEO_CS_DEFAULT
	};
	struct video_scale_info src{
		frameFormat,
		(uint32_t)frameSize.width,
		(uint32_t)frameSize.height,
		VIDEO_RANGE_DEFAULT,
		VIDEO_CS_DEFAULT
	};

	// Check if scalers already defined and release them
	destroyScalers(tf);

	blog(LOG_INFO, "Initialize scalers. Size %d x %d",
		frameSize.width, frameSize.height);

	// Create new scalers
	video_scaler_create(&tf->scalerToBGR, &dst, &src, VIDEO_SCALE_DEFAULT);
	video_scaler_create(&tf->scalerFromBGR, &src, &dst, VIDEO_SCALE_DEFAULT);
}


static cv::Mat convertFrameToBGR(
	struct obs_source_frame *frame,
	struct background_removal_filter *tf
) {
	const cv::Size frameSize(frame->width, frame->height);

	if (tf->scalerToBGR == nullptr) {
		// Lazy initialize the frame scale & color converter
		initializeScalers(frameSize, frame->format, tf);
	}

	cv::Mat imageBGR(frameSize, CV_8UC3);
	const uint32_t bgrLinesize = (uint32_t)(imageBGR.cols * imageBGR.elemSize());
	video_scaler_scale(tf->scalerToBGR,
		&(imageBGR.data), &(bgrLinesize),
		frame->data, frame->linesize);

	return imageBGR;
}


static void convertBGRToFrame(
  const cv::Mat& imageBGR,
  struct obs_source_frame *frame,
  struct background_removal_filter *tf
) {
  if (tf->scalerFromBGR == nullptr) {
    // Lazy initialize the frame scale & color converter
    initializeScalers(cv::Size(frame->width, frame->height), frame->format, tf);
  }

  const uint32_t rgbLinesize = (uint32_t)(imageBGR.cols * imageBGR.elemSize());
  video_scaler_scale(tf->scalerFromBGR,
    frame->data, frame->linesize,
    &(imageBGR.data), &(rgbLinesize));
}


static void processImageForBackground(
  struct background_removal_filter *tf,
  const cv::Mat& imageBGR,
  cv::Mat& backgroundMask)
{
  if (tf->session.get() == nullptr) {
      // Onnx runtime session is not initialized. Problem in initialization
      return;
  }
  try {
    // To RGB
    cv::Mat imageRGB;
    cv::cvtColor(imageBGR, imageRGB, cv::COLOR_BGR2RGB);

    // Resize to network input size
    uint32_t inputWidth, inputHeight;
    tf->model->getNetworkInputSize(tf->inputDims, inputWidth, inputHeight);

    //blog(LOG_INFO, )
    cv::Mat resizedImageRGB;
    cv::resize(imageRGB, resizedImageRGB, cv::Size(inputWidth, inputHeight));

    // Prepare input to network
    cv::Mat resizedImage, preprocessedImage;
    resizedImageRGB.convertTo(resizedImage, CV_32F);

    tf->model->prepareInputToNetwork(resizedImage, preprocessedImage);

    tf->model->loadInputToTensor(preprocessedImage, inputWidth, inputHeight, tf->inputTensorValues);

    // Run network inference
    tf->model->runNetworkInference(tf->session, tf->inputNames, tf->outputNames, tf->inputTensor, tf->outputTensor);

    // Get output
    // Map network output mask to cv::Mat
    cv::Mat outputImage = tf->model->getNetworkOutput(tf->outputDims, tf->outputTensorValues, tf->inputDims, tf->inputTensorValues);

    // Post-process output
    tf->model->postprocessOutput(outputImage);

    if (tf->modelSelection == MODEL_SINET || tf->modelSelection == MODEL_MEDIAPIPE) {
      if (tf->threshold < 1.0f) {
        backgroundMask = outputImage > tf->threshold;  
      } else {
        outputImage.convertTo(backgroundMask, CV_8UC1, 255);   
      }        
    } else {
      if (tf->threshold > 0.0f) {
        backgroundMask = outputImage < tf->threshold;
      } else {
        outputImage.convertTo(backgroundMask, CV_8UC1, -255, 255);   
      }        
    }

    // Contour processing
    if (tf->contourFilter > 0.0 && tf->contourFilter < 1.0) {
      std::vector<std::vector<cv::Point> > contours;
      findContours(backgroundMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
      std::vector<std::vector<cv::Point> > filteredContours;
      const int64_t contourSizeThreshold = (int64_t)(backgroundMask.total() * tf->contourFilter);
      for (auto& contour : contours) {
        if (cv::contourArea(contour) > contourSizeThreshold) {
          filteredContours.push_back(contour);
        }
      }
      backgroundMask.setTo(0);
      drawContours(backgroundMask, filteredContours, -1, cv::Scalar(255), -1);
    }

    // Resize the size of the mask back to the size of the original input.
    cv::resize(backgroundMask, backgroundMask, imageBGR.size());

    // Smooth mask with a fast filter (box).
    if (tf->smoothContour > 0.0) {
      int k_size = (int)(100 * tf->smoothContour);
      cv::boxFilter(backgroundMask, backgroundMask, backgroundMask.depth(), cv::Size(k_size, k_size));
      //backgroundMask = backgroundMask > 128;
    }
  }
  catch(const std::exception& e) {
    blog(LOG_ERROR, "%s", e.what());
  }
}


static struct obs_source_frame * filter_render(void *data, struct obs_source_frame *frame)
{
	struct background_removal_filter *tf = reinterpret_cast<background_removal_filter *>(data);

	// Convert to BGR
	cv::Mat imageBGR = convertFrameToBGR(frame, tf);

	cv::Mat backgroundMask(imageBGR.size(), CV_8UC1, cv::Scalar(255));

	tf->maskEveryXFramesCount = ++(tf->maskEveryXFramesCount) % tf->maskEveryXFrames;
	if (tf->maskEveryXFramesCount != 0 && !tf->backgroundMask.empty()) {
		// We are skipping processing of the mask for this frame.
		// Get the background mask previously generated.
		tf->backgroundMask.copyTo(backgroundMask);
	} else {
		// Process the image to find the mask.
		processImageForBackground(tf, imageBGR, backgroundMask);

		// Now that the mask is completed, save it off so it can be used on a later frame
		// if we've chosen to only process the mask every X frames.
		backgroundMask.copyTo(tf->backgroundMask);
	}

	// Apply the mask back to the main image.
	try {
		if (tf->feather > 0.0) {
			// If we're going to feather/alpha blend, we need to do some processing that
			// will combine the blended "foreground" and "masked background" images onto the main image.
			cv::Mat maskFloat;
			int k_size = (int)(40 * tf->feather);

			// Convert Mat to float and Normalize the alpha mask to keep intensity between 0 and 1.
			backgroundMask.convertTo(maskFloat, CV_32FC1, 1.0 / 255.0);
			//Feather the normalized mask.
			//cv::boxFilter(maskFloat, maskFloat, maskFloat.depth(), cv::Size(k_size, k_size));

			// Alpha blend
			cv::Mat maskFloat3c;
			cv::cvtColor(maskFloat, maskFloat3c, cv::COLOR_GRAY2BGR);
			cv::Mat tmpImage, tmpBackground;
			// Mutiply the unmasked foreground area of the image with ( 1 - alpha matte).
			cv::multiply(imageBGR, cv::Scalar(1, 1, 1) - maskFloat3c, tmpImage, 1.0, CV_32FC3);
			// Multiply the masked background area (with the background color applied) with the alpha matte.
			cv::multiply(cv::Mat(imageBGR.size(), CV_32FC3, tf->backgroundColor), maskFloat3c, tmpBackground);
			// Add the foreground and background images together, rescale back to an 8bit integer image
			// and apply onto the main image.
			cv::Mat(tmpImage + tmpBackground).convertTo(imageBGR, CV_8UC3);
		} else {
			// If we're not feathering/alpha blending, we can
			// apply the mask as-is back onto the main image.
			imageBGR.setTo(tf->backgroundColor, backgroundMask);
			
            //cvtColor(255 - backgroundMask, imageBGR, cv::COLOR_GRAY2BGR);
            //cvtColor(backgroundMask, imageBGR, cv::COLOR_GRAY2BGR);
		}
	}
	catch(const std::exception& e) {
		blog(LOG_ERROR, "%s", e.what());
	}

	// Put masked image back on frame,
	convertBGRToFrame(imageBGR, frame, tf);
	return frame;
}


static void filter_destroy(void *data)
{
	struct background_removal_filter *tf = reinterpret_cast<background_removal_filter *>(data);

	if (tf) {
		destroyScalers(tf);
		bfree(tf);
	}
}

/*
static void enum_active_sources(void *data, obs_source_enum_proc_t enum_callback, void *param)
{
  switch (enum_callback)
  {
  
  }
}

void enum_all_sources(void *data, obs_source_enum_proc_t enum_callback, void *param)
{
}

//Called to render audio of composite sources. Only used with sources that have the OBS_SOURCE_COMPOSITE output capability flag.
static bool audio_render(void *data, uint64_t *ts_out, struct obs_source_audio_mix *audio_output, uint32_t mixers, size_t channels, size_t sample_rate)
{
}
*/

struct obs_source_info background_removal_filter_info = {
	.id = "background_removal",
	.type = OBS_SOURCE_TYPE_FILTER,
	.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_ASYNC, //| OBS_SOURCE_COMPOSITE,
	.get_name = filter_getname,
	.create = filter_create,
	.destroy = filter_destroy,
	.get_defaults = filter_defaults,
	.get_properties = filter_properties,
	.update = filter_update,
	.filter_video = filter_render
	/*
	.enum_active_sources = enum_active_sources,
	.enum_all_sources = enum_all_sources,
	.audio_render = audio_render
	*/
};

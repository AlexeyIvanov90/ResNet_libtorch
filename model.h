#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "data_set.h"


struct ConvNetImpl : public torch::nn::Module 
{
	ConvNetImpl(int64_t channels, int64_t height, int64_t width)
		: conv1(torch::nn::Conv2dOptions(3 /*input channels*/, 64 /*output channels*/, 7 /*kernel size*/).stride(1)),
		
		conv64_1(torch::nn::Conv2dOptions(64, 64, 3).stride(1)),
		conv64_2(torch::nn::Conv2dOptions(64, 64, 3).stride(1)),
		conv64_3(torch::nn::Conv2dOptions(64, 64, 3).stride(1)),
		conv64_4(torch::nn::Conv2dOptions(64, 64, 3).stride(1)),

		conv64_128(torch::nn::Conv2dOptions(64, 128, 1).stride(1)),

		conv128_1(torch::nn::Conv2dOptions(64, 128, 3).stride(1)),
		conv128_2(torch::nn::Conv2dOptions(128, 128, 3).stride(1)),
		conv128_3(torch::nn::Conv2dOptions(128, 128, 3).stride(1)),
		conv128_4(torch::nn::Conv2dOptions(128, 128, 3).stride(1)),

		conv128_256(torch::nn::Conv2dOptions(128, 256, 1).stride(1)),

		conv256_1(torch::nn::Conv2dOptions(128, 256, 3).stride(1)),
		conv256_2(torch::nn::Conv2dOptions(256, 256, 3).stride(1)),
		conv256_3(torch::nn::Conv2dOptions(256, 256, 3).stride(1)),
		conv256_4(torch::nn::Conv2dOptions(256, 256, 3).stride(1)),

		conv256_512(torch::nn::Conv2dOptions(256, 512, 1).stride(1)),

		conv512_1(torch::nn::Conv2dOptions(256, 512, 3).stride(1)),
		conv512_2(torch::nn::Conv2dOptions(512, 512, 3).stride(1)),
		conv512_3(torch::nn::Conv2dOptions(512, 512, 3).stride(1)),
		conv512_4(torch::nn::Conv2dOptions(512, 512, 3).stride(1)),

		n(GetConvOutput(channels, height, width)),
		lin1(n, 2)
	{
		register_module("conv1", conv1);
		register_module("conv64_1", conv64_1);
		register_module("conv64_2", conv64_2);
		register_module("conv64_3", conv64_3);
		register_module("conv64_4", conv64_4);

		register_module("conv64_128", conv64_128);

		register_module("conv128_1", conv128_1);
		register_module("conv128_2", conv128_2);
		register_module("conv128_3", conv128_3);
		register_module("conv128_4", conv128_4);

		register_module("conv128_256", conv128_256);

		register_module("conv256_1", conv256_1);
		register_module("conv256_2", conv256_2);
		register_module("conv256_3", conv256_3);
		register_module("conv256_4", conv256_4);

		register_module("conv256_512", conv256_512);

		register_module("conv512_1", conv512_1);
		register_module("conv512_2", conv512_2);
		register_module("conv512_3", conv512_3);
		register_module("conv512_4", conv512_4);

		register_module("lin1", lin1);
	};


	torch::Tensor forward(torch::Tensor x)
	{
		torch::Tensor res;

		x = torch::relu(torch::max_pool2d(conv1(x), 2));
		res = x.clone();

		x = torch::relu(conv64_1(x));
		x = torch::relu(conv64_2(x));

		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();

		x = torch::relu(conv64_3(x));
		x = torch::relu(conv64_4(x));

		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();

		res = conv64_128(res);

		x = torch::relu(conv128_1(x));
		x = torch::relu(conv128_2(x));

		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();

		x = torch::relu(conv128_3(x));
		x = torch::relu(conv128_4(x));

		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();

		res = conv128_256(res);

		x = torch::relu(conv256_1(x));
		x = torch::relu(conv256_2(x));

		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();

		x = torch::relu(conv256_3(x));
		x = torch::relu(conv256_4(x));

		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();

		res = conv256_512(res);

		x = torch::relu(conv512_1(x));
		x = torch::relu(conv512_2(x));

		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();

		x = torch::relu(conv512_3(x));
		x = torch::relu(conv512_4(x));

		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();

		x = torch::relu(torch::avg_pool2d(x, 2));

		x = x.view({ -1, n });

		x = torch::log_softmax(lin1(x), 1/*dim*/);

		return x;
	};

	// Get number of elements of output.
	int64_t GetConvOutput(int64_t channels, int64_t height, int64_t width) {
		torch::Tensor res;

		torch::Tensor x = torch::zeros({ 1, channels, height, width });
		x = torch::max_pool2d(conv1(x), 2);
		res = x.clone();

		x = conv64_1(x);
		x = conv64_2(x);

		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();

		x = conv64_3(x);
		x = conv64_4(x);

		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();

		res = conv64_128(res);

		x = conv128_1(x);
		x = conv128_2(x);

		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();

		x = conv128_3(x);
		x = conv128_4(x);

		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();

		res = conv128_256(res);

		x = conv256_1(x);
		x = conv256_2(x);

		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();

		x = conv256_3(x);
		x = conv256_4(x);

		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();

		res = conv256_512(res);

		x = conv512_1(x);
		x = conv512_2(x);

		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();

		x = conv512_3(x);
		x = conv512_4(x);
		
		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();

		x = torch::relu(torch::avg_pool2d(x, 2));

		std::cout << x.numel() << std::endl;

		return x.numel();
	}


	torch::nn::Conv2d conv1, 
		conv64_1, conv64_2, conv64_3, conv64_4, conv64_128,
		conv128_1, conv128_2, conv128_3, conv128_4, conv128_256,
		conv256_1, conv256_2, conv256_3, conv256_4, conv256_512,
		conv512_1, conv512_2, conv512_3, conv512_4;
	int64_t n;
	torch::nn::Linear lin1;
};


TORCH_MODULE(ConvNet);


torch::Tensor classification(torch::Tensor img_tensor, ConvNet model);
double classification_accuracy(CustomDataset &scr, ConvNet model, bool save_error = false);
void train(CustomDataset &train_data_set, CustomDataset &val_data_set, ConvNet &model, int epochs, torch::data::DataLoaderOptions OptionsData, torch::Device device = torch::kCPU);
#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "data_set.h"


struct ConvNetImpl : public torch::nn::Module 
{
	ConvNetImpl(int64_t channels, int64_t height, int64_t width)
		: conv1(torch::nn::Conv2dOptions(3 /*input channels*/, 64 /*output channels*/, 7 /*kernel size*/).stride(1)),

		conv64_1(torch::nn::Conv2dOptions(64, 64, 3).stride(1)),
		bn2d_64_1(torch::nn::BatchNorm2d(64)),
		conv64_2(torch::nn::Conv2dOptions(64, 64, 3).stride(1)),
		bn2d_64_2(torch::nn::BatchNorm2d(64)),
		conv64_3(torch::nn::Conv2dOptions(64, 64, 3).stride(1)),
		bn2d_64_3(torch::nn::BatchNorm2d(64)),
		conv64_4(torch::nn::Conv2dOptions(64, 64, 3).stride(1)),
		bn2d_64_4(torch::nn::BatchNorm2d(64)),

		conv64_128(torch::nn::Conv2dOptions(64, 128, 1).stride(1)),

		conv128_1(torch::nn::Conv2dOptions(64, 128, 3).stride(1)),
		bn2d_128_1(torch::nn::BatchNorm2d(128)),
		conv128_2(torch::nn::Conv2dOptions(128, 128, 3).stride(1)),
		bn2d_128_2(torch::nn::BatchNorm2d(128)),
		conv128_3(torch::nn::Conv2dOptions(128, 128, 3).stride(1)),
		bn2d_128_3(torch::nn::BatchNorm2d(128)),
		conv128_4(torch::nn::Conv2dOptions(128, 128, 3).stride(1)),
		bn2d_128_4(torch::nn::BatchNorm2d(128)),

		conv128_256(torch::nn::Conv2dOptions(128, 256, 1).stride(1)),

		conv256_1(torch::nn::Conv2dOptions(128, 256, 3).stride(1)),
		bn2d_256_1(torch::nn::BatchNorm2d(256)),
		conv256_2(torch::nn::Conv2dOptions(256, 256, 3).stride(1)),
		bn2d_256_2(torch::nn::BatchNorm2d(256)),
		conv256_3(torch::nn::Conv2dOptions(256, 256, 3).stride(1)), 
		bn2d_256_3(torch::nn::BatchNorm2d(256)),
		conv256_4(torch::nn::Conv2dOptions(256, 256, 3).stride(1)),
		bn2d_256_4(torch::nn::BatchNorm2d(256)),

		conv256_512(torch::nn::Conv2dOptions(256, 512, 1).stride(1)),

		conv512_1(torch::nn::Conv2dOptions(256, 512, 3).stride(1)),
		bn2d_512_1(torch::nn::BatchNorm2d(512)),
		conv512_2(torch::nn::Conv2dOptions(512, 512, 3).stride(1)),
		bn2d_512_2(torch::nn::BatchNorm2d(512)),
		conv512_3(torch::nn::Conv2dOptions(512, 512, 3).stride(1)),
		bn2d_512_3(torch::nn::BatchNorm2d(512)),
		conv512_4(torch::nn::Conv2dOptions(512, 512, 3).stride(1)),
		bn2d_512_4(torch::nn::BatchNorm2d(512)),

		n(GetConvOutput(channels, height, width)),
		lin1(n, 2)
	{
		register_module("conv1", conv1);

		register_module("conv64_1", conv64_1);
		register_module("bn2d_64_1", bn2d_64_1);
		register_module("conv64_2", conv64_2);
		register_module("bn2d_64_2", bn2d_64_2);
		register_module("conv64_3", conv64_3);
		register_module("bn2d_64_3", bn2d_64_3);
		register_module("conv64_4", conv64_4);
		register_module("bn2d_64_4", bn2d_64_4);

		register_module("conv64_128", conv64_128);

		register_module("conv128_1", conv128_1);
		register_module("bn2d_128_1", bn2d_128_1);
		register_module("conv128_2", conv128_2);
		register_module("bn2d_128_2", bn2d_128_2);
		register_module("conv128_3", conv128_3);
		register_module("bn2d_128_3", bn2d_128_3);
		register_module("conv128_4", conv128_4);
		register_module("bn2d_128_4", bn2d_128_4);

		register_module("conv128_256", conv128_256);

		register_module("conv256_1", conv256_1);
		register_module("bn2d_256_1", bn2d_256_1);
		register_module("conv256_2", conv256_2);
		register_module("bn2d_256_2", bn2d_256_2);
		register_module("conv256_3", conv256_3);
		register_module("bn2d_256_3", bn2d_256_3);
		register_module("conv256_4", conv256_4);
		register_module("bn2d_256_4", bn2d_256_4);

		register_module("conv256_512", conv256_512);

		register_module("conv512_1", conv512_1);
		register_module("bn2d_512_1", bn2d_512_1);
		register_module("conv512_2", conv512_2);
		register_module("bn2d_512_2", bn2d_512_2);
		register_module("conv512_3", conv512_3);
		register_module("bn2d_512_3", bn2d_512_3);
		register_module("conv512_4", conv512_4);
		register_module("bn2d_512_4", bn2d_512_4);

		register_module("lin1", lin1);
	};


	torch::Tensor forward(torch::Tensor x)
	{
		torch::Tensor res;

		x = torch::relu(torch::max_pool2d(conv1(x), 2));
		res = x.clone();
		//-------------------------------------------------------------------------------------------
		x = torch::relu(conv64_1(x));
		x = torch::batch_norm(bn2d_64_1->forward(x), batch_norm_tensor[0], batch_norm_tensor[1], batch_norm_tensor[2], batch_norm_tensor[3], true, 0.9, 0.001, true);
		x = torch::relu(conv64_2(x));
		x = torch::batch_norm(bn2d_64_2->forward(x), batch_norm_tensor[4], batch_norm_tensor[5], batch_norm_tensor[6], batch_norm_tensor[7], true, 0.9, 0.001, true);

		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();

		x = torch::relu(conv64_3(x));
		x = torch::batch_norm(bn2d_64_3->forward(x), batch_norm_tensor[8], batch_norm_tensor[9], batch_norm_tensor[10], batch_norm_tensor[11], true, 0.9, 0.001, true);
		x = torch::relu(conv64_4(x));
		x = torch::batch_norm(bn2d_64_4->forward(x), batch_norm_tensor[12], batch_norm_tensor[13], batch_norm_tensor[14], batch_norm_tensor[15], true, 0.9, 0.001, true);

		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();

		res = conv64_128(res);
		//-------------------------------------------------------------------------------------------
		x = torch::relu(conv128_1(x));
		x = torch::batch_norm(bn2d_128_1->forward(x), batch_norm_tensor[16], batch_norm_tensor[17], batch_norm_tensor[18], batch_norm_tensor[19], true, 0.9, 0.001, true);
		x = torch::relu(conv128_2(x));
		x = torch::batch_norm(bn2d_128_2->forward(x), batch_norm_tensor[20], batch_norm_tensor[21], batch_norm_tensor[22], batch_norm_tensor[23], true, 0.9, 0.001, true);

		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();

		x = torch::relu(conv128_3(x));
		x = torch::batch_norm(bn2d_128_3->forward(x), batch_norm_tensor[24], batch_norm_tensor[25], batch_norm_tensor[26], batch_norm_tensor[27], true, 0.9, 0.001, true);
		x = torch::relu(conv128_4(x));
		x = torch::batch_norm(bn2d_128_4->forward(x), batch_norm_tensor[28], batch_norm_tensor[29], batch_norm_tensor[30], batch_norm_tensor[31], true, 0.9, 0.001, true);

		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();

		res = conv128_256(res);
		//-------------------------------------------------------------------------------------------
		x = torch::relu(conv256_1(x));
		x = torch::batch_norm(bn2d_256_1->forward(x), batch_norm_tensor[32], batch_norm_tensor[33], batch_norm_tensor[34], batch_norm_tensor[35], true, 0.9, 0.001, true);
		x = torch::relu(conv256_2(x));
		x = torch::batch_norm(bn2d_256_2->forward(x), batch_norm_tensor[36], batch_norm_tensor[37], batch_norm_tensor[38], batch_norm_tensor[39], true, 0.9, 0.001, true);

		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();

		x = torch::relu(conv256_3(x));
		x = torch::batch_norm(bn2d_256_3->forward(x), batch_norm_tensor[40], batch_norm_tensor[41], batch_norm_tensor[42], batch_norm_tensor[43], true, 0.9, 0.001, true);
		x = torch::relu(conv256_4(x));
		x = torch::batch_norm(bn2d_256_4->forward(x), batch_norm_tensor[44], batch_norm_tensor[45], batch_norm_tensor[46], batch_norm_tensor[47], true, 0.9, 0.001, true);

		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();

		res = conv256_512(res);
		//-------------------------------------------------------------------------------------------
		x = torch::relu(conv512_1(x));
		x = torch::batch_norm(bn2d_512_1->forward(x), batch_norm_tensor[48], batch_norm_tensor[49], batch_norm_tensor[50], batch_norm_tensor[51], true, 0.9, 0.001, true);
		x = torch::relu(conv512_2(x));
		x = torch::batch_norm(bn2d_512_2->forward(x), batch_norm_tensor[52], batch_norm_tensor[53], batch_norm_tensor[54], batch_norm_tensor[55], true, 0.9, 0.001, true);

		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();

		x = torch::relu(conv512_3(x));
		x = torch::batch_norm(bn2d_512_3->forward(x), batch_norm_tensor[56], batch_norm_tensor[57], batch_norm_tensor[58], batch_norm_tensor[59], true, 0.9, 0.001, true);
		x = torch::relu(conv512_4(x));
		x = torch::batch_norm(bn2d_512_4->forward(x), batch_norm_tensor[60], batch_norm_tensor[61], batch_norm_tensor[62], batch_norm_tensor[63], true, 0.9, 0.001, true);

		res.index({ torch::indexing::Slice(),torch::indexing::Slice(), torch::indexing::Slice(2, res.sizes()[2] - 2), torch::indexing::Slice(2, res.sizes()[3] - 2) }) += x;
		x = res.clone();
		//-------------------------------------------------------------------------------------------
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

		return x.numel();
	}


	torch::nn::Conv2d conv1, 
		conv64_1, conv64_2, conv64_3, conv64_4, conv64_128,
		conv128_1, conv128_2, conv128_3, conv128_4, conv128_256,
		conv256_1, conv256_2, conv256_3, conv256_4, conv256_512,
		conv512_1, conv512_2, conv512_3, conv512_4;

	torch::nn::BatchNorm2d bn2d_64_1, bn2d_64_2, bn2d_64_3, bn2d_64_4,
		bn2d_128_1, bn2d_128_2, bn2d_128_3, bn2d_128_4,
		bn2d_256_1, bn2d_256_2, bn2d_256_3, bn2d_256_4,
		bn2d_512_1, bn2d_512_2, bn2d_512_3, bn2d_512_4;

	std::vector<torch::Tensor> batch_norm_tensor = std::vector<torch::Tensor>(64);

	int64_t n;
	torch::nn::Linear lin1;
};


TORCH_MODULE(ConvNet);


torch::Tensor classification(torch::Tensor img_tensor, ConvNet model);
double classification_accuracy(CustomDataset &scr, ConvNet model, bool save_error = false);
void train(CustomDataset &train_data_set, CustomDataset &val_data_set, ConvNet &model, int epochs, torch::data::DataLoaderOptions OptionsData, torch::Device device = torch::kCPU);
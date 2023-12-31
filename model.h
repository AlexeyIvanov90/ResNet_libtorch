#pragma once

#include <torch/torch.h>
#include "data_set.h"


#define SIZE_IMG { 3, 200, 100 }
#define NUM_CLASSES 11


struct BasicBlock : torch::nn::Module {

	static const int expansion;

	int64_t stride;
	torch::nn::Conv2d conv1;
	torch::nn::BatchNorm2d bn1;
	torch::nn::Conv2d conv2;
	torch::nn::BatchNorm2d bn2;
	torch::nn::Sequential downsample;

	BasicBlock(int64_t inplanes, int64_t planes, int64_t stride_ = 1,
		torch::nn::Sequential downsample_ = torch::nn::Sequential());

	torch::Tensor forward(torch::Tensor x);
};


 struct ResNetImpl : torch::nn::Module {

	int64_t n = 0;
	int64_t inplanes = 64;
	torch::nn::Conv2d conv1;
	torch::nn::BatchNorm2d bn1;
	torch::nn::Sequential layer1;
	torch::nn::Sequential layer2;
	torch::nn::Sequential layer3;
	torch::nn::Sequential layer4;
	torch::nn::Linear fc;

	ResNetImpl(at::IntArrayRef layers);
	torch::Tensor forward(torch::Tensor x);

private:
	torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride = 1);
	int64_t _get_conv_output(at::IntArrayRef layers);
};

TORCH_MODULE(ResNet);

ResNet ResNet18();
ResNet ResNet34();

void train(CustomDataset &train_data_set, CustomDataset &val_data_set, ResNet &model, size_t epochs, torch::data::DataLoaderOptions OptionsData, torch::Device device);


#include <filesystem>
#include "model.h"
#include "utils.h"


torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
	int64_t stride = 1, int64_t padding = 0, bool with_bias = false) {
	torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
	conv_options.stride() = stride;
	conv_options.padding() = padding;
	conv_options.bias() = with_bias;
	return conv_options;
}


BasicBlock::BasicBlock(int64_t inplanes, int64_t planes, int64_t stride_ ,
	torch::nn::Sequential downsample_)
	: conv1(conv_options(inplanes, planes, 3, stride_, 1)),
	bn1(planes),
	conv2(conv_options(planes, planes, 3, 1, 1)),
	bn2(planes),
	downsample(downsample_)
{
	register_module("conv1", conv1);
	register_module("bn1", bn1);
	register_module("conv2", conv2);
	register_module("bn2", bn2);
	stride = stride_;
	if (!downsample->is_empty()) {
		register_module("downsample", downsample);
	}
}


torch::Tensor BasicBlock::forward(torch::Tensor x) {
	at::Tensor residual(x.clone());

	x = conv1->forward(x);
	x = bn1->forward(x);
	x = torch::relu(x);

	x = conv2->forward(x);
	x = bn2->forward(x);

	if (!downsample->is_empty()) {
		residual = downsample->forward(residual);
	}

	x += residual;
	x = torch::relu(x);

	return x;
}


const int BasicBlock::expansion = 1;


ResNetImpl::ResNetImpl(at::IntArrayRef layers)
	: conv1(torch::nn::Conv2dOptions(3, 64, 7).stride(1)),
	//:conv1(conv_options(3, 64, 7, 2)),
	bn1(64),
	layer1(_make_layer(64, layers[0])),
	layer2(_make_layer(128, layers[1], 2)),
	layer3(_make_layer(256, layers[2], 2)),
	layer4(_make_layer(512, layers[3], 2)),
	n(22528),	
	fc(n, NUM_CLASSES)
{
	register_module("conv1", conv1);
	register_module("bn1", bn1);
	register_module("layer1", layer1);
	register_module("layer2", layer2);
	register_module("layer3", layer3);
	register_module("layer4", layer4);
	register_module("fc", fc);
}


torch::Tensor ResNetImpl::forward(torch::Tensor x) {
	x = conv1->forward(x);
	x = bn1->forward(x);
	x = torch::relu(x);
	x = torch::max_pool2d(x, 3, 2, 1);

	x = layer1->forward(x);
	x = layer2->forward(x);
	x = layer3->forward(x);
	x = layer4->forward(x);

	x = torch::avg_pool2d(x, 3, 1);
	x = x.view({ x.sizes()[0], -1 });

	//std::cout << x.sizes() << std::endl;

	x = fc->forward(x);
	x = torch::softmax(x, 1);

	return x;
}


torch::nn::Sequential ResNetImpl::_make_layer(int64_t planes, int64_t blocks, int64_t stride) {
	torch::nn::Sequential downsample;
	if (stride != 1 || inplanes != planes * BasicBlock::expansion) {
		downsample = torch::nn::Sequential(
			torch::nn::Conv2d(conv_options(inplanes, planes * BasicBlock::expansion, 1, stride)),
			torch::nn::BatchNorm2d(planes * BasicBlock::expansion)
		);
	}
	torch::nn::Sequential layers;
	layers->push_back(BasicBlock(inplanes, planes, stride, downsample));
	inplanes = planes * BasicBlock::expansion;
	for (int64_t i = 0; i < blocks; i++) {
		layers->push_back(BasicBlock(inplanes, planes));
	}

	return layers;
}


ResNet ResNet18() {
	at::IntArrayRef layers = { 2, 2, 2, 2 };
	ResNet model(layers);
	return model;
}


ResNet ResNet34() {
	at::IntArrayRef layers = { 3, 4, 6, 3 };
	ResNet model(layers);
	return model;
}


void train(CustomDataset &train_data_set, CustomDataset &val_data_set, ResNet &model, size_t epochs, torch::data::DataLoaderOptions OptionsData, torch::Device device)
{
	if (device == torch::kCPU)
		std::cout << "Training on CPU" << std::endl;
	else
		std::cout << "Training on GPU" << std::endl;

	model->to(device);

	auto train_data_set_ = train_data_set.map(torch::data::transforms::Stack<>());

	auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		train_data_set_,
		OptionsData);

	double lr = 1e-4;
	size_t count = 0;

	torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr));
	//torch::optim::SGD optimizer{ model->parameters(), torch::optim::SGDOptions(lr).momentum(0.9).weight_decay(1e-4) };

	int dataset_size = train_data_set.size().value();
	torch::Tensor accuracy;
	float best_val_accuracy = classification_accuracy(val_data_set, model, accuracy);

	std::cout << "Initial accuracy model: " + std::to_string(best_val_accuracy * 100.) + " %" << std::endl;
	std::cout << accuracy << std::endl;

	model->train();

	for (int epoch = 1; epoch <= epochs; epoch++) {
		auto begin = std::chrono::steady_clock::now();

		size_t batch_idx = 0;
		double train_mse = 0.;
		double val_accuracy = 0.;

		for (auto& batch : *train_data_loader) {
			auto stat = "\r" + std::to_string(int((double(batch_idx * OptionsData.batch_size()) / dataset_size) * 100)) + "%";
			std::cout << stat;

			auto imgs = batch.data;
			auto labels = batch.target.squeeze();

			imgs = imgs.to(device);
			labels = labels.to(device);

			optimizer.zero_grad();
			auto output = model->forward(imgs);

			//torch::Tensor weight = torch::ones(8);
			//weight[0] = 0.54;
			//weight[1] = 0.42;
			//weight[2] = 0.08;
			//weight[3] = 1.00;			
			//weight[4] = 0.74;
			//weight[5] = 0.53;
			//weight[6] = 0.96;
			//weight[7] = 0.8;
			//auto loss = torch::cross_entropy_loss(output, labels, weight);
			
			auto loss = torch::cross_entropy_loss(output, labels);

			loss.backward();
			optimizer.step();

			train_mse += loss.template item<float>();

			batch_idx++;
		}

		train_mse /= (float)batch_idx;

		auto end = std::chrono::steady_clock::now();
		auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
		std::cout << "\rTime for epoch: " << elapsed_ms.count() << " ms\n";

		model->eval();
		model->to(torch::kCPU);

		val_accuracy = classification_accuracy(val_data_set, model);

		std::string stat = "\rEpoch [" + std::to_string(epoch) + "/" +
			std::to_string(epochs) + "] Train MSE: " + std::to_string(train_mse) +
			" Val accuracy: " + std::to_string(val_accuracy * 100.) + " %";

		std::string model_file_name = "../models/epoch_" + std::to_string(epoch);

		if (val_accuracy > best_val_accuracy)
		{
			stat += "\nbest_model";
			torch::save(model, "../best_model.pt");
			model_file_name += "best_model";
			best_val_accuracy = val_accuracy;
			count = 0;
		}
		else {
			count++;
			if (count == 10 && lr > 1e-5 * 1.5) {
				count = 0;
				lr = lr / 10.;

				std::cout << "new learning rate: " << lr << std::endl;

				auto options = static_cast<torch::optim::AdamOptions&>(optimizer.defaults());
				auto lr = options.lr();
				options.lr(lr);
			}
		}

		torch::save(model, model_file_name + ".pt");

		std::ofstream out;
		out.open("../models/stat.txt", std::ios::app);
		if (out.is_open())
			out << stat;
		out.close();

		std::cout << stat << std::endl;

		if (epoch != epochs) {
			model->to(device);
			model->train();
		}
	}
}


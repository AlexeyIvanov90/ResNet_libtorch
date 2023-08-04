#include <torch/torch.h>
#include <iostream>
#include  "data_set.h"
#include "model.h"

int main()
{
	std::string train_file_csv = "../data_set/data_train.csv";
	std::string val_file_csv = "../data_set/data_val.csv";
	std::string test_file_csv = "../data_set/data_test.csv";

	int64_t img_size[] = { 3, 100, 200 };
	torch::Device device("cpu");
	auto epochs = 3;

	CustomDataset train_data_set(train_file_csv);
	CustomDataset val_data_set(val_file_csv);
	CustomDataset test_data_set(test_file_csv);

	ResNet model = resnet34();

	torch::data::DataLoaderOptions OptionsData;
	OptionsData.batch_size(64).workers(12);

	train(train_data_set, val_data_set, model, epochs, OptionsData, device);

	std::cout << "Test error: " << classification_accuracy(test_data_set, model, false) << std::endl;
	std::cout << "Val error: " << classification_accuracy(val_data_set, model, false) << std::endl;
	std::cout << "Train error: " << classification_accuracy(train_data_set, model, false) << std::endl;

	return 0;
}
#include <torch/torch.h>
#include <iostream>
#include  "data_set.h"
#include "model.h"
#include  "utils.h"

int main()
{
	std::string train_file_csv = "../data_set/data_train.csv";
	std::string val_file_csv = "../data_set/data_val.csv";
	std::string test_file_csv = "../data_set/data_test.csv";
	torch::Tensor accuracy;

	torch::Device device("cpu");
	size_t epochs = 100000;

	CustomDataset train_data_set(train_file_csv);
	CustomDataset val_data_set(val_file_csv);
	CustomDataset test_data_set(test_file_csv);

	ResNet model = ResNet18();
	torch::data::DataLoaderOptions OptionsData;
	OptionsData.batch_size(64).workers(12);

	torch::load(model, "../best_model.pt");

	train(train_data_set, val_data_set, model, epochs, OptionsData, device);

	torch::load(model, "../best_model.pt");

	std::cout << "Test accuracy: " << classification_accuracy(test_data_set, model, accuracy) << std::endl;
	std::cout << accuracy << std::endl;

	std::cout << "Val accuracy: " << classification_accuracy(val_data_set, model, accuracy) << std::endl;
	std::cout << accuracy << std::endl;

	std::cout << "Train accuracy: " << classification_accuracy(train_data_set, model) << std::endl;
	std::cout << accuracy << std::endl;

	return 0;
}
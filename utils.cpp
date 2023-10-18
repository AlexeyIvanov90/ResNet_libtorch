#include "utils.h"


torch::Tensor classification(torch::Tensor img_tensor, ResNet model)
{
	model->eval();
	model->to(torch::kCPU);
	img_tensor.to(torch::kCPU);
	img_tensor = img_tensor.unsqueeze(0);

	torch::Tensor log_prob = model->forward(img_tensor);
	torch::Tensor prob = torch::exp(log_prob);

	return torch::argmax(prob);
}


double classification_accuracy(CustomDataset &scr, ResNet model, bool confusion_matrix) {
	std::vector<size_t> category_size;
	scr.get_category_size(&category_size);
	torch::Tensor matrix = torch::zeros({ (int)category_size.size(), (int)category_size.size() });

	for (int i = 0; i < scr.size().value(); i++) {
		auto obj = scr.get(i);

		torch::Tensor result = classification(obj.data, model);

		if (result.item<int>() == obj.target.item<int>())
			matrix[obj.target.item<int>()][obj.target.item<int>()] = matrix[obj.target.item<int>()][obj.target.item<int>()].item<int>() + 1;
		else
			matrix[obj.target.item<int>()][result.item<int>()] = matrix[obj.target.item<int>()][result.item<int>()].item<int>() +  1;
	}

	double error = matrix.diag().sum().item<float>() / matrix.sum().item<float>();

	if (confusion_matrix) {
		for (int i = 0; i < category_size.size(); i++)//confusion matrix
			matrix[i] = matrix[i].div((int)category_size.at(i));
		std::cout << matrix << std::endl;
	}

	return error;
} 

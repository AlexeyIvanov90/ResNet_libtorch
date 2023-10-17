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


double classification_accuracy(CustomDataset &scr, ResNet model, bool save_error)
{
	int error = 0;
	std::ofstream out;
	out.open("../error_CNN/error_CNN.csv", std::ios::out);
	for (int i = 0; i < scr.size().value(); i++) {
		auto obj = scr.get(i);

		torch::Tensor result = classification(obj.data, model);

		if (result.item<int>() != obj.target.item<int>()) {
			error++;
			if (save_error) {
				Element elem = scr.get_element(i);

				if (out.is_open())
					out << elem.img + "," +
					std::to_string(elem.label) + "," +
					std::to_string(result.item<int>()) +
					"\n";
			}
		}
	}
	out.close();

	return (double)error / scr.size().value();
}
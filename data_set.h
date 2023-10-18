#pragma once

#include <torch/torch.h>
#include <vector>


namespace cv {
	class Mat;
}


struct Element
{
	Element() {};
	Element(std::string img, int label) :img{ img }, label{ label } {};

	std::string img;
	int label;
};


class CustomDataset : public torch::data::Dataset<CustomDataset>
{
private:
	std::vector<Element> _data;
	std::vector<size_t> _category_size;

public:
	CustomDataset(std::string& file_names_csv);
	torch::data::Example<> get(size_t index) override;
	Element get_element(size_t index);
	void get_category_size(std::vector<size_t>* category_size);

	torch::optional<size_t> size() const override;
};


torch::Tensor img_to_tensor(cv::Mat scr);
torch::Tensor img_to_tensor(std::string path);
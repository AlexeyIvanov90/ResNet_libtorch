#include "data_set.h"
#include <sstream>
#include <fstream>
#include <opencv2/opencv.hpp>


torch::Tensor img_to_tensor(cv::Mat scr) {
	cv::cvtColor(scr, scr, cv::COLOR_BGR2RGB); // camera out - RGB, openCV - BGR
	
	torch::Tensor img_tensor = torch::from_blob(scr.data, { scr.rows, scr.cols, 3 }, torch::kByte).clone();

	img_tensor = img_tensor.toType(torch::kFloat);
	img_tensor = img_tensor.div(255);
	img_tensor = img_tensor.permute({ 2,0,1 });
	return img_tensor;
}


torch::Tensor img_to_tensor(std::string path) {
	cv::Mat img = cv::imread(path);
	return img_to_tensor(img);
}


void read_csv(std::string& location, std::vector<Element>* data, std::vector<size_t>* category_size) {
	std::fstream in(location, std::ios::in);
	std::string line;
	std::string name;
	std::string label;
	data->clear();
	category_size->clear();

	while (getline(in, line))
	{
		std::stringstream s(line);
		getline(s, name, ',');
		getline(s, label, ',');

		if (category_size->size() < stoi(label) + 1)
			category_size->resize(stoi(label) + 1);

		category_size->at(stoi(label))++;

		data->push_back(Element(name, stoi(label)));
	}
}


CustomDataset::CustomDataset(std::string& file_names_csv) {
	read_csv(file_names_csv, &_data, &_category_size);
}


torch::data::Example<> CustomDataset::get(size_t index) {

	std::string file_location = _data[index].img;
	int64_t label = _data[index].label;

	torch::Tensor img_tensor = img_to_tensor(file_location);

	torch::Tensor label_tensor = torch::full({ 1 }, label);
	label_tensor.to(torch::kInt64);

	return { img_tensor, label_tensor };
}


 void CustomDataset::get_category_size(std::vector<size_t>* category_size) {
	category_size->resize(_category_size.size());
	std::copy(_category_size.begin(), _category_size.end(), category_size->begin());
}


Element CustomDataset::get_element(size_t index) {
	return _data[index];
}


torch::optional<size_t> CustomDataset::size() const{
	return _data.size();
};
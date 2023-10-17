#pragma once

#include <torch/torch.h>
#include "model.h"
#include "data_set.h"


torch::Tensor classification(torch::Tensor img_tensor, ResNet model);
double classification_accuracy(CustomDataset &scr, ResNet model, bool save_error = false);

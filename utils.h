#pragma once

#include <torch/torch.h>
#include "model.h"
#include "data_set.h"


torch::Tensor classification(torch::Tensor img_tensor, ResNet model);
double classification_accuracy(CustomDataset &scr, ResNet model);
torch::Tensor confusion_matrix(CustomDataset &scr, ResNet model);

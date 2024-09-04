// Copyright (c) 2024 wesep team. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SEPARATE_SEPARATE_ENGINE_H_
#define SEPARATE_SEPARATE_ENGINE_H_

#include <memory>
#include <string>
#include <vector>

#include "torch/script.h"
#include "torch/torch.h"

#include "frontend/feature_pipeline.h"

namespace wesep {

class SeparateEngine {
 public:
  explicit SeparateEngine(const std::string& model_path, const int feat_dim,
                          const int sample_rate);

  void InitEngineThreads(int num_threads = 1);

  void ForwardFunc(const std::vector<int16_t>& mix_wav, const int16_t* spk1_emb,
                   const int16_t* spk2_emb, int data_size,
                   std::vector<std::vector<float>>* output);

  void ExtractFeature(const int16_t* data, int data_size,
                      std::vector<std::vector<float>>* feat);

  void ApplyMean(std::vector<std::vector<float>>* feat);

 private:
  std::shared_ptr<torch::jit::script::Module> model_ = nullptr;
  std::shared_ptr<wenet::FeaturePipelineConfig> feature_config_ = nullptr;
  std::shared_ptr<wenet::FeaturePipeline> feature_pipeline_ = nullptr;
  int sample_rate_ = 16000;
  int feat_dim_ = 80;
};

}  // namespace wesep

#endif  // SEPARATE_SEPARATE_ENGINE_H_



#include "separate/separate_engine.h"

#include "torch/script.h"
#include "torch/torch.h"
#include "glog/logging.h"
#include "gflags/gflags.h"

namespace wesep {

void SeparateEngine::InitEngineThreads(int num_threads) {
  // for multi-thread performance
  at::set_num_threads(num_threads);
  VLOG(1) << "Num intra-op threads: " << at::get_num_threads();
}

SeparateEngine::SeparateEngine(const std::string& model_path,
                               const int feat_dim,
                               const int sample_rate) {

  sample_rate_  = sample_rate;
  feat_dim_ = feat_dim;
  feature_config_ = std::make_shared<wenet::FeaturePipelineConfig>(feat_dim, sample_rate);
  feature_pipeline_ = std::make_shared<wenet::FeaturePipeline>(*feature_config_);
  feature_pipeline_->Reset();

  InitEngineThreads(1);
  torch::jit::script::Module model = torch::jit::load(model_path);
  model_ = std::make_shared<torch::jit::script::Module> (std::move(model));
  model_->eval();
}

void SeparateEngine::ExtractFeature(const int16_t* data,
                               int data_size,
                               std::vector<std::vector<float>>* feat) {
  feature_pipeline_->AcceptWaveform(std::vector<int16_t>(data, data + data_size));
  feature_pipeline_->set_input_finished();
  feature_pipeline_->Read(feature_pipeline_->num_frames(), feat);
  feature_pipeline_->Reset();
  this->ApplyMean(feat);
}

void SeparateEngine::ApplyMean(std::vector<std::vector<float>>* feat) {
  std::vector<float> mean(feat_dim_, 0);
  for (auto& i : *feat) {
    std::transform(i.begin(), i.end(), mean.begin(), mean.begin(),
                   std::plus<>{});
  }
  std::transform(mean.begin(), mean.end(), mean.begin(),
                 [&](const float d) { return d / feat->size(); });
  for (auto& i : *feat) {
    std::transform(i.begin(), i.end(), mean.begin(), i.begin(), std::minus<>{});
  }
}

void SeparateEngine::ForwardFunc(const std::vector<int16_t> &mix_wav,
                                 const int16_t* spk1_emb,
                                 const int16_t* spk2_emb,
                                 int data_size,
                                 std::vector<std::vector<float>> *output) {
  // pre-process
  std::vector<float> input_wav(mix_wav.size());
  for (int i = 0; i < mix_wav.size(); i++) {
    input_wav[i] = static_cast<float>(mix_wav[i]) / (1 << 15);
  }
  std::vector<std::vector<float>> spk1_emb_feat;
  this->ExtractFeature(spk1_emb, data_size, &spk1_emb_feat);
  std::vector<std::vector<float>> spk2_emb_feat;
  this->ExtractFeature(spk2_emb, data_size, &spk2_emb_feat);

  // torch mix_wav
  torch::Tensor torch_wav = torch::zeros({2, mix_wav.size()}, torch::kFloat32);
  for (size_t i = 0; i < 2; i++){
    torch::Tensor row = torch::from_blob(input_wav.data(), {input_wav.size()},
                                         torch::kFloat32).clone();
    torch_wav[i] = std::move(row);
  }

  // torch spk_emb_feat
  torch::Tensor torch_spk_emb_feat = torch::zeros(
    {2, spk1_emb_feat.size(), feat_dim_}, torch::kFloat32);
  for (size_t i = 0; i < spk1_emb_feat.size(); i++) {
    torch::Tensor row1 = torch::from_blob(spk1_emb_feat[i].data(), {feat_dim_},
                                          torch::kFloat32);
    torch_spk_emb_feat[0][i] = std::move(row1);
    torch::Tensor row2 = torch::from_blob(spk2_emb_feat[i].data(), {feat_dim_},
                                          torch::kFloat32);
    torch_spk_emb_feat[1][i] = std::move(row2);
  }

  // forward
  torch::NoGradGuard no_grad;
  auto outputs = model_->forward(
    {torch_wav, torch_spk_emb_feat}).toTuple()->elements();
  torch::Tensor wav_out = outputs[0].toTensor();
  auto accessor = wav_out.accessor<float, 2>();

  output->resize(2, std::vector<float>(wav_out.size(1), 0.0));
  for (int i = 0; i < wav_out.size(1); i++) {
    (*output)[0][i] = accessor[0][i];
    (*output)[1][i] = accessor[1][i];
  }
}

}  // namespace wesep

#include <iostream>
#include <fstream>
#include <string>

#include "glog/logging.h"
#include "gflags/gflags.h"

#include "frontend/wav.h"
#include "separate/separate_engine.h"
#include "utils/utils.h"
#include "utils/timer.h"


DEFINE_string(wav_path, "", "the path of mixing audio.");
DEFINE_string(spk1_emb, "", "the emb of spk1.");
DEFINE_string(spk2_emb, "", "the emb of spk2.");
DEFINE_string(wav_scp, "", "input wav scp.");
DEFINE_string(model, "", "the path of wesep model.");
DEFINE_string(output_dir, "", "output path.");
DEFINE_int32(sample_rate, 16000, "sample rate");
DEFINE_int32(feat_dim, 80, "fbank feature dimension.");


int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  std::vector<std::vector<std::string>> waves;
  if (!FLAGS_wav_path.empty() && !FLAGS_spk1_emb.empty() &&
      !FLAGS_spk2_emb.empty()) {
    waves.push_back(std::vector<std::string>({
      "test", FLAGS_wav_path, FLAGS_spk1_emb, FLAGS_spk2_emb}));
  } else {
    std::ifstream wav_scp(FLAGS_wav_scp);
    std::string line;
    while (getline(wav_scp, line)) {
      std::vector<std::string> strs;
      wesep::SplitString(line, &strs);
      CHECK_EQ(strs.size(), 4);
      waves.push_back(std::vector<std::string>({
        strs[0], strs[1], strs[2], strs[3]}));
    }
    if (waves.empty()) {
      LOG(FATAL) << "Please provide non-empty wav scp.";
    }
  }

  if (FLAGS_output_dir.empty()) {
    LOG(FATAL) << "Invalid output path.";
  }

  int g_total_waves_dur = 0;
  int g_total_process_time = 0;

  auto model = std::make_shared<wesep::SeparateEngine>(
    FLAGS_model, FLAGS_feat_dim, FLAGS_sample_rate);

  for (auto wav : waves) {
    // mix wav
    wenet::WavReader wav_reader(wav[1]);
    CHECK_EQ(wav_reader.sample_rate(), 16000);
    int16_t* mix_wav_data = const_cast<int16_t*>(wav_reader.data());

    int wave_dur = static_cast<int>(static_cast<float>(
      wav_reader.num_sample()) / wav_reader.sample_rate() * 1000);

    // spk1
    wenet::WavReader spk1_reader(wav[2]);
    CHECK_EQ(spk1_reader.sample_rate(), 16000);
    int16_t* spk1_data = const_cast<int16_t*>(spk1_reader.data());

    // spk2
    wenet::WavReader spk2_reader(wav[3]);
    CHECK_EQ(spk2_reader.sample_rate(), 16000);
    int16_t* spk2_data = const_cast<int16_t*>(spk2_reader.data());

    // forward
    std::vector<std::vector<float>> outputs;
    int process_time = 0;
    wenet::Timer timer;
    model->ForwardFunc(std::vector<int16_t>(
      mix_wav_data, mix_wav_data + wav_reader.num_sample()),
      spk1_data, spk2_data,
      std::min(spk1_reader.num_sample(), spk2_reader.num_sample()),
      &outputs);
    process_time = timer.Elapsed();
    LOG(INFO) << "process: " << wav[0]
              << " RTF: " << static_cast<float>(process_time) / wave_dur;
    // 保存音频
    wenet::WriteWavFile(outputs[0].data(),  outputs[0].size(), 16000,
                        FLAGS_output_dir + "/" + wav[0] + "-spk1.wav");
    wenet::WriteWavFile(outputs[1].data(),  outputs[1].size(), 16000,
                        FLAGS_output_dir + "/" + wav[0] + "-spk2.wav");
    g_total_process_time += process_time;
    g_total_waves_dur += wave_dur;
  }
  LOG(INFO) << "Total: process " << g_total_waves_dur << "ms audio taken "
            << g_total_process_time << "ms.";
  LOG(INFO) << "RTF: " << std::setprecision(4)
            << static_cast<float>(g_total_process_time) / g_total_waves_dur;
  return 0;
}

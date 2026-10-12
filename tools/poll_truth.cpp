// 调试用：轮询打印能量机关真值（模式/角度/转速/5叶激活值）。
// 用法: poll_truth [seconds] [--ipc-dir PATH]
#include <daedalus_sim_sdk/talos_metadata_reader.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <unistd.h>

using namespace daedalus::sim::sdk::v1;

static std::string defaultIpcDir() {
  const char* xdg = std::getenv("XDG_RUNTIME_DIR");
  const std::string base = xdg != nullptr ? xdg : "/tmp";
  return base + "/daedalus-learning-" + std::to_string(::getuid());
}

int main(int argc, char** argv) {
  double seconds = 10.0;
  std::string ipc_dir = defaultIpcDir();
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--ipc-dir" && i + 1 < argc) ipc_dir = argv[++i];
    else seconds = std::atof(a.c_str());
  }

  TalosMetadataMapping mapping;
  const ClientStatus open_status = mapping.open(ipc_dir + "/talos_ipc_meta");
  if (!open_status.ok()) {
    std::fprintf(stderr, "metadata open failed: %s\n", open_status.message.c_str());
    return 1;
  }
  auto reader = mapping.reader();
  if (!reader.ok()) {
    std::fprintf(stderr, "reader failed: %s\n", reader.status.message.c_str());
    return 1;
  }

  const auto t0 = std::chrono::steady_clock::now();
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() <
         seconds) {
    const auto truth = reader.value->readLatestGroundTruth();
    if (!truth.ok()) {
      std::fprintf(stderr, "truth read failed: %s\n", truth.status.message.c_str());
    } else {
      const auto& batch = *truth.value;
      if (batch.rune_count == 0) {
        std::printf("t=%.1f rune_count=0\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      } else {
        const auto& r = batch.runes[0];
        std::printf("t=%.1f mode=%u angle=%+.3f v_roll=%.3f mech=%u acts=[%u,%u,%u,%u,%u]\n",
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                    r.rune_mode, static_cast<double>(r.current_angle),
                    static_cast<double>(r.v_roll), r.mechanism_state,
                    r.target_activations[0], r.target_activations[1],
                    r.target_activations[2], r.target_activations[3],
                    r.target_activations[4]);
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
  }
  return 0;
}

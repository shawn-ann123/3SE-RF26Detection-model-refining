// 受控标定工具：把能量机关冻结（Static 运动），并指定各叶片外观状态。
// 用法: rune_static [--mode small|large] [--leaf N] [--all-deactivated] [--restore]
//                   [--dir clock|counter] [--ipc-dir PATH]
//   --leaf N            仅激活第 N 号叶片（其余 Deactivated）
//   --all-deactivated   五叶全部 Deactivated
//   --restore           恢复规则驱动大符旋转（默认场景）
#include <daedalus_sim_sdk/contest_client.hpp>
#include <daedalus_sim_sdk/scene_control_client.hpp>
#include <daedalus_sim_sdk/talos_metadata_reader.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unistd.h>

namespace {

using namespace daedalus::sim::sdk::v1;

std::string defaultIpcDir() {
  const char* xdg = std::getenv("XDG_RUNTIME_DIR");
  const std::string base = xdg != nullptr ? xdg : "/tmp";
  return base + "/daedalus-learning-" + std::to_string(::getuid());
}

void usage() {
  std::printf(
      "Usage: rune_static [--mode small|large] [--leaf N] "
      "[--all-deactivated] [--restore]\n"
      "                   [--dir clock|counter] [--ipc-dir PATH]\n"
      "  freeze rune in Static motion with chosen leaf states\n");
}

}  // namespace

int main(int argc, char** argv) {
  std::string ipc_dir = defaultIpcDir();
  RuneScenario scenario;
  scenario.mode = RuneMode::Large;
  scenario.motion = RuneMotion::Static;
  scenario.red_face_direction = RuneDirection::Clockwise;
  scenario.leaf_states.assign(5, RuneLeafState::Deactivated);
  bool restore = false;
  int leaf = -1;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--ipc-dir") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "--ipc-dir requires a value\n");
        return 2;
      }
      ipc_dir = argv[++i];
    } else if (arg == "--mode") {
      if (i + 1 >= argc) return 2;
      const std::string m = argv[++i];
      scenario.mode = m == "small" ? RuneMode::Small : RuneMode::Large;
    } else if (arg == "--leaf") {
      if (i + 1 >= argc) return 2;
      leaf = std::atoi(argv[++i]);
      if (leaf < 0 || leaf > 4) {
        std::fprintf(stderr, "--leaf must be in [0,4]\n");
        return 2;
      }
      scenario.leaf_states[static_cast<std::size_t>(leaf)] =
          RuneLeafState::Activated;
    } else if (arg == "--all-deactivated") {
      scenario.leaf_states.assign(5, RuneLeafState::Deactivated);
    } else if (arg == "--dir") {
      if (i + 1 >= argc) return 2;
      scenario.red_face_direction =
          std::strcmp(argv[++i], "counter") == 0
              ? RuneDirection::CounterClockwise
              : RuneDirection::Clockwise;
    } else if (arg == "--restore") {
      restore = true;
    } else if (arg == "--help" || arg == "-h") {
      usage();
      return 0;
    } else {
      usage();
      return 2;
    }
  }
  if (restore) {
    scenario.motion = RuneMotion::RuleDriven;
    scenario.leaf_states.clear();
    scenario.mode = RuneMode::Large;
  }

  ContestClientOptions options;
  options.ipc_directory = ipc_dir;
  ContestClient client(options);
  const ClientStatus connect_status = client.connect();
  if (!connect_status.ok()) {
    std::fprintf(stderr, "connect failed: %s\n",
                 connect_status.message.c_str());
    return 1;
  }

  const auto scene = client.selectScene(ContestScene::Energy);
  if (!scene.ok()) {
    std::fprintf(stderr, "selectScene(energy) failed: %s\n",
                 scene.status.message.c_str());
    return 1;
  }
  const auto resp = client.setRuneScenario(scenario);
  if (!resp.ok()) {
    std::fprintf(stderr, "setRuneScenario failed: %s\n",
                 resp.status.message.c_str());
    return 1;
  }
  std::printf("ok applied_frame_seq=%llu message=%s\n",
              static_cast<unsigned long long>(resp.value->applied_frame_seq),
              resp.value->message.c_str());

  TalosMetadataMapping mapping;
  const ClientStatus open_status = mapping.open(ipc_dir + "/talos_ipc_meta");
  if (open_status.ok()) {
    auto reader = mapping.reader();
    if (reader.ok()) {
      auto truth = reader.value->readLatestGroundTruth();
      if (truth.ok()) {
        for (std::uint32_t i = 0; i < truth.value->rune_count; ++i) {
          const auto& r = truth.value->runes[i];
          std::printf("rune[%u] team=%u mode=%u angle=%.3f rad activations="
                      "[%u,%u,%u,%u,%u]\n",
                      i, r.team, r.rune_mode,
                      static_cast<double>(r.current_angle),
                      r.target_activations[0], r.target_activations[1],
                      r.target_activations[2], r.target_activations[3],
                      r.target_activations[4]);
        }
      }
    }
  }
  return 0;
}

// 能量机关小符/大符切换工具（单可执行文件）。
// 用法: switch_rune small|large|small-activated|large-activated|status [--static] [--activated LIST] [--ipc-dir PATH]
//   small/large       小符/大符（规则驱动旋转）
//   small-activated   小符：旋转 + 指定扇叶亮起（--activated LIST，缺省五臂全亮；--static 则静止冻结）
//   large-activated   大符：旋转 + 指定扇叶亮起（--activated LIST，缺省五臂全亮；--static 则静止冻结）
//   status            打印能量机关真值（模式/角度/转速/扇叶激活状态）
#include <daedalus_sim_sdk/contest_client.hpp>
#include <daedalus_sim_sdk/scene_control_client.hpp>
#include <daedalus_sim_sdk/talos_metadata_reader.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

const char* runeModeName(std::uint8_t mode) {
  switch (mode) {
    case 0: return "small";
    case 1: return "large";
    default: return "unknown";
  }
}

const char* teamName(std::uint8_t team) {
  return team == 0 ? "red" : (team == 1 ? "blue" : "unknown");
}

void usage() {
  std::printf("Usage: switch_rune small|large|small-activated|large-activated|status "
              "[--static] [--activated LIST] [--ipc-dir PATH]\n"
              "  small             小符（规则驱动旋转）\n"
              "  large             大符（规则驱动旋转）\n"
              "  small-activated   小符：旋转 + 指定扇叶亮起（--activated LIST，缺省五臂全亮）\n"
              "  large-activated   大符：旋转 + 指定扇叶亮起（--activated LIST，缺省五臂全亮）\n"
              "  status            打印能量机关真值（模式/角度/转速/扇叶激活状态）\n"
              "  --static          仅对 -activated 命令有效：停止旋转，冻结在当前角度\n"
              "  --activated LIST  仅对 -activated 命令有效：亮起的扇叶索引，0-4 逗号分隔，\n"
              "                    如 \"0,1,2\" 表示三个灯臂亮起（逐个亮起的中间状态）\n");
}

// 解析 "0,1,2" 形式的扇叶索引列表；空字符串 = 不激活任何扇叶。
// 要求每个索引在 0-4 且不重复。
bool parseLeaves(const std::string& list, std::vector<std::uint8_t>& out) {
  out.clear();
  if (list.empty()) return true;
  std::size_t i = 0;
  while (i < list.size()) {
    if (list[i] < '0' || list[i] > '9') return false;
    std::size_t j = i;
    while (j < list.size() && list[j] >= '0' && list[j] <= '9') ++j;
    const int v = std::atoi(list.substr(i, j - i).c_str());
    if (v < 0 || v > 4) return false;
    out.push_back(static_cast<std::uint8_t>(v));
    i = j;
    if (i < list.size()) {
      if (list[i] != ',') return false;
      ++i;
    }
  }
  std::vector<std::uint8_t> sorted = out;
  std::sort(sorted.begin(), sorted.end());
  return std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
}

int printStatus(const std::string& ipc_dir) {
  TalosMetadataMapping mapping;
  const ClientStatus open_status = mapping.open(ipc_dir + "/talos_ipc_meta");
  if (!open_status.ok()) {
    std::fprintf(stderr, "metadata open failed: %s\n",
                 open_status.message.c_str());
    return 1;
  }
  auto reader = mapping.reader();
  if (!reader.ok()) {
    std::fprintf(stderr, "reader failed: %s\n", reader.status.message.c_str());
    return 1;
  }
  const auto truth = reader.value->readLatestGroundTruth();
  if (!truth.ok()) {
    std::fprintf(stderr, "truth read failed: %s\n",
                 truth.status.message.c_str());
    return 1;
  }
  const auto& batch = *truth.value;
  std::printf("rune_count=%u\n", batch.rune_count);
  for (std::uint32_t i = 0; i < batch.rune_count; ++i) {
    const auto& rune = batch.runes[i];
    std::printf("  rune[%u] team=%s(%u) mode=%s(%u) angle=%.2f rad v_roll=%.2f rad/s "
                "mech_state=%u activations=[%u,%u,%u,%u,%u]\n",
                i, teamName(rune.team), rune.team, runeModeName(rune.rune_mode),
                rune.rune_mode, static_cast<double>(rune.current_angle),
                static_cast<double>(rune.v_roll), rune.mechanism_state,
                rune.target_activations[0], rune.target_activations[1],
                rune.target_activations[2], rune.target_activations[3],
                rune.target_activations[4]);
  }
  if (batch.rune_count == 0) {
    std::printf("  (无真值数据：当前不在能量机关场景，或 DAEDALUS_LEARNING_TRUTH=0)\n");
  }
  return 0;
}

// 规则驱动旋转时场景 API 不接受调用方指定的扇叶状态（"simulator-owned"），
// 已激活外观通过 legacy set_rune_state 接口下发：mode + activated_targets。
// 该接口不含 motion 字段，不会影响旋转状态。
int applyActivatedLeaves(RuneMode mode,
                         const std::vector<std::uint8_t>& activated_leaves) {
  RuneState state;
  state.mode = mode;
  state.pending_targets.clear();
  state.activated_targets = activated_leaves;

  SceneControlOptions sc_opts;
  sc_opts.session_id = "switch-rune-" + std::to_string(::getpid());
  SceneControlClient sc(sc_opts);

  const auto session = sc.createSession();
  if (!session.ok()) {
    std::fprintf(stderr, "createSession failed: %s\n",
                 session.status.message.c_str());
    return 1;
  }
  const auto resp = sc.setRuneState(state);
  if (!resp.ok()) {
    std::fprintf(stderr, "setRuneState failed: %s\n",
                 resp.status.message.c_str());
    return 1;
  }
  std::printf("rune_state applied_frame_seq=%llu message=%s\n",
              static_cast<unsigned long long>(resp.value->applied_frame_seq),
              resp.value->message.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string ipc_dir = defaultIpcDir();
  std::string action;
  bool static_freeze = false;
  bool has_activated_arg = false;
  std::vector<std::uint8_t> activated_leaves;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--ipc-dir") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "--ipc-dir requires a value\n");
        return 2;
      }
      ipc_dir = argv[++i];
    } else if (arg == "--static") {
      static_freeze = true;
    } else if (arg == "--activated") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "--activated requires a value\n");
        return 2;
      }
      has_activated_arg = true;
      if (!parseLeaves(argv[++i], activated_leaves)) {
        std::fprintf(stderr,
                     "--activated 必须是 0-4 的逗号分隔列表且不重复，如 0,1,2\n");
        return 2;
      }
    } else if (arg == "--help" || arg == "-h") {
      usage();
      return 0;
    } else if (!action.empty()) {
      usage();
      return 2;
    } else {
      action = arg;
    }
  }

  RuneMode mode = RuneMode::Large;
  bool activated = false;
  if (action == "small") {
    mode = RuneMode::Small;
  } else if (action == "large") {
    mode = RuneMode::Large;
  } else if (action == "small-activated") {
    mode = RuneMode::Small;
    activated = true;
  } else if (action == "large-activated") {
    mode = RuneMode::Large;
    activated = true;
  } else if (action == "status") {
    // status 不需要场景控制连接
  } else {
    usage();
    return 2;
  }
  if (static_freeze && !activated) {
    std::fprintf(stderr, "--static 仅对 small-activated / large-activated 有效\n");
    return 2;
  }
  if (has_activated_arg && !activated) {
    std::fprintf(stderr, "--activated 仅对 small-activated / large-activated 有效\n");
    return 2;
  }
  if (activated && !has_activated_arg) {
    activated_leaves = {0, 1, 2, 3, 4};
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

  if (action == "status") {
    return printStatus(ipc_dir);
  }

  const auto scene = client.selectScene(ContestScene::Energy);
  if (!scene.ok()) {
    std::fprintf(stderr, "selectScene(energy) failed: %s\n",
                 scene.status.message.c_str());
    return 1;
  }
  std::printf("scene=energy applied_frame_seq=%llu message=%s\n",
              static_cast<unsigned long long>(scene.value->applied_frame_seq),
              scene.value->message.c_str());

  RuneScenario scenario;
  scenario.mode = mode;
  scenario.red_face_direction = RuneDirection::Clockwise;
  if (activated && static_freeze) {
    scenario.motion = RuneMotion::Static;
    scenario.leaf_states.assign(5, RuneLeafState::Deactivated);
    for (const std::uint8_t leaf : activated_leaves) {
      scenario.leaf_states[leaf] = RuneLeafState::Activated;
    }
  } else {
    scenario.motion = RuneMotion::RuleDriven;
    scenario.leaf_states.clear();
  }
  const auto resp = client.setRuneScenario(scenario);
  if (!resp.ok()) {
    std::fprintf(stderr, "setRuneScenario failed: %s\n",
                 resp.status.message.c_str());
    return 1;
  }
  std::printf("rune=%s%s applied_frame_seq=%llu message=%s\n", action.c_str(),
              static_freeze ? "(static)" : "",
              static_cast<unsigned long long>(resp.value->applied_frame_seq),
              resp.value->message.c_str());

  if (activated && !static_freeze) {
    if (applyActivatedLeaves(mode, activated_leaves) != 0) return 1;
  }

  return printStatus(ipc_dir);
}

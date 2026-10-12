// 一次性场景状态设置小工具（调试用）：
// set_state --mode small|large [--pending LIST] [--activated LIST] [--scenario]
// 全程 SceneControlClient（避免 ContestClient session mismatch 坑）。
#include <daedalus_sim_sdk/contest_client.hpp>
#include <daedalus_sim_sdk/scene_control_client.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <unistd.h>

using namespace daedalus::sim::sdk::v1;

static std::string defaultIpcDir() {
  const char* xdg = std::getenv("XDG_RUNTIME_DIR");
  const std::string base = xdg != nullptr ? xdg : "/tmp";
  return base + "/daedalus-learning-" + std::to_string(::getuid());
}

static bool parseLeaves(const std::string& s, std::vector<std::uint8_t>& out) {
  out.clear();
  if (s.empty()) return true;
  std::size_t i = 0;
  while (i < s.size()) {
    const int v = std::atoi(s.substr(i).c_str());
    if (v < 0 || v > 4) return false;
    out.push_back(static_cast<std::uint8_t>(v));
    while (i < s.size() && s[i] != ',') ++i;
    if (i < s.size()) ++i;
  }
  return true;
}

int main(int argc, char** argv) {
  RuneMode mode = RuneMode::Small;
  std::vector<std::uint8_t> pending, activated;
  bool do_scenario = false;
  std::string ipc_dir = defaultIpcDir();
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--mode" && i + 1 < argc) mode = std::string(argv[++i]) == "large" ? RuneMode::Large : RuneMode::Small;
    else if (a == "--pending" && i + 1 < argc) parseLeaves(argv[++i], pending);
    else if (a == "--activated" && i + 1 < argc) parseLeaves(argv[++i], activated);
    else if (a == "--scenario") do_scenario = true;
    else if (a == "--ipc-dir" && i + 1 < argc) ipc_dir = argv[++i];
  }

  ContestClientOptions copt;
  copt.ipc_directory = ipc_dir;
  ContestClient client(copt);
  const auto conn = client.connect();
  if (!conn.ok()) { std::fprintf(stderr, "connect failed: %s\n", conn.message.c_str()); return 1; }
  const auto scene = client.selectScene(ContestScene::Energy);
  if (!scene.ok()) { std::fprintf(stderr, "selectScene failed: %s\n", scene.status.message.c_str()); return 1; }

  SceneControlOptions sc_opts;
  sc_opts.session_id = "set-state-" + std::to_string(::getpid());
  SceneControlClient sc(sc_opts);
  const auto session = sc.createSession();
  if (!session.ok()) { std::fprintf(stderr, "createSession failed: %s\n", session.status.message.c_str()); return 1; }

  if (do_scenario) {
    RuneScenario scenario;
    scenario.mode = mode;
    scenario.motion = RuneMotion::RuleDriven;
    scenario.red_face_direction = RuneDirection::Clockwise;
    scenario.leaf_states.clear();
    const auto resp = sc.setRuneScenario(scenario);
    if (!resp.ok()) { std::fprintf(stderr, "setRuneScenario failed: %s\n", resp.status.message.c_str()); return 1; }
    std::printf("scenario applied: %s\n", resp.value->message.c_str());
  }

  RuneState st;
  st.mode = mode;
  st.pending_targets = pending;
  st.activated_targets = activated;
  const auto resp = sc.setRuneState(st);
  if (!resp.ok()) { std::fprintf(stderr, "setRuneState failed: %s\n", resp.status.message.c_str()); return 1; }
  std::printf("state applied: %s\n", resp.value->message.c_str());
  return 0;
}

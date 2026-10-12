// set_rune_state pending/activated 语义探针 v2（开发验证用）。
// 全程使用单一 SceneControlClient session（ContestClient 只 connect+selectScene；
// 实测：SceneControlClient.createSession 之后 ContestClient.setRuneScenario 会
// session mismatch，反之亦然）。
// 步骤：RuleDriven 旋转 -> pending 单叶保持性（旋转中）-> pending+activated 混搭 ->
//       Small 模式 2 pending（约束）-> Large 模式 2 pending（约束）-> 恢复。
#include <daedalus_sim_sdk/contest_client.hpp>
#include <daedalus_sim_sdk/scene_control_client.hpp>
#include <daedalus_sim_sdk/talos_metadata_reader.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

namespace {

using namespace daedalus::sim::sdk::v1;

std::string defaultIpcDir() {
  const char* xdg = std::getenv("XDG_RUNTIME_DIR");
  const std::string base = xdg != nullptr ? xdg : "/tmp";
  return base + "/daedalus-learning-" + std::to_string(::getuid());
}

const char* modeName(std::uint8_t m) {
  switch (m) { case 0: return "small"; case 1: return "large"; default: return "off"; }
}

void printTruth(const std::string& tag, TalosMetadataMapping& mapping) {
  auto reader = mapping.reader();
  if (!reader.ok()) { std::printf("[%s] reader failed: %s\n", tag.c_str(), reader.status.message.c_str()); return; }
  const auto truth = reader.value->readLatestGroundTruth();
  if (!truth.ok()) { std::printf("[%s] truth failed: %s\n", tag.c_str(), truth.status.message.c_str()); return; }
  const auto& batch = *truth.value;
  for (std::uint32_t i = 0; i < batch.rune_count; ++i) {
    const auto& r = batch.runes[i];
    std::printf("[%s] rune[%u] mode=%s(%u) angle=%+.3f v_roll=%+.3f mech=%u acts=[%u,%u,%u,%u,%u]\n",
                tag.c_str(), i, modeName(r.rune_mode), r.rune_mode,
                static_cast<double>(r.current_angle), static_cast<double>(r.v_roll),
                r.mechanism_state, r.target_activations[0], r.target_activations[1],
                r.target_activations[2], r.target_activations[3], r.target_activations[4]);
  }
  if (batch.rune_count == 0) std::printf("[%s] (no rune truth)\n", tag.c_str());
}

void wait(const std::string& tag, int ms, TalosMetadataMapping& mapping) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
  printTruth(tag, mapping);
}

}  // namespace

int main(int argc, char** argv) {
  std::string ipc_dir = defaultIpcDir();
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--ipc-dir" && i + 1 < argc) ipc_dir = argv[++i];
  }

  TalosMetadataMapping mapping;
  const ClientStatus open_status = mapping.open(ipc_dir + "/talos_ipc_meta");
  if (!open_status.ok()) {
    std::fprintf(stderr, "metadata open failed: %s\n", open_status.message.c_str());
    return 1;
  }

  ContestClientOptions options;
  options.ipc_directory = ipc_dir;
  ContestClient client(options);
  const ClientStatus conn = client.connect();
  if (!conn.ok()) { std::fprintf(stderr, "connect failed: %s\n", conn.message.c_str()); return 1; }
  const auto scene = client.selectScene(ContestScene::Energy);
  if (!scene.ok()) { std::fprintf(stderr, "selectScene failed: %s\n", scene.status.message.c_str()); return 1; }
  std::printf("scene=energy message=%s\n", scene.value->message.c_str());

  SceneControlOptions sc_opts;
  sc_opts.session_id = "probe-rune-" + std::to_string(::getpid());
  SceneControlClient sc(sc_opts);
  const auto session = sc.createSession();
  if (!session.ok()) { std::fprintf(stderr, "createSession failed: %s\n", session.status.message.c_str()); return 1; }
  std::printf("session: %s\n", session.value->message.c_str());

  printTruth("start", mapping);

  // 1. Small + RuleDriven：确认旋转起来（v_roll 非零）
  {
    RuneScenario scenario;
    scenario.mode = RuneMode::Small;
    scenario.motion = RuneMotion::RuleDriven;
    scenario.red_face_direction = RuneDirection::Clockwise;
    scenario.leaf_states.clear();
    const auto resp = sc.setRuneScenario(scenario);
    if (!resp.ok()) { std::fprintf(stderr, "setRuneScenario(small) failed: %s\n", resp.status.message.c_str()); return 1; }
    std::printf("setRuneScenario(small rule-driven): %s\n", resp.value->message.c_str());
  }
  wait("t+2s rotate?", 2000, mapping);
  wait("t+5s rotate?", 3000, mapping);

  // 2. pending=[2]：期望 acts[2]=1，旋转中保持
  {
    RuneState st;
    st.mode = RuneMode::Small;
    st.pending_targets = {2};
    st.activated_targets.clear();
    const auto resp = sc.setRuneState(st);
    if (!resp.ok()) { std::fprintf(stderr, "setRuneState(pending=[2]) failed: %s\n", resp.status.message.c_str()); return 1; }
    std::printf("setRuneState(pending=[2]): %s\n", resp.value->message.c_str());
  }
  wait("t+2s pending[2]", 2000, mapping);
  wait("t+7s pending[2] held?", 5000, mapping);

  // 3. pending=[1] + activated=[3] 混搭
  {
    RuneState st;
    st.mode = RuneMode::Small;
    st.pending_targets = {1};
    st.activated_targets = {3};
    const auto resp = sc.setRuneState(st);
    if (!resp.ok()) { std::fprintf(stderr, "setRuneState(mix) failed: %s\n", resp.status.message.c_str()); return 1; }
    std::printf("setRuneState(pending=[1] activated=[3]): %s\n", resp.value->message.c_str());
  }
  wait("t+2s mix", 2000, mapping);
  wait("t+7s mix held?", 5000, mapping);

  // 4. Small 模式 2 个 pending：期望拒绝
  {
    RuneState st;
    st.mode = RuneMode::Small;
    st.pending_targets = {0, 1};
    st.activated_targets.clear();
    const auto resp = sc.setRuneState(st);
    if (!resp.ok()) {
      std::printf("setRuneState(small pending x2) REJECTED: %s\n", resp.status.message.c_str());
    } else {
      std::printf("setRuneState(small pending x2): %s\n", resp.value->message.c_str());
    }
  }
  wait("t+2s small pending x2", 2000, mapping);

  // 5. Large + RuleDriven，然后 2 个 pending：测 Large 模式约束
  {
    RuneScenario scenario;
    scenario.mode = RuneMode::Large;
    scenario.motion = RuneMotion::RuleDriven;
    scenario.red_face_direction = RuneDirection::Clockwise;
    scenario.leaf_states.clear();
    const auto resp = sc.setRuneScenario(scenario);
    if (!resp.ok()) { std::fprintf(stderr, "setRuneScenario(large) failed: %s\n", resp.status.message.c_str()); return 1; }
    std::printf("setRuneScenario(large rule-driven): %s\n", resp.value->message.c_str());
  }
  wait("t+2s large rotate?", 2000, mapping);
  {
    RuneState st;
    st.mode = RuneMode::Large;
    st.pending_targets = {0, 1};
    st.activated_targets.clear();
    const auto resp = sc.setRuneState(st);
    if (!resp.ok()) {
      std::printf("setRuneState(large pending x2) REJECTED: %s\n", resp.status.message.c_str());
    } else {
      std::printf("setRuneState(large pending x2): %s\n", resp.value->message.c_str());
    }
  }
  wait("t+2s large pending x2", 2000, mapping);
  {
    RuneState st;
    st.mode = RuneMode::Large;
    st.pending_targets = {2};
    st.activated_targets.clear();
    const auto resp = sc.setRuneState(st);
    if (!resp.ok()) { std::fprintf(stderr, "setRuneState(large pending[2]) failed: %s\n", resp.status.message.c_str()); return 1; }
    std::printf("setRuneState(large pending[2]): %s\n", resp.value->message.c_str());
  }
  wait("t+2s large pending[2]", 2000, mapping);
  wait("t+7s large pending[2] held?", 5000, mapping);

  // 6. 恢复：Large RuleDriven + pending=[0] activated=[1,3]（探针前状态）
  {
    RuneState st;
    st.mode = RuneMode::Large;
    st.pending_targets = {0};
    st.activated_targets = {1, 3};
    const auto resp = sc.setRuneState(st);
    if (!resp.ok()) { std::fprintf(stderr, "setRuneState(restore) failed: %s\n", resp.status.message.c_str()); return 1; }
    std::printf("setRuneState(restore pending=[0] activated=[1,3]): %s\n", resp.value->message.c_str());
  }
  wait("t+2s restored", 2000, mapping);
  return 0;
}

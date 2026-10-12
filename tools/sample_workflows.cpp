// 能量机关工况采样程序：遍历可达工况空间，每工况设置状态 -> 真值核对生效 ->
// 调 export_labeled_frames 采一个旋转周期（--per-roll 每圈 N 帧），
// 统一 out 目录合并追加（labels.jsonl 自动合并，文件名/标签带工况编号）。
//
// 工况空间（探针实测规则语义，2026-10-12）：
//   RuleDriven 旋转下，无 pending 固定时规则自动点亮"第一个未激活叶"；
//   有 pending 固定时亮集=固定集；Small pending<=1、Large<=2（服务器强制）。
//   因此可达状态 = 亮叶位置 x 激活子集，另加 5 叶全激活后的完成态 [3,3,3,3,3]。
//   全灭 / 0亮+激活 在旋转下不可达，已按用户决定放弃。
// 用法示例：
//   ./sample_workflows --out /path/data/runeset --per-roll 24 --dry-run   # 预览工况
//   ./sample_workflows --out /path/data/runeset --per-roll 24             # 正式采样
//   ./sample_workflows --out /path/data/runeset --per-roll 24 --start 50  # 断点续跑
#include <daedalus_sim_sdk/contest_client.hpp>
#include <daedalus_sim_sdk/scene_control_client.hpp>
#include <daedalus_sim_sdk/talos_metadata_reader.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>

namespace {

using namespace daedalus::sim::sdk::v1;

constexpr int kLeaves = 5;

std::string defaultIpcDir() {
  const char* xdg = std::getenv("XDG_RUNTIME_DIR");
  const std::string base = xdg != nullptr ? xdg : "/tmp";
  return base + "/daedalus-learning-" + std::to_string(::getuid());
}

// 本程序所在目录（/proc/self/exe），用于把相对路径解析成绝对路径，
// 保证从任意工作目录启动都能找到 export 与 GLB 资源。
std::string exeDir() {
  char buf[4096];
  const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n > 0) {
    buf[n] = '\0';
    const std::string p(buf);
    const std::size_t slash = p.rfind('/');
    if (slash != std::string::npos) return p.substr(0, slash);
  }
  return ".";
}

struct Options {
  std::string out_dir;
  std::string ipc_dir;
  std::string export_bin = "./export_labeled_frames";
  int per_roll = 24;
  int start = 0;
  int limit = 0;  // 0 = 全部
  bool dry_run = false;
};

bool parseOptions(int argc, char** argv, Options* o) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto value = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", arg.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--out") o->out_dir = value();
    else if (arg == "--ipc-dir") o->ipc_dir = value();
    else if (arg == "--export-bin") o->export_bin = value();
    else if (arg == "--per-roll") o->per_roll = std::atoi(value());
    else if (arg == "--start") o->start = std::atoi(value());
    else if (arg == "--limit") o->limit = std::atoi(value());
    else if (arg == "--dry-run") o->dry_run = true;
    else if (arg == "--help" || arg == "-h") {
      std::printf(
          "Usage: sample_workflows --out DIR [options]\n"
          "  --out DIR         采样输出目录（统一合并，labels.jsonl 追加）\n"
          "  --per-roll N      每工况每旋转周期采样帧数（默认 24，每 15 度一帧）\n"
          "  --start N         从第 N 个工况开始（断点续跑）\n"
          "  --limit N         最多采样 N 个工况（0 = 全部）\n"
          "  --export-bin PATH export_labeled_frames 路径（默认同目录）\n"
          "  --dry-run         只打印工况列表不采样\n"
          "  --ipc-dir PATH    真值 IPC 目录（默认 /tmp/daedalus-learning-<uid>）\n");
      return false;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      std::exit(2);
    }
  }
  if (o->out_dir.empty() && !o->dry_run) {
    std::fprintf(stderr, "--out is required (use --dry-run to preview without sampling)\n");
    std::exit(2);
  }
  return true;
}

struct Workflow {
  int index = 0;                       // 全局编号（000 起，写入文件名与标签行）
  RuneMode mode = RuneMode::Small;
  bool completed = false;              // 5 叶全激活 -> 完成态 [3,3,3,3,3]
  std::uint8_t want_acts[kLeaves] = {0, 0, 0, 0, 0};  // 期望真值 5 叶激活值
  std::vector<std::uint8_t> pending;
  std::vector<std::uint8_t> activated;
};

// 掩码展开：mask 的第 k 位置 1 -> 取 cand[k]
std::vector<std::uint8_t> subsetFromMask(const std::vector<std::uint8_t>& cand,
                                         int mask) {
  std::vector<std::uint8_t> out;
  for (std::size_t k = 0; k < cand.size(); ++k)
    if (mask & (1 << k)) out.push_back(cand[k]);
  return out;
}

// 亮叶位置 p 与激活子集（在非 p 叶上取）组合出一个工况
Workflow makeWorkflow(int index, RuneMode mode,
                      const std::vector<std::uint8_t>& pending,
                      const std::vector<std::uint8_t>& activated) {
  Workflow w;
  w.index = index;
  w.mode = mode;
  w.pending = pending;
  w.activated = activated;
  for (const std::uint8_t p : pending) w.want_acts[p] = 1;
  for (const std::uint8_t a : activated) w.want_acts[a] = 2;
  return w;
}

// 完成态工况：activated=[0..4]，真值将自动转为 [3,3,3,3,3]
Workflow makeCompletedWorkflow(int index, RuneMode mode) {
  Workflow w;
  w.index = index;
  w.mode = mode;
  w.completed = true;
  w.activated = {0, 1, 2, 3, 4};
  for (int i = 0; i < kLeaves; ++i) w.want_acts[i] = 3;
  return w;
}

std::vector<Workflow> buildWorkflows() {
  std::vector<Workflow> out;
  int idx = 0;

  // 非 p 叶的索引列表
  auto complement = [](std::uint8_t p) {
    std::vector<std::uint8_t> c;
    for (std::uint8_t i = 0; i < kLeaves; ++i)
      if (i != p) c.push_back(i);
    return c;
  };

  // Small：5 亮叶位置 x 其余 4 叶的 16 种激活子集 = 80
  for (std::uint8_t p = 0; p < kLeaves; ++p) {
    const auto comp = complement(p);
    for (int mask = 0; mask < 16; ++mask)
      out.push_back(makeWorkflow(idx++, RuneMode::Small, {p},
                                 subsetFromMask(comp, mask)));
  }
  // Large 单亮叶：80
  for (std::uint8_t p = 0; p < kLeaves; ++p) {
    const auto comp = complement(p);
    for (int mask = 0; mask < 16; ++mask)
      out.push_back(makeWorkflow(idx++, RuneMode::Large, {p},
                                 subsetFromMask(comp, mask)));
  }
  // Large 双亮叶：10 对 x 其余 3 叶的 8 种激活子集 = 80
  for (std::uint8_t p = 0; p < kLeaves; ++p) {
    for (std::uint8_t q = static_cast<std::uint8_t>(p + 1); q < kLeaves; ++q) {
      std::vector<std::uint8_t> comp;
      for (std::uint8_t i = 0; i < kLeaves; ++i)
        if (i != p && i != q) comp.push_back(i);
      for (int mask = 0; mask < 8; ++mask)
        out.push_back(makeWorkflow(idx++, RuneMode::Large, {p, q},
                                   subsetFromMask(comp, mask)));
    }
  }
  // 完成态：每模式 1 个
  out.push_back(makeCompletedWorkflow(idx++, RuneMode::Small));
  out.push_back(makeCompletedWorkflow(idx++, RuneMode::Large));
  return out;
}

std::string wfName(int index) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%03d", index);
  return buf;
}

void printWorkflow(const Workflow& w) {
  std::printf("wf%03d mode=%s pending=[%s] activated=[%s] expect=[%u,%u,%u,%u,%u]\n",
              w.index, w.mode == RuneMode::Small ? "small" : "large",
              [&]() { std::string s; for (auto v : w.pending) s += char('0' + v); return s; }().c_str(),
              [&]() { std::string s; for (auto v : w.activated) s += char('0' + v); return s; }().c_str(),
              w.want_acts[0], w.want_acts[1], w.want_acts[2], w.want_acts[3],
              w.want_acts[4]);
}

int countWorkflowFrames(const std::string& out_dir, int index) {
  const std::string suffix = "_wf" + wfName(index) + ".png";
  int n = 0;
  std::error_code ec;
  for (const auto& e : std::filesystem::directory_iterator(out_dir, ec)) {
    if (ec) break;
    const std::string nm = e.path().filename().string();
    if (nm.size() >= suffix.size() &&
        nm.compare(nm.size() - suffix.size(), suffix.size(), suffix) == 0)
      ++n;
  }
  return n;
}

// 读最新真值里第一个符的模式与 5 叶激活值。返回 false = 读不到。
bool readRuneTruth(TalosMetadataMapping& mapping, std::uint8_t* mode, std::uint8_t* acts) {
  auto reader = mapping.reader();
  if (!reader.ok()) return false;
  const auto truth = reader.value->readLatestGroundTruth();
  if (!truth.ok()) return false;
  const auto& batch = *truth.value;
  if (batch.rune_count == 0) return false;
  *mode = batch.runes[0].rune_mode;
  for (int i = 0; i < kLeaves; ++i) acts[i] = batch.runes[0].target_activations[i];
  return true;
}

bool actsMatch(const Workflow& w, const std::uint8_t* acts) {
  for (int i = 0; i < kLeaves; ++i)
    if (static_cast<int>(acts[i]) != static_cast<int>(w.want_acts[i])) return false;
  return true;
}

// 等真值与工况一致（轮询 --timeout 秒）。返回是否达成。
bool waitForTruth(const Workflow& w, TalosMetadataMapping& mapping, double timeout_s) {
  const auto t0 = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - t0 <
         std::chrono::seconds(1) * static_cast<long long>(timeout_s)) {
    std::uint8_t mode = 255, acts[kLeaves] = {0};
    if (readRuneTruth(mapping, &mode, acts)) {
      const std::uint8_t want = w.mode == RuneMode::Small ? 0 : 1;
      if (mode == want && actsMatch(w, acts)) return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  return false;
}

int runExport(const Options& opts, const Workflow& w) {
  const std::string per_roll = std::to_string(opts.per_roll);
  const std::string wf = wfName(w.index);
  char acts_arg[32];
  std::snprintf(acts_arg, sizeof(acts_arg), "%u,%u,%u,%u,%u", w.want_acts[0],
                w.want_acts[1], w.want_acts[2], w.want_acts[3], w.want_acts[4]);
  const std::string glb = exeDir() + "/../assets/POWER.glb";
  const char* argv[] = {opts.export_bin.c_str(), "--out",       opts.out_dir.c_str(),
                        "--per-roll",           per_roll.c_str(), "--workflow",
                        wf.c_str(),             "--expect-acts", acts_arg,
                        "--glb",                glb.c_str(),    "--img-net",
                        nullptr};
  const pid_t pid = fork();
  if (pid < 0) {
    std::fprintf(stderr, "fork failed\n");
    return -1;
  }
  if (pid == 0) {
    // 关闭从父进程继承的 SDK 连接句柄（ContestClient/SceneControlClient 的
    // UDP socket 与 metadata 映射），否则子进程的图像流收不到帧
    for (int fd = 3; fd < 256; ++fd) ::close(fd);
    execvp(argv[0], const_cast<char* const*>(argv));
    std::fprintf(stderr, "exec %s failed\n", argv[0]);
    _exit(127);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

}  // namespace

int main(int argc, char** argv) {
  Options opts;
  if (!parseOptions(argc, argv, &opts)) return 0;
  if (opts.ipc_dir.empty()) opts.ipc_dir = defaultIpcDir();

  // 相对路径（如 ./export_labeled_frames）按本程序所在目录解析，与启动目录无关
  const std::string exe_dir = exeDir();
  if (!opts.export_bin.empty() && opts.export_bin[0] != '/')
    opts.export_bin = exe_dir + "/" + opts.export_bin;
  if (!opts.out_dir.empty() && opts.out_dir[0] != '/')
    opts.out_dir = std::filesystem::absolute(opts.out_dir).string();

  const std::vector<Workflow> workflows = buildWorkflows();
  std::printf("工况总数: %zu（Small 81 + Large 161；全枚举 242）\n", workflows.size());

  if (opts.dry_run) {
    for (const auto& w : workflows) printWorkflow(w);
    return 0;
  }

  TalosMetadataMapping mapping;
  const ClientStatus open_status = mapping.open(opts.ipc_dir + "/talos_ipc_meta");
  if (!open_status.ok()) {
    std::fprintf(stderr, "metadata open failed: %s\n", open_status.message.c_str());
    return 1;
  }

  ContestClientOptions options;
  options.ipc_directory = opts.ipc_dir;
  ContestClient client(options);
  const ClientStatus conn = client.connect();
  if (!conn.ok()) { std::fprintf(stderr, "connect failed: %s\n", conn.message.c_str()); return 1; }
  const auto scene = client.selectScene(ContestScene::Energy);
  if (!scene.ok()) { std::fprintf(stderr, "selectScene failed: %s\n", scene.status.message.c_str()); return 1; }
  // ContestClient 的 connect 会订阅 5602 图像流，且父进程自己从不取帧，
  // 会抢占 fork 出的 export 子进程的图像流（子进程连上后收不到帧）。
  // 场景已选定，此后不再需要该连接。
  client.close();

  SceneControlOptions sc_opts;
  sc_opts.session_id = "sample-workflows-" + std::to_string(::getpid());
  SceneControlClient sc(sc_opts);
  const auto session = sc.createSession();
  if (!session.ok()) { std::fprintf(stderr, "createSession failed: %s\n", session.status.message.c_str()); return 1; }

  std::filesystem::create_directories(opts.out_dir);

  const int total = opts.limit > 0
                        ? std::min<int>(opts.limit, static_cast<int>(workflows.size()) - opts.start)
                        : static_cast<int>(workflows.size()) - opts.start;
  int done = 0, skipped = 0, failed = 0;
  RuneMode cur_mode = RuneMode::Off;
  const auto t_all0 = std::chrono::steady_clock::now();

  for (int i = opts.start; i < static_cast<int>(workflows.size()); ++i) {
    if (opts.limit > 0 && i - opts.start >= opts.limit) break;
    const Workflow& w = workflows[i];

    if (countWorkflowFrames(opts.out_dir, w.index) >= opts.per_roll) {
      std::printf("[%3d/%3d] wf%03d 已有 %d 帧，跳过（断点续跑）\n", i - opts.start + 1, total,
                  w.index, opts.per_roll);
      ++skipped;
      continue;
    }

    // 模式变化时重设场景（RuleDriven 旋转）
    if (cur_mode != w.mode) {
      RuneScenario scenario;
      scenario.mode = w.mode;
      scenario.motion = RuneMotion::RuleDriven;
      scenario.red_face_direction = RuneDirection::Clockwise;
      scenario.leaf_states.clear();
      const auto resp = sc.setRuneScenario(scenario);
      if (!resp.ok()) {
        std::fprintf(stderr, "wf%03d setRuneScenario failed: %s\n", w.index,
                     resp.status.message.c_str());
        ++failed;
        continue;
      }
      cur_mode = w.mode;
      std::this_thread::sleep_for(std::chrono::seconds(2));  // 场景重建稳定期
    }

    RuneState st;
    st.mode = w.mode;
    st.pending_targets = w.pending;
    st.activated_targets = w.activated;
    bool applied = false;
    for (int attempt = 0; attempt < 2 && !applied; ++attempt) {
      const auto resp = sc.setRuneState(st);
      if (!resp.ok()) {
        std::fprintf(stderr, "wf%03d setRuneState failed: %s\n", w.index,
                     resp.status.message.c_str());
        break;
      }
      applied = waitForTruth(w, mapping, 8.0);
      if (!applied) std::fprintf(stderr, "wf%03d 真值未达成，重试 setRuneState\n", w.index);
    }
    if (!applied) {
      std::fprintf(stderr, "wf%03d 状态未生效，跳过\n", w.index);
      ++failed;
      continue;
    }

    std::printf("[%3d/%3d] wf%03d 状态生效，开始采样（每圈 %d 帧）...\n", i - opts.start + 1,
                total, w.index, opts.per_roll);
    const int rc = runExport(opts, w);
    if (rc != 0) {
      std::fprintf(stderr, "wf%03d 采样失败（export 退出码 %d），跳过\n", w.index, rc);
      ++failed;
    } else {
      ++done;
    }
  }

  // 结束：恢复 Large 规则驱动、全不亮
  {
    RuneScenario scenario;
    scenario.mode = RuneMode::Large;
    scenario.motion = RuneMotion::RuleDriven;
    scenario.red_face_direction = RuneDirection::Clockwise;
    scenario.leaf_states.clear();
    const auto r1 = sc.setRuneScenario(scenario);
    if (!r1.ok()) std::fprintf(stderr, "restore scenario failed: %s\n", r1.status.message.c_str());
    RuneState st;
    st.mode = RuneMode::Large;
    st.pending_targets.clear();
    st.activated_targets.clear();
    const auto r2 = sc.setRuneState(st);
    if (!r2.ok()) std::fprintf(stderr, "restore state failed: %s\n", r2.status.message.c_str());
  }

  const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all0).count();
  std::printf("采样结束：完成 %d、跳过（已有数据）%d、失败 %d，用时 %.1f 分钟\n",
              done, skipped, failed, dt / 60.0);
  return failed > 0 ? 1 : 0;
}

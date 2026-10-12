// 自动标注采集：抓取模拟器相机帧 + 按帧序对齐真值。
// 关键点 = POWER.glb 已知 3D 几何（叶板/三叉戟顶点、R 标中心）
//         + 真值 current_angle 绕符轴旋转 + 相机位姿/内参成像投影。
// 坐标写入 640x480 网络输入空间（labels.jsonl）。
//
// 用法: export_labeled_frames --out DIR [--count N] [选项]
//
// 类别语义（已与用户确认）：
//   class 0 = 亮起的叶片（act==1；act==0 未激活不亮，不标——用户看 debug 图后校正）
//   class 1 = 小符已激活（Small 且 act==2/3）
//   class 2 = 大符已激活三叉戟（Large 且 act==2；Large+Completed 不标记）
//   其余状态不标记。
//
// 关键点摆法：
//   class 0/1 叶板: 0=离R最远边缘中点 4=离R最近边缘中点 1=左边缘中点 3=右边缘中点
//   class 2 三叉戟: 0=叉尖最左 4=叉尖最右 1=横杠最左 3=横杠最右
//   2 均为 R 标中心；左右以该面亮起侧为基准（两符面互为镜像，各自取自己的亮起侧）。
//   相机处于符面亮起侧背面时不标注该面（该面不可见）。
#include <daedalus_sim_sdk/talos_metadata_reader.hpp>
#include <daedalus_sim_sdk/tcp_image_client.hpp>

#include <png.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

namespace {

using namespace daedalus::sim::sdk::v1;

volatile std::sig_atomic_t g_stop = 0;
void onSignal(int) { g_stop = 1; }

constexpr double kNetScale = 640.0 / 1440.0;  // 1440x1080 -> 640x480
constexpr double kPi = 3.14159265358979323846;
constexpr int kTargetCount = 5;

struct Options {
  std::string out_dir = "frames_labeled";
  std::string ipc_dir;
  std::string glb_path = "../assets/POWER.glb";
  bool glb_explicit = false;
  long count = 0;            // 0 = 直到 Ctrl+C
  long per_roll = 0;         // >0：按真值角度每转 2π/per_roll 存一帧，转满一圈停止
  std::string workflow;      // 工况编号：写入文件名与标签行 workflow 字段
  bool img_net = false;      // 存 640x480 缩放图（标签坐标即网络空间）
  std::string expect_acts;   // "1,2,2,0,0"：per-roll 采样前等真值达到该 5 叶状态
                             // （跳过 setRuneState 前的旧缓冲帧）
  std::uint8_t expect_acts_v[5] = {0, 0, 0, 0, 0};
  bool has_expect_acts = false;
  bool debug_draw = false;   // 输出叠加关键点的 _debug.png
  bool self_test = false;    // 只解析 glb 打印几何后退出
  std::string host = "127.0.0.1";
  std::uint16_t port = daedalus::sim::sdk::v1::kTcpImagePort;
};

std::string defaultIpcDir() {
  const char* xdg = std::getenv("XDG_RUNTIME_DIR");
  const std::string base = xdg != nullptr ? xdg : "/tmp";
  return base + "/daedalus-learning-" + std::to_string(::getuid());
}

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
    else if (arg == "--glb") { o->glb_path = value(); o->glb_explicit = true; }
    else if (arg == "--count") o->count = std::strtol(value(), nullptr, 10);
    else if (arg == "--per-roll") o->per_roll = std::strtol(value(), nullptr, 10);
    else if (arg == "--workflow") o->workflow = value();
    else if (arg == "--expect-acts") {
      o->expect_acts = value();
      // 解析 "1,2,2,0,0"
      std::string s = o->expect_acts;
      int idx = 0;
      for (std::size_t i = 0; i < s.size() && idx < 5; ++i) {
        if (s[i] == ',') { ++idx; continue; }
        if (s[i] < '0' || s[i] > '9') { std::fprintf(stderr, "--expect-acts 格式错误\n"); std::exit(2); }
        o->expect_acts_v[idx] = static_cast<std::uint8_t>(s[i] - '0');
      }
      o->has_expect_acts = true;
    }
    else if (arg == "--img-net") o->img_net = true;
    else if (arg == "--debug-draw") o->debug_draw = true;
    else if (arg == "--self-test") o->self_test = true;
    else if (arg == "--host") o->host = value();
    else if (arg == "--port") o->port = static_cast<std::uint16_t>(std::strtol(value(), nullptr, 10));
    else if (arg == "--help" || arg == "-h") {
      std::printf(
          "Usage: export_labeled_frames --out DIR [--count N] [options]\n"
          "  抓帧 + 真值投影 -> RGB PNG + labels.jsonl（640x480 坐标）\n"
          "  --ipc-dir PATH    真值 IPC 目录（默认 /tmp/daedalus-learning-<uid>）\n"
          "  --glb PATH        POWER.glb 路径（默认 ../assets/POWER.glb）\n"
          "  --per-roll N      按真值角度采样：每转 2*pi/N 存一帧，转满一圈退出\n"
          "  --workflow STR    工况编号：写入文件名（_wf<STR>）与标签行 workflow 字段\n"
          "  --expect-acts S   如 \"1,2,2,0,0\"：per-roll 模式先等真值 5 叶达到该状态\n"
          "                    （跳过状态切换前的旧缓冲帧）再开始计角度\n"
          "  --img-net         存 640x480 缩放图（标签坐标即网络空间；默认存原尺寸）\n"
          "  --debug-draw      额外输出叠加角点的 _debug.png 供校准\n"
          "  --self-test       只解析 glb 打印几何常量后退出\n");
      return false;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      std::exit(2);
    }
  }
  return true;
}

// ---------------- 基础数学 ----------------

struct Mat4 {
  double m[4][4];  // 行主序: m[row][col]
};

Mat4 mat4Identity() {
  Mat4 r{};
  for (int i = 0; i < 4; ++i) r.m[i][i] = 1.0;
  return r;
}

Mat4 mat4Mul(const Mat4& a, const Mat4& b) {
  Mat4 r{};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      for (int k = 0; k < 4; ++k) r.m[i][j] += a.m[i][k] * b.m[k][j];
  return r;
}

void transformPoint(const Mat4& w, const double p[3], double out[3]) {
  for (int i = 0; i < 3; ++i)
    out[i] = w.m[i][0] * p[0] + w.m[i][1] * p[1] + w.m[i][2] * p[2] + w.m[i][3];
}

double dot3(const double a[3], const double b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void cross3(const double a[3], const double b[3], double out[3]) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

double norm3(const double v[3]) {
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

bool normalize3(double v[3]) {
  const double len = norm3(v);
  if (len < 1e-12) return false;
  v[0] /= len; v[1] /= len; v[2] /= len;
  return true;
}

// 绕单位轴 a（过原点 o）按右手定则旋转角 angle，原地修改 p
void rotateAroundAxis(const double a[3], const double o[3], double angle, double p[3]) {
  double v[3] = {p[0] - o[0], p[1] - o[1], p[2] - o[2]};
  const double c = std::cos(angle), s = std::sin(angle);
  const double d = dot3(a, v);
  double cr[3];
  cross3(a, v, cr);
  for (int i = 0; i < 3; ++i)
    p[i] = o[i] + v[i] * c + cr[i] * s + a[i] * d * (1.0 - c);
}

// 平面内旋转：u' = c*u + s*w, w' = -s*u + c*w（逆变换 unrot2 取 -s）
void rot2(double c, double s, double& uu, double& ww) {
  const double u2 = c * uu + s * ww;
  const double w2 = -s * uu + c * ww;
  uu = u2;
  ww = w2;
}

void unrot2(double c, double s, double& uu, double& ww) {
  const double u2 = c * uu - s * ww;
  const double w2 = s * uu + c * ww;
  uu = u2;
  ww = w2;
}

// odom <-> 世界（Bevy Y-up）：
//   odom = (-z_world, -x_world, y_world)；逆：world = (y_odom, z_odom, -x_odom)
void worldToOdom(const double w[3], double o[3]) {
  o[0] = -w[2];
  o[1] = -w[0];
  o[2] = w[1];
}

void odomToWorld(const double o[3], double w[3]) {
  w[0] = -o[1];
  w[1] = o[2];
  w[2] = -o[0];
}

// ---------------- 最小 JSON 解析器（glTF JSON chunk 用） ----------------

struct Json {
  enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
  bool boolean = false;
  double num = 0.0;
  std::string str;
  std::vector<Json> arr;
  std::vector<std::pair<std::string, Json>> obj;

  const Json* find(const char* key) const {
    if (type != Obj) return nullptr;
    for (const auto& kv : obj)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
  double numOr(double def) const { return type == Num ? num : def; }
  long long intOr(long long def) const { return type == Num ? static_cast<long long>(num) : def; }
};

class JsonParser {
 public:
  JsonParser(const char* begin, const char* end) : cur_(begin), end_(end) {}

  bool parse(Json* out) {
    skipWs();
    return parseValue(out);
  }

 private:
  const char* cur_;
  const char* end_;

  void skipWs() {
    while (cur_ < end_ && (*cur_ == ' ' || *cur_ == '\t' || *cur_ == '\n' || *cur_ == '\r')) ++cur_;
  }

  bool match(const char* lit) {
    const std::size_t n = std::strlen(lit);
    if (static_cast<std::size_t>(end_ - cur_) < n || std::memcmp(cur_, lit, n) != 0) return false;
    cur_ += n;
    return true;
  }

  bool parseValue(Json* out) {
    if (cur_ >= end_) return false;
    const char c = *cur_;
    if (c == '{') return parseObject(out);
    if (c == '[') return parseArray(out);
    if (c == '"') {
      out->type = Json::Str;
      return parseString(&out->str);
    }
    if (c == 't') {
      if (!match("true")) return false;
      out->type = Json::Bool;
      out->boolean = true;
      return true;
    }
    if (c == 'f') {
      if (!match("false")) return false;
      out->type = Json::Bool;
      return true;
    }
    if (c == 'n') {
      if (!match("null")) return false;
      out->type = Json::Null;
      return true;
    }
    return parseNumber(out);
  }

  bool parseNumber(Json* out) {
    const char* s = cur_;
    while (cur_ < end_ &&
           (std::isdigit(static_cast<unsigned char>(*cur_)) || *cur_ == '-' || *cur_ == '+' ||
            *cur_ == '.' || *cur_ == 'e' || *cur_ == 'E'))
      ++cur_;
    if (cur_ == s) return false;
    out->type = Json::Num;
    out->num = std::strtod(std::string(s, cur_ - s).c_str(), nullptr);
    return true;
  }

  static unsigned hex4(const char* p) {
    unsigned v = 0;
    for (int i = 0; i < 4; ++i) {
      v <<= 4;
      const char c = p[i];
      if (c >= '0' && c <= '9') v |= static_cast<unsigned>(c - '0');
      else if (c >= 'a' && c <= 'f') v |= static_cast<unsigned>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= static_cast<unsigned>(c - 'A' + 10);
    }
    return v;
  }

  static void appendUtf8(std::string* out, unsigned cp) {
    if (cp < 0x80) {
      out->push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
      out->push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      out->push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out->push_back(static_cast<char>(0xF0 | (cp >> 18)));
      out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }

  bool parseString(std::string* out) {
    if (*cur_ != '"') return false;
    ++cur_;
    out->clear();
    while (cur_ < end_) {
      const char c = *cur_++;
      if (c == '"') return true;
      if (c != '\\') {
        out->push_back(c);
        continue;
      }
      if (cur_ >= end_) return false;
      const char e = *cur_++;
      switch (e) {
        case '"': out->push_back('"'); break;
        case '\\': out->push_back('\\'); break;
        case '/': out->push_back('/'); break;
        case 'b': out->push_back('\b'); break;
        case 'f': out->push_back('\f'); break;
        case 'n': out->push_back('\n'); break;
        case 'r': out->push_back('\r'); break;
        case 't': out->push_back('\t'); break;
        case 'u': {
          if (cur_ + 4 > end_) return false;
          unsigned cp = hex4(cur_);
          cur_ += 4;
          if (cp >= 0xD800 && cp <= 0xDBFF && cur_ + 6 <= end_ && cur_[0] == '\\' && cur_[1] == 'u') {
            const unsigned lo = hex4(cur_ + 2);
            if (lo >= 0xDC00 && lo <= 0xDFFF) {
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              cur_ += 6;
            }
          }
          appendUtf8(out, cp);
          break;
        }
        default: return false;
      }
    }
    return false;
  }

  bool parseArray(Json* out) {
    ++cur_;  // '['
    out->type = Json::Arr;
    skipWs();
    if (cur_ < end_ && *cur_ == ']') {
      ++cur_;
      return true;
    }
    for (;;) {
      Json v;
      if (!parseValue(&v)) return false;
      out->arr.push_back(std::move(v));
      skipWs();
      if (cur_ >= end_) return false;
      if (*cur_ == ',') {
        ++cur_;
        continue;
      }
      if (*cur_ == ']') {
        ++cur_;
        return true;
      }
      return false;
    }
  }

  bool parseObject(Json* out) {
    ++cur_;  // '{'
    out->type = Json::Obj;
    skipWs();
    if (cur_ < end_ && *cur_ == '}') {
      ++cur_;
      return true;
    }
    for (;;) {
      skipWs();
      std::string key;
      if (!parseString(&key)) return false;
      skipWs();
      if (cur_ >= end_ || *cur_ != ':') return false;
      ++cur_;
      Json v;
      if (!parseValue(&v)) return false;
      out->obj.emplace_back(std::move(key), std::move(v));
      skipWs();
      if (cur_ >= end_) return false;
      if (*cur_ == ',') {
        ++cur_;
        continue;
      }
      if (*cur_ == '}') {
        ++cur_;
        return true;
      }
      return false;
    }
  }
};

// ---------------- glb 读取 ----------------

struct GlbFile {
  bool ok = false;
  std::string owned;  // 持有文件内容（bin 指针指向其内部）
  Json root;
  const std::uint8_t* bin = nullptr;
  std::size_t bin_len = 0;
};

bool loadGlb(const std::string& path, GlbFile* out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (data.size() < 20 || std::memcmp(data.data(), "glTF", 4) != 0) return false;
  std::size_t off = 12;
  if (off + 8 > data.size()) return false;
  const std::uint32_t json_len = *reinterpret_cast<const std::uint32_t*>(data.data() + off);
  off += 8;
  if (off + json_len > data.size()) return false;
  JsonParser parser(data.data() + off, data.data() + off + json_len);
  if (!parser.parse(&out->root)) return false;
  off += json_len;
  if (off + 8 > data.size()) return false;
  const std::uint32_t bin_len32 = *reinterpret_cast<const std::uint32_t*>(data.data() + off);
  off += 8;
  if (off + bin_len32 > data.size()) return false;
  out->owned = std::move(data);
  out->bin = reinterpret_cast<const std::uint8_t*>(out->owned.data()) + off;
  out->bin_len = bin_len32;
  out->ok = true;
  return true;
}

const Json* jget(const Json& obj, const char* key) { return obj.find(key); }

long long jint(const Json* v, long long def) { return v ? v->intOr(def) : def; }

std::string nameOf(const Json& node) {
  const Json* nm = jget(node, "name");
  return nm && nm->type == Json::Str ? nm->str : std::string();
}

// 节点局部矩阵：matrix 优先，否则 T*R*S
Mat4 nodeLocalMatrix(const Json& n) {
  const Json* m = jget(n, "matrix");
  if (m && m->type == Json::Arr && m->arr.size() == 16) {
    Mat4 r{};
    for (int c = 0; c < 4; ++c)
      for (int row = 0; row < 4; ++row) r.m[row][c] = m->arr[c * 4 + row].numOr(0.0);
    return r;
  }
  Mat4 t = mat4Identity(), r = mat4Identity(), s = mat4Identity();
  const Json* tv = jget(n, "translation");
  if (tv && tv->type == Json::Arr && tv->arr.size() >= 3)
    for (int i = 0; i < 3; ++i) t.m[i][3] = tv->arr[i].numOr(0.0);
  const Json* q = jget(n, "rotation");
  if (q && q->type == Json::Arr && q->arr.size() >= 4) {
    const double x = q->arr[0].numOr(0.0), y = q->arr[1].numOr(0.0);
    const double z = q->arr[2].numOr(0.0), w = q->arr[3].numOr(0.0);
    r.m[0][0] = 1 - 2 * (y * y + z * z);
    r.m[0][1] = 2 * (x * y - z * w);
    r.m[0][2] = 2 * (x * z + y * w);
    r.m[1][0] = 2 * (x * y + z * w);
    r.m[1][1] = 1 - 2 * (x * x + z * z);
    r.m[1][2] = 2 * (y * z - x * w);
    r.m[2][0] = 2 * (x * z - y * w);
    r.m[2][1] = 2 * (y * z + x * w);
    r.m[2][2] = 1 - 2 * (x * x + y * y);
  }
  const Json* sv = jget(n, "scale");
  if (sv && sv->type == Json::Arr && sv->arr.size() >= 3)
    for (int i = 0; i < 3; ++i) s.m[i][i] = sv->arr[i].numOr(1.0);
  return mat4Mul(mat4Mul(t, r), s);
}

struct Verts {
  std::vector<double> x, y, z;
  std::size_t size() const { return x.size(); }
  void add(double px, double py, double pz) {
    x.push_back(px);
    y.push_back(py);
    z.push_back(pz);
  }
};

// 读取 mesh 的 POSITION 顶点并变换到世界坐标（追加到 out）
bool loadPositions(const Json& root, const std::uint8_t* bin, long long mesh_idx,
                   const Mat4& world, Verts* out) {
  const Json* meshes = jget(root, "meshes");
  if (!meshes || mesh_idx < 0 || static_cast<std::size_t>(mesh_idx) >= meshes->arr.size())
    return false;
  const Json& mesh = meshes->arr[static_cast<std::size_t>(mesh_idx)];
  const Json* prims = jget(mesh, "primitives");
  const Json* accs = jget(root, "accessors");
  const Json* bvs = jget(root, "bufferViews");
  if (!prims || !accs || !bvs) return false;
  bool any = false;
  for (const Json& prim : prims->arr) {
    const Json* attrs = jget(prim, "attributes");
    if (!attrs) continue;
    const Json* pos = jget(*attrs, "POSITION");
    if (!pos || pos->type != Json::Num) continue;
    const long long acc_idx = pos->intOr(-1);
    if (acc_idx < 0 || static_cast<std::size_t>(acc_idx) >= accs->arr.size()) continue;
    const Json& acc = accs->arr[static_cast<std::size_t>(acc_idx)];
    const long long bv_idx = jint(jget(acc, "bufferView"), -1);
    const long long byte_off = jint(jget(acc, "byteOffset"), 0);
    const long long count = jint(jget(acc, "count"), 0);
    const long long ct = jint(jget(acc, "componentType"), 0);
    if (bv_idx < 0 || static_cast<std::size_t>(bv_idx) >= bvs->arr.size()) continue;
    const Json& bv = bvs->arr[static_cast<std::size_t>(bv_idx)];
    const long long buf_off = jint(jget(bv, "byteOffset"), 0);
    long long stride = jint(jget(bv, "byteStride"), 0);
    if (stride == 0) stride = (ct == 5126) ? 12 : 6;
    if (bin == nullptr || buf_off < 0 || byte_off < 0 || count < 0) continue;
    const std::uint8_t* base = bin + buf_off + byte_off;
    for (long long i = 0; i < count; ++i) {
      const std::uint8_t* p = base + i * stride;
      double v[3];
      if (ct == 5126) {
        float fv[3];
        std::memcpy(fv, p, sizeof(fv));
        v[0] = fv[0]; v[1] = fv[1]; v[2] = fv[2];
      } else if (ct == 5123) {
        std::uint16_t uv[3];
        std::memcpy(uv, p, sizeof(uv));
        v[0] = uv[0]; v[1] = uv[1]; v[2] = uv[2];
      } else {
        continue;
      }
      double wv[3];
      transformPoint(world, v, wv);
      out->add(wv[0], wv[1], wv[2]);
      any = true;
    }
  }
  return any;
}

// ---------------- 符面几何 ----------------

struct TargetGeom {
  double phi = 0.0;   // 叶臂在符面内的方位角（相对 TARGET_1）
  double plate_u_min = 0, plate_u_max = 0;
  double plate_w_min = 0, plate_w_max = 0;
  double plate_n_mid = 0;
  double tri_n_mid = 0;
  double k0[2] = {0, 0}, k4[2] = {0, 0}, k1[2] = {0, 0}, k3[2] = {0, 0};  // 三叉戟关键点（符面 u,w）
  double tri_verts[12][2] = {};  // 三叉戟全部顶点（符面 u,w，调试叠加用）
  int tri_vert_count = 0;
};

struct FaceGeom {
  double origin[3] = {0, 0, 0};   // 世界坐标（Bevy Y-up）
  double axis[3] = {0, 0, 0};     // 旋转轴（世界，单位）
  double u[3] = {0, 0, 0};        // 符面径向基
  double w[3] = {0, 0, 0};
  double n[3] = {0, 0, 0};        // 符面法向（≈轴）
  int lit_sign = 1;               // 亮起侧方向 = lit_sign * n
  double r_center[3] = {0, 0, 0}; // R 标中心（符面 u,w,n 坐标）
  TargetGeom targets[kTargetCount];
};

void toFaceCoords(const FaceGeom& f, const double p[3], double out[3]) {
  const double d[3] = {p[0] - f.origin[0], p[1] - f.origin[1], p[2] - f.origin[2]};
  out[0] = dot3(f.u, d);
  out[1] = dot3(f.w, d);
  out[2] = dot3(f.n, d);
}

// 符面 (u,w,n) -> 世界
void facePointWorld(const FaceGeom& f, double uu, double ww, double nn, double out[3]) {
  for (int i = 0; i < 3; ++i)
    out[i] = f.origin[i] + uu * f.u[i] + ww * f.w[i] + nn * f.n[i];
}

// 由三角形绕序求 mesh 朝外法向，与面法向点积取符号（退化时用 R 侧兜底）。
// 注意：多 primitive 的 mesh 索引相对各自 POSITION accessor，必须逐 primitive 累计叉积。
int litSignFromWinding(const Json& root, const std::uint8_t* bin, long long mesh_idx,
                       const Mat4& world, const double n[3], double r_center_n) {
  const Json* meshes = jget(root, "meshes");
  const Json* accs = jget(root, "accessors");
  const Json* bvs = jget(root, "bufferViews");
  if (!meshes || !accs || !bvs || mesh_idx < 0 ||
      static_cast<std::size_t>(mesh_idx) >= meshes->arr.size())
    return r_center_n > 0 ? 1 : -1;
  const Json& mesh = meshes->arr[static_cast<std::size_t>(mesh_idx)];
  const Json* prims = jget(mesh, "primitives");
  if (!prims) return r_center_n > 0 ? 1 : -1;
  double sum[3] = {0, 0, 0};
  bool any = false;
  for (const Json& prim : prims->arr) {
    const Json* attrs = jget(prim, "attributes");
    const Json* idxj = jget(prim, "indices");
    if (!attrs || !idxj || idxj->type != Json::Num) continue;
    const Json* pos = jget(*attrs, "POSITION");
    if (!pos || pos->type != Json::Num) continue;

    const long long acc_idx = pos->intOr(-1);
    if (acc_idx < 0 || static_cast<std::size_t>(acc_idx) >= accs->arr.size()) continue;
    const Json& acc = accs->arr[static_cast<std::size_t>(acc_idx)];
    const long long bv_idx = jint(jget(acc, "bufferView"), -1);
    const long long byte_off = jint(jget(acc, "byteOffset"), 0);
    const long long count = jint(jget(acc, "count"), 0);
    const long long ct = jint(jget(acc, "componentType"), 0);
    if (bv_idx < 0 || static_cast<std::size_t>(bv_idx) >= bvs->arr.size()) continue;
    const Json& bv = bvs->arr[static_cast<std::size_t>(bv_idx)];
    const long long buf_off = jint(jget(bv, "byteOffset"), 0);
    long long stride = jint(jget(bv, "byteStride"), 0);
    if (stride == 0) stride = (ct == 5126) ? 12 : 6;
    if (buf_off < 0 || byte_off < 0 || count < 0) continue;
    const std::uint8_t* base = bin + buf_off + byte_off;
    std::vector<double> px, py, pz;
    px.reserve(static_cast<std::size_t>(count));
    py.reserve(static_cast<std::size_t>(count));
    pz.reserve(static_cast<std::size_t>(count));
    for (long long i = 0; i < count; ++i) {
      const std::uint8_t* q = base + i * stride;
      double v[3];
      if (ct == 5126) {
        float fv[3];
        std::memcpy(fv, q, sizeof(fv));
        v[0] = fv[0]; v[1] = fv[1]; v[2] = fv[2];
      } else if (ct == 5123) {
        std::uint16_t uv[3];
        std::memcpy(uv, q, sizeof(uv));
        v[0] = uv[0]; v[1] = uv[1]; v[2] = uv[2];
      } else {
        continue;
      }
      double wv[3];
      transformPoint(world, v, wv);
      px.push_back(wv[0]); py.push_back(wv[1]); pz.push_back(wv[2]);
    }

    const long long i_acc = idxj->intOr(-1);
    if (i_acc < 0 || static_cast<std::size_t>(i_acc) >= accs->arr.size()) continue;
    const Json& iacc = accs->arr[static_cast<std::size_t>(i_acc)];
    const long long ibv_idx = jint(jget(iacc, "bufferView"), -1);
    const long long ibyte_off = jint(jget(iacc, "byteOffset"), 0);
    const long long icount = jint(jget(iacc, "count"), 0);
    const long long ict = jint(jget(iacc, "componentType"), 0);
    if (ibv_idx < 0 || static_cast<std::size_t>(ibv_idx) >= bvs->arr.size()) continue;
    const Json& ibv = bvs->arr[static_cast<std::size_t>(ibv_idx)];
    const long long ibuf_off = jint(jget(ibv, "byteOffset"), 0);
    long long istride = jint(jget(ibv, "byteStride"), 0);
    if (istride == 0) istride = (ict == 5125) ? 4 : 2;
    if (ibuf_off < 0 || ibyte_off < 0 || icount < 0) continue;
    const std::uint8_t* ibase = bin + ibuf_off + ibyte_off;
    for (long long i = 0; i + 2 < icount; i += 3) {
      std::uint32_t t[3];
      bool bad = false;
      for (int k = 0; k < 3; ++k) {
        const std::uint8_t* q = ibase + (i + k) * istride;
        if (ict == 5123) {
          std::uint16_t v16;
          std::memcpy(&v16, q, sizeof(v16));
          t[k] = v16;
        } else if (ict == 5125) {
          std::uint32_t v32;
          std::memcpy(&v32, q, sizeof(v32));
          t[k] = v32;
        } else {
          bad = true;
          break;
        }
      }
      if (bad || t[0] >= px.size() || t[1] >= px.size() || t[2] >= px.size()) continue;
      const double a[3] = {px[t[0]], py[t[0]], pz[t[0]]};
      const double b[3] = {px[t[1]], py[t[1]], pz[t[1]]};
      const double c[3] = {px[t[2]], py[t[2]], pz[t[2]]};
      const double ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
      const double ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
      double cr[3];
      cross3(ab, ac, cr);
      sum[0] += cr[0]; sum[1] += cr[1]; sum[2] += cr[2];
      any = true;
    }
  }
  if (!any) return r_center_n > 0 ? 1 : -1;
  if (normalize3(sum)) return dot3(sum, n) > 0 ? 1 : -1;
  return r_center_n > 0 ? 1 : -1;
}

struct GlbScene {
  const Json* nodes = nullptr;
  std::vector<Mat4> world;
  std::vector<char> visited;
};

bool buildScene(const GlbFile& glb, GlbScene* scene, std::string* err) {
  scene->nodes = jget(glb.root, "nodes");
  if (!scene->nodes || scene->nodes->type != Json::Arr || scene->nodes->arr.empty()) {
    *err = "glb has no nodes";
    return false;
  }
  const std::size_t n = scene->nodes->arr.size();
  scene->world.assign(n, Mat4{});
  std::vector<char> visited(n, 0);
  std::vector<int> stack;
  bool have_roots = false;
  const Json* scenes = jget(glb.root, "scenes");
  const Json* scene_idx = jget(glb.root, "scene");
  if (scenes && scene_idx && scene_idx->type == Json::Num) {
    const long long si = scene_idx->intOr(-1);
    if (si >= 0 && static_cast<std::size_t>(si) < scenes->arr.size()) {
      const Json* roots = jget(scenes->arr[static_cast<std::size_t>(si)], "nodes");
      if (roots && roots->type == Json::Arr) {
        for (const Json& rn : roots->arr) {
          const long long idx = rn.intOr(-1);
          if (idx >= 0 && static_cast<std::size_t>(idx) < n) {
            scene->world[static_cast<std::size_t>(idx)] = mat4Identity();
            stack.push_back(static_cast<int>(idx));
            have_roots = true;
          }
        }
      }
    }
  }
  if (!have_roots) {
    std::vector<char> is_child(n, 0);
    for (const Json& node : scene->nodes->arr) {
      const Json* ch = jget(node, "children");
      if (ch && ch->type == Json::Arr)
        for (const Json& c : ch->arr) {
          const long long idx = c.intOr(-1);
          if (idx >= 0 && static_cast<std::size_t>(idx) < n) is_child[static_cast<std::size_t>(idx)] = 1;
        }
    }
    for (std::size_t i = 0; i < n; ++i)
      if (!is_child[i]) {
        scene->world[i] = mat4Identity();
        stack.push_back(static_cast<int>(i));
      }
  }
  while (!stack.empty()) {
    const int p = stack.back();
    stack.pop_back();
    if (visited[p]) continue;
    visited[p] = 1;
    const Json& node = scene->nodes->arr[p];
    const Json* ch = jget(node, "children");
    if (ch && ch->type == Json::Arr)
      for (const Json& c : ch->arr) {
        const long long idx = c.intOr(-1);
        if (idx < 0 || static_cast<std::size_t>(idx) >= n) continue;
        scene->world[static_cast<std::size_t>(idx)] =
            mat4Mul(scene->world[p], nodeLocalMatrix(scene->nodes->arr[static_cast<std::size_t>(idx)]));
        stack.push_back(static_cast<int>(idx));
      }
  }
  scene->visited.swap(visited);
  return true;
}

bool buildFaceGeometry(const GlbFile& glb, const GlbScene& scene, std::size_t node_idx,
                       FaceGeom* out, std::string* err) {
  const Json& fn = scene.nodes->arr[node_idx];
  const Mat4& fw = scene.world[node_idx];
  const std::string fname = nameOf(fn);

  out->origin[0] = fw.m[0][3];
  out->origin[1] = fw.m[1][3];
  out->origin[2] = fw.m[2][3];
  // 旋转轴：实体局部 Dir3(-1,0,-1) 经面旋转矩阵变换到世界
  double a_world[3];
  for (int i = 0; i < 3; ++i)
    a_world[i] = fw.m[i][0] * -0.7071067811865476 + fw.m[i][2] * -0.7071067811865476;
  if (!normalize3(a_world)) {
    *err = fname + ": bad axis";
    return false;
  }
  out->axis[0] = a_world[0]; out->axis[1] = a_world[1]; out->axis[2] = a_world[2];
  // 符面法向 = 轴（叶板顶点 PCA 已验证 dot≈1）
  out->n[0] = a_world[0]; out->n[1] = a_world[1]; out->n[2] = a_world[2];

  auto idx_by_name = [&](const std::string& name) -> long long {
    for (std::size_t k = 0; k < scene.nodes->arr.size(); ++k)
      if (nameOf(scene.nodes->arr[k]) == name) return static_cast<long long>(k);
    return -1;
  };
  auto gather_mesh = [&](const std::string& node_name, Verts* outv) -> bool {
    const long long ni = idx_by_name(node_name);
    if (ni < 0 || !scene.visited[static_cast<std::size_t>(ni)]) return false;
    const Json* m = jget(scene.nodes->arr[static_cast<std::size_t>(ni)], "mesh");
    if (!m || m->type != Json::Num) return false;
    return loadPositions(glb.root, glb.bin, m->intOr(-1), scene.world[static_cast<std::size_t>(ni)], outv);
  };

  // 叶板（TARGET_1 四个状态 mesh）质心 -> 径向基 u
  Verts pv;
  for (const char* s : {"DISABLED", "ACTIVE", "ACTIVATED", "COMPLETED"})
    gather_mesh(fname + "_TARGET_1_" + s, &pv);
  if (pv.size() == 0) {
    *err = fname + ": no plate meshes";
    return false;
  }
  double pc[3] = {0, 0, 0};
  for (std::size_t i = 0; i < pv.size(); ++i) {
    pc[0] += pv.x[i]; pc[1] += pv.y[i]; pc[2] += pv.z[i];
  }
  pc[0] /= static_cast<double>(pv.size());
  pc[1] /= static_cast<double>(pv.size());
  pc[2] /= static_cast<double>(pv.size());
  double u0[3] = {pc[0] - out->origin[0], pc[1] - out->origin[1], pc[2] - out->origin[2]};
  const double dn = dot3(u0, out->n);
  u0[0] -= dn * out->n[0]; u0[1] -= dn * out->n[1]; u0[2] -= dn * out->n[2];
  if (!normalize3(u0)) {
    *err = fname + ": degenerate radial direction";
    return false;
  }
  double wv[3];
  cross3(u0, out->n, wv);
  if (!normalize3(wv)) {
    *err = fname + ": degenerate w basis";
    return false;
  }
  double uv[3];
  cross3(out->n, wv, uv);
  out->u[0] = uv[0]; out->u[1] = uv[1]; out->u[2] = uv[2];
  out->w[0] = wv[0]; out->w[1] = wv[1]; out->w[2] = wv[2];

  // R 标质心（三片 mesh 合并）
  Verts rv;
  gather_mesh(fname + "_R_POWERED", &rv);
  gather_mesh(fname + "_R_UNPOWERED", &rv);
  gather_mesh(fname + "_R_PADDING", &rv);
  if (rv.size() == 0) {
    *err = fname + ": no R meshes";
    return false;
  }
  double rc[3] = {0, 0, 0};
  for (std::size_t i = 0; i < rv.size(); ++i) {
    rc[0] += rv.x[i]; rc[1] += rv.y[i]; rc[2] += rv.z[i];
  }
  rc[0] /= static_cast<double>(rv.size());
  rc[1] /= static_cast<double>(rv.size());
  rc[2] /= static_cast<double>(rv.size());
  toFaceCoords(*out, rc, out->r_center);

  // 亮起侧：TARGET_1 ACTIVE 叶板绕序
  {
    const long long ni = idx_by_name(fname + "_TARGET_1_ACTIVE");
    long long mesh_idx = -1;
    if (ni >= 0) {
      const Json* m = jget(scene.nodes->arr[static_cast<std::size_t>(ni)], "mesh");
      if (m && m->type == Json::Num) mesh_idx = m->intOr(-1);
    }
    if (mesh_idx >= 0)
      out->lit_sign = litSignFromWinding(glb.root, glb.bin, mesh_idx,
                                         scene.world[static_cast<std::size_t>(ni)], out->n,
                                         out->r_center[2]);
    else
      out->lit_sign = out->r_center[2] > 0 ? 1 : -1;
  }

  // 每个叶臂目标
  for (int j = 1; j <= kTargetCount; ++j) {
    TargetGeom& t = out->targets[j - 1];
    const std::string tag = fname + "_TARGET_" + std::to_string(j) + "_";

    Verts tp;
    for (const char* s : {"DISABLED", "ACTIVE", "ACTIVATED", "COMPLETED"})
      gather_mesh(tag + s, &tp);
    if (tp.size() == 0) {
      *err = tag + "<plate> missing";
      return false;
    }
    // 符面坐标 + 臂方位角
    std::vector<double> fu(tp.size()), fw(tp.size()), fn(tp.size());
    double sum_u = 0, sum_w = 0;
    for (std::size_t i = 0; i < tp.size(); ++i) {
      const double p[3] = {tp.x[i], tp.y[i], tp.z[i]};
      double lc[3];
      toFaceCoords(*out, p, lc);
      fu[i] = lc[0]; fw[i] = lc[1]; fn[i] = lc[2];
      sum_u += lc[0]; sum_w += lc[1];
    }
    sum_u /= static_cast<double>(tp.size());
    sum_w /= static_cast<double>(tp.size());
    t.phi = std::atan2(sum_w, sum_u);
    const double c = std::cos(t.phi), s = std::sin(t.phi);
    for (std::size_t i = 0; i < tp.size(); ++i) rot2(c, s, fu[i], fw[i]);
    t.plate_u_min = fu[0]; t.plate_u_max = fu[0];
    t.plate_w_min = fw[0]; t.plate_w_max = fw[0];
    double n_min = fn[0], n_max = fn[0];
    for (std::size_t i = 1; i < tp.size(); ++i) {
      t.plate_u_min = std::min(t.plate_u_min, fu[i]);
      t.plate_u_max = std::max(t.plate_u_max, fu[i]);
      t.plate_w_min = std::min(t.plate_w_min, fw[i]);
      t.plate_w_max = std::max(t.plate_w_max, fw[i]);
      n_min = std::min(n_min, fn[i]);
      n_max = std::max(n_max, fn[i]);
    }
    t.plate_n_mid = 0.5 * (n_min + n_max);

    // 三叉戟（LEGGING_1，12 顶点）
    Verts tv;
    gather_mesh(tag + "LEGGING_1", &tv);
    if (tv.size() < 10) {
      *err = tag + "LEGGING_1 missing";
      return false;
    }
    std::vector<std::pair<double, double>> pts;  // 对齐后 (u', w')
    double n_sum = 0;
    for (std::size_t i = 0; i < tv.size(); ++i) {
      const double p[3] = {tv.x[i], tv.y[i], tv.z[i]};
      double lc[3];
      toFaceCoords(*out, p, lc);
      n_sum += lc[2];
      double uu = lc[0], ww = lc[1];
      if (i < 12) {  // 调试叠加用（符面坐标，未对齐）
        t.tri_verts[i][0] = uu;
        t.tri_verts[i][1] = ww;
      }
      rot2(c, s, uu, ww);
      pts.emplace_back(uu, ww);
    }
    t.tri_vert_count = static_cast<int>(std::min<std::size_t>(tv.size(), 12));
    t.tri_n_mid = n_sum / static_cast<double>(tv.size());
    std::sort(pts.begin(), pts.end(),
              [](const std::pair<double, double>& a, const std::pair<double, double>& b) {
                return a.first < b.first;
              });
    // 上簇 = u 最大 6 点（叉尖），下簇 = u 最小 2 点（横杠）
    auto w_extremes = [&](std::size_t lo, std::size_t hi,
                          std::pair<double, double>* wmin,
                          std::pair<double, double>* wmax) {
      std::size_t imin = lo, imax = lo;
      for (std::size_t i = lo; i < hi; ++i) {
        if (pts[i].second < pts[imin].second) imin = i;
        if (pts[i].second > pts[imax].second) imax = i;
      }
      *wmin = pts[imin];
      *wmax = pts[imax];
    };
    std::pair<double, double> top_min, top_max, bot_min, bot_max;
    w_extremes(pts.size() - 6, pts.size(), &top_min, &top_max);
    w_extremes(0, 2, &bot_min, &bot_max);
    const std::pair<double, double> top_left = out->lit_sign > 0 ? top_min : top_max;
    const std::pair<double, double> top_right = out->lit_sign > 0 ? top_max : top_min;
    const std::pair<double, double> bot_left = out->lit_sign > 0 ? bot_min : bot_max;
    const std::pair<double, double> bot_right = out->lit_sign > 0 ? bot_max : bot_min;
    auto back_to_face = [&](std::pair<double, double> p) {
      unrot2(c, s, p.first, p.second);
      return p;
    };
    const auto k0f = back_to_face(top_left);
    const auto k4f = back_to_face(top_right);
    const auto k1f = back_to_face(bot_left);
    const auto k3f = back_to_face(bot_right);
    t.k0[0] = k0f.first; t.k0[1] = k0f.second;
    t.k4[0] = k4f.first; t.k4[1] = k4f.second;
    t.k1[0] = k1f.first; t.k1[1] = k1f.second;
    t.k3[0] = k3f.first; t.k3[1] = k3f.second;
  }
  return true;
}

// ---------------- 投影 ----------------

// 点（odom 系）-> 像素。发布的相机位姿在 ROS odom 系（Z 向上），四元数是
// q_ros = M·R_cam·M^T 相似变换，因此 p_cam = M^T·R_ros^T·(p_o - t_o)；
// Bevy 相机前向为 -Z，投影 px = cx - fx·x/z，py = cy + fy·y/z。返回是否成功。
bool projectOdom(const double p_w[3], const ExposureState& exp, const CameraInfo& cam,
                 double* px, double* py) {
  const double qw = exp.camera_quaternion_world_wxyz[0];
  const double qx = exp.camera_quaternion_world_wxyz[1];
  const double qy = exp.camera_quaternion_world_wxyz[2];
  const double qz = exp.camera_quaternion_world_wxyz[3];
  // R_ros（wxyz）
  const double R[3][3] = {
      {1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)},
      {2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)},
      {2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)},
  };
  const double dx = p_w[0] - exp.camera_position_world[0];
  const double dy = p_w[1] - exp.camera_position_world[1];
  const double dz = p_w[2] - exp.camera_position_world[2];
  // v = R^T * (p_o - t_o)
  const double vx = R[0][0] * dx + R[1][0] * dy + R[2][0] * dz;
  const double vy = R[0][1] * dx + R[1][1] * dy + R[2][1] * dz;
  const double vz = R[0][2] * dx + R[1][2] * dy + R[2][2] * dz;
  // p_cam = M^T * v，M^T = [[0,-1,0],[0,0,1],[-1,0,0]]
  const double x = -vy;
  const double y = vz;
  const double z = -vx;
  if (z >= -0.01) return false;  // 相机后方（Bevy 前向为 -Z）
  *px = cam.cx - cam.fx * x / z;
  *py = cam.cy + cam.fy * y / z;
  return true;
}

// 符面 (u,w,n) -> 绕轴旋转 current_angle -> odom -> 像素
bool facePointToPixels(const FaceGeom& f, double uu, double ww, double nn, double angle,
                       const ExposureState& exp, const CameraInfo& cam,
                       double* px, double* py) {
  double pw[3];
  facePointWorld(f, uu, ww, nn, pw);
  rotateAroundAxis(f.axis, f.origin, angle, pw);
  double po[3];
  worldToOdom(pw, po);
  return projectOdom(po, exp, cam, px, py);
}

// ---------------- PNG 输出与调试叠加 ----------------

bool saveRgbPng(const std::string& path, const std::uint8_t* payload,
                std::uint32_t width, std::uint32_t height,
                daedalus::sim::sdk::v1::tcp_image::PixelFormat format) {
  const bool rgba = format == daedalus::sim::sdk::v1::tcp_image::PixelFormat::Rgba32;
  FILE* fp = std::fopen(path.c_str(), "wb");
  if (fp == nullptr) return false;
  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  png_infop info = png_create_info_struct(png);
  if (png == nullptr || info == nullptr) {
    if (png) png_destroy_write_struct(&png, nullptr);
    std::fclose(fp);
    return false;
  }
  if (setjmp(png_jmpbuf(png))) {
    png_destroy_write_struct(&png, &info);
    std::fclose(fp);
    return false;
  }
  png_init_io(png, fp);
  png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGB,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png, info);
  std::vector<std::uint8_t> row(static_cast<std::size_t>(width) * 3U);
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      const std::uint32_t i = y * width + x;
      const std::uint32_t base = rgba ? i * 4U : i * 3U;
      row[x * 3U] = payload[base];
      row[x * 3U + 1U] = payload[base + 1U];
      row[x * 3U + 2U] = payload[base + 2U];
    }
    png_write_row(png, row.data());
  }
  png_write_end(png, info);
  png_destroy_write_struct(&png, &info);
  std::fclose(fp);
  return true;
}

void drawDot(std::vector<std::uint8_t>& img, std::uint32_t width, std::uint32_t height,
             int x, int y, std::uint8_t r, std::uint8_t g, std::uint8_t b, int rad = 3) {
  if (x < 0 || y < 0 || static_cast<std::uint32_t>(x) >= width ||
      static_cast<std::uint32_t>(y) >= height)
    return;
  for (int dy = -rad; dy <= rad; ++dy) {
    for (int dx = -rad; dx <= rad; ++dx) {
      if (dx * dx + dy * dy > rad * rad + 1) continue;
      const int xx = x + dx, yy = y + dy;
      if (xx < 0 || yy < 0 || static_cast<std::uint32_t>(xx) >= width ||
          static_cast<std::uint32_t>(yy) >= height)
        continue;
      const std::size_t i = (static_cast<std::size_t>(yy) * width + xx) * 3U;
      img[i] = r; img[i + 1] = g; img[i + 2] = b;
    }
  }
}

bool saveRgbPngBytes(const std::string& path, const std::vector<std::uint8_t>& img,
                     std::uint32_t width, std::uint32_t height) {
  FILE* fp = std::fopen(path.c_str(), "wb");
  if (fp == nullptr) return false;
  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  png_infop info = png_create_info_struct(png);
  if (png == nullptr || info == nullptr) {
    if (png) png_destroy_write_struct(&png, nullptr);
    std::fclose(fp);
    return false;
  }
  if (setjmp(png_jmpbuf(png))) {
    png_destroy_write_struct(&png, &info);
    std::fclose(fp);
    return false;
  }
  png_init_io(png, fp);
  png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGB,
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png, info);
  for (std::uint32_t y = 0; y < height; ++y)
    png_write_row(png, const_cast<png_bytep>(img.data() + static_cast<std::size_t>(y) * width * 3U));
  png_write_end(png, info);
  png_destroy_write_struct(&png, &info);
  std::fclose(fp);
  return true;
}

// 双线性缩放 RGB 图：src[sh*sw*3] -> dst[dh*dw*3]
void resizeRgb(const std::vector<std::uint8_t>& src, std::uint32_t sw, std::uint32_t sh,
               std::vector<std::uint8_t>& dst, std::uint32_t dw, std::uint32_t dh) {
  dst.assign(static_cast<std::size_t>(dw) * dh * 3U, 0);
  const double sx = static_cast<double>(sw) / dw;
  const double sy = static_cast<double>(sh) / dh;
  for (std::uint32_t y = 0; y < dh; ++y) {
    const double fy = (y + 0.5) * sy - 0.5;
    const int y0 = static_cast<int>(std::floor(fy));
    const int y1 = std::min<int>(y0 + 1, sh - 1);
    const double wy = fy - y0;
    for (std::uint32_t x = 0; x < dw; ++x) {
      const double fx = (x + 0.5) * sx - 0.5;
      const int x0 = static_cast<int>(std::floor(fx));
      const int x1 = std::min<int>(x0 + 1, sw - 1);
      const double wx = fx - x0;
      const std::size_t p00 = (static_cast<std::size_t>(std::max(y0, 0)) * sw + std::max(x0, 0)) * 3U;
      const std::size_t p01 = (static_cast<std::size_t>(std::max(y0, 0)) * sw + x1) * 3U;
      const std::size_t p10 = (static_cast<std::size_t>(y1) * sw + std::max(x0, 0)) * 3U;
      const std::size_t p11 = (static_cast<std::size_t>(y1) * sw + x1) * 3U;
      const std::size_t o = (static_cast<std::size_t>(y) * dw + x) * 3U;
      for (int c = 0; c < 3; ++c) {
        const double v = src[p00 + c] * (1 - wx) * (1 - wy) + src[p01 + c] * wx * (1 - wy) +
                         src[p10 + c] * (1 - wx) * wy + src[p11 + c] * wx * wy;
        dst[o + c] = static_cast<std::uint8_t>(v + 0.5);
      }
    }
  }
}

// ---------------- 对象发射 ----------------

struct ObjOut {
  int class_id = -1;
  double kp[5][2] = {};  // 全分辨率像素（640/1440 缩放到网络空间发生在写标签时）
  int face_idx = -1;
  int target_idx = -1;
  int rune_idx = -1;     // 所属 rune（两符面旋转方向相反，各用各的角度）
  bool trident = false;
};

// 相机是否处于符面亮起侧（背面不标注，该面不可见）
bool faceFacingCamera(const FaceGeom& f, const ExposureState& exp) {
  const double c_odom[3] = {exp.camera_position_world[0], exp.camera_position_world[1],
                            exp.camera_position_world[2]};
  double camw[3];
  odomToWorld(c_odom, camw);
  const double dcam[3] = {camw[0] - f.origin[0], camw[1] - f.origin[1], camw[2] - f.origin[2]};
  const double dlen = norm3(dcam);
  return dlen <= 1e-9 || f.lit_sign * dot3(f.n, dcam) >= 0.02 * dlen;
}

void emitObjects(const GroundTruthBatch& batch, const ExposureState& exp, const CameraInfo& cam,
                 const std::vector<FaceGeom>& faces, std::vector<ObjOut>* out) {
  for (std::uint32_t ri = 0; ri < batch.rune_count; ++ri) {
    const auto& rune = batch.runes[ri];
    // 用 r_center_odom 匹配符面（真值顺序不保证）
    const double r_odom[3] = {rune.r_center_odom[0], rune.r_center_odom[1],
                              rune.r_center_odom[2]};
    double rw[3];
    odomToWorld(r_odom, rw);
    int fi = -1;
    double best = 1e18;
    for (std::size_t f = 0; f < faces.size(); ++f) {
      const double d[3] = {rw[0] - faces[f].origin[0], rw[1] - faces[f].origin[1],
                           rw[2] - faces[f].origin[2]};
      const double dist = norm3(d);
      if (dist < best) {
        best = dist;
        fi = static_cast<int>(f);
      }
    }
    if (fi < 0 || best > 0.1) {
      std::fprintf(stderr, "  rune[%u] no matching face (dist=%.3f), skipped\n", ri, best);
      continue;
    }
    const FaceGeom& f = faces[static_cast<std::size_t>(fi)];
    // 亮起侧背面不标注（该面不可见）
    if (!faceFacingCamera(f, exp)) {
      std::printf("  rune[%u] (FACE_%d) 背面朝向相机，跳过\n", ri, fi + 1);
      continue;
    }

    for (int j = 0; j < kTargetCount; ++j) {
      const int act = rune.target_activations[j];
      const int mode = rune.rune_mode;
      bool trident = false;
      int cls = -1;
      if (act == 1) cls = 0;
      else if (mode == 0 && (act == 2 || act == 3)) cls = 1;
      else if (mode == 1 && act == 2) { cls = 2; trident = true; }
      else if (mode == 1 && act == 3) cls = 1;  // Large 完成态也按 class 1 标（用户确认）
      else continue;

      const TargetGeom& t = f.targets[j];
      double uu[5], ww[5], nn[5];
      if (trident) {
        uu[0] = t.k0[0]; ww[0] = t.k0[1];
        uu[4] = t.k4[0]; ww[4] = t.k4[1];
        uu[1] = t.k1[0]; ww[1] = t.k1[1];
        uu[3] = t.k3[0]; ww[3] = t.k3[1];
        nn[0] = nn[1] = nn[3] = nn[4] = t.tri_n_mid;
      } else {
        const double umid = 0.5 * (t.plate_u_min + t.plate_u_max);
        const double wleft = f.lit_sign > 0 ? t.plate_w_min : t.plate_w_max;
        const double wright = f.lit_sign > 0 ? t.plate_w_max : t.plate_w_min;
        uu[0] = t.plate_u_max; ww[0] = 0.0;
        uu[4] = t.plate_u_min; ww[4] = 0.0;
        uu[1] = umid; ww[1] = wleft;
        uu[3] = umid; ww[3] = wright;
        nn[0] = nn[1] = nn[3] = nn[4] = t.plate_n_mid;
        const double c = std::cos(t.phi), s = std::sin(t.phi);
        for (int k = 0; k < 5; ++k)
          if (k != 2) unrot2(c, s, uu[k], ww[k]);
      }
      uu[2] = f.r_center[0]; ww[2] = f.r_center[1]; nn[2] = f.r_center[2];

      ObjOut obj;
      obj.class_id = cls;
      obj.face_idx = fi;
      obj.target_idx = j;
      obj.rune_idx = static_cast<int>(ri);
      obj.trident = trident;
      bool ok = true;
      for (int k = 0; k < 5; ++k) {
        double px, py;
        if (!facePointToPixels(f, uu[k], ww[k], nn[k], rune.current_angle, exp, cam, &px, &py)) {
          ok = false;
          break;
        }
        if (px < -4 || px > cam.width + 4 || py < -4 || py > cam.height + 4) {
          ok = false;
          break;
        }
        obj.kp[k][0] = px;
        obj.kp[k][1] = py;
      }
      if (!ok) continue;
      out->push_back(obj);
      std::printf("  rune[%u] target[%d] class=%d mode=%d act=%d pts: (%.0f,%.0f) (%.0f,%.0f) "
                  "(%.0f,%.0f) (%.0f,%.0f) (%.0f,%.0f)\n",
                  ri, j, cls, mode, act, obj.kp[0][0], obj.kp[0][1], obj.kp[1][0], obj.kp[1][1],
                  obj.kp[2][0], obj.kp[2][1], obj.kp[3][0], obj.kp[3][1], obj.kp[4][0], obj.kp[4][1]);
    }
  }
}

// ---------------- 主流程 ----------------

void printGeometry(const std::vector<FaceGeom>& faces) {
  for (std::size_t i = 0; i < faces.size(); ++i) {
    const FaceGeom& f = faces[i];
    std::printf("FACE_%zu origin=(%.4f,%.4f,%.4f) axis=(%.4f,%.4f,%.4f) lit_sign=%+d\n",
                i + 1, f.origin[0], f.origin[1], f.origin[2], f.axis[0], f.axis[1], f.axis[2],
                f.lit_sign);
    std::printf("  u=(%.4f,%.4f,%.4f) w=(%.4f,%.4f,%.4f) n=(%.4f,%.4f,%.4f)\n",
                f.u[0], f.u[1], f.u[2], f.w[0], f.w[1], f.w[2], f.n[0], f.n[1], f.n[2]);
    std::printf("  R center (u,w,n) = (%.4f,%.4f,%.4f)\n", f.r_center[0], f.r_center[1],
                f.r_center[2]);
    for (int j = 0; j < kTargetCount; ++j) {
      const TargetGeom& t = f.targets[j];
      std::printf("  TARGET_%d phi=%+.1fdeg plate u[%.3f,%.3f] w[%.3f,%.3f] n_mid=%.4f\n", j + 1,
                  t.phi * 180.0 / kPi, t.plate_u_min, t.plate_u_max, t.plate_w_min, t.plate_w_max,
                  t.plate_n_mid);
      std::printf("     trident n_mid=%.4f 0=(%.4f,%.4f) 4=(%.4f,%.4f) 1=(%.4f,%.4f) 3=(%.4f,%.4f)\n",
                  t.tri_n_mid, t.k0[0], t.k0[1], t.k4[0], t.k4[1], t.k1[0], t.k1[1], t.k3[0],
                  t.k3[1]);
    }
  }
}

}  // namespace

// 可执行文件所在目录（/proc/self/exe），用于按 exe 位置兜底找默认 glb
namespace {
std::string executableDir() {
  char buf[4096];
  const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0) return ".";
  buf[n] = '\0';
  const std::string path(buf);
  const std::size_t slash = path.find_last_of('/');
  return slash == std::string::npos ? "." : path.substr(0, slash);
}
}  // namespace

int main(int argc, char** argv) {
  Options opts;
  if (!parseOptions(argc, argv, &opts)) return 0;

  GlbFile glb;
  if (!loadGlb(opts.glb_path, &glb)) {
    // 默认相对路径随 cwd 变化；从项目根目录运行 ./tools/... 时按 exe 位置兜底
    if (!opts.glb_explicit && loadGlb(executableDir() + "/../assets/POWER.glb", &glb)) {
      std::printf("loaded POWER.glb from exe-relative path\n");
    } else {
      std::fprintf(stderr, "cannot load glb: %s\n", opts.glb_path.c_str());
      return 1;
    }
  }
  std::printf("loaded %s (bin %zu bytes)\n", opts.glb_path.c_str(), glb.bin_len);

  GlbScene scene;
  std::string err;
  if (!buildScene(glb, &scene, &err)) {
    std::fprintf(stderr, "scene build failed: %s\n", err.c_str());
    return 1;
  }

  std::vector<FaceGeom> faces;
  for (std::size_t i = 0; i < scene.nodes->arr.size(); ++i) {
    const std::string nm = nameOf(scene.nodes->arr[i]);
    if (nm.rfind("FACE_", 0) != 0) continue;
    const std::string rest = nm.substr(5);
    if (rest.empty() || rest.find('_') != std::string::npos) continue;
    bool digits = true;
    for (const char c : rest) digits = digits && std::isdigit(static_cast<unsigned char>(c));
    if (!digits || !scene.visited[i]) continue;
    FaceGeom fg;
    if (!buildFaceGeometry(glb, scene, i, &fg, &err)) {
      std::fprintf(stderr, "face geometry failed (%s): %s\n", nm.c_str(), err.c_str());
      return 1;
    }
    faces.push_back(fg);
  }
  if (faces.empty()) {
    std::fprintf(stderr, "no FACE_* nodes found in glb\n");
    return 1;
  }
  std::printf("parsed %zu rune face(s)\n", faces.size());
  if (opts.self_test) {
    printGeometry(faces);
    return 0;
  }

  if (opts.ipc_dir.empty()) opts.ipc_dir = defaultIpcDir();

  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);

  // 真值元数据（只读）
  TalosMetadataMapping mapping;
  const ClientStatus open_status = mapping.open(opts.ipc_dir + "/talos_ipc_meta");
  if (!open_status.ok()) {
    std::fprintf(stderr, "metadata open failed: %s\n", open_status.message.c_str());
    return 1;
  }
  auto reader = mapping.reader();
  if (!reader.ok()) {
    std::fprintf(stderr, "reader failed: %s\n", reader.status.message.c_str());
    return 1;
  }
  auto cam_res = reader.value->readCameraInfo();
  if (!cam_res.ok()) {
    std::fprintf(stderr, "camera info failed: %s\n", cam_res.status.message.c_str());
    return 1;
  }
  const CameraInfo cam = *cam_res.value;
  std::printf("camera: %.1fx%.1f fx=%.2f fy=%.2f cx=%.1f cy=%.1f\n",
              static_cast<double>(cam.width), static_cast<double>(cam.height),
              cam.fx, cam.fy, cam.cx, cam.cy);

  TcpImageClient client({opts.host, opts.port});
  const ClientStatus conn = client.connect();
  if (!conn.ok()) {
    std::fprintf(stderr, "image connect failed: %s\n", conn.message.c_str());
    return 1;
  }
  std::printf("connected to %s:%u\n", opts.host.c_str(), opts.port);

  std::filesystem::create_directories(opts.out_dir);
  // 与已有数据共存：统计 labels.jsonl 现有条数，末尾无换行时修复（上次异常中断的残留），
  // 完整行补换行保留，不完整残留截断到最后一个完整行。
  const std::string labels_path = opts.out_dir + "/labels.jsonl";
  std::size_t existing_lines = 0;
  {
    std::ifstream in(labels_path, std::ios::binary);
    if (in) {
      std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      in.close();
      if (!data.empty()) {
        const bool ends_nl = data.back() == '\n';
        bool kept_tail = false;
        if (!ends_nl) {
          const std::size_t last_nl = data.find_last_of('\n');
          const std::size_t tail_start = (last_nl == std::string::npos) ? 0 : last_nl + 1;
          const std::string tail = data.substr(tail_start);
          const bool tail_ok = tail.find("\"file\"") != std::string::npos && tail.back() == '}';
          if (tail_ok) {
            // 完整行缺末尾换行：补换行，避免追加时与新行粘连
            std::ofstream fix(labels_path, std::ios::binary | std::ios::app);
            fix.write("\n", 1);
            fix.close();
            kept_tail = true;
          } else {
            std::ofstream fix(labels_path, std::ios::binary | std::ios::trunc);
            fix.write(data.data(), static_cast<std::streamsize>(tail_start));
            fix.close();
            std::fprintf(stderr,
                         "labels.jsonl 末尾 %zu 字节不完整（上次中断残留），已截断修复\n",
                         data.size() - tail_start);
          }
        }
        for (char ch : data)
          if (ch == '\n') ++existing_lines;
        if (kept_tail) ++existing_lines;
      }
    }
  }
  std::ofstream labels(labels_path, std::ios::app);
  if (!labels) {
    std::fprintf(stderr, "cannot open labels.jsonl\n");
    return 1;
  }
  if (existing_lines > 0)
    std::printf("labels.jsonl 已有 %zu 条标注，本次继续追加（新帧与旧帧共存）\n",
                existing_lines);

  std::uint64_t after_seq = 0;
  long saved = 0;
  const double kRollStep = opts.per_roll > 0 ? 2.0 * kPi / opts.per_roll : 0.0;
  double roll_total = 0.0;      // 累计转过的角度（绝对值）
  double next_mark = kRollStep; // 下一次存帧的角度阈值
  bool have_prev = false;
  double prev_angle = 0.0;
  const bool roll_mode = opts.per_roll > 0;
  const auto roll_start = std::chrono::steady_clock::now();
  auto last_progress = roll_start;
  while (!g_stop && (roll_mode ? (roll_total < 2.0 * kPi - 1e-6)
                               : (opts.count == 0 || saved < opts.count))) {
    if (roll_mode &&
        std::chrono::steady_clock::now() - last_progress > std::chrono::seconds(30)) {
      std::fprintf(stderr, "角度 30 秒无变化（符未旋转？），中止 per-roll 采样\n");
      return 2;
    }
    auto result = client.waitForLatest(after_seq, std::chrono::milliseconds(500));
    if (!result.ok()) {
      if (result.status.error == ClientError::Timeout) continue;
      std::fprintf(stderr, "frame error: %s\n", result.status.message.c_str());
      break;
    }
    const auto& frame = *result.value;
    const auto& h = frame.header;

    auto snap = reader.value->readGroundTruthForFrame(h.source_sequence);
    if (!snap.ok()) {
      std::fprintf(stderr, "truth for frame %llu failed: %s\n",
                   static_cast<unsigned long long>(h.source_sequence),
                   snap.status.message.c_str());
      after_seq = h.source_sequence;
      continue;
    }
    const auto& batch = snap.value->ground_truth;
    const auto& exp = snap.value->exposure_state;

    if (roll_mode) {
      if (batch.rune_count == 0) {
        after_seq = h.source_sequence;
        continue;
      }
      // 等真值达到期望工况状态（跳过 setRuneState 前的旧缓冲帧）
      if (opts.has_expect_acts) {
        const auto& t = batch.runes[0].target_activations;
        bool match = true;
        for (int i = 0; i < 5; ++i)
          if (t[i] != opts.expect_acts_v[i]) { match = false; break; }
        if (!match) {
          after_seq = h.source_sequence;
          continue;
        }
      }
      const double ang = batch.runes[0].current_angle;
      if (have_prev) {
        // 真值角度定义域约 (-2π, 2π]，跨边界 wrap 时 |Δ| 会虚大，修正 ±4π
        double d = ang - prev_angle;
        if (d > 2.0 * kPi) d -= 4.0 * kPi;
        if (d < -2.0 * kPi) d += 4.0 * kPi;
        roll_total += std::fabs(d);
      }
      prev_angle = ang;
      have_prev = true;
      last_progress = std::chrono::steady_clock::now();
      if (roll_total < next_mark) {
        after_seq = h.source_sequence;
        continue;  // 还没转到下一个采样角度
      }
      next_mark += kRollStep;
    }

    char wf_suffix[64] = "";
    if (!opts.workflow.empty())
      std::snprintf(wf_suffix, sizeof(wf_suffix), "_wf%s", opts.workflow.c_str());
    char name[128];
    std::snprintf(name, sizeof(name), "frame_%06llu_%llu%s.png",
                  static_cast<unsigned long long>(h.source_sequence),
                  static_cast<unsigned long long>(h.capture_timestamp_ns), wf_suffix);
    const std::string path = opts.out_dir + "/" + name;
    if (std::filesystem::exists(path)) {
      std::fprintf(stderr, "skip duplicate frame %s (同名文件已存在，不覆盖)\n", name);
      after_seq = h.source_sequence;
      continue;
    }

    std::vector<ObjOut> objs;
    emitObjects(batch, exp, cam, faces, &objs);

    if (objs.empty() && batch.rune_count > 0) {
      std::printf("  [no objects] truth:");
      for (std::uint32_t ri = 0; ri < batch.rune_count; ++ri) {
        const auto& r = batch.runes[ri];
        std::printf(" rune%u(mode=%u mech=%u acts=", ri, r.rune_mode, r.mechanism_state);
        for (int j = 0; j < 5; ++j) std::printf("%u", r.target_activations[j]);
        std::printf(" angle=%.2f dir=%d)", r.current_angle, r.direction);
      }
      std::printf("\n");
    }

    if (opts.img_net) {
      const bool rgba = h.format == tcp_image::PixelFormat::Rgba32;
      std::vector<std::uint8_t> rgb(static_cast<std::size_t>(h.width) * h.height * 3U);
      for (std::uint32_t y = 0; y < h.height; ++y) {
        for (std::uint32_t x = 0; x < h.width; ++x) {
          const std::uint32_t i = y * h.width + x;
          const std::uint32_t base = rgba ? i * 4U : i * 3U;
          const std::size_t o = static_cast<std::size_t>(i) * 3U;
          rgb[o] = frame.payload[base];
          rgb[o + 1] = frame.payload[base + 1];
          rgb[o + 2] = frame.payload[base + 2];
        }
      }
      std::vector<std::uint8_t> small;
      resizeRgb(rgb, h.width, h.height, small, 640, 480);
      if (!saveRgbPngBytes(path, small, 640, 480)) {
        std::fprintf(stderr, "write failed: %s\n", path.c_str());
        break;
      }
    } else {
      if (!saveRgbPng(path, frame.payload.data(), h.width, h.height, h.format)) {
        std::fprintf(stderr, "write failed: %s\n", path.c_str());
        break;
      }
    }

    if (opts.debug_draw) {
      std::vector<std::uint8_t> img(static_cast<std::size_t>(h.width) * h.height * 3U);
      const bool rgba = h.format == tcp_image::PixelFormat::Rgba32;
      for (std::uint32_t y = 0; y < h.height; ++y) {
        for (std::uint32_t x = 0; x < h.width; ++x) {
          const std::uint32_t i = y * h.width + x;
          const std::uint32_t base = rgba ? i * 4U : i * 3U;
          const std::size_t o = static_cast<std::size_t>(i) * 3U;
          img[o] = frame.payload[base];
          img[o + 1] = frame.payload[base + 1];
          img[o + 2] = frame.payload[base + 2];
        }
      }
      // 校准参考：正面朝向相机的符面始终画 R 中心（白）+ 叶板四角（灰），与是否标注无关
      for (std::size_t fi2 = 0; fi2 < faces.size(); ++fi2) {
        const FaceGeom& fr = faces[fi2];
        if (!faceFacingCamera(fr, exp)) continue;
        double rune_angle = 0.0;
        for (std::uint32_t ri = 0; ri < batch.rune_count; ++ri) {
          const double r_odom[3] = {batch.runes[ri].r_center_odom[0],
                                    batch.runes[ri].r_center_odom[1],
                                    batch.runes[ri].r_center_odom[2]};
          double rw[3];
          odomToWorld(r_odom, rw);
          const double d[3] = {rw[0] - fr.origin[0], rw[1] - fr.origin[1], rw[2] - fr.origin[2]};
          if (norm3(d) < 0.1) {
            rune_angle = batch.runes[ri].current_angle;
            break;
          }
        }
        double px, py;
        if (facePointToPixels(fr, fr.r_center[0], fr.r_center[1], fr.r_center[2], rune_angle, exp,
                              cam, &px, &py))
          drawDot(img, h.width, h.height, static_cast<int>(px), static_cast<int>(py),
                  255, 255, 255, 5);
        for (int j = 0; j < kTargetCount; ++j) {
          const TargetGeom& tr = fr.targets[j];
          const double c = std::cos(tr.phi), s = std::sin(tr.phi);
          for (int cu = 0; cu < 2; ++cu) {
            for (int cw = 0; cw < 2; ++cw) {
              double uu = cu ? tr.plate_u_max : tr.plate_u_min;
              double ww = cw ? tr.plate_w_max : tr.plate_w_min;
              unrot2(c, s, uu, ww);
              if (facePointToPixels(fr, uu, ww, tr.plate_n_mid, rune_angle, exp, cam, &px, &py))
                drawDot(img, h.width, h.height, static_cast<int>(px), static_cast<int>(py),
                        120, 120, 120, 1);
            }
          }
        }
      }
      for (const ObjOut& obj : objs) {
        const FaceGeom& f = faces[static_cast<std::size_t>(obj.face_idx)];
        const TargetGeom& t = f.targets[obj.target_idx];
        const double rune_angle = batch.runes[static_cast<std::size_t>(obj.rune_idx)].current_angle;
        // 参考几何：三叉戟全部顶点 / 叶板四角
        if (obj.trident) {
          for (int v = 0; v < t.tri_vert_count; ++v) {
            double px, py;
            if (facePointToPixels(f, t.tri_verts[v][0], t.tri_verts[v][1], t.tri_n_mid,
                                  rune_angle, exp, cam, &px, &py))
              drawDot(img, h.width, h.height, static_cast<int>(px), static_cast<int>(py),
                      220, 220, 220, 1);
          }
        } else {
          const double c = std::cos(t.phi), s = std::sin(t.phi);
          for (int cu = 0; cu < 2; ++cu) {
            for (int cw = 0; cw < 2; ++cw) {
              double uu = cu ? t.plate_u_max : t.plate_u_min;
              double ww = cw ? t.plate_w_max : t.plate_w_min;
              unrot2(c, s, uu, ww);
              double px, py;
              if (facePointToPixels(f, uu, ww, t.plate_n_mid, rune_angle, exp,
                                    cam, &px, &py))
                drawDot(img, h.width, h.height, static_cast<int>(px), static_cast<int>(py),
                        0, 255, 255, 1);
            }
          }
        }
        static const std::uint8_t kCol[5][3] = {{0, 255, 0}, {255, 255, 0}, {255, 128, 0},
                                                {255, 0, 255}, {0, 0, 255}};
        for (int k = 0; k < 5; ++k)
          drawDot(img, h.width, h.height, static_cast<int>(obj.kp[k][0]),
                  static_cast<int>(obj.kp[k][1]), kCol[k][0], kCol[k][1], kCol[k][2], 3);
      }
      std::string debug_path = path;
      debug_path.replace(debug_path.size() - 4, 4, "_debug.png");
      saveRgbPngBytes(debug_path, img, h.width, h.height);
    }

    labels << "{\"file\":\"" << name << "\"";
    if (!opts.workflow.empty()) labels << ",\"workflow\":\"" << opts.workflow << "\"";
    labels << ",\"objects\":[";
    for (std::size_t i = 0; i < objs.size(); ++i) {
      if (i) labels << ',';
      labels << "{\"class_id\":" << objs[i].class_id << ",\"keypoints\":[";
      for (int k = 0; k < 5; ++k) {
        if (k) labels << ',';
        char buf[64];
        std::snprintf(buf, sizeof(buf), "[%.2f,%.2f]", objs[i].kp[k][0] * kNetScale,
                      objs[i].kp[k][1] * kNetScale);
        labels << buf;
      }
      labels << "]}";
    }
    labels << "]}\n";

    after_seq = h.source_sequence;
    ++saved;
    std::printf("saved %s (%ux%u, %zu object(s))\n", name, h.width, h.height, objs.size());
  }

  std::printf("done: 本次 %ld frame(s)，labels.jsonl 共 %zu 条，目录 %s\n", saved,
              existing_lines + static_cast<std::size_t>(saved), opts.out_dir.c_str());
  return 0;
}

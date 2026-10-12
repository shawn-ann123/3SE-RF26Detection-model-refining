// 导出模拟器摄像头图像：通过 SDK TcpImageClient 抓取帧，保存为 PPM、PNG(BGR 序) 或原始 RGBA/RGB。
// 用法: export_camera_images [--out DIR] [--count N] [--format ppm|raw|bgr]
//   --count 0 表示持续抓取直到 Ctrl+C。
#include <daedalus_sim_sdk/tcp_image_client.hpp>

#include <png.h>

#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

volatile std::sig_atomic_t g_stop = 0;
void onSignal(int) { g_stop = 1; }

struct Options {
  std::string out_dir = "frames";
  std::string format = "ppm";  // ppm | raw | bgr
  long count = 0;              // 0 = 无限
  std::string host = "127.0.0.1";
  std::uint16_t port = daedalus::sim::sdk::v1::kTcpImagePort;
};

bool parseOptions(int argc, char** argv, Options* opts) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto value = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", arg.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--out") {
      opts->out_dir = value();
    } else if (arg == "--format") {
      opts->format = value();
    } else if (arg == "--count") {
      opts->count = std::strtol(value(), nullptr, 10);
    } else if (arg == "--host") {
      opts->host = value();
    } else if (arg == "--port") {
      opts->port = static_cast<std::uint16_t>(std::strtol(value(), nullptr, 10));
    } else if (arg == "--help" || arg == "-h") {
      std::printf(
          "Usage: export_camera_images [--out DIR] [--count N] "
          "[--format ppm|raw|bgr] [--host H] [--port P]\n"
          "  ppm: RGB, P6 (通用预览)  raw: 模拟器原始 RGBA32/RGB24\n"
          "  bgr: PNG 文件, 像素通道序为 BGR (cv::imread 读取后字节序不变)\n");
      return false;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      std::exit(2);
    }
  }
  if (opts->format != "ppm" && opts->format != "raw" &&
      opts->format != "bgr") {
    std::fprintf(stderr, "--format must be ppm, raw or bgr\n");
    std::exit(2);
  }
  return true;
}

std::string formatName(daedalus::sim::sdk::v1::tcp_image::PixelFormat f) {
  using PF = daedalus::sim::sdk::v1::tcp_image::PixelFormat;
  return f == PF::Rgba32 ? "rgba32" : "rgb24";
}

std::string extensionFor(const Options& opts,
                         daedalus::sim::sdk::v1::tcp_image::PixelFormat f) {
  if (opts.format == "ppm") return ".ppm";
  if (opts.format == "bgr") return ".png";
  return f == daedalus::sim::sdk::v1::tcp_image::PixelFormat::Rgba32 ? ".rgba"
                                                                     : ".rgb";
}

bool savePngBgr(const std::string& path, const std::uint8_t* payload,
                std::uint32_t width, std::uint32_t height,
                daedalus::sim::sdk::v1::tcp_image::PixelFormat format) {
  const bool rgba =
      format == daedalus::sim::sdk::v1::tcp_image::PixelFormat::Rgba32;
  FILE* fp = std::fopen(path.c_str(), "wb");
  if (fp == nullptr) return false;
  png_structp png = png_create_write_struct(
      PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (png == nullptr) {
    std::fclose(fp);
    return false;
  }
  png_infop info = png_create_info_struct(png);
  if (info == nullptr) {
    png_destroy_write_struct(&png, nullptr);
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
               PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png, info);

  std::vector<std::uint8_t> row(static_cast<std::size_t>(width) * 3U);
  for (std::uint32_t y = 0; y < height; ++y) {
    for (std::uint32_t x = 0; x < width; ++x) {
      const std::uint32_t i = y * width + x;
      const std::uint32_t base = rgba ? i * 4U : i * 3U;
      row[x * 3U] = payload[base + 2U];      // B
      row[x * 3U + 1U] = payload[base + 1U]; // G
      row[x * 3U + 2U] = payload[base + 0U]; // R
    }
    png_write_row(png, row.data());
  }
  png_write_end(png, info);
  png_destroy_write_struct(&png, &info);
  std::fclose(fp);
  return true;
}

bool savePpm(const std::string& path, const std::uint8_t* payload,
             std::uint32_t width, std::uint32_t height,
             daedalus::sim::sdk::v1::tcp_image::PixelFormat format) {
  const bool rgba =
      format == daedalus::sim::sdk::v1::tcp_image::PixelFormat::Rgba32;
  const std::uint32_t channels = rgba ? 4U : 3U;
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  out << "P6\n" << width << ' ' << height << "\n255\n";
  if (!rgba) {
    out.write(reinterpret_cast<const char*>(payload),
              static_cast<std::streamsize>(width) * height * channels);
  } else {
    std::string rgb;
    rgb.reserve(static_cast<std::size_t>(width) * height * 3U);
    for (std::uint32_t i = 0; i < width * height; ++i) {
      rgb.push_back(static_cast<char>(payload[i * 4U]));
      rgb.push_back(static_cast<char>(payload[i * 4U + 1U]));
      rgb.push_back(static_cast<char>(payload[i * 4U + 2U]));
    }
    out.write(rgb.data(), static_cast<std::streamsize>(rgb.size()));
  }
  return static_cast<bool>(out);
}

}  // namespace

int main(int argc, char** argv) {
  using namespace daedalus::sim::sdk::v1;

  Options opts;
  if (!parseOptions(argc, argv, &opts)) return 0;

  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);

  TcpImageClient client({opts.host, opts.port});
  const ClientStatus status = client.connect();
  if (!status.ok()) {
    std::fprintf(stderr, "connect failed: %s\n", status.message.c_str());
    return 1;
  }
  std::printf("connected to %s:%u\n", opts.host.c_str(), opts.port);

  const std::string mkdir_cmd = "mkdir -p '" + opts.out_dir + "'";
  if (std::system(mkdir_cmd.c_str()) != 0) {
    std::fprintf(stderr, "cannot create output directory: %s\n",
                 opts.out_dir.c_str());
    return 1;
  }

  std::ofstream manifest(opts.out_dir + "/manifest.jsonl");
  if (!manifest) {
    std::fprintf(stderr, "cannot open manifest\n");
    return 1;
  }

  std::uint64_t after_seq = 0;
  long saved = 0;
  while (!g_stop && (opts.count == 0 || saved < opts.count)) {
    auto result = client.waitForLatest(after_seq, std::chrono::milliseconds(500));
    if (!result.ok()) {
      if (result.status.error == ClientError::Timeout) continue;
      std::fprintf(stderr, "frame error: %s\n", result.status.message.c_str());
      break;
    }
    const auto& frame = *result.value;
    const auto& h = frame.header;

    char name[128];
    std::snprintf(name, sizeof(name), "frame_%06llu_%llu%s",
                  static_cast<unsigned long long>(h.source_sequence),
                  static_cast<unsigned long long>(h.capture_timestamp_ns),
                  extensionFor(opts, h.format).c_str());
    const std::string path = opts.out_dir + "/" + name;

    bool ok = false;
    if (opts.format == "ppm") {
      ok = savePpm(path, frame.payload.data(), h.width, h.height, h.format);
    } else if (opts.format == "bgr") {
      ok = savePngBgr(path, frame.payload.data(), h.width, h.height, h.format);
    } else {
      std::ofstream out(path, std::ios::binary);
      out.write(reinterpret_cast<const char*>(frame.payload.data()),
                static_cast<std::streamsize>(frame.payload.size()));
      ok = static_cast<bool>(out);
    }
    if (!ok) {
      std::fprintf(stderr, "write failed: %s\n", path.c_str());
      break;
    }

    manifest << "{\"schema_version\":\"daedalus.offline-frame-capture/1\","
             << "\"file\":\"" << name << "\","
             << "\"format\":\""
             << (opts.format == "bgr" ? "png_bgr24" : formatName(h.format))
             << "\","
             << "\"width\":" << h.width << ','
             << "\"height\":" << h.height << ','
             << "\"producer_epoch\":" << h.producer_epoch << ','
             << "\"source_sequence\":" << h.source_sequence << ','
             << "\"capture_timestamp_ns\":" << h.capture_timestamp_ns << "}\n";

    after_seq = h.source_sequence;
    ++saved;
    std::printf("saved %s (%ux%u, %u bytes)\n", name, h.width, h.height,
                h.payload_bytes);
  }

  std::printf("done: %ld frame(s) in %s\n", saved, opts.out_dir.c_str());
  return 0;
}

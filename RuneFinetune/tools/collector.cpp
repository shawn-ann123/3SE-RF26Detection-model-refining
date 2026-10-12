// 采集器：原子化抓帧 + 同步读取真值。
// 每帧输出: frame_<seq>.ppm (RGB P6) + frame_<seq>.json (真值) + manifest.jsonl
// 用法: collector --out DIR [--count N] [--every-ms T] [--ipc-dir PATH]
#include <daedalus_sim_sdk/talos_metadata_reader.hpp>
#include <daedalus_sim_sdk/tcp_image_client.hpp>
#include <daedalus_sim_sdk/client_result.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include <unistd.h>

using namespace daedalus::sim::sdk::v1;

static std::string defaultIpcDir() {
    const char* xdg = std::getenv("XDG_RUNTIME_DIR");
    const std::string base = xdg != nullptr ? xdg : "/tmp";
    return base + "/daedalus-learning-" + std::to_string(::getuid());
}

static void writePpm(const std::string& path, const TcpImageFrame& f) {
    const std::uint32_t w = f.header.width, h = f.header.height;
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << w << " " << h << "\n255\n";
    const std::uint8_t* p = f.payload.data();
    // 支持 RGBA32 / RGB24
    const std::size_t bpp = (f.payload.size() == (std::size_t)w * h * 4) ? 4 : 3;
    std::vector<char> row((std::size_t)w * 3);
    for (std::uint32_t y = 0; y < h; ++y) {
        const std::uint8_t* src = p + (std::size_t)y * w * bpp;
        for (std::uint32_t x = 0; x < w; ++x) {
            row[(std::size_t)x * 3 + 0] = (char)src[(std::size_t)x * bpp + 0];
            row[(std::size_t)x * 3 + 1] = (char)src[(std::size_t)x * bpp + 1];
            row[(std::size_t)x * 3 + 2] = (char)src[(std::size_t)x * bpp + 2];
        }
        out.write(row.data(), row.size());
    }
}

static void writeJson(const std::string& path, std::uint64_t seq,
                      std::uint64_t ts, const GroundTruthBatch& batch,
                      const ExposureState& es, const std::string& ppm_file,
                      std::uint32_t w, std::uint32_t h) {
    std::ofstream out(path);
    out << "{\n";
    out << "  \"frame_seq\": " << seq << ",\n";
    out << "  \"timestamp_ns\": " << ts << ",\n";
    out << "  \"image\": {\"file\": \"" << ppm_file << "\", \"width\": " << w
        << ", \"height\": " << h << "},\n";
    out << "  \"exposure\": {\n";
    out << "    \"camera_position_world\": [" << es.camera_position_world[0] << ", "
        << es.camera_position_world[1] << ", " << es.camera_position_world[2] << "],\n";
    out << "    \"camera_quat_world_wxyz\": [" << es.camera_quaternion_world_wxyz[0]
        << ", " << es.camera_quaternion_world_wxyz[1] << ", "
        << es.camera_quaternion_world_wxyz[2] << ", "
        << es.camera_quaternion_world_wxyz[3] << "],\n";
    out << "    \"gimbal_position_world\": [" << es.gimbal_position_world[0] << ", "
        << es.gimbal_position_world[1] << ", " << es.gimbal_position_world[2] << "],\n";
    out << "    \"gimbal_quat_world_wxyz\": [" << es.gimbal_quaternion_world_wxyz[0]
        << ", " << es.gimbal_quaternion_world_wxyz[1] << ", "
        << es.gimbal_quaternion_world_wxyz[2] << ", "
        << es.gimbal_quaternion_world_wxyz[3] << "],\n";
    out << "    \"chassis_position_world\": [" << es.chassis_position_world[0] << ", "
        << es.chassis_position_world[1] << ", " << es.chassis_position_world[2] << "],\n";
    out << "    \"chassis_rpy_world\": [" << es.chassis_rpy_world[0] << ", "
        << es.chassis_rpy_world[1] << ", " << es.chassis_rpy_world[2] << "],\n";
    out << "    \"gimbal_yaw_rad\": " << es.gimbal_yaw_rad
        << ", \"gimbal_pitch_rad\": " << es.gimbal_pitch_rad << "\n";
    out << "  },\n";
    out << "  \"runes\": [\n";
    for (std::uint32_t r = 0; r < batch.rune_count; ++r) {
        const GroundTruthRune& ru = batch.runes[r];
        out << (r ? ",\n" : "") << "    {\n";
        out << "      \"team\": " << (unsigned)ru.team
            << ", \"rune_mode\": " << (unsigned)ru.rune_mode
            << ", \"mechanism_state\": " << (unsigned)ru.mechanism_state << ",\n";
        out << "      \"center\": [" << ru.r_center_odom[0] << ", "
            << ru.r_center_odom[1] << ", " << ru.r_center_odom[2] << "],\n";
        out << "      \"radius\": " << ru.radius << ", \"current_angle\": "
            << ru.current_angle << ", \"v_roll\": " << ru.v_roll
            << ", \"direction\": " << ru.direction << ",\n";
        out << "      \"sin\": {\"amp\": " << ru.sin_amplitude << ", \"omega\": "
            << ru.sin_omega << ", \"phase\": " << ru.sin_phase << ", \"offset\": "
            << ru.sin_offset << ", \"rel_time\": " << ru.relative_time << "},\n";
        out << "      \"blade_id\": " << ru.blade_id
            << ", \"target_activations\": [" << (unsigned)ru.target_activations[0]
            << ", " << (unsigned)ru.target_activations[1] << ", "
            << (unsigned)ru.target_activations[2] << ", "
            << (unsigned)ru.target_activations[3] << ", "
            << (unsigned)ru.target_activations[4] << "]\n";
        out << "    }";
    }
    out << "\n  ]\n}\n";
}

int main(int argc, char** argv) {
    std::string out_dir = ".";
    std::string ipc_dir = defaultIpcDir();
    int count = 10;
    int every_ms = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_dir = argv[++i];
        else if (std::strcmp(argv[i], "--count") == 0 && i + 1 < argc) count = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--every-ms") == 0 && i + 1 < argc) every_ms = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ipc-dir") == 0 && i + 1 < argc) ipc_dir = argv[++i];
    }

    TalosMetadataMapping mapping;
    ClientStatus st = mapping.open(ipc_dir + "/talos_ipc_meta");
    if (!st.ok()) { std::fprintf(stderr, "meta open failed: %s\n", st.message.c_str()); return 1; }
    auto rr = mapping.reader();
    if (!rr.ok()) { std::fprintf(stderr, "reader failed: %s\n", rr.status.message.c_str()); return 1; }
    TalosMetadataReader& reader = *rr.value;

    TcpImageClient img;
    ClientStatus cs = img.connect();
    if (!cs.ok()) { std::fprintf(stderr, "image connect failed: %s\n", cs.message.c_str()); return 1; }

    std::ofstream manifest(out_dir + "/manifest.jsonl", std::ios::app);
    std::uint64_t last_seq = 0;
    for (int i = 0; i < count; ++i) {
        auto f = img.waitForLatest(last_seq, std::chrono::seconds(5));
        if (!f.ok()) { std::fprintf(stderr, "wait frame failed: %s\n", f.status.message.c_str()); return 1; }
        const std::uint64_t seq = f.value->header.source_sequence;
        const std::uint64_t ts = f.value->header.capture_timestamp_ns;
        last_seq = seq;

        char name[128];
        std::snprintf(name, sizeof(name), "frame_%06llu.ppm", (unsigned long long)seq);
        writePpm(out_dir + "/" + name, *f.value);

        // 轮询等真值发布
        ClientResult<GroundTruthExposureSnapshot> gt;
        for (int t = 0; t < 60; ++t) {
            gt = reader.readGroundTruthForFrame(seq);
            if (gt.ok()) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if (!gt.ok()) {
            std::fprintf(stderr, "frame %llu: GT failed: %s\n",
                         (unsigned long long)seq, gt.status.message.c_str());
        }

        char jname[128];
        std::snprintf(jname, sizeof(jname), "frame_%06llu.json", (unsigned long long)seq);
        if (gt.ok()) {
            writeJson(out_dir + "/" + jname, seq, ts, gt.value->ground_truth,
                      gt.value->exposure_state, name,
                      f.value->header.width, f.value->header.height);
        }
        manifest << "{\"frame_seq\": " << seq << ", \"timestamp_ns\": " << ts
                 << ", \"file\": \"" << name << "\", \"gt\": \""
                 << (gt.ok() ? jname : "") << "\"}\n";
        manifest.flush();
        std::printf("captured frame_%llu gt=%s\n", (unsigned long long)seq,
                    gt.ok() ? "ok" : "FAILED");
        if (every_ms > 0 && i + 1 < count)
            std::this_thread::sleep_for(std::chrono::milliseconds(every_ms));
    }
    return 0;
}

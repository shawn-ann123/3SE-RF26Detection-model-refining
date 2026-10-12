// 读取模拟器真值：GroundTruthRune + ExposureState 完整 dump
// 用法: truth_dump [--ipc-dir PATH]
#include <daedalus_sim_sdk/talos_metadata_reader.hpp>
#include <daedalus_sim_sdk/client_result.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>

using namespace daedalus::sim::sdk::v1;

static std::string defaultIpcDir() {
    const char* xdg = std::getenv("XDG_RUNTIME_DIR");
    const std::string base = xdg != nullptr ? xdg : "/tmp";
    return base + "/daedalus-learning-" + std::to_string(::getuid());
}

int main(int argc, char** argv) {
    std::string ipc_dir = defaultIpcDir();
    long long frame_seq = -1;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--ipc-dir") == 0 && i + 1 < argc) {
            ipc_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--frame") == 0 && i + 1 < argc) {
            frame_seq = std::atoll(argv[++i]);
        }
    }

    TalosMetadataMapping mapping;
    ClientStatus st = mapping.open(ipc_dir + "/talos_ipc_meta");
    if (!st.ok()) {
        std::fprintf(stderr, "open failed: %s\n", st.message.c_str());
        return 1;
    }
    auto reader_result = mapping.reader();
    if (!reader_result.ok()) {
        std::fprintf(stderr, "reader failed: %s\n", reader_result.status.message.c_str());
        return 1;
    }
    TalosMetadataReader& reader = *reader_result.value;

    const bool want_frame = frame_seq >= 0;
    auto gt_frame = want_frame
        ? reader.readGroundTruthForFrame(static_cast<std::uint64_t>(frame_seq))
        : ClientResult<GroundTruthExposureSnapshot>();
    auto gt_latest = want_frame
        ? ClientResult<GroundTruthBatch>()
        : reader.readLatestGroundTruth();
    if ((want_frame && !gt_frame.ok()) || (!want_frame && !gt_latest.ok())) {
        const std::string msg = want_frame ? gt_frame.status.message : gt_latest.status.message;
        std::fprintf(stderr, "read ground truth failed: %s\n", msg.c_str());
        return 1;
    }
    const GroundTruthBatch& batch = want_frame ? gt_frame.value->ground_truth : *gt_latest.value;

    std::printf("== GroundTruthBatch ==\n");
    std::printf("frame_seq=%llu timestamp_ns=%llu target_count=%u rune_count=%u\n",
                (unsigned long long)batch.frame_seq,
                (unsigned long long)batch.timestamp_ns,
                batch.target_count, batch.rune_count);
    if (want_frame) {
        std::printf("producer_epoch=%llu publication=%llu\n",
                    (unsigned long long)gt_frame.value->producer_epoch,
                    (unsigned long long)gt_frame.value->publication);
    }

    for (std::uint32_t r = 0; r < batch.rune_count; ++r) {
        const GroundTruthRune& rune = batch.runes[r];
        std::printf("-- rune[%u] --\n", r);
        std::printf("  frame_seq=%llu timestamp_ns=%llu\n",
                    (unsigned long long)rune.frame_seq,
                    (unsigned long long)rune.timestamp_ns);
        std::printf("  team=%u rune_mode=%u mechanism_state=%u\n",
                    rune.team, rune.rune_mode, rune.mechanism_state);
        std::printf("  r_center_odom=[%.4f, %.4f, %.4f]\n",
                    rune.r_center_odom[0], rune.r_center_odom[1], rune.r_center_odom[2]);
        std::printf("  radius=%.4f current_angle=%.4f v_roll=%.4f direction=%d\n",
                    rune.radius, rune.current_angle, rune.v_roll, rune.direction);
        std::printf("  sin: amp=%.4f omega=%.4f phase=%.4f offset=%.4f rel_time=%.4f\n",
                    rune.sin_amplitude, rune.sin_omega, rune.sin_phase,
                    rune.sin_offset, rune.relative_time);
        std::printf("  blade_id=%d target_activations=[%u,%u,%u,%u,%u]\n",
                    rune.blade_id,
                    rune.target_activations[0], rune.target_activations[1],
                    rune.target_activations[2], rune.target_activations[3],
                    rune.target_activations[4]);
    }

    if (want_frame) {
        const ExposureState& es = gt_frame.value->exposure_state;
        std::printf("== ExposureState ==\n");
        std::printf("  frame_seq=%llu timestamp_ns=%llu\n",
                    (unsigned long long)es.frame_seq,
                    (unsigned long long)es.timestamp_ns);
        std::printf("  state_flags=0x%x world_frame=%u\n",
                    es.state_flags, es.world_frame);
        std::printf("  chassis_position_world=[%.4f, %.4f, %.4f]\n",
                    es.chassis_position_world[0], es.chassis_position_world[1],
                    es.chassis_position_world[2]);
        std::printf("  chassis_rpy_world=[%.4f, %.4f, %.4f]\n",
                    es.chassis_rpy_world[0], es.chassis_rpy_world[1],
                    es.chassis_rpy_world[2]);
        std::printf("  chassis_quat_world_wxyz=[%.6f, %.6f, %.6f, %.6f]\n",
                    es.chassis_quaternion_world_wxyz[0],
                    es.chassis_quaternion_world_wxyz[1],
                    es.chassis_quaternion_world_wxyz[2],
                    es.chassis_quaternion_world_wxyz[3]);
        std::printf("  gimbal_position_world=[%.4f, %.4f, %.4f]\n",
                    es.gimbal_position_world[0], es.gimbal_position_world[1],
                    es.gimbal_position_world[2]);
        std::printf("  gimbal_quat_world_wxyz=[%.6f, %.6f, %.6f, %.6f]\n",
                    es.gimbal_quaternion_world_wxyz[0],
                    es.gimbal_quaternion_world_wxyz[1],
                    es.gimbal_quaternion_world_wxyz[2],
                    es.gimbal_quaternion_world_wxyz[3]);
        std::printf("  camera_position_world=[%.4f, %.4f, %.4f]\n",
                    es.camera_position_world[0], es.camera_position_world[1],
                    es.camera_position_world[2]);
        std::printf("  camera_quat_world_wxyz=[%.6f, %.6f, %.6f, %.6f]\n",
                    es.camera_quaternion_world_wxyz[0],
                    es.camera_quaternion_world_wxyz[1],
                    es.camera_quaternion_world_wxyz[2],
                    es.camera_quaternion_world_wxyz[3]);
        std::printf("  gimbal_yaw_rad=%.4f gimbal_pitch_rad=%.4f\n",
                    es.gimbal_yaw_rad, es.gimbal_pitch_rad);
    } else {
        auto exposure = reader.readExposureStateForFrame(batch.frame_seq);
        if (!exposure.ok()) {
            std::fprintf(stderr, "readExposureStateForFrame failed: %s\n",
                         exposure.status.message.c_str());
        } else {
            const ExposureState& es = *exposure.value;
            std::printf("== ExposureState ==\n");
            std::printf("  frame_seq=%llu timestamp_ns=%llu\n",
                        (unsigned long long)es.frame_seq,
                        (unsigned long long)es.timestamp_ns);
            std::printf("  state_flags=0x%x world_frame=%u\n",
                        es.state_flags, es.world_frame);
            std::printf("  chassis_position_world=[%.4f, %.4f, %.4f]\n",
                        es.chassis_position_world[0], es.chassis_position_world[1],
                        es.chassis_position_world[2]);
            std::printf("  chassis_rpy_world=[%.4f, %.4f, %.4f]\n",
                        es.chassis_rpy_world[0], es.chassis_rpy_world[1],
                        es.chassis_rpy_world[2]);
            std::printf("  chassis_quat_world_wxyz=[%.6f, %.6f, %.6f, %.6f]\n",
                        es.chassis_quaternion_world_wxyz[0],
                        es.chassis_quaternion_world_wxyz[1],
                        es.chassis_quaternion_world_wxyz[2],
                        es.chassis_quaternion_world_wxyz[3]);
            std::printf("  gimbal_position_world=[%.4f, %.4f, %.4f]\n",
                        es.gimbal_position_world[0], es.gimbal_position_world[1],
                        es.gimbal_position_world[2]);
            std::printf("  gimbal_quat_world_wxyz=[%.6f, %.6f, %.6f, %.6f]\n",
                        es.gimbal_quaternion_world_wxyz[0],
                        es.gimbal_quaternion_world_wxyz[1],
                        es.gimbal_quaternion_world_wxyz[2],
                        es.gimbal_quaternion_world_wxyz[3]);
            std::printf("  camera_position_world=[%.4f, %.4f, %.4f]\n",
                        es.camera_position_world[0], es.camera_position_world[1],
                        es.camera_position_world[2]);
            std::printf("  camera_quat_world_wxyz=[%.6f, %.6f, %.6f, %.6f]\n",
                        es.camera_quaternion_world_wxyz[0],
                        es.camera_quaternion_world_wxyz[1],
                        es.camera_quaternion_world_wxyz[2],
                        es.camera_quaternion_world_wxyz[3]);
            std::printf("  gimbal_yaw_rad=%.4f gimbal_pitch_rad=%.4f\n",
                        es.gimbal_yaw_rad, es.gimbal_pitch_rad);
        }
    }

    auto gs = reader.readGimbalState();
    if (gs.ok()) {
        const GimbalState& g = *gs.value;
        std::printf("== GimbalState ==\n  yaw=%.4f pitch=%.4f yaw_vel=%.4f pitch_vel=%.4f\n",
                    g.yaw_deg, g.pitch_deg, g.yaw_velocity_deg_s, g.pitch_velocity_deg_s);
    }
    return 0;
}

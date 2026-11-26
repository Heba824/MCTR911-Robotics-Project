// Exercise the actual bridge callbacks without relying on DDS discovery.
#include <mujoco_ros2/mujoco_ros.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <thread>

struct TorqueBridgeTest
{

    static void require(bool value, const char *message)
    {
        if (!value) throw std::runtime_error(message);
    }

    static void send(MuJoCoROS &bridge, const std::vector<double> &packet,
                     std::chrono::milliseconds source_age = std::chrono::milliseconds(0))
    {
        auto message = std::make_shared<std_msgs::msg::Float64MultiArray>();
        message->data = packet;
        rclcpp::MessageInfo info{rmw_message_info_t{}};
        if (source_age.count())
            info.get_rmw_message_info().source_timestamp =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now().time_since_epoch() - source_age).count();
        bridge.joint_command_callback(message, info);
    }

    static void effort(const MuJoCoROS &bridge, const std::vector<double> &expected)
    {
        for (int i = 0; i < 7; ++i)
            require(std::abs(bridge._jointStateMessage.effort[i] - expected[i]) < 1e-10,
                    "Requested and applied arm torques differ");
    }

    static void torque(MuJoCoROS &bridge)
    {
        const auto *data = bridge._jointState;
        bridge.update_simulation();
        const auto home = bridge._jointStateMessage.position;
        send(bridge, std::vector<double>(8, 0));
        bridge.update_simulation();
        require(data->time == 0 && bridge._jointStateMessage.position == home,
                "Startup must remain paused after invalid command");

        send(bridge, std::vector<double>(9, 0), std::chrono::milliseconds(60));
        bridge.update_simulation();
        require(data->time == 0 && bridge._torqueState == MuJoCoROS::TorqueState::Waiting,
                "An expired DDS sample must not start the bridge");

        // Test-only reference at the home pose. The bridge never adds qfrc_bias.
        std::vector<double> hold(9, 0);
        for (int i = 0; i < 7; ++i) hold[i] = data->qfrc_bias[bridge._armDofs[i]];
        hold[7] = 255;
        std::array<double, 7> peak{};
        for (int tick = 0; tick < 200; ++tick)
        {
            send(bridge, hold);
            bridge.update_simulation();
            effort(bridge, hold);
            for (int i = 0; i < 7; ++i)
                peak[i] = std::max(peak[i], std::abs(home[i] - bridge._jointStateMessage.position[i]));
        }
        require(std::abs(data->time - 2.0) < 1e-9, "Wrong number of physics substeps");
        require(*std::max_element(peak.begin(), peak.end()) < 0.01, "Home hold drift exceeded 0.01 rad");
        std::cout << "Two simulated seconds at home. Peak deviation (rad):";
        for (double value : peak) std::cout << ' ' << value;
        std::cout << "\nRequested and applied hold torque (N m):";
        for (int i = 0; i < 7; ++i) std::cout << ' ' << hold[i];
        std::cout << '\n';

        for (int i = 0; i < 7; ++i)
        {
            for (double sign : {1.0, -1.0})
            {
                auto pulse = hold;
                pulse[i] += sign * 0.5;
                const double before = data->qvel[bridge._armDofs[i]];
                send(bridge, pulse);
                bridge.update_simulation();
                effort(bridge, pulse);
                const double change = data->qvel[bridge._armDofs[i]] - before;
                require(sign * change > 0, "Joint response sign mismatch");
                std::cout << "Joint " << i + 1 << " requested " << pulse[i]
                          << " applied " << bridge._jointStateMessage.effort[i]
                          << " N m, delta velocity " << change << " rad/s\n";
                // A second batch consumes the same command without another packet.
                bridge.update_simulation();
                effort(bridge, pulse);
                send(bridge, hold);
                bridge.update_simulation();
            }
        }
        for (double sign : {1.0, -1.0})
        {
            std::vector<double> bound{87, 87, 87, 87, 12, 12, 12, 255, 0};
            for (int i = 0; i < 7; ++i) bound[i] *= sign;
            send(bridge, bound);
            bridge.update_simulation();
            effort(bridge, bound);
        }
        send(bridge, hold);
        for (int c = 0; c < 7; ++c)
        {
            auto invalid = hold;
            invalid[7] = 0;
            if (c == 0) invalid.resize(8);
            if (c == 1) invalid.resize(10);
            if (c == 2) invalid[0] = std::nan("");
            if (c == 3) invalid[6] = 12.001;
            if (c == 4) invalid[7] = 256;
            if (c == 5) invalid[8] = 1;
            if (c == 6) invalid[0] = -INFINITY;
            send(bridge, invalid);
            bridge.update_simulation();
            effort(bridge, hold);
            require(data->ctrl[bridge._gripperActuator] == 255, "Invalid packet changed gripper");
        }

        const auto last_valid = bridge._lastTorqueCommand;
        send(bridge, std::vector<double>(9, 0), std::chrono::milliseconds(60));
        require(bridge._lastTorqueCommand == last_valid,
                "An expired DDS sample must not refresh the watchdog");

        hold[7] = 0;
        for (int tick = 0; tick < 25; ++tick)
        {
            send(bridge, hold);
            bridge.update_simulation();
            effort(bridge, hold);
        }
        require(bridge._jointStateMessage.position[7] < 0.025 &&
                bridge._jointStateMessage.position[8] < 0.025, "Gripper did not close independently");
        std::cout << "Finger positions (m): " << bridge._jointStateMessage.position[7]
                  << ' ' << bridge._jointStateMessage.position[8] << '\n';

        const std::vector<double> zero(9, 0);
        send(bridge, zero);
        bridge.update_simulation();
        effort(bridge, zero);
        const auto before_timeout = data->time;
        std::this_thread::sleep_for(std::chrono::milliseconds(110));
        send(bridge, std::vector<double>(8, 0));
        bridge.update_simulation();
        require(bridge._torqueState == MuJoCoROS::TorqueState::TimedOut, "Watchdog failed to latch");
        send(bridge, hold);
        bridge.update_simulation();
        require(data->time == before_timeout, "Timed-out simulation resumed");
        effort(bridge, zero);
        std::cout << "Accepted zero and signed limits, rejected seven malformed packets. "
                     "One watchdog event, latched after a 110 ms gap.\n";
    }

    static void legacy(MuJoCoROS &bridge)
    {
        bridge.update_simulation();
        auto target = bridge._jointStateMessage.position;
        target[0] += 0.08;
        if (bridge._controlMode == POSITION)
        {
            auto command = target;
            command[7] = 255;
            command[8] = 0;
            send(bridge, command);
        }
        else
        {
            for (int i = 0; i < 8; ++i)
                send(bridge, {0.2, 0, 0, 0, 0, 0, 0, 255, 0});
            send(bridge, {0, 0, 0, 0, 0, 0, 0, 255, 0});
        }
        for (int tick = 0; tick < 50; ++tick) bridge.update_simulation();
        double peak = 0;
        for (int i = 0; i < 7; ++i)
            peak = std::max(peak, std::abs(target[i] - bridge._jointStateMessage.position[i]));
        require(peak < 0.02, "Legacy arm target error exceeded 0.02 rad");
        std::cout << "Legacy peak arm error (rad): " << peak << '\n';
    }

    static void exchange(MuJoCoROS &bridge)
    {
        require(bridge._controlMode == TORQUE, "Exchange requires TORQUE");
        const auto report = [&bridge]() {
            const auto &state = bridge._jointStateMessage;
            std::cout << "STATE " << std::setprecision(17) << bridge._jointState->time
                      << ' ' << static_cast<int>(bridge._torqueState)
                      << ' ' << bridge._jointState->ncon;
            for (const auto *field : {&state.position, &state.velocity, &state.effort})
                for (int i = 0; i < 7; ++i) std::cout << ' ' << (*field)[i];
            std::cout << std::endl;
        };
        bridge.update_simulation();
        report();
        std::vector<double> packet(9);
        while (std::cin >> packet[0])
        {
            for (int i = 1; i < 9; ++i)
                require(static_cast<bool>(std::cin >> packet[i]), "Incomplete test packet");
            send(bridge, packet);
            bridge.update_simulation();
            report();
        }
    }

    static void run(const std::string &scene, bool exchange_commands)
    {
        MuJoCoROS bridge(scene);
        if (exchange_commands) exchange(bridge);
        else if (bridge._controlMode == TORQUE) torque(bridge);
        else legacy(bridge);
    }
};

int main(int argc, char **argv)
{
    if (argc < 2) return 2;
    rclcpp::init(argc, argv);
    std::cout << std::setprecision(9);
    int result = 0;
    try
    {
        TorqueBridgeTest::run(argv[1], argc > 2 && std::string(argv[2]) == "--exchange");
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        result = 1;
    }
    rclcpp::shutdown();
    return result;
}

#include "rclcpp/rclcpp.hpp"
#include <math.h>
#include <cmath>
#include <string>
#include <vector>
#include "sensor_msgs/msg/laser_scan.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "std_msgs/msg/color_rgba.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"

class ReactiveFollowGap : public rclcpp::Node {
public:
    ReactiveFollowGap() : Node("reactive_node") {
        // Subscribers
        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10,
            std::bind(&ReactiveFollowGap::scanCB, this, std::placeholders::_1));
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/ego_racecar/odom", 10,
            std::bind(&ReactiveFollowGap::odomCB, this, std::placeholders::_1));
        
        // Publishers
        drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
            "/drive", 10);
        viz_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "/gap_target", 10);
        gap_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>(
            "/gap_scan", 10);

        // Parameters
        this->declare_parameter("kernel_size", 3);
        kernel_size_ = this->get_parameter("kernel_size").as_int();
        this->declare_parameter("gap_lookahead", 3.);
        gap_lookahead_ = this->get_parameter("gap_lookahead").as_double();        
        this->declare_parameter("obstacle_buffer", 0.25);
        obstacle_buffer_ = this->get_parameter("obstacle_buffer").as_double();
        this->declare_parameter("fov", 180.);
        fov_ = this->get_parameter("fov").as_double();
        fov_ *= M_PI / 180.;

        param_timer_ = this->create_wall_timer(std::chrono::milliseconds(100),
            std::bind(&ReactiveFollowGap::paramTimer, this));
        
        // TF
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        tf_timer_ = this->create_wall_timer(std::chrono::milliseconds(100),
            std::bind(&ReactiveFollowGap::tfTimer, this));
    }

    void paramTimer() {
        kernel_size_ = this->get_parameter("kernel_size").as_int();
        gap_lookahead_ = this->get_parameter("gap_lookahead").as_double();
        obstacle_buffer_ = this->get_parameter("obstacle_buffer").as_double();
        fov_ = this->get_parameter("fov").as_double();
        fov_ *= M_PI / 180.;
    }

    void tfTimer() {
        std::string fromFrameRel = laser_frame_.c_str();
        std::string toFrameRel = target_frame_.c_str();

        try {
            lidar_transform_ = tf_buffer_->lookupTransform(
                toFrameRel, fromFrameRel,
                tf2::TimePointZero);
        } catch (const tf2::TransformException & ex) {
            RCLCPP_INFO(this->get_logger(), "Could not transform %s to %s: %s",
                        toFrameRel.c_str(), fromFrameRel.c_str(), ex.what());
            return;
        }
    }

private:
    int kernel_size_;
    double gap_lookahead_;
    double obstacle_buffer_;
    double fov_;

    nav_msgs::msg::Odometry curr_odom_;

    // ROS
    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    
    // Publishers
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr viz_pub_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr gap_pub_;

    // Timers
    rclcpp::TimerBase::SharedPtr param_timer_;
    rclcpp::TimerBase::SharedPtr tf_timer_;

    // Transforms
    geometry_msgs::msg::TransformStamped lidar_transform_;
    std::string laser_frame_ = "ego_racecar/laser";
    std::string target_frame_ = "ego_racecar/base_link";
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

    void convolve(std::vector<float>& signal, std::vector<float> kernel) {
        // Create result
        std::vector<float> result(signal.size(), 0.0);

        // Get kernel size and halfsize for boundary handling
        std::size_t kernelSize = kernel.size();
        std::size_t kernelHalfSize = kernelSize / 2;

        // Do convolution
        for (std::size_t i = 0; i < signal.size(); ++i) {
            // Iterate over kernel
            for (std::size_t j = 0; j < kernelSize; ++j) {
                // Find index for of signal for convolution operation
                std::size_t idx = i + j - kernelHalfSize;

                // Make sure idx within bounds
                if (idx < signal.size()) {
                    result[i] = signal[idx] * kernel[j];
                }
            }
        }

        signal = result;
        return;
    }

    void preprocess_lidar(std::vector<float>& ranges)
    {   
        // double lookahead_ = 3.0;
        // Find moving average of values over some kernel_size_ window
        // This is the same as a convolution with kernel of ones array
        std::vector<float> kernel(kernel_size_, 1.0);
        convolve(ranges, kernel);

        // // Trim all ranges to below lookahead
        // for (std::size_t i = 0; i < ranges.size(); ++i) {
        //     if (ranges[i] > lookahead_) {
        //         ranges[i] = lookahead_;
        //     }
        // }

        return;
    }

    void find_max_gap(std::vector<float> ranges, std::pair<int, int>& gap_idxs) {   
        int max_gap = 0;
        int gap = 0;
        bool new_gap = true;

        // Relative start idx, to the ranges vector
        int gap_start_idx = 0;
        
        for (std::size_t i = 0; i < ranges.size(); ++i) {
            // Definition of gap: Values further away than lookahead point
            if (ranges[i] >= gap_lookahead_) {
                // Set beginning index for gap
                if (new_gap) {
                    gap_start_idx = i;
                    new_gap = false;
                }
                // Count how big gap is
                gap++;

                // Determine if max
                if (gap > max_gap) {
                    max_gap = gap;
                    // int buff = obstacle_buffer_;

                    // Apply buffer
                    // Remove any scans whose range < obstacle buffer distance
                    if (ranges[i] < obstacle_buffer_) {
                        ranges[i] = 0;
                    }
                    gap_idxs = {gap_start_idx, i};
                }
            } else {
                gap = 0;
                new_gap = true;
            }
        }
        
        return; // Pass gap_idxs back by reference, relative to the ranges vector
    }

    int get_best_point(std::vector<float> ranges, std::pair<int, int> idxs)
    {   
        // idxs is relative to ranges vector
        double max = 0;
        int best_point = (idxs.first + idxs.second)/2;
        // for (int i = idxs.first; i <= idxs.second; ++i) {
        //     if (ranges[i] > max) {
        //         best_point = i;
        //         max = ranges[i];
        //     }
        // }
        return best_point;
    }

    void scanCB(const sensor_msgs::msg::LaserScan::ConstSharedPtr msg) 
    {   
        double current_fov = msg->angle_max - msg->angle_min;
        double crop_fov = fov_;

        // Calculate the new start/end index of the FOV
        int crop_start_idx = static_cast<int>((current_fov - crop_fov) / (2 * msg->angle_increment));
        int crop_end_idx = static_cast<int>(crop_fov / msg->angle_increment) + crop_start_idx;

        // Extract the scans within the new FOV
        std::vector<float> scans = msg->ranges;
        std::vector<float> cropped_scans(scans.begin() + crop_start_idx, scans.begin() + crop_end_idx);
        
        // Process
        // preprocess_lidar(cropped_scans);

        // Find max gap
        std::pair<int, int> gap;
        find_max_gap(cropped_scans, gap);

        // Find target
        int gap_idx = get_best_point(cropped_scans, gap);
        double target_angle = msg->angle_min + (gap_idx + crop_start_idx) * msg->angle_increment;

        // Command
        ackermann_msgs::msg::AckermannDriveStamped command;
        command.drive.steering_angle = target_angle;
        command.drive.speed = 0.25*cropped_scans[gap_idx];
        command.header.stamp = this->get_clock()->now();
        drive_pub_->publish(command);

        // Visualization
        // Find target point (Trim to threshold distance)
        tf2::Vector3 target_point;
        tf2::Vector3 target_transformed;
        double target_range = cropped_scans[gap_idx];
        if (target_range > gap_lookahead_) {
            target_range = gap_lookahead_;
        }
        target_point[0] = target_range * std::cos(target_angle);
        target_point[1] = target_range * std::sin(target_angle);
        target_point[2] = 0.;

        tf2::Transform l_trans;
        tf2::fromMsg(lidar_transform_.transform, l_trans);
        target_transformed = l_trans.getBasis() * target_point;

        geometry_msgs::msg::Pose target_pose;
        target_pose.position.x = target_transformed[0];
        target_pose.position.y = target_transformed[1];
        target_pose.position.z = target_transformed[2];

        // Gap target marker
        markerViz(target_pose);
        
        // Gap visualization
        sensor_msgs::msg::LaserScan gap_scan = *msg;
        gap_scan.header.stamp = this->get_clock()->now();
        gap_scan.header.frame_id = target_frame_.c_str();
        gap_scan.ranges = cropped_scans;
        gap_scan.angle_min = msg->angle_min + crop_start_idx * msg->angle_increment;
        gap_scan.angle_min = msg->angle_max - crop_end_idx * msg->angle_increment;
        gap_pub_->publish(gap_scan);

    }

    void odomCB(const nav_msgs::msg::Odometry::ConstSharedPtr odom_msg) {
        curr_odom_ = *odom_msg;
    }

    void markerViz(geometry_msgs::msg::Pose pose) {
        visualization_msgs::msg::Marker marker;
        marker.header.stamp = this->get_clock()->now();
        marker.header.frame_id = target_frame_.c_str();
        marker.type = 2;
        marker.pose = pose;
        geometry_msgs::msg::Vector3 scale;
        scale.x = 0.25;
        scale.y = 0.25;
        scale.z = 0.25;
        marker.scale = scale;
        std_msgs::msg::ColorRGBA color;
        color.r = 255;
        color.g = 0.;
        color.b = 0.;
        color.a = 1.;
        marker.color = color;
        viz_pub_->publish(marker);
        return;
    }

};
int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ReactiveFollowGap>());
    rclcpp::shutdown();
    return 0;
}

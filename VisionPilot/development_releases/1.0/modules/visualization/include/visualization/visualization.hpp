//
// Created by atanasko on 27.4.26.
//

#ifndef VISIONPILOT_VISUALIZATION_HPP
#define VISIONPILOT_VISUALIZATION_HPP

#include <opencv2/opencv.hpp>

#include <optional>
#include <string>
#include <vector>

namespace visualization {

inline constexpr int kFrameWidth = 1024;
inline constexpr int kFrameHeight = 512;
inline constexpr int kVisualizationPanelWidth = 360;
inline constexpr float kDetectionOverlayAlpha = 0.40F;
inline constexpr float kRightPanelAlpha = 0.40F;
inline constexpr float kDrivablePathAlpha = 0.45F;
inline constexpr float kPathPreviewMaxDistanceMeters = 100.0F;

inline const cv::Scalar kCipoColor(0, 0, 255);
inline const cv::Scalar kCuttingInColor(0, 255, 255);
inline const cv::Scalar kOtherCarsColor(255, 0, 0);
inline const cv::Scalar kPositiveAccelerationColor(0, 200, 0);
inline const cv::Scalar kNegativeAccelerationColor(0, 0, 255);
inline const cv::Scalar kWhiteColor(255, 255, 255);
inline const cv::Scalar kPanelTextColor(35, 35, 35);

const cv::Mat homography_matrix = (
	cv::Mat_<float>(3, 3) <<
		0.00209514907F, -0.000941721466F, -9.24906396F,
		0.00662758637F, -0.000352940531F, -3.33396502F,
		0.000120077371F, -0.00411343505F, 1.0F
);

struct YoloBoundingBox {
	int class_id = 0;
	float center_x = 0.0F;
	float center_y = 0.0F;
	float width = 0.0F;
	float height = 0.0F;
};

struct LaneShapeVisualization {
	bool has_cipo_object = false;
	std::optional<float> distance_to_cipo;
	std::optional<float> relative_cipo_velocity;
	// Normal image coordinates in the 1024x512 frame.
	std::vector<cv::Point2f> tracked_waypoints;
};

struct DesiredControlVisualization {
	float steering_angle = 0.0F;
	float velocity = 0.0F;
	float acceleration = 0.0F;
};

bool render_frame(
	const cv::Mat &frame,
	const std::string &window_name = "VisionPilot",
	const std::vector<std::string> &overlay_lines = {}
);

cv::Mat visualize_frame(
	const cv::Mat &frame,
	const std::vector<YoloBoundingBox> &bounding_boxes,
	const LaneShapeVisualization &lane_shape,
	const DesiredControlVisualization &desired_control
);

void close_windows();

}  // namespace visualization

#endif //VISIONPILOT_VISUALIZATION_HPP

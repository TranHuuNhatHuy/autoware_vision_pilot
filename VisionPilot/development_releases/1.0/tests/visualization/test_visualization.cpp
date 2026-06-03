//
// Created by Copilot.
//

#include <visualization/visualization.hpp>

#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

template <typename T>
std::vector<T> read_scalar_sequence(const cv::FileNode &node) {
	std::vector<T> values;
	if (node.empty() || !node.isSeq()) {
		return values;
	}

	for (auto it = node.begin(); it != node.end(); ++it) {
		T value{};
		*it >> value;
		values.push_back(value);
	}

	return values;
}

std::vector<visualization::YoloBoundingBox> load_bounding_boxes(const cv::FileNode &node) {
	std::vector<visualization::YoloBoundingBox> bounding_boxes;
	if (node.empty() || !node.isSeq()) {
		return bounding_boxes;
	}

	for (auto it = node.begin(); it != node.end(); ++it) {
		const std::vector<float> values = read_scalar_sequence<float>(*it);
		if (values.size() < 5) {
			continue;
		}

		visualization::YoloBoundingBox box;
		box.class_id = static_cast<int>(values[0]);
		box.center_x = values[1];
		box.center_y = values[2];
		box.width = values[3];
		box.height = values[4];
		bounding_boxes.push_back(box);
	}

	return bounding_boxes;
}

visualization::LaneShapeVisualization load_lane_shape(const cv::FileNode &node) {
	visualization::LaneShapeVisualization lane_shape;
	if (node.empty()) {
		return lane_shape;
	}

	node["has_cipo_object"] >> lane_shape.has_cipo_object;

	double distance_to_cipo = 0.0;
	node["distance_to_cipo"] >> distance_to_cipo;
	lane_shape.distance_to_cipo = static_cast<float>(distance_to_cipo);

	double relative_velocity = 0.0;
	node["relative_cipo_velocity"] >> relative_velocity;
	lane_shape.relative_cipo_velocity = static_cast<float>(relative_velocity);

	const cv::FileNode waypoints = node["tracked_waypoints"];
	if (!waypoints.empty() && waypoints.isSeq()) {
		for (auto it = waypoints.begin(); it != waypoints.end(); ++it) {
			const std::vector<float> values = read_scalar_sequence<float>(*it);
			if (values.size() < 2) {
				continue;
			}
			lane_shape.tracked_waypoints.emplace_back(values[0], values[1]);
		}
	}

	return lane_shape;
}

visualization::DesiredControlVisualization load_desired_control(const cv::FileNode &node) {
	visualization::DesiredControlVisualization desired_control;
	if (node.empty()) {
		return desired_control;
	}

	node["steering_angle"] >> desired_control.steering_angle;
	node["velocity"] >> desired_control.velocity;
	node["acceleration"] >> desired_control.acceleration;
	return desired_control;
}

bool should_show_window() {
	const char *show_flag = std::getenv("VISUALIZATION_TEST_SHOW");
	return show_flag != nullptr && std::string(show_flag) == "1";
}

}  // namespace

int main(int argc, char **argv) {
	const std::filesystem::path asset_dir = argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::path(VISUALIZATION_TEST_ASSET_DIR);
	const std::filesystem::path frame_path = argc > 2 ? std::filesystem::path(argv[2]) : asset_dir / "test_frame.png";
	const std::filesystem::path yaml_path = argc > 3 ? std::filesystem::path(argv[3]) : asset_dir / "test_upstream_data.yaml";
	const std::filesystem::path output_path = argc > 4 ? std::filesystem::path(argv[4]) : std::filesystem::path(VISUALIZATION_TEST_OUTPUT_PATH);

	const cv::Mat frame = cv::imread(frame_path.string(), cv::IMREAD_COLOR);
	if (frame.empty()) {
		std::cerr << "Failed to load frame: " << frame_path << '\n';
		return 1;
	}

	const cv::FileStorage file_storage(yaml_path.string(), cv::FileStorage::READ);
	if (!file_storage.isOpened()) {
		std::cerr << "Failed to open upstream YAML: " << yaml_path << '\n';
		return 1;
	}

	const cv::FileNode autospeed = file_storage["autospeed"];
	const cv::FileNode lane_shape_node = file_storage["lane_shape"];
	const cv::FileNode desired_planning_node = file_storage["desired_planning"];

	const std::vector<visualization::YoloBoundingBox> bounding_boxes = load_bounding_boxes(autospeed);
	const visualization::LaneShapeVisualization lane_shape = load_lane_shape(lane_shape_node);
	const visualization::DesiredControlVisualization desired_control = load_desired_control(desired_planning_node);

	const cv::Mat rendered = visualization::visualize_frame(frame, bounding_boxes, lane_shape, desired_control);
	if (rendered.empty()) {
		std::cerr << "Visualization produced an empty frame.\n";
		return 1;
	}

	if (!cv::imwrite(output_path.string(), rendered)) {
		std::cerr << "Failed to write output image: " << output_path << '\n';
		return 1;
	}

	std::cout << "Saved rendered visualization to: " << output_path << '\n';

	if (should_show_window()) {
		visualization::render_frame(rendered, "VisionPilot Visualization Test");
		std::cout << "Press any key in the window to close it.\n";
		cv::waitKey(0);
		visualization::close_windows();
	} else {
		std::cout << "GUI display is disabled by default. Set VISUALIZATION_TEST_SHOW=1 to open a window.\n";
	}

	return 0;
}

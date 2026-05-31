//
// Created by atanasko on 27.4.26.
//

#include <visualization/visualization.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>

namespace visualization {

namespace {

cv::Scalar class_color(int class_id) {
	switch (class_id) {
		case 1:
			return kCipoColor;
		case 2:
			return kCuttingInColor;
		case 3:
			return kOtherCarsColor;
		default:
			return cv::Scalar(180, 180, 180);
	}
}

std::string format_float(float value, int precision) {
	std::ostringstream stream;
	stream << std::fixed << std::setprecision(precision) << value;
	return stream.str();
}

cv::Mat blend_overlay(const cv::Mat &base, const cv::Mat &overlay, float alpha) {
	cv::Mat result;
	cv::addWeighted(overlay, alpha, base, 1.0F - alpha, 0.0, result);
	return result;
}

cv::Point2f project_bev_to_image(const cv::Point2f &point) {
	const cv::Vec3f homogeneous(point.x, point.y, 1.0F);
	const cv::Vec3f projected = kBevToNormal * homogeneous;
	if (std::abs(projected[2]) < 1e-6F) {
		return cv::Point2f(-1.0F, -1.0F);
	}
	return cv::Point2f(projected[0] / projected[2], projected[1] / projected[2]);
}

cv::Mat load_wheel_icon() {
	const std::vector<std::filesystem::path> candidates = {
		std::filesystem::current_path() / "Media" / "wheel.png",
		std::filesystem::current_path() / ".." / "Media" / "wheel.png",
		std::filesystem::current_path() / ".." / ".." / "Media" / "wheel.png",
		std::filesystem::current_path() / ".." / ".." / ".." / "Media" / "wheel.png"
	};

	for (const auto &candidate : candidates) {
		cv::Mat icon = cv::imread(candidate.string(), cv::IMREAD_UNCHANGED);
		if (!icon.empty()) {
			return icon;
		}
	}

	return cv::Mat();
}

cv::Mat rotate_icon(const cv::Mat &icon, float angle_degrees) {
	if (icon.empty()) {
		return cv::Mat();
	}

	const cv::Point2f center(icon.cols * 0.5F, icon.rows * 0.5F);
	const cv::Mat rotation = cv::getRotationMatrix2D(center, angle_degrees, 1.0);

	const cv::Rect2f bounds = cv::RotatedRect(cv::Point2f(), icon.size(), angle_degrees).boundingRect2f();
	cv::Mat adjusted_rotation = rotation.clone();
	adjusted_rotation.at<double>(0, 2) += bounds.width * 0.5 - center.x;
	adjusted_rotation.at<double>(1, 2) += bounds.height * 0.5 - center.y;

	cv::Mat rotated;
	const cv::Scalar border_color = icon.channels() == 4 ? cv::Scalar(0, 0, 0, 0) : cv::Scalar(255, 255, 255);
	cv::warpAffine(
		icon,
		rotated,
		adjusted_rotation,
		bounds.size(),
		cv::INTER_LINEAR,
		cv::BORDER_CONSTANT,
		border_color
	);

	return rotated;
}

void overlay_icon(cv::Mat &canvas, const cv::Mat &icon, const cv::Point &top_left) {
	if (icon.empty()) {
		return;
	}

	const int x = std::clamp(top_left.x, 0, std::max(0, canvas.cols - 1));
	const int y = std::clamp(top_left.y, 0, std::max(0, canvas.rows - 1));
	const int width = std::min(icon.cols, canvas.cols - x);
	const int height = std::min(icon.rows, canvas.rows - y);
	if (width <= 0 || height <= 0) {
		return;
	}

	const cv::Rect dst_rect(x, y, width, height);
	const cv::Rect src_rect(0, 0, width, height);
	cv::Mat dst_roi = canvas(dst_rect);
	const cv::Mat src_roi = icon(src_rect);

	if (src_roi.channels() == 4) {
		std::vector<cv::Mat> channels;
		cv::split(src_roi, channels);
		const cv::Mat alpha = channels[3];
		cv::Mat src_bgr;
		cv::merge(std::vector<cv::Mat>{channels[0], channels[1], channels[2]}, src_bgr);

		cv::Mat src_float;
		cv::Mat dst_float;
		src_bgr.convertTo(src_float, CV_32FC3);
		dst_roi.convertTo(dst_float, CV_32FC3);

		cv::Mat alpha_float;
		alpha.convertTo(alpha_float, CV_32FC1, 1.0 / 255.0);
		std::vector<cv::Mat> alpha_channels{alpha_float, alpha_float, alpha_float};
		cv::Mat alpha_3;
		cv::merge(alpha_channels, alpha_3);

		const cv::Mat ones(alpha_3.size(), alpha_3.type(), cv::Scalar::all(1.0));
		const cv::Mat blended = src_float.mul(alpha_3) + dst_float.mul(ones - alpha_3);
		blended.convertTo(dst_roi, dst_roi.type());
	} else {
		src_roi.copyTo(dst_roi);
	}
}

void draw_text_centered(cv::Mat &canvas, const std::string &text, const cv::Rect &rect, double scale, const cv::Scalar &color, int thickness = 1) {
	int baseline = 0;
	const cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
	const int x = rect.x + std::max(0, (rect.width - text_size.width) / 2);
	const int y = rect.y + std::max(text_size.height, (rect.height + text_size.height) / 2);
	cv::putText(canvas, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv::LINE_AA);
}

void draw_boxed_value(cv::Mat &canvas, const cv::Rect &rect, const std::string &title, const std::string &value) {
	cv::rectangle(canvas, rect, cv::Scalar(255, 255, 255), cv::FILLED);
	cv::rectangle(canvas, rect, cv::Scalar(200, 200, 200), 1);

	const int title_y = rect.y + 22;
	const int value_y = rect.y + rect.height / 2 + 24;
	cv::putText(canvas, title, cv::Point(rect.x + 12, title_y), cv::FONT_HERSHEY_SIMPLEX, 0.6, kPanelTextColor, 1, cv::LINE_AA);
	cv::putText(canvas, value, cv::Point(rect.x + 12, value_y), cv::FONT_HERSHEY_SIMPLEX, 0.72, kPanelTextColor, 2, cv::LINE_AA);
}

cv::Mat make_translucent_panel(int width, int height) {
	cv::Mat base(height, width, CV_8UC3, cv::Scalar(245, 245, 245));
	cv::Mat overlay(height, width, CV_8UC3, kWhiteColor);
	return blend_overlay(base, overlay, kRightPanelAlpha);
}

cv::Rect make_clamped_rect(const cv::Rect &rect, const cv::Size &size) {
	const int x = std::clamp(rect.x, 0, std::max(0, size.width - 1));
	const int y = std::clamp(rect.y, 0, std::max(0, size.height - 1));
	const int right = std::clamp(rect.x + rect.width, 0, size.width);
	const int bottom = std::clamp(rect.y + rect.height, 0, size.height);
	return cv::Rect(x, y, std::max(0, right - x), std::max(0, bottom - y));
}

cv::Rect yolo_to_rect(const YoloBoundingBox &box, const cv::Size &size) {
	const float cx = box.center_x * static_cast<float>(size.width);
	const float cy = box.center_y * static_cast<float>(size.height);
	const float width = box.width * static_cast<float>(size.width);
	const float height = box.height * static_cast<float>(size.height);
	const cv::Rect rect(
		static_cast<int>(std::lround(cx - width * 0.5F)),
		static_cast<int>(std::lround(cy - height * 0.5F)),
		std::max(1, static_cast<int>(std::lround(width))),
		std::max(1, static_cast<int>(std::lround(height)))
	);
	return make_clamped_rect(rect, size);
}

void draw_detection_boxes(cv::Mat &frame, const std::vector<YoloBoundingBox> &bounding_boxes) {
	if (bounding_boxes.empty()) {
		return;
	}

	cv::Mat overlay = frame.clone();
	for (const auto &box : bounding_boxes) {
		const cv::Rect rect = yolo_to_rect(box, frame.size());
		if (rect.width <= 0 || rect.height <= 0) {
			continue;
		}
		cv::rectangle(overlay, rect, class_color(box.class_id), cv::FILLED);
	}

	frame = blend_overlay(frame, overlay, kDetectionOverlayAlpha);
}

std::vector<cv::Point> build_path_polygon(const std::vector<cv::Point2f> &centerline, const cv::Size &size) {
	std::vector<cv::Point> left_side;
	std::vector<cv::Point> right_side;
	left_side.reserve(centerline.size());
	right_side.reserve(centerline.size());

	for (std::size_t index = 0; index < centerline.size(); ++index) {
		const cv::Point2f current = centerline[index];
		const cv::Point2f previous = index > 0 ? centerline[index - 1] : centerline[index];
		const cv::Point2f next = index + 1 < centerline.size() ? centerline[index + 1] : centerline[index];

		cv::Point2f tangent = next - previous;
		const float tangent_norm = std::sqrt(tangent.x * tangent.x + tangent.y * tangent.y);
		if (tangent_norm > 1e-4F) {
			tangent.x /= tangent_norm;
			tangent.y /= tangent_norm;
		} else {
			tangent = cv::Point2f(0.0F, -1.0F);
		}

		const cv::Point2f normal(-tangent.y, tangent.x);
		const float y_ratio = std::clamp(current.y / std::max(1.0F, static_cast<float>(size.height - 1)), 0.0F, 1.0F);
		const float half_width = size.width * 0.125F * y_ratio;

		left_side.emplace_back(cv::Point(
			static_cast<int>(std::lround(current.x + normal.x * half_width)),
			static_cast<int>(std::lround(current.y + normal.y * half_width))
		));
		right_side.emplace_back(cv::Point(
			static_cast<int>(std::lround(current.x - normal.x * half_width)),
			static_cast<int>(std::lround(current.y - normal.y * half_width))
		));
	}

	std::vector<cv::Point> polygon;
	polygon.reserve(left_side.size() + right_side.size());
	polygon.insert(polygon.end(), left_side.begin(), left_side.end());
	for (auto it = right_side.rbegin(); it != right_side.rend(); ++it) {
		polygon.push_back(*it);
	}

	return polygon;
}

void draw_main_drivable_path(cv::Mat &frame, const std::vector<cv::Point2f> &tracked_waypoints, float acceleration) {
	if (tracked_waypoints.size() < 2) {
		return;
	}

	std::vector<cv::Point2f> projected_points;
	projected_points.reserve(tracked_waypoints.size());
	for (const auto &waypoint : tracked_waypoints) {
		const cv::Point2f projected = project_bev_to_image(waypoint);
		if (projected.x >= 0.0F && projected.y >= 0.0F) {
			projected_points.push_back(projected);
		}
	}

	if (projected_points.size() < 2) {
		return;
	}

	const cv::Scalar path_color = acceleration >= 0.0F ? kPositiveAccelerationColor : kNegativeAccelerationColor;
	const std::vector<cv::Point> polygon = build_path_polygon(projected_points, frame.size());
	if (polygon.size() < 3) {
		return;
	}

	cv::Mat overlay = frame.clone();
	std::vector<std::vector<cv::Point>> polygons{polygon};
	cv::fillPoly(overlay, polygons, path_color);
	frame = blend_overlay(frame, overlay, kDrivablePathAlpha);
}

cv::Point2f interpolate_waypoint(
	const std::vector<cv::Point2f> &waypoints,
	float target_distance
) {
	if (waypoints.empty()) {
		return cv::Point2f();
	}

	if (target_distance <= waypoints.front().x) {
		return waypoints.front();
	}

	for (std::size_t index = 0; index + 1 < waypoints.size(); ++index) {
		const cv::Point2f &a = waypoints[index];
		const cv::Point2f &b = waypoints[index + 1];
		if (target_distance < a.x || target_distance > b.x) {
			continue;
		}

		const float span = b.x - a.x;
		if (std::abs(span) < 1e-4F) {
			return a;
		}

		const float ratio = (target_distance - a.x) / span;
		return cv::Point2f(
			target_distance,
			a.y + ratio * (b.y - a.y)
		);
	}

	return waypoints.back();
}

void draw_bev_ruler(cv::Mat &canvas, const cv::Rect &area, float max_distance_m) {
	const int ruler_width = 62;
	const cv::Rect ruler_rect(area.x + area.width - ruler_width, area.y, ruler_width, area.height);
	cv::rectangle(canvas, ruler_rect, cv::Scalar(236, 236, 236), cv::FILLED);
	cv::line(canvas, cv::Point(ruler_rect.x, ruler_rect.y), cv::Point(ruler_rect.x, ruler_rect.y + ruler_rect.height), cv::Scalar(170, 170, 170), 1);

	for (int tick = 0; tick <= 10; ++tick) {
		const float ratio = static_cast<float>(tick) / 10.0F;
		const int y = ruler_rect.y + ruler_rect.height - static_cast<int>(std::lround(ratio * static_cast<float>(ruler_rect.height)));
		const int tick_length = (tick % 5 == 0) ? 16 : 9;
		cv::line(canvas, cv::Point(ruler_rect.x, y), cv::Point(ruler_rect.x + tick_length, y), cv::Scalar(120, 120, 120), 1);

		if (tick % 2 == 0) {
			const int distance = static_cast<int>(std::lround(ratio * max_distance_m));
			cv::putText(canvas, std::to_string(distance) + "m", cv::Point(ruler_rect.x + tick_length + 4, y + 5), cv::FONT_HERSHEY_SIMPLEX, 0.42, kPanelTextColor, 1, cv::LINE_AA);
		}
	}
}

void draw_bev_waypoints(cv::Mat &canvas, const std::vector<cv::Point2f> &tracked_waypoints, const cv::Rect &area, float max_distance_m) {
	if (tracked_waypoints.size() < 2) {
		return;
	}

	const float lateral_half_span_m = kBevLateralHalfSpanMeters;
	const int ruler_width = 62;
	const cv::Rect path_rect(area.x + 8, area.y + 8, std::max(1, area.width - ruler_width - 16), std::max(1, area.height - 16));

	std::vector<cv::Point> polyline_points;
	polyline_points.reserve(tracked_waypoints.size());
	for (const auto &waypoint : tracked_waypoints) {
		const float forward_m = std::clamp(waypoint.x, 0.0F, max_distance_m);
		const float lateral_m = std::clamp(waypoint.y, -lateral_half_span_m, lateral_half_span_m);
		const float x_ratio = (lateral_m + lateral_half_span_m) / (2.0F * lateral_half_span_m);
		const float y_ratio = forward_m / max_distance_m;
		const int x = path_rect.x + static_cast<int>(std::lround(x_ratio * static_cast<float>(path_rect.width)));
		const int y = path_rect.y + path_rect.height - static_cast<int>(std::lround(y_ratio * static_cast<float>(path_rect.height)));
		polyline_points.emplace_back(x, y);
	}

	if (polyline_points.size() >= 2) {
		cv::Mat overlay = canvas.clone();
		std::vector<std::vector<cv::Point>> polylines{polyline_points};
		const cv::Scalar lane_color(40, 180, 90);
		cv::polylines(overlay, polylines, false, lane_color, 4, cv::LINE_AA);
		canvas = blend_overlay(canvas, overlay, 0.35F);
	}
}

void draw_cipo_marker(cv::Mat &canvas, const std::vector<cv::Point2f> &tracked_waypoints, const LaneShapeVisualization &lane_shape, const cv::Rect &area, float max_distance_m) {
	if (!lane_shape.has_cipo_object || !lane_shape.distance_to_cipo.has_value()) {
		return;
	}

	const float distance_m = std::clamp(*lane_shape.distance_to_cipo, 0.0F, max_distance_m);
	const float lateral_half_span_m = kBevLateralHalfSpanMeters;
	const int ruler_width = 62;
	const cv::Rect path_rect(area.x + 8, area.y + 8, std::max(1, area.width - ruler_width - 16), std::max(1, area.height - 16));

	const cv::Point2f cipo_waypoint = interpolate_waypoint(tracked_waypoints, distance_m);
	const float x_ratio = (std::clamp(cipo_waypoint.y, -lateral_half_span_m, lateral_half_span_m) + lateral_half_span_m) / (2.0F * lateral_half_span_m);
	const int x = path_rect.x + static_cast<int>(std::lround(x_ratio * static_cast<float>(path_rect.width)));
	const int y = path_rect.y + path_rect.height - static_cast<int>(std::lround(distance_m / max_distance_m * static_cast<float>(path_rect.height)));

	const cv::Rect marker_rect(x - 12, y - 16, 24, 20);
	cv::rectangle(canvas, marker_rect, cv::Scalar(60, 60, 230), cv::FILLED);
	cv::rectangle(canvas, marker_rect, cv::Scalar(255, 255, 255), 1);

	const std::string distance_text = format_float(*lane_shape.distance_to_cipo, 1) + " m";
	const std::string velocity_text = lane_shape.relative_cipo_velocity.has_value() ? format_float(*lane_shape.relative_cipo_velocity, 1) + " km/h" : "-- km/h";
	cv::putText(canvas, distance_text, cv::Point(x - 24, std::max(area.y + 14, y - 22)), cv::FONT_HERSHEY_SIMPLEX, 0.42, kPanelTextColor, 1, cv::LINE_AA);
	cv::putText(canvas, velocity_text, cv::Point(x - 24, std::max(area.y + 28, y - 8)), cv::FONT_HERSHEY_SIMPLEX, 0.42, kPanelTextColor, 1, cv::LINE_AA);
}

void draw_right_panel(cv::Mat &canvas, const std::vector<cv::Point2f> &tracked_waypoints, const LaneShapeVisualization &lane_shape, const DesiredControlVisualization &desired_control) {
	const int panel_width = kVisualizationPanelWidth;
	const cv::Rect panel_rect(canvas.cols - panel_width, 0, panel_width, canvas.rows);
	if (panel_rect.x < 0) {
		return;
	}

	cv::Mat panel = make_translucent_panel(panel_rect.width, panel_rect.height);

	cv::rectangle(panel, cv::Rect(12, 12, panel_rect.width - 24, 150), cv::Scalar(255, 255, 255), cv::FILLED);
	cv::rectangle(panel, cv::Rect(12, 12, panel_rect.width - 24, 150), cv::Scalar(210, 210, 210), 1);
	draw_text_centered(panel, "Desired planning values", cv::Rect(12, 20, panel_rect.width - 24, 30), 0.58, kPanelTextColor, 2);

	const cv::Rect velocity_rect(30, 64, 120, 68);
	draw_boxed_value(panel, velocity_rect, "Velocity", format_float(desired_control.velocity, 1) + " km/h");

	const cv::Rect steering_rect(160, 64, 120, 68);
	draw_boxed_value(panel, steering_rect, "Steering", format_float(desired_control.steering_angle, 1) + " deg");

	const cv::Rect wheel_rect(192, 86, 130, 130);
	const cv::Mat wheel_icon = load_wheel_icon();
	if (!wheel_icon.empty()) {
		const cv::Mat rotated = rotate_icon(wheel_icon, desired_control.steering_angle);
		const int wheel_x = panel_rect.width - rotated.cols - 18;
		const int wheel_y = 38;
		overlay_icon(panel, rotated, cv::Point(wheel_x, wheel_y));
	} else {
		cv::circle(panel, cv::Point(wheel_rect.x + wheel_rect.width / 2, wheel_rect.y + wheel_rect.height / 2), 40, cv::Scalar(80, 80, 80), 3);
	}

	cv::putText(panel, format_float(desired_control.velocity, 1) + " km/h", cv::Point(24, 120), cv::FONT_HERSHEY_SIMPLEX, 0.76, kPanelTextColor, 2, cv::LINE_AA);
	cv::putText(panel, "Acceleration", cv::Point(24, 168), cv::FONT_HERSHEY_SIMPLEX, 0.56, kPanelTextColor, 1, cv::LINE_AA);
	const cv::Scalar acceleration_color = desired_control.acceleration >= 0.0F ? cv::Scalar(50, 190, 80) : cv::Scalar(70, 70, 230);
	cv::putText(panel, format_float(desired_control.acceleration, 1) + " m/s2", cv::Point(24, 190), cv::FONT_HERSHEY_SIMPLEX, 0.54, acceleration_color, 2, cv::LINE_AA);

	const cv::Rect bev_rect(12, 178, panel_rect.width - 24, panel_rect.height - 190);
	cv::rectangle(panel, bev_rect, cv::Scalar(255, 255, 255), cv::FILLED);
	cv::rectangle(panel, bev_rect, cv::Scalar(210, 210, 210), 1);
	cv::putText(panel, "BEV path", cv::Point(bev_rect.x + 10, bev_rect.y + 22), cv::FONT_HERSHEY_SIMPLEX, 0.55, kPanelTextColor, 1, cv::LINE_AA);

	const cv::Rect path_area(bev_rect.x + 10, bev_rect.y + 32, bev_rect.width - 20, bev_rect.height - 42);
	draw_bev_ruler(panel, path_area, kBevRulerMaxDistanceMeters);
	draw_bev_waypoints(panel, tracked_waypoints, path_area, kBevRulerMaxDistanceMeters);
	draw_cipo_marker(panel, tracked_waypoints, lane_shape, path_area, kBevRulerMaxDistanceMeters);

	panel.copyTo(canvas(panel_rect));
}

void draw_main_overlay(cv::Mat &frame, const std::vector<YoloBoundingBox> &bounding_boxes, const LaneShapeVisualization &lane_shape, const DesiredControlVisualization &desired_control) {
	draw_detection_boxes(frame, bounding_boxes);
	draw_main_drivable_path(frame, lane_shape.tracked_waypoints, desired_control.acceleration);
}

}  // namespace

bool render_frame(
	const cv::Mat &frame,
	const std::string &window_name,
	const std::vector<std::string> &overlay_lines
) {
	if (frame.empty()) {
		return false;
	}

	cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, frame.cols, frame.rows);

	cv::Mat display = frame.clone();

	if (!overlay_lines.empty()) {
		const int font_face = cv::FONT_HERSHEY_SIMPLEX;
		const double font_scale = 0.55;
		const int thickness = 1;
		const int line_gap = 8;
		const int left_padding = 12;
		const int top_padding = 24;

		int box_width = 0;
		int box_height = 0;
		for (const auto &line : overlay_lines) {
			int baseline = 0;
			cv::Size text_size = cv::getTextSize(line, font_face, font_scale, thickness, &baseline);
			box_width = std::max(box_width, text_size.width);
			box_height += text_size.height + line_gap;
		}

		cv::rectangle(
			display,
			cv::Rect(6, 6, box_width + left_padding * 2, box_height + top_padding),
			cv::Scalar(0, 0, 0),
			cv::FILLED
		);

		int y = 6 + top_padding - 6;
		for (const auto &line : overlay_lines) {
			int baseline = 0;
			cv::Size text_size = cv::getTextSize(line, font_face, font_scale, thickness, &baseline);
			y += text_size.height;
			cv::putText(
				display,
				line,
				cv::Point(12, y),
				font_face,
				font_scale,
				cv::Scalar(0, 255, 255),
				thickness,
				cv::LINE_AA
			);
			y += line_gap;
		}
	}

	cv::imshow(window_name, display);
	cv::waitKey(1);
	return true;
}

cv::Mat visualize_frame(
	const cv::Mat &frame,
	const std::vector<YoloBoundingBox> &bounding_boxes,
	const LaneShapeVisualization &lane_shape,
	const DesiredControlVisualization &desired_control
) {
	if (frame.empty()) {
		return cv::Mat();
	}

	const int panel_width = kVisualizationPanelWidth;
	cv::Mat output(frame.rows, frame.cols + panel_width, frame.type(), cv::Scalar(28, 28, 30));
	frame.copyTo(output(cv::Rect(0, 0, frame.cols, frame.rows)));

	cv::Mat left_frame = output(cv::Rect(0, 0, frame.cols, frame.rows));
	draw_main_overlay(left_frame, bounding_boxes, lane_shape, desired_control);
	draw_right_panel(output, lane_shape.tracked_waypoints, lane_shape, desired_control);

	return output;
}

void close_windows() {
	cv::destroyAllWindows();
}

}  // namespace visualization

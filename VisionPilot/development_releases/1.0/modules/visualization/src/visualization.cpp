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

cv::Mat load_wheel_icon() {
	const std::vector<std::filesystem::path> candidates = {
		std::filesystem::current_path() / "modules" / "visualization" / "src" / "assets" / "wheel.png",
		std::filesystem::current_path() / ".." / "modules" / "visualization" / "src" / "assets" / "wheel.png",
		std::filesystem::current_path() / ".." / ".." / "modules" / "visualization" / "src" / "assets" / "wheel.png",
		std::filesystem::current_path() / ".." / ".." / ".." / "modules" / "visualization" / "src" / "assets" / "wheel.png"
	};
	// ... (keep the rest of the function)
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

void draw_boxed_value(cv::Mat &canvas, const cv::Rect &rect, const std::string &title, const std::string &value, const cv::Scalar &value_color = kPanelTextColor) {
	cv::rectangle(canvas, rect, cv::Scalar(255, 255, 255), cv::FILLED);
	cv::rectangle(canvas, rect, cv::Scalar(200, 200, 200), 1);

	const int title_y = rect.y + 22;
	const int value_y = rect.y + rect.height / 2 + 24;
	cv::putText(canvas, title, cv::Point(rect.x + 12, title_y), cv::FONT_HERSHEY_SIMPLEX, 0.6, kPanelTextColor, 1, cv::LINE_AA);
	cv::putText(canvas, value, cv::Point(rect.x + 12, value_y), cv::FONT_HERSHEY_SIMPLEX, 0.72, value_color, 2, cv::LINE_AA);
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
	if (bounding_boxes.empty()) return;

	cv::Mat overlay = frame.clone();
	for (const auto &box : bounding_boxes) {
		const cv::Rect rect = yolo_to_rect(box, frame.size());
		if (rect.width <= 0 || rect.height <= 0) continue;
		cv::rectangle(overlay, rect, class_color(box.class_id), cv::FILLED);
	}
	// Write directly back into the referenced memory block
	cv::addWeighted(overlay, kDetectionOverlayAlpha, frame, 1.0F - kDetectionOverlayAlpha, 0.0, frame);
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
		projected_points.emplace_back(
			std::clamp(waypoint.x, 0.0F, static_cast<float>(frame.cols - 1)),
			std::clamp(waypoint.y, 0.0F, static_cast<float>(frame.rows - 1))
		);
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
	cv::addWeighted(overlay, kDrivablePathAlpha, frame, 1.0F - kDrivablePathAlpha, 0.0, frame);
	std::vector<cv::Point> centerline_points;
	centerline_points.reserve(projected_points.size());
	for (const auto &point : projected_points) {
		centerline_points.emplace_back(cv::Point(
			static_cast<int>(std::lround(point.x)),
			static_cast<int>(std::lround(point.y))
		));
	}
	cv::polylines(frame, std::vector<std::vector<cv::Point>>{centerline_points}, false, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
}

float polyline_length(const std::vector<cv::Point2f> &points) {
	float total = 0.0F;
	for (std::size_t index = 1; index < points.size(); ++index) {
		const cv::Point2f delta = points[index] - points[index - 1];
		total += std::sqrt(delta.x * delta.x + delta.y * delta.y);
	}
	return total;
}

cv::Point2f point_along_polyline(const std::vector<cv::Point2f> &points, float target_ratio) {
	if (points.empty()) {
		return cv::Point2f();
	}

	if (points.size() == 1) {
		return points.front();
	}

	const float clamped_ratio = std::clamp(target_ratio, 0.0F, 1.0F);
	const float total_length = polyline_length(points);
	if (total_length <= 1e-4F) {
		return points.front();
	}

	const float target_length = total_length * clamped_ratio;
	float accumulated = 0.0F;
	for (std::size_t index = 1; index < points.size(); ++index) {
		const cv::Point2f delta = points[index] - points[index - 1];
		const float segment_length = std::sqrt(delta.x * delta.x + delta.y * delta.y);
		if (accumulated + segment_length >= target_length) {
			const float segment_ratio = (target_length - accumulated) / std::max(segment_length, 1e-4F);
			return points[index - 1] + delta * segment_ratio;
		}
		accumulated += segment_length;
	}

	return points.back();
}

void draw_path_preview_ruler(cv::Mat &canvas, const cv::Rect &area, float max_distance_m) {
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

void draw_path_preview(cv::Mat &canvas, const std::vector<cv::Point2f> &tracked_waypoints, const cv::Rect &area) {
	if (tracked_waypoints.size() < 2) return;

	std::vector<cv::Point2f> bev_waypoints;
	cv::perspectiveTransform(tracked_waypoints, bev_waypoints, homography_matrix);

	const int ruler_width = 62;
	const cv::Rect path_rect(area.x + 8, area.y + 8, std::max(1, area.width - ruler_width - 16), std::max(1, area.height - 16));

	std::vector<cv::Point> polyline_points;
	polyline_points.reserve(bev_waypoints.size());
	
	const float max_lateral = 15.0F; // Assume 15m left/right for the BEV view

	for (const auto &bev_pt : bev_waypoints) {
		// bev_pt.x is longitudinal (0 to 100m)
		// bev_pt.y is lateral (+Y is left, -Y is right)
		const float x_ratio = std::clamp(bev_pt.x / kPathPreviewMaxDistanceMeters, 0.0F, 1.0F);
		const float y_ratio = std::clamp((bev_pt.y + max_lateral) / (2.0F * max_lateral), 0.0F, 1.0F);

		// Flip y_ratio because pixel 0 is the left edge, matching +Y
		const int px = path_rect.x + static_cast<int>(std::lround((1.0F - y_ratio) * static_cast<float>(path_rect.width)));
		const int py = path_rect.y + static_cast<int>(std::lround((1.0F - x_ratio) * static_cast<float>(path_rect.height)));
		polyline_points.emplace_back(px, py);
	}

	cv::Mat overlay = canvas.clone();
	cv::polylines(overlay, std::vector<std::vector<cv::Point>>{polyline_points}, false, cv::Scalar(40, 180, 90), 4, cv::LINE_AA);
	cv::addWeighted(overlay, 0.8F, canvas, 0.2F, 0.0, canvas);
}

void draw_cipo_marker(cv::Mat &canvas, const std::vector<cv::Point2f> &tracked_waypoints, const LaneShapeVisualization &lane_shape, const cv::Rect &area, float max_distance_m) {
	if (!lane_shape.has_cipo_object || !lane_shape.distance_to_cipo.has_value() || tracked_waypoints.empty()) return;

	std::vector<cv::Point2f> bev_waypoints;
	cv::perspectiveTransform(tracked_waypoints, bev_waypoints, homography_matrix);

	const float distance_m = std::clamp(*lane_shape.distance_to_cipo, 0.0F, max_distance_m);
	
	// Interpolate lateral (Y) position at the given longitudinal (X) distance
	float lateral_y = bev_waypoints.front().y;
	for (size_t i = 1; i < bev_waypoints.size(); ++i) {
		if (bev_waypoints[i].x >= distance_m) {
			const float ratio = (distance_m - bev_waypoints[i-1].x) / std::max(1e-4F, bev_waypoints[i].x - bev_waypoints[i-1].x);
			lateral_y = bev_waypoints[i-1].y + ratio * (bev_waypoints[i].y - bev_waypoints[i-1].y);
			break;
		}
		lateral_y = bev_waypoints.back().y;
	}

	const int ruler_width = 62;
	const cv::Rect path_rect(area.x + 8, area.y + 8, std::max(1, area.width - ruler_width - 16), std::max(1, area.height - 16));
	const float max_lateral = 15.0F;

	const float x_ratio = std::clamp(distance_m / max_distance_m, 0.0F, 1.0F);
	const float y_ratio = std::clamp((lateral_y + max_lateral) / (2.0F * max_lateral), 0.0F, 1.0F);

	const int px = path_rect.x + static_cast<int>(std::lround((1.0F - y_ratio) * static_cast<float>(path_rect.width)));
	const int py = path_rect.y + static_cast<int>(std::lround((1.0F - x_ratio) * static_cast<float>(path_rect.height)));

	const cv::Rect marker_rect(px - 12, py - 16, 24, 20);
	cv::rectangle(canvas, marker_rect, cv::Scalar(60, 60, 230), cv::FILLED);
	cv::rectangle(canvas, marker_rect, cv::Scalar(255, 255, 255), 1);

	const std::string distance_text = format_float(*lane_shape.distance_to_cipo, 1) + " m";
	const std::string velocity_text = lane_shape.relative_cipo_velocity.has_value() ? format_float(*lane_shape.relative_cipo_velocity, 1) + " km/h" : "-- km/h";
	cv::putText(canvas, distance_text, cv::Point(px - 24, std::max(area.y + 14, py - 22)), cv::FONT_HERSHEY_SIMPLEX, 0.42, kPanelTextColor, 1, cv::LINE_AA);
	cv::putText(canvas, velocity_text, cv::Point(px - 24, std::max(area.y + 28, py - 8)), cv::FONT_HERSHEY_SIMPLEX, 0.42, kPanelTextColor, 1, cv::LINE_AA);
}

void draw_right_panel(cv::Mat &canvas, const std::vector<cv::Point2f> &tracked_waypoints, const LaneShapeVisualization &lane_shape, const DesiredControlVisualization &desired_control) {
	const int panel_width = kVisualizationPanelWidth;
	const cv::Rect panel_rect(canvas.cols - panel_width, 0, panel_width, canvas.rows);
	if (panel_rect.x < 0) return;

	cv::Mat panel = canvas(panel_rect); // Point directly to the canvas ROI
	cv::Mat white_bg(panel.size(), panel.type(), kWhiteColor);
	cv::addWeighted(white_bg, kRightPanelAlpha, panel, 1.0F - kRightPanelAlpha, 0.0, panel);

	const cv::Rect top_card(12, 12, panel_rect.width - 24, 176);
	cv::rectangle(panel, top_card, cv::Scalar(255, 255, 255), cv::FILLED);
	cv::rectangle(panel, top_card, cv::Scalar(210, 210, 210), 1);
	draw_text_centered(panel, "Desired planning values", cv::Rect(12, 20, panel_rect.width - 24, 30), 0.58, kPanelTextColor, 2);

	const cv::Rect wheel_area(24, 56, 96, 96);
	const cv::Mat wheel_icon = load_wheel_icon();
	if (!wheel_icon.empty()) {
		cv::Mat rotated = rotate_icon(wheel_icon, desired_control.steering_angle);
		const float max_width = static_cast<float>(wheel_area.width);
		const float max_height = static_cast<float>(wheel_area.height);
		const float scale = std::min(max_width / std::max(1, rotated.cols), max_height / std::max(1, rotated.rows));
		if (scale < 1.0F) {
			cv::resize(rotated, rotated, cv::Size(), scale, scale, cv::INTER_AREA);
		}
		overlay_icon(panel, rotated, cv::Point(wheel_area.x + (wheel_area.width - rotated.cols) / 2, wheel_area.y + (wheel_area.height - rotated.rows) / 2));
	} else {
		cv::circle(panel, cv::Point(wheel_area.x + wheel_area.width / 2, wheel_area.y + wheel_area.height / 2), 34, cv::Scalar(80, 80, 80), 3);
	}

	const cv::Rect velocity_rect(136, 56, 180, 44);
	draw_boxed_value(panel, velocity_rect, "Velocity", format_float(desired_control.velocity, 1) + " km/h");

	const cv::Rect steering_rect(136, 108, 180, 44);
	draw_boxed_value(panel, steering_rect, "Steering", format_float(desired_control.steering_angle, 1) + " deg");

	const cv::Rect accel_rect(136, 160, 180, 44);
	const cv::Scalar acceleration_color = desired_control.acceleration >= 0.0F ? cv::Scalar(50, 190, 80) : cv::Scalar(70, 70, 230);
	draw_boxed_value(panel, accel_rect, "Acceleration", format_float(desired_control.acceleration, 1) + " m/s2", acceleration_color);

	const cv::Rect bev_rect(12, 214, panel_rect.width - 24, panel_rect.height - 226);
	cv::rectangle(panel, bev_rect, cv::Scalar(255, 255, 255), cv::FILLED);
	cv::rectangle(panel, bev_rect, cv::Scalar(210, 210, 210), 1);
	cv::putText(panel, "Path preview", cv::Point(bev_rect.x + 10, bev_rect.y + 22), cv::FONT_HERSHEY_SIMPLEX, 0.55, kPanelTextColor, 1, cv::LINE_AA);

	const cv::Rect path_area(bev_rect.x + 10, bev_rect.y + 32, bev_rect.width - 20, bev_rect.height - 42);
	draw_path_preview_ruler(panel, path_area, kPathPreviewMaxDistanceMeters);
	draw_path_preview(panel, tracked_waypoints, path_area);
	draw_cipo_marker(panel, tracked_waypoints, lane_shape, path_area, kPathPreviewMaxDistanceMeters);

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
	if (frame.empty()) return cv::Mat();

	// Don't expand the frame, overlay directly on top
	cv::Mat output = frame.clone();

	draw_main_overlay(output, bounding_boxes, lane_shape, desired_control);
	draw_right_panel(output, lane_shape.tracked_waypoints, lane_shape, desired_control);

	return output;
}

void close_windows() {
	cv::destroyAllWindows();
}

}  // namespace visualization

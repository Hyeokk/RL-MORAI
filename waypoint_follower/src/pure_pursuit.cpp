#include <ros/ros.h>
#include <waypointFollower.h>
double WaypointFollower::calcSteeringAngle()
{
	getClosestWaypoint(cur_pose_);

    double t_dist = .0;    // get dist between current pose and target waypoint
    for(int i = 0; i < waypoints_size_; i++) 
    {
        double dist = calcPlaneDist(cur_pose_, waypoints_[i].pose);
        if(dist > lookahead_dist_)
        {
            target_index_ = i;
            t_dist = dist;
            break;
        }
    }
    
	double steering_angle;
	double target_x = waypoints_[target_index_].pose.pose.position.x;
	double target_y = waypoints_[target_index_].pose.pose.position.y;
    double cur_x = cur_pose_.pose.position.x;
    double cur_y = cur_pose_.pose.position.y;
    double t_x = waypoints_[target_index_].pose.pose.position.x;
    double t_y = waypoints_[target_index_].pose.pose.position.y;
    double dx = t_x - cur_x;
    double dy = t_y - cur_y;
    double dd = sqrt(pow(dx, 2) + powf(dy, 2));
    double temp_theta = atan2(dy,dx) * 180.0/M_PI;

    double deg_alpha = (temp_theta - cur_course_);
    if (deg_alpha <= -180.0) deg_alpha += 360.0;
    else if(deg_alpha > 180.0) deg_alpha -= 360.0;

    double alpha = deg_alpha * M_PI/180.0;
    if(is_backward_) 
        cur_course_ += -180.0 ? cur_course_ >= 180.0 : 180.0;
		// if(cur_course_ >= 180.0) cur_course_ -= 180.0;
		// else cur_course_ += 180.0;
	
	
    return atan2((2.0*WHEEL_BASE*sin(alpha) / lookahead_dist_), 1.0);
}


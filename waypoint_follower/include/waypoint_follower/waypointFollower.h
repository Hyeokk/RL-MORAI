#ifndef WAYPOINTFOLLOWER_H
#define WAYPOINTFOLLOWER_H

#include <ros/ros.h>
#include <tf/tf.h>

// C, C++ header
#include <iostream>
#include <vector>
#include <numeric>

// msg OpenSource
#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Float32.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose2D.h>
#include <ackermann_msgs/AckermannDriveStamped.h>

// msgs from waypont_Loader
#include <waypoint_maker/Lane.h>
#include <waypoint_maker/Waypoint.h>
#include <waypoint_maker/State.h>


#include <math.h>

#include <nav_msgs/Path.h>


// header for tf
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>

// 1_2
#include "sensor_msgs/NavSatFix.h"

#include <new_lane/lidar_topic_msg.h>

using namespace std;



class WaypointFollower
{
private:
	int closest_waypoint_;
	bool is_course_gps_;
	double change_dist_;

	// parameters in header
	double prev_lpf;
	vector<waypoint_maker::Waypoint> waypoints_;
	double max_search_dist_;
	int loader_number_;
	int waypoint_min;

	// parameters for process
	double spd_state_;
	double dist_;
	int current_mission_state_;
	bool is_pose_;
	bool is_course_;
	bool is_lane_;
	bool is_control_;
	ros::Time temp_time;
	ros::Duration time_difference;


	// parameters for platform_control
	double input_speed_;
	double input_steer_;

	// parameters for pure pursuit
	geometry_msgs::PoseStamped cur_pose_front,cur_pose_back;
	const double LD_RATIO = 0.4; 
	double lookahead_dist_;
	double cur_speed_;
	int waypoints_size_;
	int target_index_;
	double ego_cur_speed_;
	// parameters for stanly
	double prev_ryaw_;
    double prev_yaw_;
	double cur_course_;	

	//parameters for tf
	tf::TransformBroadcaster tf_odom_;

    tf::Quaternion quar_ego;

	tf::TransformBroadcaster lidar2map;
	tf::TransformBroadcaster lidarodom2map;
	
	tf::TransformListener listener;
	
	nav_msgs::Path past_path_rviz_;

	ros::Time start_;
	ros::Time end_;
	ros::Duration sec_;

	ros::Time start_2;
	ros::Time end_2;
	ros::Duration sec_2;

	int lane_number_;
	double brake_;

	const double WHEEL_BASE = 1.212;
	const double MAX_SEARCH_DIST = 10.0;
	const double MAX_SUM = 58.0;
	const double MIN_SUM = -150.0;

	//time fit
	double n_gps_start_sec_;
	double n_gps_during_sec_;
	double n_gps_start_sec2_;

	// fix
	bool fix_ = false;
	int fix_num_;

	// lidar
    bool is_acc;
    int acc_index;

	// ROS
	ros::NodeHandle nh_;
	ros::NodeHandle private_nh_;


// publisher
	// to waypoint_loader
	ros::Publisher lane_number_pub_;
	ros::Publisher vis_pub_front;
	ros::Publisher vis_pub_back;
	ros::Publisher pp_target_pub;
	ros::Publisher st_target_pub;

	ros::Publisher course_pub;


// subscriber
	// from nmea_parser
	ros::Subscriber speed_sub_; 
    ros::Subscriber lidar_topic_sub_;
	// from utm_odometry_node
	ros::Subscriber odom_sub_front_;
	ros::Subscriber odom_sub_back_;

	// from Waypoint_loader
	ros::Subscriber lane_sub_;
	ros::Subscriber state_sub_;
	ros::Subscriber dist_sub_;
	// fix
	ros::Subscriber fix_front_sub_,fix_back_sub_;
	//platform_control_sub
	ros::Subscriber ego_speed_sub_;


	std_msgs::Float64 course_msg_;

	double start_sec_;
	double during_sec_;
	bool gear_flag;


public:
	WaypointFollower()
	{
		initSetup();
	}

	~WaypointFollower()
	{
		waypoints_.clear();

	}

	void initSetup()
	{

		gear_flag = false;
		start_sec_ = .0;
		during_sec_ = .0;
		change_dist_ = .0;
		closest_waypoint_ = 0;
		is_course_gps_ = false;

		// parameters in header
		prev_lpf = 0;

		loader_number_ = 0;
		max_search_dist_ = 10.0;
		waypoint_min = -1;

		// parameters for process
		spd_state_ = 0.0;
		dist_ = 100.0;
		current_mission_state_ = -1;
		is_pose_ = false;
		is_course_ = false;
		is_lane_ = false;
		is_control_ = false;

		// parameters for platform_control
		input_speed_ = 0.0;
		input_steer_ = 0.0;

		// parameters for pure pursuit
		lookahead_dist_ = 0.0;
		waypoints_size_ = 0;
		target_index_ = 0;
		cur_speed_ = 0.0;
		ego_cur_speed_ = 0.0;
	
		// parameters for stanly
		prev_ryaw_ = 0.0;
    	prev_yaw_ = 0.0;
		cur_course_ = 0.0;
	

		// parameters for Parking Mission
		lane_number_ = 0;
		n_gps_start_sec_ = .0;
		n_gps_during_sec_ = .0;
		n_gps_start_sec2_ = .0;

		brake_ = 0;

		// 1_2
		fix_num_=0;

		// lidar
		is_acc = false;
		acc_index = 0;

	// ROS
		// Publisher
		vis_pub_front = nh_.advertise<visualization_msgs::Marker>("front_pose",1);
		vis_pub_back = nh_.advertise<visualization_msgs::Marker>("back_pose",1);
		pp_target_pub = nh_.advertise<visualization_msgs::Marker>("target_pose",1);
		st_target_pub = nh_.advertise<visualization_msgs::Marker>("target_pose",1);

		lane_number_pub_ = nh_.advertise<waypoint_maker::State>("lane_number_msg_", 1);

		course_pub = nh_.advertise<std_msgs::Float64>("course_path", 1);	
	
		// Subscriber for gps
		odom_sub_front_ = nh_.subscribe("odom_front", 1, &WaypointFollower::OdomCallbackfront, this);
		odom_sub_back_ = nh_.subscribe("odom_back", 1, &WaypointFollower::OdomCallbackback, this);
		speed_sub_ = nh_.subscribe("cur_speed", 1, &WaypointFollower::SpeedCallback, this);
		dist_sub_ = nh_.subscribe("/change_dist", 1, &WaypointFollower::ChangeDistCallback, this);
		
		// Subscriber for loader
		lane_sub_ = nh_.subscribe("final_waypoints", 1, &WaypointFollower::LaneCallback, this);
		state_sub_ = nh_.subscribe("gps_state", 1, &WaypointFollower::StateCallback, this);
		
		// fix
		fix_front_sub_= nh_.subscribe("ublox_gps/fix_front", 10, &WaypointFollower::fixCallback, this);
		fix_back_sub_= nh_.subscribe("ublox_gps_2/fix_back", 10, &WaypointFollower::fixCallback2, this);
		//platfrom-control
		ego_speed_sub_ = nh_.subscribe("/ego_speed", 1, &WaypointFollower::EgoSpeedCallback, this);
		//lidar
		lidar_topic_sub_=  nh_.subscribe("lidar_topic", 1, &WaypointFollower::LidarLaneCallback, this);
	}

	void ChangeDistCallback(const std_msgs::Float64::ConstPtr &dist_msg)
	{	
		change_dist_ = dist_msg->data;
		// cout<<"change_dist_"<<change_dist_<<endl;
	}

	void EgoSpeedCallback(const std_msgs::Float32::ConstPtr &speed_msg)
	{	
		ego_cur_speed_ = speed_msg->data ; 
		cout<<"ego_cur_speed_"<<ego_cur_speed_<<endl;
	}

	double calcPlaneDist(const geometry_msgs::PoseStamped pose1, const geometry_msgs::PoseStamped pose2)
	{	
		return (double)(sqrt(pow(pose1.pose.position.x - pose2.pose.position.x, 2) + pow(pose1.pose.position.y - pose2.pose.position.y, 2)));
	}

	void getClosestWaypoint(geometry_msgs::PoseStamped current_pose)
	{ // 큰 정적 미션용 getClosestWaypoint
		if (!waypoints_.empty())
		{
			float dist_min = max_search_dist_;
			for (int i = 0; i < waypoints_.size(); i++)
			{
				float dist = calcPlaneDist(current_pose, waypoints_[i].pose);
				if (dist < dist_min)
				{
					dist_min = dist;
					waypoint_min = i;
				}
			}
			closest_waypoint_ = waypoint_min;
		}
		else
			cout << "------ NO CLOSEST WAYPOINT -------" << endl;
	}


	// callBack for gps

	void fixCallback(const sensor_msgs::NavSatFix::ConstPtr &fix_msg)
	{
		fix_num_ = fix_msg->status.status;
		if (fix_msg->position_covariance[0] > 0.00022){
			fix_ = false;
		}
		else{
			fix_ = true;
		}
	}

	void fixCallback2(const sensor_msgs::NavSatFix::ConstPtr &fix_msg)
	{
		fix_num_ = fix_msg->status.status;
		// cout << "fix_msg" << fix_msg->position_covariance[0] << endl;
		if (fix_msg->position_covariance[0] > 0.00022) {
			fix_ = false;
		}
		else{
			fix_ = true;
		}
	}

	void cal_course(){
		double dx = cur_pose_front.pose.position.x - cur_pose_back.pose.position.x;
		double dy = cur_pose_front.pose.position.y - cur_pose_back.pose.position.y;		

		cur_course_ = atan2(dy,dx)*(180/M_PI);

		course_msg_.data = cur_course_;
		course_pub.publish(course_msg_);
		
		// cout<<"cur_course: "<<cur_course_<<endl;
	}
	
	void SpeedCallback(const std_msgs::Float32::ConstPtr &speed_msg)
	{
		cur_speed_ = speed_msg->data;
	}



	void OdomCallbackfront(const nav_msgs::Odometry::ConstPtr &odom_msg)
	{
		cur_pose_front.header = odom_msg->header;
		cur_pose_front.pose.position = odom_msg->pose.pose.position;
		is_pose_ = true;
		visualization_msgs::Marker marker;
		marker.header.frame_id = "map";
		marker.header.stamp = ros::Time();
		marker.ns = "my_namespace";
		marker.id = 0;
		marker.type = visualization_msgs::Marker::SPHERE;
		marker.action = visualization_msgs::Marker::ADD;
		marker.pose.position.x = odom_msg->pose.pose.position.x;
		marker.pose.position.y = odom_msg->pose.pose.position.y;
		marker.pose.position.z = odom_msg->pose.pose.position.z;
		marker.pose.orientation.x = 0.0;
		marker.pose.orientation.y = 0.0;
		marker.pose.orientation.z = 0.0;
		marker.pose.orientation.w = 1.0;
		marker.scale.x = 0.3;
		marker.scale.y = 0.3;
		marker.scale.z = 0.1;
		marker.color.a = 1.0; // Don't forget to set the alpha!
		marker.color.r = 1.0;
		marker.color.g = 0.0;
		marker.color.b = 0.0;

		vis_pub_front.publish( marker );
	}

	void OdomCallbackback(const nav_msgs::Odometry::ConstPtr &odom_msg)
	{
		cur_pose_back.header = odom_msg->header;
		cur_pose_back.pose.position = odom_msg->pose.pose.position;
		is_pose_ = true;

		visualization_msgs::Marker marker;
		marker.header.frame_id = "map";
		marker.header.stamp = ros::Time();
		marker.ns = "my_namespace";
		marker.id = 0;
		marker.type = visualization_msgs::Marker::SPHERE;
		marker.action = visualization_msgs::Marker::ADD;
		marker.pose.position.x = odom_msg->pose.pose.position.x;
		marker.pose.position.y = odom_msg->pose.pose.position.y;
		marker.pose.position.z = odom_msg->pose.pose.position.z;
		marker.pose.orientation.x = 0.0;
		marker.pose.orientation.y = 0.0;
		marker.pose.orientation.z = 0.0;
		marker.pose.orientation.w = 1.0;
		marker.scale.x = 0.3;
		marker.scale.y = 0.3;
		marker.scale.z = 0.1;
		marker.color.a = 1.0; // Don't forget to set the alpha!
		marker.color.r = 0.0;
		marker.color.g = 0.0;
		marker.color.b = 1.0;

		vis_pub_back.publish( marker );
	}

	void StateCallback(const waypoint_maker::State::ConstPtr &state_msg)
	{
		dist_ = state_msg->dist;
		current_mission_state_ = state_msg->current_state;
		loader_number_ = state_msg->lane_number;
	}

	void LaneCallback(const waypoint_maker::Lane::ConstPtr &lane_msg)
	{
		waypoints_.clear();
		vector<waypoint_maker::Waypoint>().swap(waypoints_);
		waypoints_ = lane_msg->waypoints;
		waypoints_size_ = waypoints_.size();
		if (waypoints_size_ != 0)
		{
			is_lane_ = true;
		}
	}

    void LidarLaneCallback(const new_lane::lidar_topic_msg::ConstPtr &msg)
	{
		is_acc = msg->ACC_flag;
        acc_index = msg->ACC_index_gap;

	}



	void process();
	double getSpeed() { return input_speed_; }
	double getSteer() { return input_steer_; }
	int get_Mission_State() { return current_mission_state_; }

	double hybrid_control();
	double pure_pursuit(); 
	double stanly();
	int find_cloestindex(geometry_msgs::PoseStamped pose);
	double cross_error(geometry_msgs::PoseStamped pose);
	

};

#endif

#ifndef WAYPOINTFOLLOWER_H
#define WAYPOINTFOLLOWER_H

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <ackermann_msgs/AckermannDriveStamped.h>
#include <waypoint_maker/Lane.h>
#include <waypoint_maker/Waypoint.h>
#include <waypoint_maker/State.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Int32.h>
#include <geometry_msgs/Pose2D.h>

#include <std_msgs/Float32.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <math.h>
#include <numeric>

#include <sensor_msgs/Imu.h>
#include <morai_msgs/CtrlCmd.h>
#include <morai_msgs/CollisionData.h>
//#include <morai_msgs/EgoVehicleStatus.h>
#include <tf/tf.h>
#include "ros/time.h"

//#include <traffic_light/gostop.h>

//service
#include <morai_msgs/MoraiEventCmdSrv.h>
#include <morai_msgs/EventInfo.h>
#include <morai_msgs/Lamps.h>

#define INTER_TIME_PLUS 1000000000
#define INTER_TIME_MIN 90000000
#define INTER_TIME_MAX 200000000
#define INTER_SPEED_TIME_MAX 3600000000


using namespace std;

class WaypointFollower 
{
private:
	enum Gear_Num
	{
		Parking = 1,
		Rear = 2,
		Drive = 4
	};

	enum Service_Opt
	{
		Gear = 2,
		Light = 4
	};

	enum Light_Opt
	{
		Turn_Off = 0,
		Left_Emergency_Light = 1,
		Right_Light = 2
	};

	enum Speed_Opt
	{
		Stop = 0,
		Vel_10 = 1,
		Vel_15 = 2,
		Vel_17 = 3,
		Vel_20 = 4,
		Parking_vel = 5,
		Vertical = 6,
		Fast = 7,
		Slow = 8,
		Vel_19 = 9
		
	};

	double j_speed = 0;// for Jeong Min
	double j_steer = 0;// for Jeong Min

	bool static_flag = false;// for Jeong Min

	const double WHEEL_BASE = 3.00;
	const double MAX_SEARCH_DIST = 10.0;
	const double MAX_SUM = 58.0;
	const double MIN_SUM = -150.0;

	const double KI = 0.09; //0.09
	const double PID_FIT = 1.0;

	const double STRAGHT_LD = 5.0;
	const double CURVE_LD = 2.5;
	const double BIG_CURVE_LD = 3.0;
	const double PARKING_LD = 2.0;

	int waypoints_size_;
	int target_index_;
	
	vector<waypoint_maker::Waypoint> waypoints_;
	
	double dist_;

	int current_mission_state_;
	int loader_number_;

	int waypoint_min_;
	int closest_waypoint_;

	int spd_state_;

	bool is_pose_;
	bool is_course_;
	bool is_lane_;
	bool is_control_;
	bool is_obs_detect_;
	bool vision_check_;

	geometry_msgs::PoseStamped cur_pose_;
	
	double lookahead_dist_;
	double cur_course_;
	double cur_speed_; //speed over ground -> nmea sentence GPRMC 7

	double dynamic_speed_ ;
	
	ros::NodeHandle nh_;
	ros::NodeHandle private_nh_;
	
	ros::Publisher ackermann_pub_;
	ros::Publisher lane_number_pub_;

	ros::Subscriber odom_sub_;
	ros::Subscriber lane_sub_;
	ros::Subscriber vision_sub_;
	ros::Subscriber state_sub_;
	ros::Subscriber imu_sub_;

	ros::Subscriber line_sub_;
	ros::Subscriber vertical_sub_;
	ros::Subscriber obst_sub_;

	ros::Subscriber static_sub_;
	ros::Subscriber static_steer_;

	
	morai_msgs::CtrlCmd ackermann_msg_;
	waypoint_maker::State lane_number_msg_;
	vector<geometry_msgs::PoseStamped> traffic_stop_;
	
	ros::ServiceClient service_client_ ;//service
	morai_msgs::MoraiEventCmdSrv srv_;
	
	//동적
	bool is_obs_detect_dy_;
	
	//굴절
	bool is_line_vertical_;
	//오르막
	bool is_hill_stop_done_;
	bool start_hill_stop_;

	bool is_first_stop_done_;
	bool start_first_stop_;
	bool nogps_steering_;
	
	bool is_third_stop_done_;
	bool start_third_stop_;
	bool nogps_steering_2_;

	//time fit
	double start_sec_;
	double during_sec_;
	bool gear_flag;
	double n_gps_start_sec_;
	double n_gps_during_sec_;
	double n_gps_start_sec2_;

	//service
    bool do_service_once_;
	
	double ex_x_, ex_y_;
    unsigned int ex_time_;
    int inter_time_;
    
    double sum_error_;
    double accel_, brake_;

	int parking_count_;
	bool is_backward_;
    
    bool is_dynamic_finished_;
	bool dynamic_check_flag_;


	bool is_vertical_;

	double parking_dist_;

	public:
	WaypointFollower() 
	{
		initSetup();
	}

	~WaypointFollower() 
	{
		waypoints_.clear();
	}
	
	double speed_;
	double cur_steer;
    double camera_angle_;
	double lidar_angle_;

	void initSetup() 
	{
	    ros::Time::init();
		ackermann_pub_ = nh_.advertise<morai_msgs::CtrlCmd>("/ctrl_cmd", 1);

		lane_number_pub_ = nh_.advertise<waypoint_maker::State>("lane_number_msg_",1);
		odom_sub_ = nh_.subscribe("odom", 1, &WaypointFollower::OdomCallback, this);

		lane_sub_ = nh_.subscribe("final_waypoints", 1, &WaypointFollower::LaneCallback, this);
		state_sub_ = nh_.subscribe("gps_state",1,&WaypointFollower::StateCallback,this);
		vision_sub_ = nh_.subscribe("/vision_check", 10, &WaypointFollower::TrafficSignCallback,this);
		imu_sub_ = nh_.subscribe("imu",1,&WaypointFollower::ImuCallback,this);
		line_sub_ = nh_.subscribe("/lane_detector/camera_ackermann",1,&WaypointFollower::LineCallback,this);
		obst_sub_ = nh_.subscribe("/CollisionData",1,&WaypointFollower::CollisionCallback,this);

// for Jeong Min
		static_sub_ = nh_.subscribe("/static_flag_topic",1,&WaypointFollower::StaticFlagCallback,this);
		static_steer_ = nh_.subscribe("/static_steer_topic",1,&WaypointFollower::StaticSteerCallback,this);
// for Jeong Min


		// vertical_sub_ = nh_.subscribe("/dynamic_stop/lidar_ackermann",1,&WaypointFollower::VerticalCallback,this);
		//vertical_sub_ = nh_.subscribe("/ctrl_gps",1,&WaypointFollower::VerticalCallback,this);
		
		service_client_ = nh_.serviceClient<morai_msgs::MoraiEventCmdSrv>("/Service_MoraiEventCmd"); //service
		
		waypoints_size_ = 0;

		dist_ = 100.0;
		lookahead_dist_ = 5.0;
		current_mission_state_ = -1;
		waypoint_min_ = -1;
		parking_count_ = -3;

		is_pose_ = false;
		is_course_ = false;
		is_lane_ = false;
		is_control_ = false;

		is_obs_detect_dy_ = false;	//동적

		//오르막
		is_hill_stop_done_ = false;
		start_hill_stop_ = false;

		//음영구역
		is_first_stop_done_ = false;
		start_first_stop_ = false;
		nogps_steering_ = false;
	
		is_third_stop_done_ = false;
		start_third_stop_ = false;
		nogps_steering_2_ = false;

		gear_flag = false;
		is_vertical_ = false;
		is_line_vertical_ = false;
/////////////////////////////vision

		vision_check_ = true;

		spd_state_ = 0;

		is_obs_detect_ = false;
		
		//service
   		do_service_once_ = false;
    	inter_time_ = 0;

		cur_course_ = .0;
		cur_speed_ = .0;
		loader_number_ = 0;
		ex_x_ = 0;
		ex_y_ = 0;
		ex_time_ = 0;
		sum_error_ = 0;
		n_gps_start_sec_ = .0;
		n_gps_during_sec_ = .0;
		n_gps_start_sec2_ = .0;

		is_backward_ = false;
		dynamic_check_flag_ = false;
		
		speed_ = .0;
		cur_steer = .0;
    	camera_angle_ = .0;
		lidar_angle_ = .0;

		accel_ = 0;
		brake_ = 0;
        nh_.getParam("/waypoint_follower_node/is_dynamic_finished", is_dynamic_finished_);
        nh_.getParam("/waypoint_follower_node/is_parking_dist", parking_dist_);
	}

	double calcPlaneDist(const geometry_msgs::PoseStamped pose1, const geometry_msgs::PoseStamped pose2) 
	{
		return sqrt(pow(pose1.pose.position.x - pose2.pose.position.x, 2) + pow(pose1.pose.position.y - pose2.pose.position.y, 2));
	}






	void CollisionCallback(const morai_msgs::CollisionData::ConstPtr& _collision_msg)
	{
		if(!_collision_msg->collision_object.empty())
		{
			for (auto i = 0; i < _collision_msg->collision_object.size(); i++)
				cout << "!!!!!!!!!!!!!!!!!!!!!!!" << _collision_msg->collision_object[i].name << endl;
		}
	}
// for Jeong Min
	void StaticFlagCallback(const std_msgs::Bool::ConstPtr& msg){
		static_flag = msg->data;
	}

	void StaticSteerCallback(const ackermann_msgs::AckermannDriveStamped::ConstPtr& acker_msg){
		j_speed = acker_msg->drive.speed;
		j_steer = acker_msg->drive.steering_angle;
	}
// for Jeong Min

	void OdomCallback(const nav_msgs::Odometry::ConstPtr &odom_msg) 
	{
		cur_pose_.header = odom_msg->header;
		cur_pose_.pose.position = odom_msg->pose.pose.position;
	    inter_time_ = cur_pose_.header.stamp.nsec - ex_time_;
        if(inter_time_ <= 0) inter_time_ += INTER_TIME_PLUS; 
        if(INTER_TIME_MIN < inter_time_ && inter_time_ < INTER_TIME_MAX) 
            cur_speed_ = getSpeed(ex_x_, ex_y_, cur_pose_.pose.position.x, cur_pose_.pose.position.y); 
		//cout <<cur_speed_ << endl;
	    ex_time_ = cur_pose_.header.stamp.nsec;
	    ex_x_ = cur_pose_.pose.position.x;
        ex_y_ = cur_pose_.pose.position.y;
		is_pose_ = true;
	}

	void LaneCallback(const waypoint_maker::Lane::ConstPtr &lane_msg) 
	{
		waypoints_.clear();
		vector<waypoint_maker::Waypoint>().swap(waypoints_);
		waypoints_ = lane_msg->waypoints;
		waypoints_size_ = waypoints_.size();
		if (waypoints_size_ != 0) is_lane_ = true;
	}

	void TrafficSignCallback(const std_msgs::Bool::ConstPtr& vision_msg)
	{
		vision_check_ = vision_msg->data;
	}
	
	void StateCallback(const waypoint_maker::State::ConstPtr &state_msg)
	{
		dist_ = state_msg->dist;
		current_mission_state_ = state_msg->current_state;
		loader_number_ = state_msg->lane_number;
	}
	


	
	void ImuCallback(const sensor_msgs::Imu::ConstPtr &imu_msg)
	{
		tf::Quaternion q(imu_msg->orientation.x, imu_msg->orientation.y,
			imu_msg->orientation.z, imu_msg->orientation.w);
		tf::Matrix3x3 m(q);
		double roll, pitch, yaw;
		m.getRPY(roll,pitch,yaw);
		cur_course_ = yaw * (180.0 / M_PI);
		is_course_ = true;
	}
	
	void LineCallback(const ackermann_msgs::AckermannDriveStamped::ConstPtr &camera_msg)
	{
		camera_angle_ = camera_msg->drive.steering_angle * -(M_PI / 180.0) * 2;
	}

	// void VerticalCallback(const ackermann_msgs::AckermannDriveStamped::ConstPtr &lidar_msg)
	// {
	// 	lidar_angle_ = lidar_msg->drive.steering_angle * -(M_PI / 180.0) * 2;
	// }


	void getClosestWaypoint(geometry_msgs::PoseStamped current_pose) 
	{
		if (!waypoints_.empty()) 
		{
			double dist_min = MAX_SEARCH_DIST;
			for (int i = 0; i < waypoints_.size(); i++)
			{
				double dist = calcPlaneDist(current_pose, waypoints_[i].pose);
				if (dist < dist_min) 
				{
					dist_min = dist;
					waypoint_min_ = i;
				}
			}
			closest_waypoint_ = waypoint_min_;
		}
		else cout << "------ NO CLOSEST WAYPOINT -------" << endl;
	}
	
	// service
	
	void clientCall_turnSignal_off() 
	{
	    if(do_service_once_)
		{
	        srv_.request.request.option = Service_Opt::Light;
            srv_.request.request.lamps.turnSignal = Light_Opt::Turn_Off;
	        service_client_.call(srv_);
	        do_service_once_ = false;
	    }
	}
	
	void clientCall_left_on() 
	{
	    if(!do_service_once_)
		{
	        srv_.request.request.option = Service_Opt::Light;
            srv_.request.request.lamps.turnSignal = Light_Opt::Left_Emergency_Light;
	        service_client_.call(srv_);
	        do_service_once_ = true;
	    }
	}
	void clientCall_right_on() 
	{
	    if(!do_service_once_)
		{
	        srv_.request.request.option = Service_Opt::Light;
            srv_.request.request.lamps.turnSignal = Light_Opt::Right_Light;
	        service_client_.call(srv_);
	        do_service_once_ = true;
			cout << "SSSSSSSSSSSSSSSSSSSs" << endl;
	    }
	}
	void clientCall_emergency_off() 
	{
	    if(do_service_once_)
		{
	        srv_.request.request.option = Service_Opt::Light;
            srv_.request.request.lamps.emergencySignal = Light_Opt::Turn_Off;
	        service_client_.call(srv_);
	        do_service_once_ = false;
	    }
	}
	void clientCall_emergency_on() 
	{
	    if(!do_service_once_)
		{
	        srv_.request.request.option = Service_Opt::Light;
            srv_.request.request.lamps.emergencySignal = Light_Opt::Left_Emergency_Light;
	        service_client_.call(srv_);
	        do_service_once_ = true;
	    }
	}
	void gear_drive()
	{
		srv_.request.request.option = Service_Opt::Gear;
		srv_.request.request.gear = Gear_Num::Drive;
		service_client_.call(srv_);
	}

	void gear_rear()
	{
		srv_.request.request.option = Service_Opt::Gear;
		srv_.request.request.gear = Gear_Num::Rear;
		service_client_.call(srv_);
	}

	void gear_parking()
	{
		srv_.request.request.option = Service_Opt::Gear;
		srv_.request.request.gear = Gear_Num::Parking;
		service_client_.call(srv_);
	}
	
	double getSpeed(double& ex_x, double& ex_y, double cur_x, double cur_y)
	{
        double distance = sqrt(pow((cur_x - ex_x ), 2) + pow((cur_y - ex_y), 2));
        double speed = distance / inter_time_ * INTER_SPEED_TIME_MAX;
 
        // cout << "distance   :: " << distance << endl;
        // cout << "time   "  << inter_time_ <<  "    ::    speed  " << speed << endl;
        return speed;
	}
	
    double PID(double target, double cur_speed) 
	{
        if (target == 0) return 0;
        target = target - PID_FIT;
        double error = target - cur_speed;
        if (error < 0) sum_error_ = 0;
        
        sum_error_ += error;
        
        if (sum_error_ > MAX_SUM) sum_error_ = MAX_SUM;
        else if (sum_error_ < MIN_SUM) sum_error_ = MIN_SUM;
        
        double result = KI * sum_error_ + target;
        if (result >= 20.0) result = 20.0;
        
        return result;
    }

    double calc_longitude_V(double default_speed);
	double calc_ld(double longitudinal_speed);
	int find_cloestindex(geometry_msgs::PoseStamped pose);

	double calcSteeringAngle();
	void process();
};

#endif

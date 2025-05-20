#include <ros/ros.h>
#include <waypoint_maker/Lane.h> //For Parking Mission
#include <waypoint_maker/Waypoint.h> // Waypoints
#include <waypoint_maker/State.h> //For Mission (미션에 필요한 msg 헤더 모음)
#include <std_msgs/Header.h>
//For 자료형
#include <geometry_msgs/PoseStamped.h> 
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Bool.h>
#include <vector>
#include <string>

// For CSV file loading
#include <fstream>
#include <sys/types.h> // 시스템 자료형 타입 정리용 헤더
#include <dirent.h> // 디렉토리 다루기 위한 헤더

using namespace std;

class WaypointLoader 
{
private:
	const double MAX_SEARCH_DIST = 10.0;
	const char* CSV_PATH = "/home/kuuve/catkin_ws/src/data/";

	ros::NodeHandle nh_;

	// publisher	
	ros::Publisher waypoint_pub_;
	ros::Publisher state_pub_;
	ros::Publisher rviz_global_path_;
	ros::Publisher rviz_currnet_path_;
	
	// subscriber
	ros::Subscriber pose_sub_;
	
	//CSV Loading variables
	ifstream is_;
	std::vector<std::string> all_csv_;
	vector<vector<waypoint_maker::Waypoint>> all_new_waypoints_;
	vector<waypoint_maker::Waypoint> new_waypoints_;
	vector<int> lane_size_; //Waypoints  
	vector< vector<int> > all_state_index_;
	vector<int> state_index_;

	
	//Vector size
	int size_;
	int final_size_;
	
	//State용 변수
	int state_inspection_;	
	int current_mission_state_;
	
	//Final Waypoint
	
	vector<waypoint_maker::Waypoint> final_waypoints_;

	//Closest Waypoint용 변수들 
	vector<int> closest_waypoint_candidates;

	float dist_;
	int closest_waypoint_;
	
	//For Parking
	int parking_state_;
	int lane_number_;
	
	bool is_state_;//middle start

	//msg들
	geometry_msgs::PoseStamped current_pose; //지역 변수 -> 전역 변수로 변경
	waypoint_maker::Lane lane_msg_;
	waypoint_maker::State state_msg_;
	nav_msgs::Path past_path_rviz_;
	vector<geometry_msgs::PoseStamped> past_path_;
	
public:
	bool is_pose_;
	
	WaypointLoader() 
	{
		initSetup();
		ROS_INFO("WAYPOINT LOADER INITIALIZED.");
	}
	
	~WaypointLoader() 
	{
		new_waypoints_.clear();
		all_new_waypoints_.clear();
		final_waypoints_.clear();
		ROS_INFO("WAYPOINT LOADER TERMINATED.");
	}
	
	void initSetup() 
	{
		nh_.getParam("/waypoint_loader_node/state_inspection", state_inspection_);
		nh_.getParam("/waypoint_loader_node/parking_count", parking_state_);

		final_size_ = 120;
		current_mission_state_ = state_inspection_;
		closest_waypoint_ = 0;
		
		get_csvs_inDirectory();
		getNewWaypoints();
		
		waypoint_pub_ = nh_.advertise<waypoint_maker::Lane>("final_waypoints", 1);
		state_pub_ = nh_.advertise<waypoint_maker::State>("gps_state", 1);
		rviz_global_path_ = nh_.advertise<nav_msgs::Path>("/rviz_global_path", 1);
		rviz_currnet_path_ = nh_.advertise<nav_msgs::Path>("/rviz_currnet_path_", 1);
		
		pose_sub_ = nh_.subscribe("odom", 10, &WaypointLoader::poseCallback, this);

		lane_number_ = 0;

		is_state_ = false; //middle start

	}
	
	
	/*
	수정 계획
	1. poseCallback 안에서는 current_pose와 lang_msg_header만 수신한다
	2. Callback 수신 알림용 bool 자료형 하나 만든다.
	2. 나머지 과정은 모두 main문으로 뺀다
	*/
	void poseCallback(const nav_msgs::Odometry::ConstPtr &msg) 
	{
		current_pose.pose.position = msg->pose.pose.position;
		lane_msg_.header = msg->header;	
	}
	
	void poseProcess()
	{
		final_waypoints_.clear();
			
		getClosestWaypoint(current_pose);
		getFinalWaypoint();

		nh_.getParam("/waypoint_loader_node/parking_count",parking_state_);
	   	if(parking_state_ > -2)
			current_mission_state_ = parking_state_ + 19;
		else
			current_mission_state_ = final_waypoints_[0].mission_state;
		
	    // current_mission_state_ = final_waypoints_[0].mission_state;
		if(!is_state_ && closest_waypoint_candidates.empty())
		    current_mission_state_ = state_inspection_;

		closest_waypoint_candidates.clear();
		dist_ = calcPlaneDist(current_pose, new_waypoints_[all_state_index_[0][current_mission_state_ + 1]].pose);
		state_msg_.dist = dist_;
		state_msg_.current_state = current_mission_state_;

		state_msg_.header.stamp = ros::Time::now();
		
		state_pub_.publish(state_msg_);
		
		lane_msg_.waypoints = final_waypoints_;
		
		waypoint_pub_.publish(lane_msg_);

		int final_waypoints_size = lane_msg_.waypoints.size();
		ROS_INFO("FINAL WAYPOINTS NUMBER=%d PUBLISHED.", final_waypoints_size);	
	}
	
	//Functions in initSetup (코드 시작하면 실행되는 함수들)
	void get_csvs_inDirectory() // /home/kuuve/data/ 안에 있는 csv파일 전부 가져옴
	{
	
		DIR* dirp = opendir(CSV_PATH);
		
		if (dirp == NULL)
		{
			perror("UNABLE TO OPEN FOLDER");
			return;
		}

		struct dirent* dp;
		while((dp = readdir(dirp)) != NULL)
		{
			string address(CSV_PATH);
			string filename (dp->d_name);

			address.append(filename);
			if (filename.size() > 2)
			{	
				all_csv_.emplace_back(address);
				ROS_INFO("SAVED CSV");
			}
		} 
		
		sort(all_csv_.begin(), all_csv_.end());
		closedir(dirp);
	}

	// 가져온 여러개의 csv파일 내의 데이터를 2차원 벡터로 정리
	void getNewWaypoints() 
	{
		string str_buf;
		int pos;

		vector<int> state_index_;
		vector<waypoint_maker::Waypoint> temp_new_waypoints;
		waypoint_maker::Waypoint temp_waypoint;

		int temp_size = all_csv_.size();

		for (auto i = 0; i < temp_size; i++)
		{ // 파일 갯수만큼

			state_index_.emplace_back(5); // current_mission_state 0번 시작 : 반드시 5
										  // 이유 : getClosestWaypoint()의 조사 시작 범위를 위해
										  // ctrl+f --> void getClosestWaypoint()

			is_.open(all_csv_[i]); // 파일 열기

			cout << "OPEN CSV" << all_csv_[i] << endl;

			temp_waypoint.mission_state = 0;

			while (getline(is_, str_buf))
			{ // 파일 내용을 str_buf에 저장

				if (str_buf != "")
				{ // temp_waypoint = [index, x, y, mission_state]
					pos = str_buf.find(",");
					temp_waypoint.waypoint_index = stoi(str_buf.substr(0, pos));

					str_buf = str_buf.substr(++pos);
					pos = str_buf.find(",");
					temp_waypoint.pose.pose.position.x = stod(str_buf.substr(0, pos));

					str_buf = str_buf.substr(++pos);
					pos = str_buf.find(",");
					temp_waypoint.pose.pose.position.y = stod(str_buf.substr(0, pos));

					str_buf = str_buf.substr(++pos);
					pos = str_buf.find(",");

					if (temp_waypoint.mission_state != stoi(str_buf.substr(0, pos))) // mission state 변화시 따로 저장
						state_index_.emplace_back(temp_waypoint.waypoint_index);
					temp_waypoint.mission_state = stoi(str_buf.substr(0, pos));

					//ryaw
					str_buf = str_buf.substr(++pos); // str_buf의 첫번째 문자부터 ++pos (str_buf에서 콤마가 몇 번째 있는지 + 1)길이 만큼의 문자열을 리턴
					pos = str_buf.find(","); // CSV 파일에서 콤마 찾기 >> ryaw 추출하기 위해
					temp_waypoint.ryaw = stod(str_buf.substr(0, pos)); // temp_waypoint에 해당 ryaw 저장
					
					//rk
					str_buf = str_buf.substr(++pos); // str_buf의 첫번째 문자부터 ++pos (str_buf에서 콤마가 몇 번째 있는지 + 1)길이 만큼의 문자열을 리턴
					pos = str_buf.find(","); // CSV 파일에서 콤마 찾기 >> rk 추출하기 위해
					temp_waypoint.rk = stod(str_buf.substr(0, pos)); // temp_waypoint에 해당 rk 저장

					temp_new_waypoints.emplace_back(temp_waypoint);
				}
			}
			is_.close();

			size_ = temp_new_waypoints.size();
			lane_size_.emplace_back(size_); // lane_size 정리

			ROS_INFO("%d WAYPOINTS HAVE BEEN SAVED.", size_);

			all_new_waypoints_.emplace_back(temp_new_waypoints); // all_new_waypoints_ 정리
			all_state_index_.emplace_back(state_index_);		 // all_state_index_ 정리

			temp_new_waypoints.clear();
			state_index_.clear();
		}

		new_waypoints_.assign(all_new_waypoints_[0].begin(), all_new_waypoints_[0].end());
		size_ = lane_size_[0];
		for (vector<int> vec : all_state_index_) {
			for (int num : vec) {
				cout << num <<" ";
			}
		cout << endl;
		}
	}
	
	
	//Functions in poseProcess
	/*
	수정사항 
	*/
	void getClosestWaypoint(geometry_msgs::PoseStamped current_pose) 
	{
		nh_.getParam("/waypoint_loader_node/parking_count", parking_state_);

		//all_state_index_ 1부터담김 인덱스 0 0으로 초기화하고 푸쉬백 해야할듯.
		for(int i = all_state_index_[0][current_mission_state_] - 4; i < size_; i++) 
		{
			double dist = calcPlaneDist(current_pose, new_waypoints_[i].pose);
			int t_state_check = new_waypoints_[i].mission_state - current_mission_state_;
			// 주차시
			if (parking_state_ > -2)
			{
				if (dist < MAX_SEARCH_DIST && (new_waypoints_[i].mission_state == parking_state_ + 19))
					closest_waypoint_candidates.emplace_back(i);
			}
			// 평시
			else if (dist < MAX_SEARCH_DIST && (t_state_check == 2 || t_state_check == 1 || t_state_check == 0))
				closest_waypoint_candidates.emplace_back(i);
			
			
			if (dist > MAX_SEARCH_DIST  && t_state_check > 2) break;
		}
		
		if(!closest_waypoint_candidates.empty()) 
		{
			int waypoint_min = -1;
			double dist_min = MAX_SEARCH_DIST;

			for(int i = 0; i < closest_waypoint_candidates.size(); i++) 
			{
				double dist = calcPlaneDist(current_pose, new_waypoints_[closest_waypoint_candidates[i]].pose);
				if(dist < dist_min) 
				{
					dist_min = dist;
					waypoint_min = closest_waypoint_candidates[i];
				}
			}

			closest_waypoint_ = waypoint_min;
			
			ROS_INFO("CLOSEST WAYPOINT INDEX=%d, X=%f, Y=%f", closest_waypoint_, new_waypoints_[closest_waypoint_].pose.pose.position.x, new_waypoints_[closest_waypoint_].pose.pose.position.y);
			is_state_ = true;//middle start
		}
		else ROS_INFO("THERE IS NO CLOSEST WAYPOINT CANDIDATE.");
	
	}

	// -1
	void getFinalWaypoint()
	{ 
		if((size_ - closest_waypoint_ - 1) < final_size_) final_size_ = size_ - closest_waypoint_ - 1;
	 	for(int i = 0; i < final_size_; i++)
			final_waypoints_.emplace_back(new_waypoints_[closest_waypoint_ + i]);
	}
	
	//Function in getClosetWaypoint
	double calcPlaneDist(const geometry_msgs::PoseStamped pose1, const geometry_msgs::PoseStamped pose2) 
	{
		return sqrt(pow(pose1.pose.position.x - pose2.pose.position.x, 2) + pow(pose1.pose.position.y - pose2.pose.position.y, 2));
	}

	void loader_rviz()
	{
		//1. 경로 총갯수 파악
		int total_size = 0;
		nav_msgs::Path global_path_rviz;
		global_path_rviz.header.frame_id = "map";
		global_path_rviz.header.stamp = ros::Time::now();
		// for(int i =0; i< all_csv_.size();i++)
		// {
		// 	total_size+=all_new_waypoints_[i].size();
		// }
		// // global_path_rviz.poses.resize(total_size);
		// if(all_csv_.size()!=0)
		// {
		// 	int idx_rowsize = 0;
		// 	for (int i = 0; i < all_csv_.size(); i++)
		// 	{
		// 		for(int j=0;j<all_new_waypoints_[i].size();j++)
		// 		{
		// 			geometry_msgs::PoseStamped loc;
		// 			loc.header.frame_id = "map";
		// 			loc.header.stamp = ros::Time::now();
		// 			loc.pose.position.x = (double)all_new_waypoints_[i][j].pose.pose.position.x;
		// 			loc.pose.position.y = (double)all_new_waypoints_[i][j].pose.pose.position.y;
		// 			// cout << "loc.pose.position.x" << (double)loc.pose.position.x << endl;
		// 			// cout << "all new waypoints[i][j] : " << (double)all_new_waypoints_[i][j].pose.pose.position.x<<endl;
		// 			// ROS_INFO("loc x : %lf",loc.pose.position.x);
		// 			// ROS_INFO("loc y : %lf",loc.pose.position.y);
		// 			loc.pose.position.z = 0.0;
		// 			if(j % 10 == 0){
		// 				global_path_rviz.poses.emplace_back(loc);
		// 			}
		// 		}
		// 			idx_rowsize += all_new_waypoints_[i].size();
		// 	}
		// }	
		for(int i =0; i< all_csv_.size();i++)
		{
			total_size+=all_new_waypoints_[i].size();
		}
		if(all_csv_.size()!=0)
		{
			int idx_rowsize = 0;
			for (int i = 0; i < all_csv_.size(); i++)
			{
				for(int j=0;j<all_new_waypoints_[i].size();j++)
				{
					geometry_msgs::PoseStamped loc;
					loc.header.frame_id = "map";
					loc.header.stamp = ros::Time::now();
					loc.pose.position.x = (double)all_new_waypoints_[i][j].pose.pose.position.x;
					loc.pose.position.y = (double)all_new_waypoints_[i][j].pose.pose.position.y;
					// cout << "loc.pose.position.x" << (double)loc.pose.position.x << endl;
					// cout << "all new waypoints[i][j] : " << (double)all_new_waypoints_[i][j].pose.pose.position.x<<endl;
					// ROS_INFO("loc x : %lf",loc.pose.position.x);
					// ROS_INFO("loc y : %lf",loc.pose.position.y);
					loc.pose.position.z = 0.0;
					if(j % 10 == 0){
						global_path_rviz.poses.emplace_back(loc);
					}
				}
					idx_rowsize += all_new_waypoints_[i].size();
			}
		}
		rviz_global_path_.publish(global_path_rviz);

		past_path_.emplace_back(current_pose);

		past_path_rviz_.header.frame_id = "map";
        past_path_rviz_.header.stamp = ros::Time::now();
        past_path_rviz_.poses = past_path_;

		rviz_currnet_path_.publish(past_path_rviz_);
	}  
			
};

int main(int argc, char** argv) 
{
	ros::init(argc, argv, "waypoint_loader");
	WaypointLoader wl;
	ros::Rate loop_rate(10);//10
	
	while(ros::ok())
	{
		ros::spinOnce();
		wl.poseProcess();	
		wl.loader_rviz();
		loop_rate.sleep();	
	}
	return 0;
}

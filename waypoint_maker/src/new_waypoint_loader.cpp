#include <ros/ros.h>

//msgs
#include <geometry_msgs/PoseStamped.h> 
#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <waypoint_maker/Lane.h> //For Parking Mission
#include <waypoint_maker/Waypoint.h> // Waypoints
#include <waypoint_maker/State.h> //For Mission (미션에 필요한 msg 헤더 모음)
#include <nav_msgs/Path.h>
#include <new_lane/lidar_topic_msg.h>


#include <vector>
#include <string>


// For CSV file loading
#include <fstream>
#include <sys/types.h> // 시스템 자료형 타입 정리용 헤더
#include <dirent.h> // 디렉토리 다루기 위한 헤더


#include <tf/tf.h>
#include <algorithm>
#include <ackermann_msgs/AckermannDriveStamped.h>


using namespace std;

/*
	waypoint loader algorithm : 

	1. initsetup()
		1-1 get_csvs_fromDirectory() : save all csv file's name in vector.
		1-2 get_new_waypooints() : open CSV files and save waypoints in vector.
	2. subscribe the Current_pose and the Lane_number
	3. poseprocess()
		3-1 getClosesttWaypoint() : find the closest waypoint 
		3-2 getFinalWaypoint() : decide 50 waypoints to publish.
		3-3 mission_state_ 설정
		3-4 state & waypoint publish
	4. publish waypoints

*/

class WaypointLoader {
private:
	int state_inspection_;	
	int current_mission_state_;


//ROS
	ros::NodeHandle nh_;
	ros::NodeHandle private_nh_;

	// publisher	
	ros::Publisher waypoint_pub_;
	ros::Publisher state_pub_;
	ros::Publisher rviz_global_path;
	ros::Publisher rviz_currnet_path_;
	ros::Publisher dist_pub_;
	// subscriber
	ros::Subscriber pose_sub_;
	ros::Subscriber lane_number_sub_;
	ros::Subscriber lidar_topic_sub_;

	//msg
	geometry_msgs::PoseStamped cur_pose_back;
	waypoint_maker::Lane lane_msg_;
	waypoint_maker::State state_msg_;
	std_msgs::Float64 dist_msg_;

//Functions
	//get_csvs_fromDirectory() & getNewWaypoints()
	std::vector<std::string> all_csv_;
	ifstream is_;
	vector< vector<waypoint_maker::Waypoint> > all_new_waypoints_;
	vector<waypoint_maker::Waypoint> new_waypoints_;
	vector<int> lane_size_;  
	int size_;

	vector< vector<int> > all_state_index_;
	nav_msgs::Path past_path_rviz_;
	vector<geometry_msgs::PoseStamped> past_path_;
	//getClosestWaypoint()
	vector<int> closest_waypoint_candidates;
	double max_search_dist_;
	double min_search_dist_;
	double dist_;
	int closest_waypoint_;
	int ex_closest_waypoint_;
	
	//getFinalWaypoint()
	int final_size_;
	vector<waypoint_maker::Waypoint> final_waypoints_;
	double final_dist_;
	int waypoint_count_;
	//spline
	vector<waypoint_maker::Waypoint> ryaw;
	vector<waypoint_maker::Waypoint> rk;

	const char* PATH_CSV_PATH = "/home/kuuve/catkin_ws/src/data_본선/";

//Mission
	//Parking mission
	int lane_number_;
	int ex_lane_number_;
	bool is_onlane_;
	bool is_state_;
	int lidar_lane_num_;

	//parameters for Parking mission
	const double WHEEL_BASE = 1.212;
	
	double cur_course_;
	double cur_speed_; 
	bool is_course_;
	bool is_search;

public:	
	WaypointLoader() {
		initSetup();
		ROS_INFO("WAYPOINT LOADER INITIALIZED.");
	}
	
	~WaypointLoader() {
		ROS_INFO("WAYPOINT LOADER TERMINATED.");
		past_path_.clear();
	}
	
	void initSetup() {
		private_nh_.getParam("/waypoint_loader_node/state_inspection", state_inspection_);

		// publisher
		waypoint_pub_ = nh_.advertise<waypoint_maker::Lane>("final_waypoints", 1);
		state_pub_ = nh_.advertise<waypoint_maker::State>("gps_state",1);
		dist_pub_ = nh_.advertise<std_msgs::Float64>("/change_dist", 1);
		// rviz
		rviz_global_path = nh_.advertise<nav_msgs::Path>("/rviz_global_path", 1);
		rviz_currnet_path_ = nh_.advertise<nav_msgs::Path>("/rviz_currnet_path_", 1);
		// subscriber
		pose_sub_ = nh_.subscribe("odom_back", 10, &WaypointLoader::poseCallback, this);
		lane_number_sub_ = nh_.subscribe("lane_number_msg_", 1, &WaypointLoader::lane_number_Callback, this);
		lidar_topic_sub_=  nh_.subscribe("lidar_topic", 1, &WaypointLoader::LidarLaneCallback, this);
		waypoint_count_ = 0;
		final_size_ = 200;
		size_ = 0;
		//getClosestWaypoint()
		max_search_dist_ = 15.0;
		min_search_dist_ = 0.5;
		current_mission_state_ = state_inspection_; 
		lidar_lane_num_ = 0;
		closest_waypoint_ = 0;
		lane_number_ = 0;
		ex_lane_number_ = 0;
		is_onlane_ = false;
		is_state_ = false;
		final_dist_ = .0;
		cur_course_ = .0;
		cur_speed_ = .0;
		is_course_ = false;
		is_search = false;

		get_csvs_fromDirectory();
		getNewWaypoints();
		loader_rviz();
	}
	
//Functions in initSetup

	

	void get_csvs_fromDirectory() // /home/kuuve/data/ 안에 있는 여러개의 csv파일 이름을 벡터에 저장
	{
	
		DIR* dirp = opendir(PATH_CSV_PATH); //디렉토리 저장

		if (dirp == NULL){
			perror("UNABLE TO OPEN FOLDER");
			return;
		}

		struct dirent* dp; 
		while((dp = readdir(dirp)) != NULL){ //디렉토리 안의 파일 이름 저장
			string address(PATH_CSV_PATH);

			string filename (dp->d_name);

			address.append(filename);
			if (filename.size() > 2){ //디렉토리 open시 ..과 .를 막기 위한 방어코드
				all_csv_.emplace_back(address);
				ROS_INFO("SAVED CSV");
			}
		}
		sort(all_csv_.begin(),all_csv_.end()); //정렬
		closedir(dirp);
	}

	void getNewWaypoints() { // 가져온 여러개의 CSV내 데이터를 벡터에 저장
	/*
		정리하고자 하는 벡터
		all_new_waypoints_ : Waypoint 모음 벡터
		all_state_index_ : mission state 모음 벡터
		lane_size_ : Vector 길이 모음 벡터
	*/

		string str_buf;
		int pos;
		vector<int> state_index_;
		vector<waypoint_maker::Waypoint> temp_new_waypoints;
		waypoint_maker::Waypoint temp_waypoint;

		for(auto i = 0; i < all_csv_.size(); i++){ //파일 갯수만큼

			state_index_.emplace_back(5); // current_mission_state 0번 시작 : 반드시 5 
										  // 이유 : getClosestWaypoint()의 조사 시작 범위를 위해
										  // ctrl+f --> void getClosestWaypoint()
						
			is_.open(all_csv_[i]); //파일 열기

			cout << "OPEN CSV" << all_csv_[i] << endl;
			
			temp_waypoint.mission_state = 0;
			// 이 부분 내용 알아보기
			while(getline(is_, str_buf)) {//파일 내용을 str_buf에 저장

				if(str_buf != "") { // temp_waypoint = [index, x, y, ryaw, rk, mission_state]
				//index
					pos = str_buf.find(",");
					temp_waypoint.waypoint_index = stoi(str_buf.substr(0, pos));

					//x
					str_buf = str_buf.substr(++pos);
					pos = str_buf.find(","); 
					temp_waypoint.pose.pose.position.x = stod(str_buf.substr(0, pos));

					//y
					str_buf = str_buf.substr(++pos);
					pos = str_buf.find(",");
					temp_waypoint.pose.pose.position.y = stod(str_buf.substr(0, pos));

					//mission state
					str_buf = str_buf.substr(++pos);
					pos = str_buf.find(",");
					if(temp_waypoint.mission_state != stoi(str_buf.substr(0, pos)))
					{
						state_index_.emplace_back(temp_waypoint.waypoint_index);

					} //mission state 변화시 따로 저장
					temp_waypoint.mission_state = stoi(str_buf.substr(0, pos));
				
					//ryaw
					str_buf = str_buf.substr(++pos);
					pos = str_buf.find(",");
					temp_waypoint.ryaw = stod(str_buf.substr(0, pos));
					
					//rk
					str_buf = str_buf.substr(++pos);
					pos = str_buf.find(",");
					temp_waypoint.rk = stod(str_buf.substr(0, pos));

					
					temp_new_waypoints.emplace_back(temp_waypoint);

				}
			}
			is_.close();
			
			size_ = temp_new_waypoints.size();
			lane_size_.emplace_back(size_); //lane_size 정리
				
			ROS_INFO("%d WAYPOINTS HAVE BEEN SAVED.", size_);

			all_new_waypoints_.emplace_back(temp_new_waypoints); //all_new_waypoints_ 정리
			all_state_index_.emplace_back(state_index_);//all_state_index_ 정리

			temp_new_waypoints.clear();
			state_index_.clear();
		}

		//0번 행 : 전체 경로
		//1번 행 ~  : 주차 경로
		new_waypoints_.assign(all_new_waypoints_[lane_number_].begin(), all_new_waypoints_[lane_number_].end());
		size_ = lane_size_[lane_number_];		
	}


	void poseProcess(){
		final_waypoints_.clear();
		
		getClosestWaypoint(cur_pose_back);//내 위치 기준 closest waypoint 찾기
		getFinalWaypoint();//최종 50개의 waypoint 결정
		
		current_mission_state_ = final_waypoints_[lane_number_].mission_state;

		


		if(!is_state_ && closest_waypoint_candidates.empty()){
			ROS_INFO("closest_waypoint_candidates are not Founded.");
		    current_mission_state_ = state_inspection_;
		}//중간에서 실행시 current_mission_state_ 수신하지 못하는 오류 방어 코드 -> 원인 아직 파악 못함 -> 로직 수정 필요

		

		if(lidar_lane_num_ == 1)
		{	
			// cout<<"inininin"<<endl;
			lane_number_ = 1;
			new_waypoints_.clear();
			new_waypoints_.assign(all_new_waypoints_[lane_number_].begin(), all_new_waypoints_[lane_number_].end());
			size_ = lane_size_[lane_number_];	
		}

		if(lidar_lane_num_ == 0)
		{	
			// cout<<"inininin"<<endl;
			lane_number_ = 0;
			new_waypoints_.clear();
			new_waypoints_.assign(all_new_waypoints_[lane_number_].begin(), all_new_waypoints_[lane_number_].end());
			size_ = lane_size_[lane_number_];	
		}


		closest_waypoint_candidates.clear();

		

		//dist_ 결정
		if(current_mission_state_ != 12) {	// end state 
			dist_ = calcPlaneDist(cur_pose_back, new_waypoints_[all_state_index_[lane_number_][current_mission_state_+1]].pose);
		}
		else {
			dist_ = calcPlaneDist(cur_pose_back, new_waypoints_[all_state_index_[lane_number_][0]].pose);	// path end -> back to 0 state
		}
		//현위치와 다음 mission_state 간의 거리 (각종 미션에 사용)

		state_msg_.dist = dist_;
		state_msg_.current_state = current_mission_state_;
		state_msg_.lane_number = lane_number_;
		state_pub_.publish(state_msg_);//미션 관련 정보 publish
		
		lane_msg_.waypoints = final_waypoints_;
		lane_msg_.onlane = is_onlane_;		
		waypoint_pub_.publish(lane_msg_);//waypoint publish

		int final_waypoints_size = lane_msg_.waypoints.size();
		ROS_INFO("FINAL WAYPOINTS NUMBER=%d PUBLISHED.", final_waypoints_size);	
	}




	/*
		getClosestWaypoint()
		1. 후보군 선정
		2. 최종 cloest_waypoint 선정
	*/
	void getClosestWaypoint(geometry_msgs::PoseStamped cur_pose_back) 
	{
		
		//1. 후보군 선정
		//조사 시작 범위 : (현 mission state 시작점) - 4  ~ (현 mission state + 2)번 구역까지
		//조사시 mission_state 
		/*  ex)
			current mission state 3번 : 67 ~ 190 이라면
			조사 시작 범위 = 63 ~

			예외)
			current mission state 0번 : 0 ~ 
			조사 시작 범위 = -4 ~
			-> 오류 발생

			해결 : 	state_index_.emplace_back(5)
				-> current mission state 0번 : 5 ~
				   조사 시작 범위 : 1 ~
		*/
		int t_index = all_state_index_[lane_number_][current_mission_state_]-4;
		cout << "==================================================================================" << endl;
		cout << "search_startIndex : " << t_index << endl;
		cout << "current_lane : " << lane_number_ << endl;
		cout << "currnet_mission_state : " << current_mission_state_ << endl;
		cout << "==================================================================================" << endl;


		if(current_mission_state_ != 12)		// nomal state
        {
            for (int i = t_index; i < size_; i++)
            {
                float dist = calcPlaneDist(cur_pose_back, new_waypoints_[i].pose);
                int t_state_check = new_waypoints_[i].mission_state - current_mission_state_;
                if (dist < max_search_dist_ && (t_state_check == 2 || t_state_check == 1 || t_state_check == 0))
                {
                    closest_waypoint_candidates.emplace_back(i);
                }
                if (dist > max_search_dist_ && t_state_check > 2)
                    break;
            }
        }
        else	// end state -> forced index input
        {	

            for (int i = t_index; i < size_; i++)
            {
                float dist = calcPlaneDist(cur_pose_back, new_waypoints_[i].pose);
                int t_state_check = abs(new_waypoints_[i].mission_state - current_mission_state_);
                if(dist < max_search_dist_ && (t_state_check == 6 || t_state_check == 0))
					closest_waypoint_candidates.emplace_back(i);
					// if(i < 3092) {
					// 	closest_waypoint_candidates.emplace_back(i);
					// }
            }
            for (int i = 1; i < 201; i++)	// 0 waypoint forced start
            {
                float dist = calcPlaneDist(cur_pose_back, new_waypoints_[i].pose);
                int t_state_check = abs(new_waypoints_[i].mission_state - current_mission_state_);
                if(dist < max_search_dist_ && (t_state_check == 6 || t_state_check == 0))
                    closest_waypoint_candidates.emplace_back(i);
            }
        }

		
		//2. 후보 중 최종 cloest_waypoint 선정
		if(!closest_waypoint_candidates.empty()) {
			int waypoint_min = -1;
			double dist_min = max_search_dist_; 
			for(int i=0;i<closest_waypoint_candidates.size();i++) {
				double dist = calcPlaneDist(cur_pose_back, new_waypoints_[closest_waypoint_candidates[i]].pose);
				if(dist < dist_min) {
					dist_min = dist;
					waypoint_min = closest_waypoint_candidates[i];
				}
			}

			closest_waypoint_ = waypoint_min;
			
			ROS_INFO("CLOSEST WAYPOINT INDEX=%d, X=%f, Y=%f", closest_waypoint_, new_waypoints_[closest_waypoint_].pose.pose.position.x, new_waypoints_[closest_waypoint_].pose.pose.position.y);
			is_state_ = true;//middle start

			final_dist_ = calcPlaneDist(cur_pose_back, new_waypoints_[closest_waypoint_].pose);
            dist_msg_.data = final_dist_;
			cout<<"final_dist_"<<final_dist_<<endl;
            dist_pub_.publish(dist_msg_);
		}
		else ROS_INFO("THERE IS NO CLOSEST WAYPOINT CANDIDATE.");

		
	}

// getFinalWaypoint()
	//최종 250개의 waypoint 선정
	void getFinalWaypoint(){ 

		waypoint_count_ = 0;
		for(int i = 0; i < final_size_; i++) 
		{
			
			if(closest_waypoint_ + i < size_-4) 
			{

				final_waypoints_.emplace_back(new_waypoints_[closest_waypoint_ + i]);
				waypoint_count_++;
				// cout<<"waypoint_count_waypoint_count_"<<waypoint_count_<<endl;
			}
			
		}
		for(int j = 1; j < final_size_- waypoint_count_ + 1; j++)
		{
			final_waypoints_.emplace_back(new_waypoints_[j]);
			// cout<<"jjjjjjjjjjjjv->>>>>"<<j<<endl;
		}
	}

	//Function in getClosetWaypoint()
	double calcPlaneDist(const geometry_msgs::PoseStamped& pose1, const geometry_msgs::PoseStamped& pose2) {
		return sqrt(pow(pose1.pose.position.x - pose2.pose.position.x, 2) + pow(pose1.pose.position.y - pose2.pose.position.y, 2));
	} 



//Callback function
	void poseCallback(const nav_msgs::Odometry::ConstPtr &msg) {
		cur_pose_back.pose.position = msg->pose.pose.position;
		lane_msg_.header = msg->header;		
	}

	void LidarLaneCallback(const new_lane::lidar_topic_msg::ConstPtr &msg)
	{
		lidar_lane_num_= msg->lane_num;
	}
	
	void lane_number_Callback(const waypoint_maker::State::ConstPtr &msg){
		lane_number_ = msg->lane_number;
	}

	void loader_rviz()
	{
		//1. 경로 총갯수 파악
		int total_size = 0;
		for(int i =0; i< all_csv_.size();i++)
		{
			total_size+=all_new_waypoints_[i].size();
		}

		nav_msgs::Path global_path_rviz;
		global_path_rviz.header.frame_id = "map";
		global_path_rviz.header.stamp = ros::Time::now();
		// global_path_rviz.poses.resize(total_size);
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
		rviz_global_path.publish(global_path_rviz);
		past_path_.emplace_back(cur_pose_back);

		past_path_rviz_.header.frame_id = "map";
        past_path_rviz_.header.stamp = ros::Time::now();
        past_path_rviz_.poses = past_path_;
		
        rviz_currnet_path_.publish(past_path_rviz_);
	}
	
};

int main(int argc, char** argv) {
	ros::init(argc, argv, "waypoint_loader"); //노드명 -> 생성자 void init setup
	WaypointLoader wl;//생성자 void init setup
	ros::Rate loop_rate(10);
	
	while(ros::ok()) {
		ros::spinOnce();
		wl.poseProcess(); 
		wl.loader_rviz();
		loop_rate.sleep();	
	}
	return 0;
}

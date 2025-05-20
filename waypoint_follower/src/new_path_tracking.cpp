#include <waypointFollower.h>
#include <new_platform_control.h>

/*
	pure pursuit 알고리즘

	1. 목표 추종점 찾기
		1-1 50개의 waypoint와 cur_pose 간의 거리 계산
		1-2 dist > lookahead_dist_ 될 때의 waypoint를 추종점으로
		1-3 계산에 사용할 최종 ld 설정

	*  후진에서 전진 or 전진에서 후진 전환시 heading 방향 반대로 돌려주기

	2. alpha 구하기
		2-1 atan2로 temp_theta 계산 (temp_theta = 목표 heading, atan2 좌표계)
		2-2 atan2-절대좌표계 통일
		2-3 alpha 계산 (목표 heading - 현재 heading)

	3. mission 별 코드
		3-1 큰 정적 미션
		3-2 배달 미션
		3-3 주차 미션

	4. 최종조향각

*/


double WaypointFollower::pure_pursuit()
{
	getClosestWaypoint(cur_pose_back);
	int min_index = find_cloestindex(cur_pose_back)+10;

	// 기본 LD
	//lookahead_dist_ = 5.0;

	// if (loader_number_ == 0)
	// {
	// 	if(spd_state_ > 16)
	// 	{
	// 		lookahead_dist_ = ego_cur_speed_ > 16 ? 4.0 + ego_cur_speed_*0.125 : 5.0;   // 직진구간시speed에 따른 LD
	// 	}

	// 	else
	// 	{
	// 		lookahead_dist_= 5.0;
	// 	}
	// }

	// else if (loader_number_ == 1)
	// {
	// 	if(spd_state_ > 16)
	// 	{
	// 		lookahead_dist_ = ego_cur_speed_ > 16 ? 4.0 + ego_cur_speed_*0.125 : 5.5;   // 직진구간시speed에 따른 LD
	// 	}

	// 	else
	// 	{
	// 		lookahead_dist_= 4.5;
	// 	}
	// }


	//16이상은 직선and 유려한 곡선   else -> 커브 구간 	
	// if(spd_state_>17)   
	// {
	// 	lookahead_dist_ = ego_cur_speed_ > 16 ? 4.5 + ego_cur_speed_*0.125 : 4.5;   // 직진구간시speed에 따른 LD
	// }
	// else
	// {
	// 	lookahead_dist_ = ego_cur_speed_ > 16 ? 4.5 + ego_cur_speed_*0.125 : 4.5; //커브구간시speed에 따른 LD
	// }


	//////레인체인지 피팅구간/////////////////
	if(change_dist_>0.5)
	{
		lookahead_dist_ = 15/(1+exp(-change_dist_))-2.5;
		if(lookahead_dist_ > 11)
		{
			lookahead_dist_ = 11.0;
		}
		else if(lookahead_dist_ < 6.5)
		{
			lookahead_dist_ = 6.5;
		}
	}
	

	// state별 LD fitting
	// if(current_mission_state_ == 1)
	// {
	// 	lookahead_dist_ = 5.0;
	// }
	// else if(current_mission_state_ == 4) {
	// 	lookahead_dist_ = 5.0;
	// }


// 1. 목표추종점 찾
	double final_ld = 0.0;

	for (int i = 0; i < waypoints_size_; i++)
	{
		double dist = calcPlaneDist(cur_pose_back, waypoints_[i].pose);
		if (dist > lookahead_dist_)
		{
			target_index_ = i;
			final_ld = dist; // 최종으로 사용할 ld
			break;
		}
	}

//visualize
	visualization_msgs::Marker marker;
	marker.header.frame_id = "map";
	marker.header.stamp = ros::Time();
	marker.ns = "my_namespace";
	marker.id = 0;
	marker.type = visualization_msgs::Marker::SPHERE;
	marker.action = visualization_msgs::Marker::ADD;
	marker.pose.position.x = waypoints_[target_index_].pose.pose.position.x;
	marker.pose.position.y = waypoints_[target_index_].pose.pose.position.y;
	marker.pose.position.z = 0;
	marker.pose.orientation.x = 0.0;
	marker.pose.orientation.y = 0.0;
	marker.pose.orientation.z = 0.0;
	marker.pose.orientation.w = 1.0;
	marker.scale.x = 1;
	marker.scale.y = 1;
	marker.scale.z = 0.1;
	marker.color.a = 1.0; // Don't forget to set the alpha!
	marker.color.r = 1.0;
	marker.color.g = 0.0;
	marker.color.b = 1.0;
	//only if using a MESH_RESOURCE marker type:
  	//  marker.mesh_resource = "package://pr2_description/meshes/base_v0/base.dae";
  	pp_target_pub.publish( marker );

// 2. alpha 구하기
	// temp_theta 계산
	double target_x = waypoints_[target_index_].pose.pose.position.x;
	double target_y = waypoints_[target_index_].pose.pose.position.y;

	double cur_x = cur_pose_back.pose.position.x;
	double cur_y = cur_pose_back.pose.position.y;
	double dx = (target_x - cur_x);
	double dy = (target_y - cur_y);

	double temp_theta = (atan2(dy, dx)) * 180.0 / M_PI; // temp theta : 동쪽 기준, 반시계 방향, -π ~ π

	// alpha 계산
	double deg_alpha = cur_course_ - temp_theta;
	if(temp_theta * cur_course_ < 0) // -pi -> pi로 뛰었을때.
    {
		if(cur_course_ < -90.0 && temp_theta > 0)
		{
			cout << "case 0 " << endl;
			deg_alpha = cur_course_ - temp_theta + 360.0;		

		}
		else if(cur_course_ > 0 && temp_theta < -90.0)
		{
			cout << "case 1 " << endl;
			deg_alpha = cur_course_ - temp_theta - 360.0;
		}	
	}
	double alpha = deg_alpha * M_PI / 180.0;

// 3. 최종 ld 계산
	if (final_ld > 20.0)
		final_ld = 20.0;
	if (final_ld < 2.0)
		final_ld = 2.0;

// 4. 최종 조향각
	double cur_steer = atan2((2.0 * WHEEL_BASE * sin(alpha)) / final_ld, 1.0);
	cur_steer = cur_steer * 180.0 / M_PI;
	cur_steer *= 1.3;
	// 얼라인 피팅
	// cur_steer -= 0.8;

	// cout << "====== Pure_Pursuit ======" <<endl;
	cout << "LookaheadDist    : " << lookahead_dist_ << endl;
	// cout << "final ld         : " << final_ld << endl;
	// cout << "cur_course_      : " << cur_course_ << endl;	
	// cout << "temp_theata      : " << temp_theta << endl;
	// cout << "alpha            : " << deg_alpha << endl;
	// cout << "cur_steer        : " << cur_steer << endl;

	return cur_steer;
}



double WaypointFollower::stanly()
{
    geometry_msgs::PoseStamped transformed_pose;
    double rad_car_course = cur_course_ * M_PI/180.0;
	
//1. 전륜축 위치구하기
    transformed_pose.pose.position.x = cur_pose_front.pose.position.x;
    transformed_pose.pose.position.y = cur_pose_front.pose.position.y;
    
//2. 횡방향 최단거리 계산
    int min_idx = find_cloestindex(transformed_pose);
    double cross_error = sqrt(pow(transformed_pose.pose.position.x - waypoints_[min_idx].pose.pose.position.x,2) + pow(transformed_pose.pose.position.y - waypoints_[min_idx].pose.pose.position.y,2));
	double stanley_k = 0.0;
	stanley_k = abs(waypoints_[min_idx].rk) + 2.4;
    
//3. 좌우 판단
    double dx = waypoints_[min_idx].pose.pose.position.x - transformed_pose.pose.position.x;
    double dy = waypoints_[min_idx].pose.pose.position.y - transformed_pose.pose.position.y;
	double sign_check = sin(waypoints_[min_idx].ryaw)*dx - cos(waypoints_[min_idx].ryaw)*dy; //
    if(sign_check < 0) cross_error = -cross_error;

	double delta = rad_car_course - waypoints_[min_idx].ryaw;; 
	//delta
	if(waypoints_[min_idx].ryaw * rad_car_course < 0)
    {
		if(rad_car_course < -M_PI/2 && waypoints_[min_idx].ryaw > 0)
		{
			cout << "case 0 " << endl;
			delta = rad_car_course - waypoints_[min_idx].ryaw + 2*M_PI;		

		}
		else if(rad_car_course > 0 && waypoints_[min_idx].ryaw < -M_PI/2)
		{
			cout << "case 1 " << endl;
			delta = rad_car_course - waypoints_[min_idx].ryaw - 2*M_PI;
		}	
	}

	//calc steer
    double steer = delta + atan2(stanley_k * cross_error, cur_speed_); // *cos(rad_car_course)
	double deg_steer = steer * 180.0/M_PI;

	//얼라인 피팅
	// deg_steer -= 0.8;

	//visualize
	visualization_msgs::Marker marker;
	marker.header.frame_id = "map";
	marker.header.stamp = ros::Time();
	marker.ns = "my_namespace";
	marker.id = 0;
	marker.type = visualization_msgs::Marker::SPHERE;
	marker.action = visualization_msgs::Marker::ADD;
	marker.pose.position.x = waypoints_[min_idx].pose.pose.position.x;
	marker.pose.position.y = waypoints_[min_idx].pose.pose.position.y;
	marker.pose.position.z = waypoints_[min_idx].pose.pose.position.z;
	marker.pose.orientation.x = 0.0;
	marker.pose.orientation.y = 0.0;
	marker.pose.orientation.z = 0.0;
	marker.pose.orientation.w = 1.0;
	marker.scale.x = 1;
	marker.scale.y = 1;
	marker.scale.z = 0.1;
	marker.color.a = 1.0; // Don't forget to set the alpha!
	marker.color.r = 0.0;
	marker.color.g = 0.0;
	marker.color.b = 1.0;
	//only if using a MESH_RESOURCE marker type:
  	//  marker.mesh_resource = "package://pr2_description/meshes/base_v0/base.dae";
  	st_target_pub.publish( marker );

    // cout << "====== STANLEY ====== " << endl;
    // cout << "current state : " << waypoints_[min_idx].mission_state << endl;
    // cout << " rk           : " << waypoints_[min_idx].rk << endl;
	// cout << " ryaw         : " << waypoints_[min_idx].ryaw * 180 / M_PI << endl;
    // cout << " caryaw       : " << rad_car_course * 180 / M_PI << endl;
    // cout << "stanley_k     : " << stanley_k << endl;
    // cout << "delta         : " << delta * 180.0/M_PI << endl;
	// cout << "atan2(stanley_k * cross_error, cur_speed_)" << atan2(stanley_k * cross_error, cur_speed_) * 180 / M_PI << endl;  // *cos(rad_car_course)
    // cout << "cross_error   : " << cross_error << endl;
	// cout << "sign_check    : " <<sign_check <<endl;    
	// cout << "min_idx       : " << min_idx << endl;
    // cout << "stanley_steer : " << deg_steer << "\n" << endl;

    return deg_steer;
}



int WaypointFollower::find_cloestindex(geometry_msgs::PoseStamped pose)  // 이때 pose 는 내 위치에서 앞바퀴 축으로 회전변환 시켜서 사용해야함
{
	double min_dis = DBL_MAX;
	int min_idx = 0;
	for(int i=0;i < waypoints_size_;i++)
	{
		double dist = calcPlaneDist(pose,waypoints_[i].pose);
		
		if(dist < min_dis)
		{
			min_dis = dist;
			min_idx = i;
		}
	}
	return min_idx;

}

double WaypointFollower::cross_error(geometry_msgs::PoseStamped pose) // pose 는 내 전륜축 위치 (0,0)
{
	// 횡방향 수직거리 = 최단거리
	double min_dis = DBL_MAX;
	int min_idx = 0;
	for(int i=0;i<waypoints_size_;i++)
	{
		double dist = calcPlaneDist(pose,waypoints_[i].pose);
		if(dist < min_dis)
		{
			min_dis = dist;
			min_idx = i;
		}
	}
	return min_dis;
}




double WaypointFollower::hybrid_control()
{
    /*
    곡률 범위
    ~0.01 : 유사 직진
    ~0.02 : 약커브
    ~0.03 : 중커브
    ~0.04 : 강커브
    */
    double steer = .0;
    double ratio = .0;

    //현 위치 closest idx 구하기
    int min_idx = find_cloestindex(cur_pose_front);
    double ratio_gain = abs(waypoints_[min_idx].rk);
	
	ratio = 0.15;
	// ratio = 35/(1+exp(-1.5*(abs(waypoints_[5].rk)-2.5)))-0.67; 
    // cout << "ratio : " << ratio << endl;
	// steer = value1 * pure_pursuit() + value2 * stanly() + value3 * Frenet_pure_persuit() + value4 * frenet_stanly() ;
	steer = (1-ratio) * pure_pursuit() + ratio * stanly();


	// if(ratio_gain > 0.06) steer *= 1.05;
	// cout << "====== hybrid ======" << endl;
	// cout << "ratio gain      : " << ratio_gain << endl;
	// cout << "ratio           : " << ratio << endl;
	// cout << "hybrid_steer    : " << steer <<endl;

	return steer;
}
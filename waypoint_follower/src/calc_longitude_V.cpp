#include <ros/ros.h>
#include <waypointFollower.h>


double WaypointFollower::calc_longitude_V(double default_speed) {
    static double prev_rk = 0.0;
    static double prev_speed = default_speed;

    int min_idx = find_cloestindex(cur_pose_);

    double v_min_limit = 20.0;

    
    if (waypoints_.empty()) {
        cerr << "Error: waypoints_ is empty!" << endl;
        return default_speed;
    }

    double rk = waypoints_[min_idx].rk;
    rk = (rk < 0) ? -rk : rk; // 절댓값 처리

    // 곡률 변화량 계산
    double drk = rk - prev_rk;
    drk = (drk < 0) ? -drk : drk; // 절댓값 처리
    prev_rk = rk;

    // 너무 작은 rk는 직선으로 간주하고 기본 속도 유지
    if (rk <= 1e-3) {
        cout << "speed: " << default_speed << " KPH (small rk)" << endl;
        prev_speed = default_speed;
        return default_speed;
    }

    // 물리 기반 속도 계산
    const double gravity = 9.81;
    const double tire_friction = 0.8;
    const double safety_factor = 0.80;

    double ay_max = tire_friction * gravity;
    double v_max = sqrt(ay_max / rk);
    double stable_speed = safety_factor * v_max;

    // 속도 제한
    double v_max_limit = default_speed;
    if (stable_speed > v_max_limit) stable_speed = v_max_limit;
    if (stable_speed < v_min_limit) stable_speed = v_min_limit;

    // 곡률 변화가 작을 경우 → 기존 속도 유지
    if (drk < 0.01) {
        cout << "speed: " << prev_speed << " KPH (small drk)" << endl;
        return prev_speed;
    }

    // Low-pass filter 적용 (속도 변화 완화) 노이즈에 대한 변동 억제
    double alpha = 0.2;  // 낮을수록 부드럽게 (필터 계수) 현재 속도 20%, 이전 속도 80% 비율
    double filtered_speed = alpha * stable_speed + (1.0 - alpha) * prev_speed;
    prev_speed = filtered_speed;

    cout << "speed: " << filtered_speed << " KPH (filtered)" << endl;
    return filtered_speed;
}



double WaypointFollower::calc_ld(double longitudinal_speed)
{
    const double v_max = 40.0;
    const double v_min = 20.0;
    const double ld_max = 10;
    const double ld_min = 5;

    if (longitudinal_speed < v_min) {
        return ld_min;  // 속도가 최저 속도보다 작으면 최저 LD
    }
    if (longitudinal_speed > v_max) {
        return ld_max;  // 속도가 최고 속도보다 크면 최고 LD
    }

    // 선형 보간법
    double ld = ld_min + ((longitudinal_speed - v_min) / (v_max - v_min)) * (ld_max - ld_min);
    //cout << "calc_ld: "<< ld << endl;
    return ld;
}

int WaypointFollower::find_cloestindex(geometry_msgs::PoseStamped pose)  // 이때 pose 는 내 위치에서 앞바퀴 축으로 회전변환 시켜서 사용해야함
{
	double min_dis = DBL_MAX ;
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
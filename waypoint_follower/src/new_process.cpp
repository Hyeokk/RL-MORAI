#include <waypointFollower.h>
#include <new_platform_control.h>

/*
	Process Algorithm

	1. 위치 수신, Heading 수신, 경로 수신 확인
	2. mission_state별 제어
	3. 주차 미션시 제어
	4. 제어에 따른 속도 및 조향각 결정
*/

void WaypointFollower::process()
{

	cal_course();	// Heading 수신

	// 1. 위치 수신, 경로 수신 확인
	if (is_pose_ && is_lane_)
	{
		is_control_ = true; // control 작동

	// 2. mission_state별 제어
		//15km/h = 4.167
		//10km/h = 2.78
		ros::Time::init();
		switch (current_mission_state_) // csv에서 받아오는 mision_state
		{
			case 0: // 처음 직진
				spd_state_ = 16.0;
				lookahead_dist_ = 7.0;
				
				break;

			case 1: // 처음 작은 곡선
				spd_state_ = 14.0;
				lookahead_dist_ = 5.25;
				
				break;

			case 2: //전 로터리
				spd_state_ = 16.0;
				lookahead_dist_ = 6.0;

				break;

			case 3: // 로터리 
				spd_state_ = 16.0;
				lookahead_dist_ = 6.0;

				break;

			case 4: // 방지턱 시작
				spd_state_ = 12.0;
				lookahead_dist_ = 5.25;

				break;

			case 5: // 방지턱 탈출 + 직진
				spd_state_ = 16.0;
				lookahead_dist_ = 7.0;

				break;

			case 6: // 처음 큰 곡선
				spd_state_ =  16.0;
				lookahead_dist_ = 6.0;

				break;

			case 7: // 사거리 전 직진
				spd_state_ =  16.0;
				lookahead_dist_ = 7.0;

				break;

			case 8: // 사거리 + 두번째 큰 곡선
				spd_state_ =  16.0;
				lookahead_dist_ = 6.0;

				break;

			case 9: // 사거리 탈충
				spd_state_ =  16.0;
				lookahead_dist_ = 7.0;

				break;

			case 10: // 세번째 곡선
				spd_state_ =  14.0;
				lookahead_dist_ = 5.5;

				break;

			case 11: // 세번째 곡선 탈출
				spd_state_ =  16.0;
				lookahead_dist_ = 6.0;

				break;

			case 12: // 마지막 작은 곡선
				spd_state_ =  14.0;
				lookahead_dist_ = 5.25;

				break;

			default:
				spd_state_ = 5.0;
				cerr << "!!!MISSION STATE ERROR!!!" << endl;
				break;
		}
	}
	else
	{
		// cout << "process DIED" << endl;
	}

	// 4. 제어에 따른 조향각 결정
	if (is_control_)
	{

		// Fail - Safe
		if(!fix_){
			if(spd_state_ == 0) spd_state_ = 0;
			else spd_state_ = 7.0;	// 감속
		}

		input_speed_ = spd_state_;
		input_steer_ = hybrid_control();

		//is acc, index 토픽 받아오기
        if(is_acc){
            if(acc_index > 150) input_speed_ = spd_state_;
            else if(acc_index <= 150 && acc_index > 100) {
				if(spd_state_ == 20.0) input_speed_ = input_speed_ + (49.167 / (1 + exp(-0.017 * (acc_index - 60))) - 40.5); 
				else if(spd_state_ == 18.0) input_speed_ = input_speed_ + (44.25 / (1 + exp(-0.017 * (acc_index - 60))) - 36.45);
				else if(spd_state_ == 16.0) input_speed_ = input_speed_ + (39.33 / (1 + exp(-0.017 * (acc_index - 60))) - 32.4); 
				else if(spd_state_ == 14.0) input_speed_ = input_speed_ + (34.4167 / (1 + exp(-0.017 * (acc_index - 60))) - 28.35); 
				else if(spd_state_ == 12.0) input_speed_ = input_speed_ + (29.5 / (1 + exp(-0.017 * (acc_index - 60))) - 24.3);
		        // // input_speed_ = input_speed_ + (49.167 / (1 + exp(-0.017 * (acc_index - 60))) - 40.5); // 20km
                // // input_speed_ = input_speed_ + (44.25 / (1 + exp(-0.017 * (acc_index - 60))) - 36.45); // 18km
                // // input_speed_ = input_speed_ + (39.33 / (1 + exp(-0.017 * (acc_index - 60))) - 32.4); // 16km
                // input_speed_ = input_speed_ + (29.5 / (1 + exp(-0.017 * (acc_index - 60))) - 24.3); // 12km
                cout << "speed : " << input_speed_ << endl;
            }
            else if(acc_index <= 100 && acc_index > 0){
                input_speed_ = .0;
            }
        }
        // cout << "acc index : " << acc_index << endl;

		// cout << "INPUT STEER" << input_steer_ << endl;



		// cout << "fix mode : ";
		// if(fix_){
		// 	cout << " True" << endl;
		// }
		// else{
		// 	cout << " Flase" << endl;
		// }

		// 범위 이외시 최대조향각
		if (input_steer_ >= 19.9999)
			input_steer_ = 19.9999;
		if (input_steer_ <= -19.9999)
			input_steer_ = -19.9999;
	}


	is_pose_ = false;
	is_course_ = false;
}

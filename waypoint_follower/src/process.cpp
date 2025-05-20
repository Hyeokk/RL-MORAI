#include <ros/ros.h>
#include <waypointFollower.h>


double rk = 0;

void WaypointFollower::process() 
{
	
	if(is_pose_ && is_course_ && is_lane_)
	{ 
		is_control_ = true;
		switch(current_mission_state_)
		{
			case 0:
            	dynamic_speed_ = calc_longitude_V(40.0);
				lookahead_dist_ = calc_ld(dynamic_speed_);
				// spd_state_ = (int)Speed_Opt::Vel_10;
				lookahead_dist_ = STRAGHT_LD;
				
				break;
			case 1:
				if(!static_flag){// for Jeong Min
					spd_state_ = (int)Speed_Opt::Slow;
					lookahead_dist_ = STRAGHT_LD;
					break;
				}
				else{// for Jeong Min
					speed_ = j_speed;
					cur_steer = j_steer;
					break;
				}

			case 2:
				spd_state_ = (int)Speed_Opt::Stop;
				lookahead_dist_ = STRAGHT_LD;
				break;
		}

	}
	

	
	if (is_control_) 
	{ 
		switch(spd_state_)
		{
			case (int)Speed_Opt::Stop: speed_ = 0; brake_ = 1;
				break;
			case (int)Speed_Opt::Vel_10: speed_ = 10.0;
				break;
			case (int)Speed_Opt::Vel_15: speed_ = 15.0;
				break;
			case (int)Speed_Opt::Vel_17: speed_ = 17.0;
				break;
			case (int)Speed_Opt::Vel_20: speed_ = 20.0;
				break;
			case (int)Speed_Opt::Parking_vel: speed_ = 8.0;
				break;
			case (int)Speed_Opt::Vertical: speed_ = 7.0;
				break;
			case (int)Speed_Opt::Fast: speed_ = 50.0;
				break;
			case (int)Speed_Opt::Slow: speed_ = 5.0;
				break;
			case (int)Speed_Opt::Vel_19: speed_ = 19.0;
				break;
			
			//default: cerr << "!!!Speed State Error!!!" << endl;
				//break;
		}
		
		if(1){
		//speed_ = PID(speed_, cur_speed_);
		cur_steer = calcSteeringAngle();
		}




		//가속구간
		ackermann_msg_.longlCmdType = 2;
		ackermann_msg_.velocity = dynamic_speed_;
		ackermann_msg_.brake = brake_;
		ackermann_msg_.steering = cur_steer;
		
		ackermann_pub_.publish(ackermann_msg_);
		
	}

	is_pose_ = false;
	is_course_ = false;
	// cout << "IS DYNAMIC FINISHED    ::    "   << is_dynamic_finished_ << endl;
	// cout << "CUR_SPEED  ::   "  << cur_speed_ << endl;
    //cout << "-------------------------------- " << endl;
}


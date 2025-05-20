#ifndef PLATFORMCONTROL_H
#define PLATFORMCONTROL_H


#include <ros/ros.h>
#include <fstream>
#include <string>
#include <math.h>
#include <iostream>
#include <vector>
#include <numeric>

#include <erp_driver/erpCmdMsg.h>
#include <erp_driver/erpStatusMsg.h>
#include <std_msgs/Float32.h>
#include <new_lane/lidar_topic_msg.h>
/*

1/2 platform 최신화

Cycle Time : 20 msec
Baudrate : 115200, parity : None,  Stop : 1
     PC -> ERP42 (14 Byte)                              ERP42 -> PC (18 Byte)
바이트 이름     /            값                       바이트 이름      /           값
1   S               0x53                        1   S                   0x53
2   T               0x54                        2   T                   0x54
3   X               0x58                        3   X                   0x58
4   A or M          0x00 or 0x01                4   A or M              0x00 or 0x01
5   E-stop          0x00 or 0x01                5   E-stop              0x00 or 0x01
6   GEAR            0~2                         6   GEAR                0~2
7   SPEED0┐         0~200                       7   SPEED0┐             -300~300
8   SPEED1┘                                     8   SPEED1┘
9   STEER0┐         -2000~2000                  9   STEER0┐             -2000~2000
10  STEER1┘                                     10  STEER1┘
11  BRAKE           1~200                       11  BRAKE               1~200
12  ALIVE           0~255                       12  ENC0┐
13  ETX0            0x0D                        13  ENC1│               -2^31~2^31
14  ETX1            0x0A                        14  ENC2│
                                                15  ENC3┘
                                                16  ALIVE               0~255
                                                17  ETX0                0x0D
                                                18  ETX1                0x0A

시리얼에 들어가 있는 값들
AorM -> Auto or Manual -------- 0x00 : manual mode ,      0x01 : auto mode
ESTOP -> Emergency STOP -------- 0x00 : E-STOP Off,      0x01 : E-STOP On
GEAR -> 0x00 : forward drive,      0x01 : neutral,      0x02 : backward drive
SPEED -> actual speed (KPH) * 10
STEER -> actual steering dgree (dgree) * 100, 오차율 : 4%  (negative is left steer)
BRAKE -> 1 : no braking,      100 : full braking
ENC -> encoder counting (Pulse per revolution : 48)
ALIVE -> increasing each one step (0~255)

<Returned Serial data 변환>
speed = (speed_return0 + speed_return1 * 256) / 10) km/h
steer = ((int)(serial_input[8]) + (int)(serial_input[9]) * 256) / 100) degree
brake = 1~150
*/
using namespace std;
class PlatformConnector
{
private:
    ros::NodeHandle nh_;
    ros::Subscriber course_sub_,erp42_status_sub_,lidar_topic_sub_;
    ros::Publisher erp42_ctrl_cmd_pub_;
    ros::Publisher ego_speed_pub_;

	ofstream os_;

    const double GEAR_RATIO = 5.0;
    const double WHEEL_RADIOUS = 0.29;


    double input_speed_;
    double input_steer_;
    double cur_speed_;
    double spd_state_;
    double gps_speed_;

    int   get_control_mode;
    bool  get_e_stop;
    int get_gear;
    int get_speed;
    int get_steer;
    int get_brake;
    int get_encoder;
    int get_alive;

    bool is_erp_state_;
    bool is_gps_state_;

    bool is_acc;
    int acc_index;

    bool e_stop;
    int gear;
    int speed;
    int steer;
    int brake;

    int count_;
    double prev_spd_state_;
    int current_mission_state_;
    // PID
    std_msgs::Float32 erp_speed_msgs_;

    //new PID
    const double KP = 1.6; //1.796; //1.6; 0821     //1.4345; //1.6794; |0818 유효값  //1.7868; //1.833; 
	const double KI = 0.8; //1.3321; //0.8; 0821  //0.51897; //1.0125;|0818 유효값  //1.104;  //1.1025;
	const double KD = .0;
    const double BRAKE_THRSHHOLD = .0;
	const double CONTROL_TIME = 0.05;

    double sum_error_, prev_error_;
    double Get_Kmh_Speed_;

public:
    PlatformConnector()
    {
        initSetup();
    }
    ~PlatformConnector()
    {
		os_.close();
    }
    void initSetup()
    {
        // course_sub_ = nh_.subscribe("cur_speed", 10, &PlatformConnector::SpeedCallback, this);
       	erp42_status_sub_ = nh_.subscribe("/erp42_status", 10,&PlatformConnector::ErpCallback,this);
		// lidar_topic_sub_=  nh_.subscribe("lidar_topic", 1, &PlatformConnector::LidarLaneCallback, this);
        
        erp42_ctrl_cmd_pub_ = nh_.advertise<erp_driver::erpCmdMsg>("/erp42_ctrl_cmd",1,true); 
        ego_speed_pub_ = nh_.advertise<std_msgs::Float32>("/ego_speed",1,true); 

        is_erp_state_ = false;
        is_gps_state_ = false;

        e_stop = 0;
        gear = 0;
        speed = 0;
        steer = 0;
        brake = 1;

        count_ = 0;
        prev_spd_state_ = 0.;
        current_mission_state_ = 0;
        gps_speed_ = .0;
        Get_Kmh_Speed_ = .0;
		os_.open("/home/kuuve/catkin_ws/src/speed_data/speed_data.csv", ios::app);
		os_<<setprecision(numeric_limits<double>::digits10+2);
        ROS_INFO("[PLATFORM_CONTROL] : INITIALIZED");
    }

    void inputSerial()
    {
        // set final_input_speed
        // double pid  = input_speed_;
        
        // double error = .0;

        // speed update error
        if(is_erp_state_ && is_gps_state_) {
            cur_speed_ = Get_Kmh_Speed_;
        }
        else if(!is_erp_state_ && is_gps_state_) {
            cur_speed_ = gps_speed_;
        }
        else if(is_erp_state_ && !is_gps_state_) {
            cur_speed_ = Get_Kmh_Speed_;
        }
        else {
            cur_speed_ = (gps_speed_ + Get_Kmh_Speed_) / 2.0;
        }
        

        // if(get_control_mode == 1)   
        //     pid = PID(abs(input_speed_), abs(cur_speed_), error);
        //     pid = input_speed_;

        if (input_speed_ == 0) // 정지 명령시 full brake
        {
            speed = 0;
            brake = cur_speed_ * 4.0 + 20; //100;//max value
        }
        else
        {
            if (input_speed_ > 0.0 && input_speed_ <= 21.8)
            { // 전진
                // pid값 반영 or break 피팅 여기서

                // if(input_speed_ > pid)
                //     input_speed_ = input_speed_;
                // if(input_speed_ < pid){
                // input_speed_ = pid;}
                // input_speed_ = pid;  
                // gear = 0;
                // speed = input_speed_ * 10;
                // brake = 1;
                // double temp_err = input_speed_ - cur_speed_;
                // if(-3.0 <= temp_err && temp_err <= -1.0){

                //     if(spd_state_ < 17.0)
                //     {
                //         gear = 0;
                //         speed = 0;
                //         brake = 1;      
                //     }
                //     else
                //     { 
                //         gear = 0;
                //         speed = KMH_to_RPM(input_speed_);
                //         brake = 1;                    
                //     }
                // }
                // else if(temp_err < -3.0)
                // {  
                //     if(spd_state_ < 17.0)
                //     {
                //         gear = 0;
                //         speed = 0;
                //         brake = abs(error) * 3.0 + 1.0;
                //     }
                //     else
                //     { 
                //         gear = 0;
                //         speed = KMH_to_RPM(input_speed_);
                //         brake = 1;                    
                //     }
                // }
                
                gear = 0;
                speed = KMH_to_RPM(input_speed_);
                brake = 1;

                
                // if(get_control_mode == 1){
                //     os_<<count_<<","<<(double)input_speed_<<","<<(double)cur_speed_<<","<<(double)spd_state_ <<","<< (double)brake<<","<< (double)Get_Kmh_Speed_<<"\n";
                // }
            }
            else if (input_speed_ > -20.0 && input_speed_ < 0.0)
            { // 후진
                // input_speed_ = pid;
                gear = 2;
                speed = -KMH_to_RPM(input_speed_);
                brake = 1;
                cout << "errorBBBBBBBack" << endl;
            }
            else
            { // 예외처리 -> 필요없다고 판단되나 혹시나..
                gear = 1;
                speed = 0;
                brake = 1;
            }
            count_++;
        }

        if(brake > 100) brake = 100;

        if (speed > 1000) speed = 1000;
        else if (speed < -1000) speed = -1000;

        if (input_steer_ > 19.9999)
            steer = 19.9999 * 100;
        else if (input_steer_ < -19.9999)
            steer = -19.9999 * 100;
        else{
            steer = input_steer_ * 100;
        }
        if(input_speed_ == 0){
            steer = 0;
            brake = 99;
        }

        pub_to_erp();

    }

    void pub_to_erp(){
        erp_driver::erpCmdMsg ctrl_msg;
        ctrl_msg.e_stop = e_stop;
        ctrl_msg.gear = gear;
        ctrl_msg.speed = speed;
        ctrl_msg.steer = steer;
        ctrl_msg.brake = brake;

        erp42_ctrl_cmd_pub_.publish(ctrl_msg);

    }

    void setFromWF(double wf_speed, double wf_steer, int _current_mission_state)
    {   

        this->input_speed_ = wf_speed;
        this->input_steer_ = wf_steer;
        this->current_mission_state_ = _current_mission_state;
        spd_state_ = wf_speed;
    }


    double KMH_to_RPM(double InPut_speed_)
    {
        // RPM = input_speed * 45.7573761
        return (InPut_speed_ * GEAR_RATIO * 60.0) / (2.0*WHEEL_RADIOUS * M_PI * 3.6);
    } 
    
    
    // double PID(double target, double cur_speed, double& _error)
	// {
    //     if ((int)target == 0) return .0;
	// 	double result = .0;
    //     double error = target - cur_speed;
    //     _error = error;
    //     // if(prev_spd_state_ != spd_state_) sum_error_ = 0;
	// 	sum_error_ += (error * CONTROL_TIME);
    //     if(sum_error_ > 32) sum_error_ = 32;
    //     if(target >= 1.0 && cur_speed <= 1.0)
    //     {
    //         sum_error_ = .0;
    //         cout << "SSSSSSSSSSSSSSSSSSSSSSs >> " << cur_speed << endl;
    //     }
	// 	double diff_error = (error - prev_error_) / CONTROL_TIME;
	// 	prev_error_ = error;
	// 	if(error > .01)
	// 		result = KP * error + KI * sum_error_; //+ KD * diff_error;
    //     else if(error < -.01)
	// 		result = KP * error + KI * sum_error_; // + KD * diff_error;
	// 	else
	// 		result = target;
	// 	// if(result < 0) result = 0;
    //     prev_spd_state_ = spd_state_;

    //     return result;
    // }

    // void LidarLaneCallback(const new_lane::lidar_topic_msg::ConstPtr &msg)
	// {
	// 	is_acc = msg->ACC_flag;
    //     acc_index = msg->ACC_index_gap;
	// }


    void SpeedCallback(const std_msgs::Float32::ConstPtr &speed_msg)
	{
        double prev_gps_speed_ = gps_speed_;

		gps_speed_ = speed_msg->data;

        if(gps_speed_ == prev_spd_state_) {
            gps_speed_ = prev_gps_speed_;
            is_gps_state_ = false;
        }
        else {
            is_gps_state_ = true;
        }
        
	}


    void ErpCallback(const erp_driver::erpStatusMsg::ConstPtr &msg){
        get_control_mode = msg->control_mode;
        get_e_stop = msg->e_stop;
        get_gear = msg->gear;
        get_speed = msg->speed;
        get_steer = msg->steer;
        get_brake = msg->brake;
        get_encoder = msg->encoder;
        get_alive = msg->alive;

        double prev_erp_speed_ = Get_Kmh_Speed_;
        Get_Kmh_Speed_ = (get_speed / 60.0) * (2.0 * WHEEL_RADIOUS * M_PI * 3.6);
        
        erp_speed_msgs_.data = Get_Kmh_Speed_ ;
        ego_speed_pub_.publish(erp_speed_msgs_);

        if(prev_erp_speed_ == Get_Kmh_Speed_) {
            Get_Kmh_Speed_ = prev_erp_speed_;
            is_erp_state_ = false;
        }
        else {
            is_erp_state_ = true;
        }
            
    }

};
#endif



//---------------------------위 주석은 0905 최신 플컨---------------------------------------------------------






//-----------------------------아래는 0827 백업 플컨 (박찬희가 변경함)-------------------------------------------------

// #ifndef PLATFORMCONTROL_H
// #define PLATFORMCONTROL_H


// #include <ros/ros.h>
// #include <fstream>
// #include <string>
// #include <math.h>
// #include <erp_driver/erpCmdMsg.h>
// #include <erp_driver/erpStatusMsg.h>
// /*
// Cycle Time : 20 msec
// Baud:115200, parity : None, Stop : 1
// PC -> ERP42 (14 Byte)                               ERP42 -> PC (18 Byte)
// 바이트 이름     /            값                       바이트 이름      /           값
// 1   S               0x53                        1   S                   0x53
// 2   T               0x54                        2   T                   0x54
// 3   X               0x58                        3   X                   0x58
// 4   A or M          0x00 or 0x01                4   A or M              0x00 or 0x01
// 5   E-stop          0x00 or 0x01                5   E-stop              0x00 or 0x01
// 6   GEAR            0~2                         6   GEAR                0~2
// 7   SPEED0┐         0~200                       7   SPEED0┐             0~200
// 8   SPEED1┘                                     8   SPEED1┘
// 9   STEER0┐         -2000~2000                  9   STEER0┐             -2000~2000
// 10  STEER1┘                                     10  STEER1┘
// 11  BRAKE           1~150                       11  BRAKE               1~150
// 12  ALIVE           0~255                       12  ENC0┐
// 13  ETX0            0x0D                        13  ENC1│               -2^31~2^31
// 14  ETX1            0x0A                        14  ENC2│
//                                                 15  ENC3┘
//                                                 16  ALIVE               0~255
//                                                 17  ETX0                0x0D
//                                                 18  ETX1                0x0A
// 시리얼에 들어가 있는 값들
// AorM -> Auto or Manual -------- 0x00 : manual mode ,      0x01 : auto mode
// ESTOP -> Emergency STOP -------- 0x00 : E-STOP Off,      0x01 : E-STOP On
// GEAR -> 0x00 : forward drive,      0x01 : neutral,      0x02 : backward drive
// SPEED -> actual speed (KPH) * 10
// STEER -> actual steering dgree (dgree) * 71, 오차율 : 4%  (negative is left steer)
// BRAKE -> 1 : no braking,      150 : full braking
// ENC -> encoder counting
// ALIVE -> increasing each one step
// <Returned Serial data 변환>
// speed = (speed_return0 + speed_return1 * 256) / 10) km/h
// steer = ((int)(serial_input[8]) + (int)(serial_input[9]) * 256) / 71) degree
// brake = 1~150
// */
// using namespace std;
// class PlatformConnector
// {
// private:

//     ros::NodeHandle nh_;
//     ros::Subscriber course_sub_,erp42_status_sub_;
//     ros::Publisher erp42_ctrl_cmd_pub_;

// 	ofstream os_;

//     //SaeJae Data Logging

//     double input_speed_;
//     double input_steer_;
//     double cur_speed_;
//     double spd_state_;

//     int   get_control_mode;
//     bool  get_e_stop;
//     int get_gear;
//     int get_speed;
//     int get_steer;
//     int get_brake;
//     int get_encoder;
//     int get_alive;

//     bool e_stop;
//     int gear;
//     int speed;
//     int steer;
//     int brake;

//     int count_;

//     // PID


//     //new PID
//     const double KP = 1.6; //1.796; //1.6; 0821     //1.4345; //1.6794; |0818 유효값  //1.7868; //1.833; 
// 	const double KI = 0.8; //1.3321; //0.8; 0821  //0.51897; //1.0125;|0818 유효값  //1.104;  //1.1025;
// 	const double KD = .0;
//     const double BRAKE_THRSHHOLD = .0;
// 	const double CONTROL_TIME = 0.1;

//     double sum_error_, prev_error_;

// public:
//     PlatformConnector()
//     {
//         initSetup();
//     }
//     ~PlatformConnector()
//     {
// 		os_.close();
//     }
//     void initSetup()
//     {
//         course_sub_ = nh_.subscribe("course", 10, &PlatformConnector::CourseCallback, this);
//        	erp42_status_sub_ = nh_.subscribe("/erp42_status", 10,&PlatformConnector::ErpCallback,this);
        
        
//         erp42_ctrl_cmd_pub_ = nh_.advertise<erp_driver::erpCmdMsg>("/erp42_ctrl_cmd",1,true); 

//         e_stop = 0;
//         gear = 0;
//         speed = 0;
//         steer = 0;
//         brake = 1;


//         count_ = 0;


// 		os_.open("/home/kuuve/speed_data.csv", ios::app);
// 		os_<<setprecision(numeric_limits<double>::digits10+2);
//         ROS_INFO("[PLATFORM_CONTROL] : INITIALIZED");
//     }

//     void inputSerial()
//     {

//         // set final_input_speed
//         double pid  = input_speed_;
//         double error = .0;
//         if(get_control_mode == 1)
//             pid = PID(abs(input_speed_), abs(cur_speed_), error);

//         if (input_speed_ == 0) // 정지 명령시 full brake
//         {
//             speed = 0;
//             brake = cur_speed_ * 6.0 + 33; //33;//max value
//         }
//         else
//         {
//             if (input_speed_ > 0.0 && input_speed_ < 20.0)
//             { // 전진
//                 // pid값 반영 or break 피팅 여기서

//                 // if(input_speed_ > pid)
//                 //     input_speed_ = input_speed_;
//                 // if(input_speed_ < pid){
//                 // input_speed_ = pid;}
//                 input_speed_ = pid;
//                 if(get_control_mode == 1){
//                     os_<<count_<<","<<(double)input_speed_<<","<<(double)cur_speed_<<","<<(double)spd_state_<<","<<"\n";
//                 }
//                 // gear = 0;
//                 // speed = input_speed_ * 10;
//                 // brake = 1;

//                 if(cur_speed_-3 <= input_speed_){
//                     gear = 0;
//                     speed = input_speed_ * 10;
//                     brake = 1;
//                 }
//                 else{   // break 값 여기서 넣음.
//                     gear = 0;
//                     speed = 0;
//                     brake = abs(error) * 2.5;
//                     if(brake > 200) brake = 200;
//                 }
//                 count_++;
//             }
//             else if (input_speed_ > -20.0 && input_speed_ < 0.0)
//             { // 후진
//                 // input_speed_ = pid;
//                 gear = 2;
//                 speed = -input_speed_ * 10;
//                 brake = 1;
//             }
//             else
//             { // 예외처리 -> 필요없다고 판단되나 혹시나..
//                 gear = 1;
//                 speed = 0;
//                 brake = 1;
//             }
//         }
//         if (speed > 200){
//             speed = 200;
//         }

//         if (input_steer_ > 28.0)
//             steer = 28.0 * 71;
//         else if (input_steer_ < -28.0)
//             steer = -28.0 * 71;
//         else{
//             steer = input_steer_ * 71;
//         }

//         pub_to_erp();

//     }

//     void pub_to_erp(){
//         erp_driver::erpCmdMsg ctrl_msg;
//         ctrl_msg.e_stop = e_stop;
//         ctrl_msg.gear = gear;
//         ctrl_msg.speed = speed;
//         ctrl_msg.steer = steer;
//         ctrl_msg.brake = brake;

//         erp42_ctrl_cmd_pub_.publish(ctrl_msg);

//     }

//     void setFromWF(double wf_speed, double wf_steer)
//     {
//         this->input_speed_ = wf_speed * 3.6;
//         this->input_steer_ = wf_steer;
//         spd_state_ = wf_speed * 3.6;
//     }
    
    
//     double PID(double target, double cur_speed, double& _error)
// 	{
//         if ((int)target == 0) return .0;
// 		double result = .0;
//         double error = target - cur_speed;
//         _error = error;
// 		sum_error_ += (error * CONTROL_TIME);
//         if(target >= 1.0 && cur_speed <= 1.0 && get_brake < 40)
//         {
//             sum_error_ = .0;
//             cout << "SSSSSSSSSSSSSSSSSSSSSSs >> " << cur_speed << endl;
//         }
// 		double diff_error = (error - prev_error_) / CONTROL_TIME;
// 		prev_error_ = error;
// 		if(error > .01)
// 			result = KP * error + KI * sum_error_; //+ KD * diff_error;
//         else if(error < -.01)
// 			result = KP * error + KI * sum_error_; // + KD * diff_error;
// 		else
// 			result = target;
// 		// if(result < 0) result = 0;
// 		// result = result > 19.99 ? 19.99 : result < -19.99 ? -19.99 : result;
//         return result;
//     }

//     void CourseCallback(const ackermann_msgs::AckermannDriveStamped::ConstPtr &course_msg)
//     {
//         cur_speed_ = course_msg->drive.speed; // kph
//     }

//     void ErpCallback(const erp_driver::erpStatusMsg::ConstPtr &msg){
//         get_control_mode = msg->control_mode;
//         get_e_stop = msg->e_stop;
//         get_gear = msg->gear;
//         get_speed = msg->speed;
//         get_steer = msg->steer;
//         get_brake = msg->brake;
//         get_encoder = msg->encoder;
//         get_alive = msg->alive;
//     }
// };
// #endif
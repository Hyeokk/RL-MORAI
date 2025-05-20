#ifndef LINEARKF_HJ_HPP
#define LINEARKF_HJ_HPP

#include <ros/ros.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <tf/tf.h>
#include <sensor_msgs/Imu.h>
#include <ackermann_msgs/AckermannDriveStamped.h>
#include <Eigen/Dense>


using namespace Eigen;
using namespace std;
using colVector = Matrix<double,2,1>;
using squreVector = Matrix<double,2,2>;
using rowVector = Matrix<double,1,2>;

/*
2023/07/03 Edited by HyeongJun_KUUVe
Linear Kalman Filter 

과정
1. gps, imu로부터 callback 받기
2. callback 받은 데이터에 칼만필터 적용
3. Before Kalman 과 After Kalman 값 csv 파일에 기록

-> 수정
callback과 파일스트림 다른 header로 빼기
->
*/

class LinearKF{
private:
/*
A_, H_, Q_, R_
P_ : 오차공분산
*/
    double p,p_gps;

    //imu KF
    squreVector A_, Q_;
    colVector X_, K_;
    squreVector P_, P_inv;
    rowVector H_;
    double Z_, R_, R_inv_;

    //gps KF --> 보류
    squreVector A_gps_, Q_gps_;
    colVector X_gps_, K_gps_;
    squreVector P_gps_, P_inv_gps_;
    rowVector H_gps_;   
    double Z_gps_, R_gps_, R_inv_gps_;

    //gps
    double gps_yaw_;
    double T_gps_;

    //imu
    double roll_, pitch_, yaw_;
    double T_; // 코드 hz

public:
    LinearKF();
    ~LinearKF();
    void correction(bool imu);
    void prediction(bool imu);

    void set_Yaw(double yaw){ yaw_ = yaw; };
    void set_GpsYaw(double yaw){ gps_yaw_ = yaw;}
    void set_T(double t){ T_ = t;};
    void set_T_gps(double t){ T_gps_ = t; };
    
    void logInCSV(ofstream& beforeKF,ofstream& afterKF, double time, bool imu);
    void loadCSV(ifstream& csv,const string& path, vector<double>& t_sum, vector<double>& t, vector<double>& angle);
};

//생성자
LinearKF::LinearKF(){

    p = 0.125; // 0.125는 뭐지?
    p_gps = 0.125;

    A_ << 0, 0,
          0, 0;
    Q_ << 0, 0,
          0, 0;
    A_gps_ << 0, 0,
              0, 0;
    Q_gps_ << 0, 0,
              0, 0;

    X_ << 0,
          0;
    P_ << p, 0,
          0, p;

    X_gps_ << 0,
              0;
    P_gps_ << p_gps, 0,
              0, p_gps;   

    R_ = 0.7;
    R_inv_ = 1/0.7;
    R_gps_ = 0.7;
    R_inv_gps_ = 1/0.7;

    H_ << 1, 0;
    H_gps_ << 1,0;
}

LinearKF::~LinearKF(){}

void::LinearKF::correction(bool imu)
{
    if(imu)
    {
        // cout << "imu KF " << endl;
        Z_ = yaw_;    
        double temp_inv = (H_*P_*H_.transpose() + R_);  
        K_ = P_ * H_.transpose() * 1.0/temp_inv;
        X_ = X_ + K_*(Z_-H_*X_);
        P_ = P_- K_*H_*P_;
    }
    else
    {
        // cout << "gps KF " << endl;
        Z_gps_ = gps_yaw_;
        double temp_inv_gps = (H_gps_*P_gps_*H_gps_.transpose() + R_gps_);
        K_gps_ = P_gps_ * H_gps_.transpose() * 1.0/temp_inv_gps;
        X_gps_ = X_gps_ + K_gps_*(Z_gps_-H_gps_*X_gps_);
        P_gps_ = P_gps_ - K_gps_*H_gps_*P_gps_;
    }
}

void::LinearKF::prediction(bool imu)
{
    if(imu){
        A_ << 1, T_,
              0, 1;
        Q_ << T_, 0,
              0, T_;

        X_ = A_*X_; //잡음은 없다고 가정
        P_ = A_*P_*A_.transpose() + Q_;  
    }
    else{
        A_gps_ << 1, T_gps_,
                  0, 1;
        Q_gps_ << T_gps_, 0,
                       0, T_gps_;

        X_gps_ = A_gps_*X_gps_;
        P_gps_ = A_gps_*P_gps_*A_gps_.transpose() + Q_gps_;
    }
}

void LinearKF::logInCSV(ofstream& beforeKF, ofstream& afterKF ,double time, bool imu)
{
    if(imu)
    {
        beforeKF << time << "," << yaw_*180.0/M_PI << endl;
        afterKF << time << "," << X_(0)*180.0/M_PI << endl;
    }
    else
    {
        beforeKF << time << "," << gps_yaw_ << endl;
        afterKF << time << "," << X_gps_(0) << endl;
    }
}


void LinearKF::loadCSV(ifstream& csv,const string& path, vector<double>& t_sum, vector<double>& t, vector<double>& angle)
{

    cout << "Load CSV STARTED " << endl;

    csv.open(path);
    if(!csv.is_open())
    {
       cerr << path << " OPEN ERROR" << endl;
    }

    double time = 0.0, past_time = 0.0;
    double yaw = 0.0;
    
    string str_buf;

    while(csv){
        // 한 줄 읽기
        getline(csv,str_buf);
        if(str_buf == "") break;

        int pos1 = str_buf.find(",");
        time = stod(str_buf.substr(0,pos1));
        t_sum.emplace_back(time);
        t.emplace_back(time - past_time);
        past_time = time;

        str_buf = str_buf.substr(pos1+1);
        int pos2 = str_buf.find(",");
        yaw = stod(str_buf.substr(0,pos2));
        angle.emplace_back(yaw);
    }


    csv.close();
    cout << "Load CSV END " << endl;
// 한 줄 읽기
// 값 저장 해두고 다음 루프로
// 다시 읽은 데이터 - 이전 저장값 = T
}

#endif




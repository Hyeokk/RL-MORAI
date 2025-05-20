#ifndef SPLINE_CSV_MAKER
#define SPLINE_CSV_MAKER

#include <ros/ros.h>
#include "./cubic_spline_planner.hpp"
#include <fstream>

//<<====================================>>//

#define VERSION 1 //예선 0 본선 1 시흥 2

//<<====================================>>//

#if VERSION == 1
#define MAKEFILEPATH "/home/kuuve/catkin_ws/src/data_본선/splined.csv"
#define EDITFILE "/home/kuuve/catkin_ws/src/data_본선/"
#else
#define MAKEFILEPATH "/home/kuuve/catkin_ws/src/data_시흥/splined.csv"
#define EDITFILE "/home/kuuve/catkin_ws/src/data_시흥/"
#endif


using namespace std;
using vecD = vector<double>;

class splinedCSV{
    private:
	    ros::NodeHandle nh_;
        int state_index;
        int state_0, state_1, state_2, state_3, state_4, state_5, state_6, state_7, state_8, state_9, state_10, state_11, state_12, state_13, state_14, state_15, state_16, state_17;
        string filepath;
        ifstream is_;
        ofstream os_;
    public:
        splinedCSV()
        {
            state_index = 0; //default

            nh_.getParam("/spline_csv_maker_node/state_0",state_0);
		    // nh_.getParam("/spline_csv_maker_node/state_1",state_1);
		    // nh_.getParam("/spline_csv_maker_node/state_2",state_2);
            // nh_.getParam("/spline_csv_maker_node/state_3",state_3);
		    // nh_.getParam("/spline_csv_maker_node/state_4",state_4);
            // nh_.getParam("/spline_csv_maker_node/state_5",state_5);
            // nh_.getParam("/spline_csv_maker_node/state_6",state_6);
            // nh_.getParam("/spline_csv_maker_node/state_7",state_7);
            // nh_.getParam("/spline_csv_maker_node/state_8",state_8);
            // nh_.getParam("/spline_csv_maker_node/state_9",state_9);
            // nh_.getParam("/spline_csv_maker_node/state_10",state_10);
            // nh_.getParam("/spline_csv_maker_node/state_11",state_11);
            // nh_.getParam("/spline_csv_maker_node/state_12",state_12);
            // nh_.getParam("/spline_csv_maker_node/state_13",state_13);
            // nh_.getParam("/spline_csv_maker_node/state_14",state_14);
            // nh_.getParam("/spline_csv_maker_node/state_15",state_15);
            // nh_.getParam("/spline_csv_maker_node/state_16",state_16);
            
        }

        ~splinedCSV()
        {
            is_.close();
		    os_.close();
        }

        void makeCSV(vecD& rx, vecD& ry, vecD& ryaw, vecD& rk) //새로운 파일 생성
        {
            ROS_INFO(" MAKE CSV\n");

            os_.open(MAKEFILEPATH,ios::app);
		    os_ << setprecision(numeric_limits<double>::digits10 + 2); // setprecision = 출력범위 지정
            for(int i=0;i<rx.size();i++)
            {
		    	// ROS_INFO("ADD NEW WAYPOINT XPOS = %ld, YPOS = %ld", rx[i], rx[i]);

                if(i == state_0 * 5) state_index = 0;
                // else if(i == state_1 * 5) state_index = 1;
                // else if(i == state_2 * 5) state_index = 2;
                // else if(i == state_3 * 5) state_index = 3;
                // else if(i == state_4 * 5) state_index = 4;
                // else if(i == state_5 * 5) state_index = 5;
                // else if(i == state_6 * 5) state_index = 6;
                // else if(i == state_7 * 5) state_index = 7;
                // else if(i == state_8 * 5) state_index = 8;
                // else if(i == state_9 * 5) state_index = 9;
                // else if(i == state_10 * 5) state_index = 10;
                // else if(i == state_11 * 5) state_index = 11;
                // else if(i == state_12 * 5) state_index = 12;
                // else if(i == state_13 * 5) state_index = 13;
                // else if(i == state_14 * 5) state_index = 14;
                // else if(i == state_15 * 5) state_index = 15;
                // else if(i == state_16 * 5) state_index = 16;

                os_ << i << "," << (double)rx[i] << "," << (double)ry[i] << "," <<  state_index << "," << (double)ryaw[i] << "," << (double)rk[i] << "," << "\n";
            }            

        }

};

#endif
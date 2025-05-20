/*
인덱스, 경로x, 경로y만 있는 CSV를
인덱스, 경로x, 경로y, 경로yaw값, 경로 곡률값, 미션스테이트 를 가지는 CSV로

1. 인덱스, 경로x, 경로y 로드하기
2. cubic spline 적용하기
3. splined된 경로에 mission state 부여하기
    param으로 미션스테이트 index 기준점 설정
    반복문으로 index까지 번호 부여
*/
#include <ros/ros.h>
#include "../include/load_global_path.hpp"
#include "../include/spline_csv_maker.hpp"

int main(int argc, char **argv)
{
	ros::init(argc, argv, "spline_csv_maker_node");
    splinedCSV spline_path;

    vecD W_X;
    vecD W_Y;
    vecD rx,ry,ryaw,rk;
    double ds = 0.1;

    Load_path::load_path(W_X,W_Y); //경로 업로드
	calc_spline_course(W_X, W_Y, rx, ry, ryaw, rk, ds); //spline화

    spline_path.makeCSV(rx,ry,ryaw,rk);

}
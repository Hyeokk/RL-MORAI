#ifndef LOAD_GLOBAL_PATH
#define LOAD_GLOBAL_PATH

#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>
#include <fstream>
#include <cmath>
#include <eigen3/Eigen/Dense>

//<<====================================>>//

#define VERSION 1 //예선 0 본선 1 시흥 2

//<<====================================>>//

#if VERSION == 1
#define PATH "/home/kuuve/catkin_ws/src/data_본선/"
#else
#define PATH "/home/kuuve/catkin_ws/src/data_시흥/"
#endif
#define SCALE_FACTER 0.5

using namespace std;
using namespace Eigen;

using vecD = vector<double>;

namespace Load_path
{
    void getNewWaypoints(const string& _path, vecD& _p_x, vecD& _p_y)
    {
        ifstream is;
        string str_buf;
        int pos;
                    
        is.open(_path);
        
        while(getline(is, str_buf)) 
        {
            pos = str_buf.find(",");
            str_buf.substr(0, pos);

            str_buf = str_buf.substr(++pos);
            pos = str_buf.find(","); 
            _p_x.emplace_back(stod(str_buf.substr(0, pos)));

            str_buf = str_buf.substr(++pos);
            pos = str_buf.find(",");
            _p_y.emplace_back(stod(str_buf.substr(0, pos)));
        }
        is.close();
    }

    void load_path(vecD& _p_x, vecD& _p_y)
    {
        DIR* dirp = opendir(PATH);
        
        if (dirp == nullptr)
        {
            perror("UNABLE TO OPEN FOLDER");
            return;
        }

        struct dirent* dp; 
        while((dp = readdir(dirp)) != nullptr)
        {
            string address(PATH);
            string filename (dp->d_name);

            address.append(filename);

            if (filename.size() > 2)
                getNewWaypoints(address, _p_x, _p_y);
        }
        closedir(dirp);
    }
}
    
#endif
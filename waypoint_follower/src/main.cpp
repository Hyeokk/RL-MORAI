#include <ros/ros.h>
#include <waypointFollower.h>


int main(int argc, char **argv) {
	ros::init(argc, argv, "waypoint_follower");
	WaypointFollower wf;
	ros::Rate loop_rate(10);
	while(ros::ok()) 
	{
		ros::spinOnce();
		wf.process();
		loop_rate.sleep();
	}
	return 0;
}

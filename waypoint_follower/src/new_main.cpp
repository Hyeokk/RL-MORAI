#include <waypointFollower.h>
#include <new_platform_control.h>

/*
1. wf.Process worked
2. pc get speed & steer frm wf
3. pc worked
*/
int main(int argc, char **argv)
{
	ros::init(argc, argv, "waypoint_follower");
	WaypointFollower wf;
	PlatformConnector pc;
	ros::Rate loop_rate(20); // hz 100 / 40으로 올리기 - 승규
	while (ros::ok())
	{
		ros::spinOnce();
		wf.process();// Process Operated
		// ros::param::get("/uturn_flag",wf.uturn_flag_);
		// cout <<"uturn flag " << wf.uturn_flag_ << endl;
		// if(!wf.uturn_flag_){

		pc.setFromWF(wf.getSpeed(), wf.getSteer(), wf.get_Mission_State()); // set pc_spd
		pc.inputSerial();
		// }
		loop_rate.sleep();
	}
	return 0;
}

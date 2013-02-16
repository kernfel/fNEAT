#define NUM_SENSORS 5
#define NUM_MOTORS 3

/** Motor output mapping

0 - Forward motor
1 - Left turn
2 - Right turn

*/

typedef struct Robot {
	double x, y;
	double heading;
} Robot;

struct Robot_Params {
	double radius;
	
	double	motor_sensitivity,
		motor_threshold,
		turn_sensitivity,
		turn_threshold;
	
	int	num_dist_sensors;
	double	dist_sensor_length,
		*dist_sensor_pos;
};

typedef struct Room {
	double x, y;
} Room;

void move_robot_in_room( Robot *bot, Room *room, struct Robot_Params *params, double *controller_outputs );
void get_sensor_readings_in_room( Robot *bot, Room *room, struct Robot_Params *params, double *sensor_inputs );


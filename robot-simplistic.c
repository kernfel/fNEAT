#include <math.h>

#include "util.h"
#include "robot-simplistic.h"

void move_robot_in_room( Robot *bot, Room *room, struct Robot_Params *params, double *controller_outputs ) {
	// Adjust heading
	double turn_output = controller_outputs[1] - controller_outputs[2];
	if ( fabs(turn_output) > params->turn_threshold ) {
		if ( turn_output > 0 )
			turn_output = turn_output - params->turn_threshold;
		else
			turn_output = turn_output + params->turn_threshold;
		bot->heading = fmod( bot->heading + turn_output*params->turn_sensitivity, 2*M_PI );
	}

	// Calculate forward move
	double motor_output = controller_outputs[0];
	if ( fabs(motor_output) > params->motor_threshold ) {
		if ( motor_output > 0 )
			motor_output = motor_output - params->motor_threshold;
		else
			motor_output = motor_output + params->motor_threshold;
		double m_x = cos(bot->heading) * motor_output * params->motor_sensitivity;
		double m_y = sin(bot->heading) * motor_output * params->motor_sensitivity;

		// Detect depth of collision, if any
		double new_x = bot->x + m_x, new_y = bot->y + m_y;
		double adjust_x=0, adjust_y=0;
		if ( new_x - params->radius < 0 ) {
			adjust_x = -(params->radius - new_x) / m_x;
		} else if ( new_x + params->radius > room->x ) {
			adjust_x = (new_x + params->radius - room->x) / m_x;
		}
		if ( new_y - params->radius < 0 ) {
			adjust_y = -(params->radius - new_y) / m_y;
		} else if ( new_y + params->radius > room->y ) {
			adjust_y = (new_y + params->radius - room->y) / m_y;
		}

		// Apply the adjusted movement vector
		bot->x += (1-max(adjust_x, adjust_y)) * m_x;
		bot->y += (1-max(adjust_x, adjust_y)) * m_y;
	}
}

void get_sensor_readings_in_room( Robot *bot, Room *room, struct Robot_Params *params, double *sensor_inputs ) {
	// Calculate sensor data
	int i;
	double s_length = params->dist_sensor_length + params->radius;
	for ( i=0; i<params->num_dist_sensors; i++ ) {
		double s_x = cos(bot->heading + params->dist_sensor_pos[i]) * s_length;
		double s_y = sin(bot->heading + params->dist_sensor_pos[i]) * s_length;
		double depth_x=0, depth_y=0;
		if ( bot->x + s_x < 0 )
			depth_x = -(bot->x + s_x) / s_x;
		else if ( bot->x + s_x > room->x )
			depth_x = (bot->x + s_x - room->x) / s_x;
		if ( bot->y + s_y < 0 )
			depth_y = -(bot->y + s_y) / s_y;
		else if ( bot->y + s_y > room->y )
			depth_y = (bot->y + s_y - room->y) / s_y;
		sensor_inputs[i] = max(depth_x, depth_y) * s_length / params->dist_sensor_length;
	}
}


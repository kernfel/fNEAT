#include <math.h>

#include "util.h"
#include "robot-simplistic.h"

void adjust_heading( Robot *bot, struct Robot_Params *params, double *controller_outputs ) {
	double turn_output = controller_outputs[1] - controller_outputs[2];
	if ( fabs(turn_output) > params->turn_threshold ) {
		if ( turn_output > 0 )
			turn_output = turn_output - params->turn_threshold;
		else
			turn_output = turn_output + params->turn_threshold;
		bot->heading = fmod( bot->heading + turn_output*params->turn_sensitivity, 2*M_PI );
	}
}

void move_robot_in_room( Robot *bot, Room *room, struct Robot_Params *params, double *controller_outputs ) {
	// Adjust heading
	adjust_heading( bot, params, controller_outputs );

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
		double reading = max(depth_x, depth_y) * s_length / params->dist_sensor_length;
		sensor_inputs[i] = params->invert_dist_sensors ? 1-reading : reading;
	}
}

void move_robot_in_tilemaze( Robot *bot, TileMaze *maze, struct Robot_Params *params, double *controller_outputs ) {
	adjust_heading( bot, params, controller_outputs );
	
	double motor_output = controller_outputs[0];
	if ( fabs(motor_output) > params->motor_threshold ) {
		if ( motor_output > 0 )
			motor_output = motor_output - params->motor_threshold;
		else
			motor_output = motor_output + params->motor_threshold;
		double m_x = cos(bot->heading) * motor_output * params->motor_sensitivity;
		double m_y = sin(bot->heading) * motor_output * params->motor_sensitivity;
		
		// Intercepts of the likeliest points of collision
		double forward_x = m_x>0 ? m_x + params->radius : m_x - params->radius;
		double forward_y = m_y>0 ? m_y + params->radius : m_y - params->radius;
		int curtile_x = (int)(bot->x / maze->tile_width);
		int curtile_y = (int)(bot->y / maze->tile_width);
		int cross_x = (int)((bot->x + forward_x) / maze->tile_width) - curtile_x;
		int cross_y = (int)((bot->y + forward_y) / maze->tile_width) - curtile_y;
		double depth_x = (bot->x + forward_x - (cross_x>0 ? curtile_x+1 : curtile_x)*maze->tile_width) / forward_x;
		double depth_y = (bot->y + forward_y - (cross_y>0 ? curtile_y+1 : curtile_y)*maze->tile_width) / forward_y;
		
		double adjust=0;
		
		// Collision only with an x edge
		if ( cross_x && ! cross_y
		 && (  curtile_x+cross_x == maze->x || curtile_x+cross_x == -1
		    || ! maze->tiles[curtile_x+cross_x + maze->x*curtile_y]
		    )
		) {
			adjust = depth_x;
		
		// Collision only with a y edge
		} else if ( cross_y && ! cross_x
		 && (  curtile_y+cross_y == maze->y || curtile_y+cross_y == -1
		    || ! maze->tiles[curtile_x + maze->x*(curtile_y+cross_y)]
		    )
		) {
			adjust = depth_y;
		
		// Collision with both x and y edges, necessitating checks against adjacent and diagonal tiles
		} else if ( cross_x && cross_y ) {
			// Separating axis perpendicular to the bot's path, as a unit vector:
			double sep_axis_x = m_y / sqrt(m_x*m_x + m_y*m_y);
			double sep_axis_y = -m_x / sqrt(m_x*m_x + m_y*m_y);
			// Half size of a tile when projected onto the separating axis:
			double proj_tile_halfwidth = (fabs(sep_axis_x) + fabs(sep_axis_y)) * maze->tile_width/2;
			// Bot's center projected onto the separating axis
			double proj_bot = sep_axis_x*(bot->x+m_x) + sep_axis_y*(bot->y+m_y);
			
			// Check against solid x-adjacent tile
			if ( curtile_x+cross_x == maze->x || curtile_x+cross_x == -1 ) {
				adjust = depth_x;
			} else if ( ! maze->tiles[curtile_x+cross_x + maze->x*curtile_y] ) {
				// Do tile and bot share space on the separating axis?
				// Note, this may not be the case even with cross_x appropriately set, as the border-crossing may happen only
				// at the edge between the y-adjacent and diagonally adjacent tiles.
				double proj_tile_center = maze->tile_width * (sep_axis_x*(curtile_x+cross_x+0.5) + sep_axis_y*(curtile_y+0.5));
				if ( fabs(proj_bot - proj_tile_center) < params->radius + proj_tile_halfwidth ) {
					adjust = depth_x;
				}
			}
			
			// Check against solid y-adjacent tile
			if ( curtile_y+cross_y == maze->y || curtile_y+cross_y == -1 ) {
				adjust = max(adjust, depth_y);
			} else if ( ! maze->tiles[curtile_x + maze->x*(curtile_y+cross_y)] ) {
				// Do tile and bot share space on the separating axis?
				double proj_tile_center = maze->tile_width * (sep_axis_x*(curtile_x+0.5) + sep_axis_y*(curtile_y+cross_y+0.5));
				if ( fabs(proj_bot - proj_tile_center) < params->radius + proj_tile_halfwidth ) {
					adjust = max(adjust, depth_y);
				}
			}

			// If all else fails, check against solid diagonally adjacent tile
			if ( ! adjust && ! maze->tiles[curtile_x+cross_x + maze->x*(curtile_y+cross_y)] ) {
				// Project corner onto separating axis
				double corner_x = maze->tile_width * (curtile_x + (cross_x>0?1:0)), corner_y = maze->tile_width * (curtile_y + (cross_y>0?1:0));
				double proj_corner = sep_axis_x*corner_x + sep_axis_y*corner_y;
				
				// Colliding with corner:
				if ( fabs(proj_bot - proj_corner) < params->radius ) {
					double bc_x = corner_x - bot->x, bc_y = corner_y - bot->y;
					double norm_m = sqrt(m_x*m_x + m_y*m_y);
					double bf = (bc_x*m_x + bc_y*m_y) / norm_m; // Projection of bot-to-corner onto the bot's path (pt F)
					double fc = bc_x*sep_axis_x + bc_y*sep_axis_y; // Distance from the corner to pt F; fc < r.
					double fx = sqrt(params->radius*params->radius - fc*fc); // Distance from pt F to pushback point X
					if ( norm_m > bf - fx )
						adjust = 1 - (bf - fx)/norm_m;
				
				// Colliding with the edge that continues one of curtile's y-parallel edges
				} else if ( (proj_bot > proj_corner && m_x < 0 && m_y > 0) // Bot heading UL
				 || (proj_bot < proj_corner && m_x < 0 && m_y < 0)         // Bot heading DL
				 || (proj_bot > proj_corner && m_x > 0 && m_y < 0)         // Bot heading DR
				 || (proj_bot < proj_corner && m_x > 0 && m_y > 0) ) {     // Bot heading UR
					adjust = depth_x;
				
				// Colliding with the edge that continues one of curtile's x-parallel edges
				} else if ( (proj_bot > proj_corner && m_x > 0 && m_y > 0) // Bot heading UR
				 || (proj_bot < proj_corner && m_x < 0 && m_y > 0)         // Bot heading UL
				 || (proj_bot > proj_corner && m_x < 0 && m_y < 0)         // Bot heading DL
				 || (proj_bot < proj_corner && m_x > 0 && m_y < 0) ) {     // Bot heading DR
					adjust = depth_y;
				}
			}
		}
		
		// Apply the adjusted movement vector
		bot->x += (1 - adjust) * m_x;
		bot->y += (1 - adjust) * m_y;
	}
}

void get_sensor_readings_in_tilemaze( Robot *bot, TileMaze *maze, struct Robot_Params *params, double *sensor_inputs ) {
	int i;
	for ( i=0; i<params->num_dist_sensors; i++ ) {
		double s_cos = cos(bot->heading + params->dist_sensor_pos[i]);
		double s_sin = sin(bot->heading + params->dist_sensor_pos[i]);
		double s_x = s_cos * params->dist_sensor_length;
		double s_y = s_sin * params->dist_sensor_length;
		double root_x = bot->x + s_cos * params->radius;
		double root_y = bot->y + s_sin * params->radius;
		int root_tile_x = (int)(root_x / maze->tile_width);
		int root_tile_y = (int)(root_y / maze->tile_width);
		int cross_x = (int)((root_x + s_x) / maze->tile_width) - root_tile_x;
		int cross_y = (int)((root_y + s_y) / maze->tile_width) - root_tile_y;
		
		sensor_inputs[i] = params->invert_dist_sensors ? 1 : 0;

		if ( ! cross_x && ! cross_y )
			continue;

		double reading=0;
		
		// Penetration of the sensor stub into the space outside of the root tile
		double depth_x = (root_x + s_x - (cross_x>0 ? root_tile_x+1 : root_tile_x)*maze->tile_width) / s_x;
		double depth_y = (root_y + s_y - (cross_y>0 ? root_tile_y+1 : root_tile_y)*maze->tile_width) / s_y;
		
		// X crossing only, solid/invalid target tile
		if ( cross_x && ! cross_y
		 && (  root_tile_x+cross_x == maze->x || root_tile_x+cross_x == -1
		    || ! maze->tiles[root_tile_x+cross_x + maze->x*root_tile_y]
		    )
		) {
			reading = depth_x;

		// Y crossing only, solid/invalid target tile
		} else if ( cross_y && ! cross_x
		 && (  root_tile_y+cross_y == maze->y || root_tile_y+cross_y == -1
		    || ! maze->tiles[root_tile_x + maze->x*(root_tile_y+cross_y)]
		    )
		) {
			reading = depth_y;

		// Diagonal crossing
		} else if ( cross_x && cross_y ) {
			// Larger proportion of the x vector outside the tile => sensor stub reaches into diagonal tile via x-adjacent neighbour
			if ( depth_x > depth_y ) {
				// x-adjacent neighbour is solid/invalid target
				if ( root_tile_x+cross_x == -1 || root_tile_x+cross_x == maze->x || ! maze->tiles[root_tile_x+cross_x + maze->x*root_tile_y] )
					reading = depth_x;
				// Diagonally adjacent neighbour is solid/invalid target
				else if ( root_tile_y+cross_y == -1 || root_tile_y+cross_y == maze->y \
				 || ! maze->tiles[root_tile_x+cross_x + maze->x*(root_tile_y+cross_y)] )
					reading = depth_y;
			} else {
				if ( root_tile_y+cross_y == -1 || root_tile_y+cross_y == maze->y || ! maze->tiles[root_tile_x + maze->x*(root_tile_y+cross_y)] )
					reading = depth_y;
				else if ( root_tile_x+cross_x == -1 || root_tile_x+cross_x == maze->x \
				 || ! maze->tiles[root_tile_x+cross_x + maze->x*(root_tile_y+cross_y)] )
					reading = depth_x;
			}
		}

		sensor_inputs[i] = params->invert_dist_sensors ? 1-reading : reading;
	}
}


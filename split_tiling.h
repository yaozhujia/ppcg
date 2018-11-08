#ifndef _SPLIT_TILING_H
#define _SPLIT_TILING_H

#include <isl/schedule_node.h>

#include "ppcg.h"

/*
 * Internal data structure for split tiling.
 * 
 * "list" represents the list of statements.
 * 
 * "n_stmt" represents the number statements.
 * 
 * "n_phase" is the number of phases that may be introduced by split tiling.
 * 
 * "no_constraints" is a set of "n_stmt" flags representing whether the
 * corresponding statement should have constraints in the phases.
 * 
 * "stmt" is a sequence of "n_stmt" names of statements. In particular, this
 * is used to store the string of names.
 * 
 * "expr" is a set of expression of "n_stmt" statements. Each expression is
 * represented as the substraction of time dimesion expression from that of
 * the first dimension of space. In other words, Each expr is in the form of
 * 
 * 		S - c * T
 * 
 * "S" is the multi_union_pw_aff expression of the time dimension, "T" is the
 * multi_union_pw_aff expression of the first dimension of space. "c" is the
 * coefficient of the variable of time dimension after scheduling, and it can
 * be obtained from the scheduling result of the isl scheduler.
 * 
 * "constant" records the constant term after scheduling.
 * 
 *  "bound" is the bounding face introduced by split tiling. In particular,
 * it should be in the form of
 * 
 * 		f(t, s0) - size * floor(f(t, s0)/4)
 * 
 * "f" is a  is a set of linear function of "t", the variable of time
 * dimension, and "s0", the variable of the first dimension of space. It
 * should be determined by computing the slope of maximum dependence
 * distance. "size" represents the parallelogram tiling size along the first
 * dimension of space. It should be the same when "--no-isl-tile-scale-tile-loops"
 * and/or option. "--no-isl-tile-shift-point-loops" is set are set.
 * 
 * "time_dim_name" represents the name of time dimension. It is used to
 * construct the bounds.
 * "space_dim_name" represents the name of space dimension. It is used to
 * construct the bounds.
 */
struct split_tile_phases_data{

	isl_set_list *list;

	int n_stmt;
	int n_phase;

	int *no_constraints;
	
	char **stmt;
	char **expr;
	int *constant;
	char **bound;

	char **time_dim_name;
	char **space_dim_name;

};

static void *split_tile_phases_data_free(struct split_tile_phases_data *data);

__isl_give isl_schedule_node *split_tile(__isl_take isl_schedule_node *node,
		struct ppcg_scop *scop, __isl_take isl_multi_val *sizes);

__isl_give isl_multi_val *split_tile_read_tile_sizes(__isl_keep isl_schedule_node *node,
		struct ppcg_scop *scop, int *tile_len);

#endif

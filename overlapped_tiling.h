#ifndef OVERLAPPED_TILING_H
#define OVERLAPPED_TILING_H

#include <isl/schedule_node.h>

#include "ppcg.h"

__isl_give isl_schedule_node *overlapped_tile(__isl_take isl_schedule_node *node,
		struct ppcg_scop *scop, __isl_take isl_multi_val *sizes,
		int multi_dim);

#endif

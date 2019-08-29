/*
 * Copyright 2019 INRIA Paris-Rocquencourt
 * Copyright 2019 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Jie Zhao
 * Ecole Normale Superieure, 45 rue d'Ulm, 75005 Paris, France
 */


#include <stdlib.h>
#include <string.h>

#include <isl/id.h>
#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/constraint.h>
#include <isl/schedule_node.h>

#include "ppcg.h"
#include "ppcg_options.h"
#include "overlapped_tiling.h"
#include "util.h"

__isl_give isl_schedule_node *overlapped_tile(__isl_take isl_schedule_node *node,
		struct ppcg_scop *scop, __isl_take isl_multi_val *sizes) {
    int i, j, k, n;
    int shift;
    isl_ctx *ctx;
    
    // apply parallelogram tiling without shifting point loops
    ctx = isl_schedule_node_get_ctx(node);
    shift = isl_options_get_tile_shift_point_loops(ctx);
    isl_options_set_tile_shift_point_loops(ctx, 0);
	node = isl_schedule_node_band_tile(node, sizes);
    isl_schedule_node_dump(node);
    isl_options_set_tile_shift_point_loops(ctx, shift);

    // insert expansion node

    return node;
}
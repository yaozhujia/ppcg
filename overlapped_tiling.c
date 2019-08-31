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
#include "isl_schedule_node_private.h"
#include "overlapped_tiling.h"
#include "util.h"

isl_stat drop_space_dim_constraints(__isl_take isl_map *map, void *user) {
    isl_union_map *umap = user;
    
    map = isl_map_drop_constraints_involving_dims(map, isl_dim_out, 1, 1);
    umap = isl_union_map_add_map(umap, map);

    return isl_stat_ok;
}

isl_stat update_bounds(__isl_take isl_constraint *c, void *user) {
    isl_map *map = user;
    isl_aff *aff, *var;
    isl_val *coeff;
    isl_local_space *ls;
    isl_constraint *constraint;

    if(isl_constraint_involves_dims(c, isl_dim_set, 1, 1)) {
        isl_constraint_dump(c);
        coeff = isl_constraint_get_coefficient_val(c, isl_dim_set, 1);
        c = isl_constraint_set_coefficient_si(c, isl_dim_set, 1, 0);
        aff = isl_constraint_get_aff(c);
        isl_aff_dump(aff);

        ls = isl_constraint_get_local_space(c);
        var = isl_aff_var_on_domain(ls, isl_dim_set, 3);
        var = isl_aff_scale_val(var, coeff);
        isl_aff_dump(var);

        aff = isl_aff_add(aff, var);
        isl_aff_dump(aff);
        isl_aff_free(aff);
    }
        isl_constraint_free(c);

    return isl_stat_ok;
}

isl_stat copy_bounds_from_set(__isl_take isl_basic_set *bset, void *user) {
    isl_map *map = user;

    isl_basic_set_foreach_constraint(bset, &update_bounds, map);
    isl_basic_set_free(bset);

    return isl_stat_ok;
}

isl_stat add_space_dim_bounds(__isl_take isl_map *map, void *user) {
    isl_union_set* domain = user;
    isl_set *set;
    isl_union_set *uset;

    set = isl_map_domain(isl_map_copy(map));
    uset = isl_union_set_copy(domain);
    uset = isl_union_set_intersect(uset, isl_union_set_from_set(set));
    set = isl_set_from_union_set(uset);
    
    map = isl_map_intersect_domain(map, set);
    set = isl_map_wrap(isl_map_copy(map));
    isl_set_foreach_basic_set(set, &copy_bounds_from_set, map);
    isl_set_free(set);
    isl_map_free(map);

    return isl_stat_ok;
}

static __isl_give isl_union_map* update_expansion(__isl_take isl_union_map *expansion,
    __isl_take isl_union_set *domain, __isl_take isl_multi_union_pw_aff *mupa)
{
    isl_space *space;
    isl_union_map *umap;

    isl_multi_union_pw_aff_free(mupa);

    space = isl_union_map_get_space(expansion);
    umap = isl_union_map_empty(space);

    isl_union_map_foreach_map(expansion, &drop_space_dim_constraints, umap);
    isl_union_map_foreach_map(umap, &add_space_dim_bounds, domain);
    isl_union_set_free(domain);

    isl_union_map_dump(umap);
    isl_union_map_free(umap);

    return expansion;
}

__isl_give isl_schedule_node *overlapped_tile(__isl_take isl_schedule_node *node,
		struct ppcg_scop *scop, __isl_take isl_multi_val *sizes)
{
    int i, j, k, n;
    int shift;
    isl_ctx *ctx;
    isl_union_set *domain, *empty, *universe;
    isl_union_map *expansion;
    isl_schedule_node *child;
    isl_multi_union_pw_aff *mupa;
    isl_union_pw_multi_aff *contraction;
    
    // apply parallelogram tiling without shifting point loops
    ctx = isl_schedule_node_get_ctx(node);
    shift = isl_options_get_tile_shift_point_loops(ctx);
    isl_options_set_tile_shift_point_loops(ctx, 0);
	node = isl_schedule_node_band_tile(node, sizes);
    isl_options_set_tile_shift_point_loops(ctx, shift);

    // construct an empty contraction
    domain = isl_schedule_node_get_domain(node);
    empty = isl_union_set_empty(isl_union_set_get_space(domain));
    contraction = isl_union_pw_multi_aff_from_union_set(empty);

    // construct the expansion
    universe = isl_union_set_universe(isl_union_set_copy(domain));
    expansion = isl_union_set_identity(universe);
    mupa = isl_schedule_node_band_get_partial_schedule(node);
    expansion = update_expansion(expansion, domain, mupa);
    isl_union_map_dump(expansion);
    
    // insert expension node
    child = isl_schedule_node_get_child(node, 0);
    isl_schedule_node_free(node);
    child = isl_schedule_node_insert_expansion(child, contraction, expansion);
    node = isl_schedule_node_parent(child);

    return node;
}
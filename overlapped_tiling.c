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
    isl_union_map **umap = user;
    
    map = isl_map_drop_constraints_involving_dims(map, isl_dim_out, 1, 1);
    *umap = isl_union_map_add_map(*umap, map);

    return isl_stat_ok;
}

isl_stat update_bounds(__isl_take isl_constraint *c, void *user) {
    isl_set **set = user;
    int n_dim;
    isl_bool equality;
    isl_aff *aff, *var;
    isl_val *coeff;
    isl_local_space *ls;
    isl_constraint *constraint;

    if(isl_constraint_involves_dims(c, isl_dim_set, 1, 1)) {
        ls = isl_constraint_get_local_space(c);
        equality = isl_constraint_is_equality(c);
        coeff = isl_constraint_get_coefficient_val(c, isl_dim_set, 1);

        c = isl_constraint_set_coefficient_si(c, isl_dim_set, 1, 0);
        aff = isl_constraint_get_aff(c);
        isl_constraint_free(c);

        n_dim = isl_local_space_dim(ls, isl_dim_set) / 2;
        var = isl_aff_var_on_domain(ls, isl_dim_set, 1 + n_dim);
        var = isl_aff_scale_val(var, coeff);
        aff = isl_aff_add(aff, var);

        if(equality)
            c = isl_equality_from_aff(aff);
        else
            c = isl_inequality_from_aff(aff);
        *set = isl_set_add_constraint(*set, c);
    }
    else
        isl_constraint_free(c);

    return isl_stat_ok;
}

isl_stat copy_bounds_from_set(__isl_take isl_basic_set *bset, void *user) {
    isl_set **set = user;

    isl_basic_set_foreach_constraint(bset, &update_bounds, set);
    isl_basic_set_free(bset);

    return isl_stat_ok;
}

struct expansion_data {
    isl_union_map *expansion;
    isl_union_set *domain;
};

isl_stat add_space_dim_bounds(__isl_take isl_map *map, void *user) {
    struct expansion_data* data = user;
    isl_set *set, *mapset;
    isl_space *space;
    isl_union_set *uset;

    set = isl_map_domain(isl_map_copy(map));
    uset = isl_union_set_copy(data->domain);
    uset = isl_union_set_intersect(uset, isl_union_set_from_set(set));
    set = isl_set_from_union_set(uset);
    
    mapset = isl_map_wrap(isl_map_copy(map));
    map = isl_map_intersect_domain(map, set);
    set = isl_map_wrap(map);
    isl_set_foreach_basic_set(set, &copy_bounds_from_set, &mapset);
    isl_set_free(set);
    data->expansion = isl_union_map_add_map(data->expansion,
        isl_set_unwrap(mapset));

    return isl_stat_ok;
}

static __isl_give isl_union_map* update_expansion(__isl_take isl_union_map *expansion,
    __isl_take isl_union_set *domain, __isl_take isl_multi_union_pw_aff *mupa)
{
    isl_space *space;
    isl_union_map *umap, *result;

    isl_multi_union_pw_aff_free(mupa);

    space = isl_union_map_get_space(expansion);
    umap = isl_union_map_empty(space);
    result = isl_union_map_copy(umap);

    isl_union_map_foreach_map(expansion, &drop_space_dim_constraints, &umap);
    isl_union_map_free(expansion);

    struct expansion_data data = { result, domain };
    isl_union_map_foreach_map(umap, &add_space_dim_bounds, &data);
    isl_union_map_free(umap);
    isl_union_map_dump(data.expansion);
    isl_union_set_free(data.domain);
    

    return data.expansion;
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
    //printf("after updata_expansion\n");
    //isl_union_map_dump(expansion);
    
    // insert expension node
    child = isl_schedule_node_get_child(node, 0);
    isl_schedule_node_free(node);
    child = isl_schedule_node_insert_expansion(child, contraction, expansion);
    node = isl_schedule_node_parent(child);

    return node;
}
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

struct starting_point_data {
    isl_union_map *expansion;
    isl_union_map *result;
    isl_val *size;
};

/* Assign "aff" to *user and return -1, effectively extracting
 * the first (and presumably only) affine expression in the isl_pw_aff
 * on which this function is used.
 */
static isl_stat extract_single_piece(__isl_take isl_set *set,
	__isl_take isl_aff *aff, void *user)
{
	isl_aff **p = user;

	*p = aff;
	isl_set_free(set);

	return isl_stat_error;
}

static isl_stat starting_point_cond(__isl_take isl_pw_aff *pa, void *user) {
    int i, j, m, n;
    isl_ctx *ctx;
    isl_aff *aff, *copy, *var;
    isl_map *map;
    isl_set *set;
    isl_val *val;
    isl_space *space;
    isl_constraint *c;
    isl_local_space *ls;
    isl_map_list *mlist;
    isl_val_list *list;
    struct starting_point_data *data = user;

    isl_pw_aff_foreach_piece(pa, &extract_single_piece, &aff);
    isl_pw_aff_free(pa);
    
    n = isl_aff_dim(aff, isl_dim_in);
    ctx = isl_aff_get_ctx(aff);
    list = isl_val_list_alloc(ctx, n + 1);
    for (i = 0; i < n; i++) {
        val = isl_aff_get_coefficient_val(aff, isl_dim_in, i);
        list = isl_val_list_add(list, val);
    }
    val = isl_aff_get_constant_val(aff);
    list = isl_val_list_add(list, val);
    isl_aff_free(aff);
    
    mlist = isl_union_map_get_map_list(data->expansion);
    m = isl_map_list_n_map(mlist);
    for (i = 0; i < m; i++) {
        map = isl_map_list_get_map(mlist, i);
        set = isl_map_wrap(map);
        //todo: n?
        for (j = 0; j < n + 1; j++) {
            val = isl_val_list_get_val(list, j);
            if (j < n) {
                space = isl_set_get_space(set);
                ls = isl_local_space_from_space(space);
                var = isl_aff_var_on_domain(ls, isl_dim_set, j);
                var = isl_aff_scale_val(var, val);
            }
            if (j == 0)
                aff = var;
            else if (j == n)
                aff = isl_aff_add_constant_val(aff, val);
            else
                aff = isl_aff_add(aff, var);
        }
        copy = isl_aff_copy(aff);
        copy = isl_aff_scale_down_val(copy, isl_val_copy(data->size));
        copy = isl_aff_floor(copy);
        copy = isl_aff_scale_val(copy, isl_val_copy(data->size));
        aff = isl_aff_sub(aff, copy);

        c = isl_equality_from_aff(aff);
        set = isl_set_add_constraint(set, c);
        map = isl_set_unwrap(set);
        data->result = isl_union_map_add_map(data->result, isl_map_copy(map));
    }

    isl_set_free(set);
    isl_val_list_free(list);
    isl_map_list_free(mlist);

    return isl_stat_ok;
}

static __isl_give isl_union_map* construct_starting_point(__isl_take isl_union_map *expansion,
    __isl_take isl_union_pw_aff *upa, __isl_take isl_val *size)
{
    isl_space *space;
    isl_union_map *result;

    space = isl_union_map_get_space(expansion);
    result = isl_union_map_empty(space);

    struct starting_point_data data = { expansion, result, size};

    isl_union_pw_aff_foreach_pw_aff(upa, &starting_point_cond, &data);

    isl_val_free(data.size);
    isl_union_pw_aff_free(upa);
    isl_union_map_free(data.expansion);

    return data.result;
}

static __isl_give isl_union_map* update_expansion(__isl_take isl_union_map *expansion,
    __isl_take isl_union_set *domain, __isl_take isl_multi_union_pw_aff *mupa,
    __isl_take isl_multi_val *sizes)
{
    int dim;
    isl_val *size;
    isl_space *space;
    isl_union_map *umap, *result;
    isl_union_pw_aff *upa;

    space = isl_union_map_get_space(expansion);
    umap = isl_union_map_empty(space);
    result = isl_union_map_copy(umap);

    isl_union_map_foreach_map(expansion, &drop_space_dim_constraints, &umap);
    isl_union_map_free(expansion);

    struct expansion_data data = { result, domain };
    isl_union_map_foreach_map(umap, &add_space_dim_bounds, &data);
    isl_union_map_free(umap);
    isl_union_set_free(data.domain);
    
    dim = isl_multi_union_pw_aff_dim(mupa, isl_dim_out);
    //todo: multiple overlapped cases
    upa = isl_multi_union_pw_aff_get_union_pw_aff(mupa, 1);
    isl_multi_union_pw_aff_free(mupa);
    size = isl_multi_val_get_val(sizes, 1);
    isl_multi_val_free(sizes);

    data.expansion = construct_starting_point(data.expansion, upa, size);
    isl_union_map_dump(data.expansion);

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

    // obtain original mupa
    mupa = isl_schedule_node_band_get_partial_schedule(node);
    
    // apply parallelogram tiling without shifting point loops
    ctx = isl_schedule_node_get_ctx(node);
    shift = isl_options_get_tile_shift_point_loops(ctx);
    isl_options_set_tile_shift_point_loops(ctx, 0);
	node = isl_schedule_node_band_tile(node, isl_multi_val_copy(sizes));
    isl_options_set_tile_shift_point_loops(ctx, shift);

    // construct an empty contraction
    domain = isl_schedule_node_get_domain(node);
    empty = isl_union_set_empty(isl_union_set_get_space(domain));
    contraction = isl_union_pw_multi_aff_from_union_set(empty);

    // construct the expansion
    universe = isl_union_set_universe(isl_union_set_copy(domain));
    expansion = isl_union_set_identity(universe);
    expansion = update_expansion(expansion, domain, mupa, sizes);
    //printf("after updata_expansion\n");
    //isl_union_map_dump(expansion);
    
    // insert expension node
    child = isl_schedule_node_get_child(node, 0);
    isl_schedule_node_free(node);
    child = isl_schedule_node_insert_expansion(child, contraction, expansion);
    node = isl_schedule_node_parent(child);

    return node;
}
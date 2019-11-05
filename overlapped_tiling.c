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
    isl_multi_val *sizes;
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
    const char *pw_name, *map_name;
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
    set = isl_pw_aff_domain(pa);
    pw_name = isl_set_get_tuple_name(set);
    isl_set_free(set);
    
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
        map_name = isl_map_get_tuple_name(map, isl_dim_in);
        if (map_name != pw_name) {
            isl_map_free(map);
            continue;
        }
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
        copy = isl_aff_scale_down_val(copy, isl_multi_val_get_val(data->sizes, 1));
        copy = isl_aff_floor(copy);
        copy = isl_aff_scale_val(copy, isl_multi_val_get_val(data->sizes, 1));
        aff = isl_aff_sub(aff, copy);

        c = isl_equality_from_aff(aff);
        set = isl_set_add_constraint(set, c);
        map = isl_set_unwrap(set);
        data->result = isl_union_map_add_map(data->result, isl_map_copy(map));

        isl_set_free(set);
    }

    isl_val_list_free(list);
    isl_map_list_free(mlist);

    return isl_stat_ok;
}

static __isl_give isl_union_map* construct_starting_point(__isl_take isl_union_map *expansion,
    __isl_take isl_union_pw_aff *upa, __isl_take isl_multi_val *sizes)
{
    isl_space *space;
    isl_union_map *result;

    space = isl_union_map_get_space(expansion);
    result = isl_union_map_empty(space);

    struct starting_point_data data = { expansion, result, sizes};

    isl_union_pw_aff_foreach_pw_aff(upa, &starting_point_cond, &data);

    isl_multi_val_free(data.sizes);
    isl_union_pw_aff_free(upa);
    isl_union_map_free(data.expansion);

    return data.result;
}

static isl_stat obtain_maxmin_in_bmap(__isl_take isl_basic_map *bmap, void *user) {
    int i, n, dim;
    isl_aff *aff;
    isl_val *val, *copy, *max, *min;
    isl_basic_set *bset;
    isl_constraint *c;
    isl_constraint_list *clist;
    isl_val_list **list = user;

    bset = isl_basic_map_domain(isl_basic_map_copy(bmap));
    dim = isl_basic_set_n_dim(bset);
    isl_basic_set_free(bset);

    //todo: check positon
    //if(dim < 2)
        //isl_die;
    //todo: multiple overlapped cases
    clist = isl_basic_map_get_constraint_list(bmap);
    for (i = 0; i < isl_constraint_list_n_constraint(clist); i++) {
        c = isl_constraint_list_get_constraint(clist, i);
        val = isl_constraint_get_coefficient_val(c, isl_dim_out, 1);
        isl_constraint_free(c);
        n = isl_val_list_n_val(*list);
        if (n == 0) {
            *list = isl_val_list_add(*list, isl_val_copy(val));
            *list = isl_val_list_add(*list, val);
        }
        else {
            max = isl_val_list_get_val(*list, 0);
            min = isl_val_list_get_val(*list, 1);
            if (isl_val_ge(val, max)) {
                copy = isl_val_copy(val);
                isl_val_list_set_val(*list, 0, copy);
            }
            if (isl_val_le(val, min)) {
                copy = isl_val_copy(val);
                isl_val_list_set_val(*list, 1, copy);
            }
            isl_val_free(max);
            isl_val_free(min);
            isl_val_free(val);
        }
    }
    isl_constraint_list_free(clist);
    
    isl_basic_map_free(bmap);

    return isl_stat_ok;
}

struct overlapped_data {
    isl_union_map *expansion;
    isl_union_map *result;
    isl_union_map *dep;
    isl_multi_val *sizes;
};

static isl_stat construct_overlapped_cond(__isl_take isl_map *map, void *user) {
    int i, dim;
    const char *map_name, *dep_name;
    isl_aff *aff, *sub;
    isl_ctx *ctx;
    isl_map *candidate;
    isl_set *set, *domain;
    isl_val *val, *coeff, *rev;
    isl_space *space;
    isl_map_list *list;
    isl_val_list *vlist;
    isl_constraint *c;
    isl_local_space *ls;
    struct overlapped_data *data = user;

    set = isl_map_domain(isl_map_copy(map));
    dim = isl_set_n_dim(set);
    isl_set_free(set);

    set = isl_map_wrap(map);
    space = isl_set_get_space(set);
    ls = isl_local_space_from_space(space);
    //todo: check positon
    //if(dim < 2)
        //isl_die;
    
    // construct upper bound affine expr
    aff = isl_aff_var_on_domain(isl_local_space_copy(ls), isl_dim_set, 1);
    val = isl_multi_val_get_val(data->sizes, 1);
    val = isl_val_sub_ui(val, 1);
    aff = isl_aff_add_constant_val(aff, val);
    sub = isl_aff_var_on_domain(ls, isl_dim_set, 1 + dim);
    aff = isl_aff_sub(aff, sub);
    
    // construct upper bound constraint from affine expr
    c = isl_inequality_from_aff(aff);
    set = isl_set_add_constraint(set, c);
    map = isl_set_unwrap(set);

    // compute coefficient for lower bounds
    map_name = isl_map_get_tuple_name(map, isl_dim_in);
    list = isl_union_map_get_map_list(data->dep);
    for (i = 0; i < isl_map_list_n_map(list); i++) {
        candidate = isl_map_list_get_map(list, i);
        domain = isl_map_domain(candidate);
        dep_name = isl_set_get_tuple_name(domain);
        isl_set_free(domain);
        if(map_name == dep_name)
            break;
        else
            continue;
    }
    isl_map_list_free(list);

    ctx = isl_union_map_get_ctx(data->dep);
    vlist = isl_val_list_alloc(ctx, 2);
    isl_map_foreach_basic_map(candidate, &obtain_maxmin_in_bmap, &vlist);

    coeff = isl_val_list_get_val(vlist, 0);
    coeff = isl_val_sub(coeff, isl_val_list_get_val(vlist, 1));
    isl_val_list_free(vlist);

    // construct lower bound affine expr
    set = isl_map_wrap(map);
    space = isl_set_get_space(set);
    ls = isl_local_space_from_space(space);
    
    // first construct time dim affine expr
    aff = isl_aff_var_on_domain(isl_local_space_copy(ls), isl_dim_set, 0);
    val = isl_multi_val_get_val(data->sizes, 0);
    sub = isl_aff_scale_down_val(isl_aff_copy(aff), isl_val_copy(val));
    sub = isl_aff_floor(sub);
    sub = isl_aff_scale_val(sub, val);
    sub = isl_aff_sub(aff, sub);

    // next construct lower bound affine expr
    val = isl_multi_val_get_val(data->sizes, 1);
    val = isl_val_sub_ui(val, 1);
    rev = isl_val_int_from_si(ctx, -1);
    sub = isl_aff_scale_val(sub, rev);
    sub = isl_aff_add_constant_val(sub, val);
    sub = isl_aff_scale_val(sub, coeff);
    aff = isl_aff_var_on_domain(isl_local_space_copy(ls), isl_dim_set, 1);
    sub = isl_aff_sub(aff, sub);

    // then construct lower bound constraint from affine expr
    aff = isl_aff_var_on_domain(ls, isl_dim_set, 1 + dim);
    aff = isl_aff_sub(aff, sub);
    c = isl_inequality_from_aff(aff);
    set = isl_set_add_constraint(set, c);

    map = isl_set_unwrap(set);
    
    data->result = isl_union_map_add_map(data->result, map);

    return isl_stat_ok;
}

static __isl_give isl_union_map* update_expansion(struct ppcg_scop *scop,
    __isl_take isl_union_map *expansion, __isl_take isl_union_set *domain,
    __isl_take isl_multi_union_pw_aff *mupa, __isl_take isl_multi_val *sizes)
{
    int dim;
    isl_val *size;
    isl_space *space;
    isl_union_map *umap, *result, *dep;
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

    data.expansion = construct_starting_point(data.expansion, upa, isl_multi_val_copy(sizes));

    space = isl_union_map_get_space(data.expansion);
    umap = isl_union_map_empty(space);
    dep = isl_union_map_copy(scop->dep_flow);
    dep = isl_union_map_coalesce(dep);
    dep = isl_union_map_gist_domain(dep, isl_union_set_copy(domain));
    dep = isl_union_map_gist_range(dep, isl_union_set_copy(domain));

    struct overlapped_data overlap = { data.expansion, umap, dep, sizes };
    isl_union_map_foreach_map(overlap.expansion, &construct_overlapped_cond, &overlap);
    isl_multi_val_free(overlap.sizes);
    isl_union_map_free(overlap.expansion);
    isl_union_map_free(overlap.dep);

    return overlap.result;
}

struct dim_size_data {
    isl_set *domain;
    isl_pw_aff *pa;
};

static isl_stat get_pw_aff_from_domain(__isl_take isl_pw_aff *pa, void *user) {
    const char *pa_name, *dom_name;
    isl_set *domain;
    struct dim_size_data *data = user;

    dom_name = isl_set_get_tuple_name(data->domain);
    domain = isl_pw_aff_domain(isl_pw_aff_copy(pa));
    pa_name = isl_set_get_tuple_name(domain);
    isl_set_free(domain);

    if (dom_name == pa_name)
        data->pa = pa;
    else
        isl_pw_aff_free(pa);

    return isl_stat_ok;
}

/* Obtain the space dimension size of input "domain". This size
 * should be compared with parallelogram tiling size. In case
 * the parallelogram tiling size is greater than this size,
 * overlapped tiling should not be applied.  
 */
static int obtain_space_dim_size(__isl_take isl_set *domain,
    __isl_take isl_multi_union_pw_aff *mupa, int dim)
{
	int i, n, bound;
    isl_aff *aff;
    isl_ctx *ctx;
	isl_val *ub, *lb, *val;
	isl_set *dom, *points;
	isl_point *max, *min;
    isl_pw_aff *pa;
    isl_val_list *clist;
    isl_union_pw_aff *upa;

	if(!domain)
		return -1;

    n = isl_multi_union_pw_aff_dim(mupa, isl_dim_out);
    //todo: check n > dim
    upa = isl_multi_union_pw_aff_get_union_pw_aff(mupa, dim);
    isl_multi_union_pw_aff_free(mupa);

    struct dim_size_data data = { domain, pa};
    isl_union_pw_aff_foreach_pw_aff(upa, &get_pw_aff_from_domain, &data);
    isl_union_pw_aff_free(upa);
    isl_pw_aff_foreach_piece(data.pa, &extract_single_piece, &aff);
    isl_pw_aff_free(data.pa);
    
    ctx = isl_aff_get_ctx(aff);
    clist = isl_val_list_alloc(ctx, dim + 2);
    for (i = 0; i <= dim; i++) {
        val = isl_aff_get_coefficient_val(aff, isl_dim_in, i);
        clist = isl_val_list_add(clist, val);
    }
    val = isl_aff_get_constant_val(aff);
    clist = isl_val_list_add(clist, val);
    isl_aff_free(aff);

    bound = 0;
    for (i = 0; i <= dim; i++) {
        points = isl_set_lexmax(isl_set_copy(domain));
        max = isl_set_sample_point(points);
        ub = isl_point_get_coordinate_val(max, isl_dim_set, i);

        points = isl_set_lexmin(isl_set_copy(domain));
        min = isl_set_sample_point(points);
        lb = isl_point_get_coordinate_val(min, isl_dim_set, i);

        val = isl_val_sub(ub, lb);
        val = isl_val_mul(val, isl_val_list_get_val(clist, i));
        bound += isl_val_get_num_si(val);
        isl_val_free(val);
        isl_point_free(max);
        isl_point_free(min);
    }
    val = isl_val_list_get_val(clist, i);
    bound += isl_val_get_num_si(val);
    
    isl_val_free(val);
    isl_set_free(domain);
    isl_val_list_free(clist);

	if(!bound)
		return INT32_MAX;

	return bound + 1;
}

__isl_give isl_schedule_node *overlapped_tile(__isl_take isl_schedule_node *node,
		struct ppcg_scop *scop, __isl_take isl_multi_val *sizes)
{
    int i, j, k, n;
    int shift, dim ,bound, overlapped;
    isl_ctx *ctx;
    isl_set *set;
    isl_val *val, *size;
    isl_set_list *list;
    isl_union_set *domain, *empty, *universe;
    isl_union_map *expansion;
    isl_schedule_node *child;
    isl_multi_union_pw_aff *mupa;
    isl_union_pw_multi_aff *contraction;

    // obtain original mupa
    mupa = isl_schedule_node_band_get_partial_schedule(node);

    // apply parallelogram tiling if tile size is greater than space extent
    overlapped = 1;
    ctx = isl_schedule_node_get_ctx(node);
    domain = isl_schedule_node_get_domain(node);
    list = isl_union_set_get_set_list(domain);
    isl_union_set_free(domain);
    n = isl_set_list_n_set(list);
    for (i = 0; i < n; i++) {
        set = isl_set_list_get_set(list, i);
        dim = isl_set_n_dim(set);
        //todo: check dim >= 2
        bound = obtain_space_dim_size(set, isl_multi_union_pw_aff_copy(mupa), 1);
        val = isl_val_int_from_si(ctx, bound);
        size = isl_multi_val_get_val(sizes, 1);
        if(isl_val_ge(size, val)) {
            overlapped = 0;
            isl_val_free(val);
            isl_val_free(size);
            break;
        }
        else {
            isl_val_free(val);
            isl_val_free(size);
        }
    }
    isl_set_list_free(list);

    if(!overlapped) {
        isl_multi_union_pw_aff_free(mupa);
        return isl_schedule_node_band_tile(node, sizes);
    }
    
    // apply parallelogram tiling without shifting point loops
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
    expansion = update_expansion(scop, expansion, domain, mupa, sizes);
    //isl_union_set_free(domain);
    //isl_multi_val_free(sizes);
    //isl_multi_union_pw_aff_free(mupa);
    //printf("after updata_expansion\n");
    //isl_union_map_dump(expansion);
    
    // insert expension node
    child = isl_schedule_node_get_child(node, 0);
    isl_schedule_node_free(node);
    child = isl_schedule_node_insert_expansion(child, contraction, expansion);
    node = isl_schedule_node_parent(child);

    return node;
}

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

/* data structure for multi-dimensional overlapped tiling.
 * "multi_dim" is either 0 or 1, indicating whether to implement overlapped
 * tiling on the first "multi_dim + 1" space dimensions.
 */
struct multi_dim_data {
    isl_union_map *umap;
    int multi_dim;
};

/* Remove the equality constraints of each "map". "map" should be a map
 * of an indentity expansion union_map. "user" is of multi_dim_data type.
 * 
 * Remove the equality constraints of the first "multi_dim + 1" space
 * dimensions.
 */
isl_stat drop_space_dim_constraints(__isl_take isl_map *map, void *user) {
    struct multi_dim_data *data = user;
    
    map = isl_map_drop_constraints_involving_dims(map, 
        isl_dim_out, 1, data->multi_dim + 1);
    data->umap = isl_union_map_add_map(data->umap, map);

    return isl_stat_ok;
}

struct multi_dim_set_data {
    isl_set *set;
    int multi_dim;
};

/* Update each constraint "c" accroding to "user". "user" is of 
 * multi_dim_set_data type.
 * 
 * Check whether the constraint "c" involves the first "multi_dim + 1" space
 * dimensions. The coefficient of the i-th dimension is saved and retrieved
 * for constructing the affine expression of the (i + n_dim)-th dimension.
 * "n_dim" is always a multiple of 2 since the set that "c" belongs to is a
 * wrapper of an identity map.
 * 
 * The new constraint is constructed according to "equality" flag.
 */
isl_stat update_bounds(__isl_take isl_constraint *c, void *user) {
    struct multi_dim_set_data *data = user;
    int i, n_dim, involved;
    isl_bool equality;
    isl_aff *aff, *var;
    isl_val *coeff;
    isl_local_space *ls;
    isl_constraint *constraint;

    involved = 0;
    for (i = 1; i <= data->multi_dim + 1; i++) {
        if(isl_constraint_involves_dims(c, isl_dim_set, i, 1)) {
            involved = 1;
            ls = isl_constraint_get_local_space(c);
            equality = isl_constraint_is_equality(c);
            coeff = isl_constraint_get_coefficient_val(c, isl_dim_set, i);

            c = isl_constraint_set_coefficient_si(c, isl_dim_set, i, 0);
            aff = isl_constraint_get_aff(c);
            isl_constraint_free(c);

            n_dim = isl_local_space_dim(ls, isl_dim_set) / 2;
            var = isl_aff_var_on_domain(ls, isl_dim_set, i + n_dim);
            var = isl_aff_scale_val(var, coeff);
            aff = isl_aff_add(aff, var);

            if(equality)
                c = isl_equality_from_aff(aff);
            else
                c = isl_inequality_from_aff(aff);
            data->set = isl_set_add_constraint(data->set, c);
            // We can assume a constraint is always involving less than 1
            // dimension due to the nature of stencil computations, and
            // it is safe to break off the loop when involved is set to 1.
            break;
        }
    }
    if(!involved)
        isl_constraint_free(c);

    return isl_stat_ok;
}

/* A wrapper function for "update_bounds". Check each constraint of a "bset"
 * to update the constraints of the set of "user".
 */
isl_stat copy_bounds_from_set(__isl_take isl_basic_set *bset, void *user) {
    struct multi_dim_set_data *data = user;

    isl_basic_set_foreach_constraint(bset, &update_bounds, data);
    isl_basic_set_free(bset);

    return isl_stat_ok;
}

struct expansion_data {
    isl_union_map *expansion;
    isl_union_set *domain;
    int multi_dim;
};

/* Add the bounding constraints of the space dimensions on which overlapped
 * tiling is applied.
 * 
 * "map" is a map of an expansion union_map data. "uset" is the union_set data
 * of iteration domain. The intersection of "set" and "uset" is used to guarantee
 * the constraints would be extracted from the correct set data.
 * 
 * Update the constraints of "mapset", a wrapper set of "map" by invoking
 * copy_bounds_from_set for each basic_set of "set". "setdata" is a wrapper of
 * "mapset" and "multi_dim", which is used to distinguish between one-dimensional
 * and multi-dimensional overlapped tiling.
 */
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

    struct multi_dim_set_data setdata = { mapset, data->multi_dim};
    isl_set_foreach_basic_set(set, &copy_bounds_from_set, &setdata);
    isl_set_free(set);
    data->expansion = isl_union_map_add_map(data->expansion,
        isl_set_unwrap(setdata.set));

    return isl_stat_ok;
}

struct starting_point_data {
    isl_union_map *expansion;
    isl_union_map *result;
    isl_multi_val *sizes;
    int size_dim;
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

/* Construct starting point condition using "pa". "pa" represent the schedule
 * affine expression of a statement after skewing.
 * 
 * As "pa" respresent a single statement, it is safe to extract affine expression
 * from "pa".
 * 
 * "list" is used to record the coefficients and constant of "pa"; "mlist" is the
 * map list of "data->expansion", while "data" is of starting_point_data type.
 * "data->expansion" may have multiple maps in multiple statement cases.
 * 
 * "pw_name" is the name of "pa", and "map_name" is the name of a map in "mlist".
 * Checking the equality between "pw_name" and "map_name" is used to guarantee the
 * starting point conditions can be constructed from the correct "pa" and map.
 * 
 * The starting point condition should be in the form of
 * 
 *      (t + s_i) - T_i*floor((t + s_i) / T_i) = 0
 * 
 * if the skewed schedule of the corresponding space dimension can be written as
 * 
 *       [S(t, ..., s_i, ...) -> (t + s_i)]
 * 
 * where "t" represents time dimension and "s_i" represents the i-th space
 * dimension. "T_i" is the tile size of the later.
 * 
 * Further, the starting point condition could be written equivalently as
 * 
 *      (t + s_i) mod T_i = 0
 * 
 * and we therefore construct it in this form. This equality constraint is then
 * added to the result of data "data->result" and returned.
 */
static isl_stat starting_point_cond(__isl_take isl_pw_aff *pa, void *user) {
    int i, j, m, n;
    const char *pw_name, *map_name;
    isl_ctx *ctx;
    isl_aff *aff, *var;
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
        aff = isl_aff_mod_val(aff, 
            isl_multi_val_get_val(data->sizes, data->size_dim));

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

/* Constrcut starting point using "upa", "sizes" and "size_dim".
 * "upa" represents the skewed space schedule, and "size_dim" represents the dimension of
 * tile sizes "sizes".
 * 
 * Iteratively invoking the starting_point_cond function for dealing with multiple statements.
 */
static __isl_give isl_union_map* construct_starting_point(__isl_take isl_union_map *expansion,
    __isl_take isl_union_pw_aff *upa, __isl_take isl_multi_val *sizes, int size_dim)
{
    isl_space *space;
    isl_union_map *result;

    space = isl_union_map_get_space(expansion);
    result = isl_union_map_empty(space);

    struct starting_point_data data = { expansion, result, sizes, size_dim };

    isl_union_pw_aff_foreach_pw_aff(upa, &starting_point_cond, &data);

    isl_multi_val_free(data.sizes);
    isl_union_pw_aff_free(upa);
    isl_union_map_free(data.expansion);

    return data.result;
}

struct maxmin_data {
    isl_val_list *list;
    int dim;
};

/* Compute the maximum and minimum slopes of dependences from which "bmap" is extraced.
 * "user" should be of maxmin_data type.
 * 
 * Check the coefficient of the given variable at dimension "data->dim". The first val
 * of "data->list" is used to represent the maximum slope and the second the minimum.
 */
static isl_stat obtain_maxmin_in_bmap(__isl_take isl_basic_map *bmap, void *user) {
    int i, n, dim;
    isl_aff *aff;
    isl_val *val, *copy, *max, *min;
    isl_basic_set *bset;
    isl_constraint *c;
    isl_constraint_list *clist;
    struct maxmin_data *data = user;

    bset = isl_basic_map_domain(isl_basic_map_copy(bmap));
    dim = isl_basic_set_n_dim(bset);
    isl_basic_set_free(bset);

    //todo: check positon
    //if(dim < 2)
        //isl_die;
    clist = isl_basic_map_get_constraint_list(bmap);
    for (i = 0; i < isl_constraint_list_n_constraint(clist); i++) {
        c = isl_constraint_list_get_constraint(clist, i);
        val = isl_constraint_get_coefficient_val(c, isl_dim_out, data->dim);
        isl_constraint_free(c);
        n = isl_val_list_n_val(data->list);
        if (n == 0) {
            data->list = isl_val_list_add(data->list, isl_val_copy(val));
            data->list = isl_val_list_add(data->list, val);
        }
        else {
            max = isl_val_list_get_val(data->list, 0);
            min = isl_val_list_get_val(data->list, 1);
            if (isl_val_ge(val, max)) {
                copy = isl_val_copy(val);
                isl_val_list_set_val(data->list, 0, copy);
            }
            if (isl_val_le(val, min)) {
                copy = isl_val_copy(val);
                isl_val_list_set_val(data->list, 1, copy);
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
    int multi_dim;
};

/* Construct overlapped constraints for each "map" extracted from expansion mapping.
 * "user" is of the overlapped_data type.
 * 
 * The overlapped constraints are constructed for the first "multi_dim + 1" space
 * dimensions. The schedule of space dimensions should already be skewed, i.e., in
 * the form of
 * 
 *      [S(t, ..., s_i, ...) -> (t + s_i)]
 * 
 * where "t" represents time dimension and "s_i" the i-th space dimension.
 * 
 * A "rectangle" trapezoid tile shape refers to an overlapped shape constructed by
 * expanding the left bounding faces of a parallelogram one. The overlapped constraints
 * could therefore be obtained by expanding the lower bound of the original constraints
 * and keep the upper bound as the same.
 * 
 * The upper bound is therefore constructed by copying the original constaints, with
 * expanded variables introduced into the constraints. In other words, the uppder bound
 * looks like
 * 
 *      s_i' < s_i + T_i
 * 
 * where "s_i'" represents the expanded variable and "T_i" the tile size.
 * 
 * The lower bound is constructed by expanding the left bounding face of the original
 * parallelogram tile, with all dependence sources along the time-tile dimension
 * considered into account.
 * 
 * The dependence relation about the current statement is first extracted from the
 * dependences "data->dep". The maximum and minimum slopes are computed by invoking
 * the obtain_maxmin_in_bmap function. The differences between the maximum and
 * minimum slopes determines how much the left bounding face of the original
 * parallelogram tile should be expanded.
 * 
 * The next step is to construct the affine expression of time dimension. In particular,
 * this affine expression should be in the form of
 * 
 *      t - T_t*floor(t/T_t)
 * 
 * where "T_t" is the tile size of time dimension.
 * 
 * For each space point dimension "d" within a parallelogram tile, the time dimension
 * difference between "d" and the live-out dimension of the current tile can be expressed
 * as
 * 
 *      T_t - 1 - (t - T_t*floor(t/T_t))
 * 
 * The set of expanded points along each "d" dimension could therefore be written as
 * 
 *      coeff * (T_t - 1 - (t - T_t*floor(t/T_t)))
 * 
 * where "coeff" is the difference between the maximum and minimum slopes of dependences.
 * The expanded lower bound is therefore be expressed as
 * 
 *      s_i' >= s_i - coeff * (T_t - 1 - (t - T_t*floor(t/T_t)))
 * 
 * The lower and upper bounds together contributes to the overlapped constraints and added
 * to "map".
 */
static isl_stat construct_overlapped_cond(__isl_take isl_map *map, void *user) {
    int i, j, dim;
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

    for (j = 1; j <= data->multi_dim + 1; j++) {
        set = isl_map_wrap(map);
        space = isl_set_get_space(set);
        ls = isl_local_space_from_space(space);
        //todo: check positon
        //if(dim < 2)
            //isl_die;
        
        // construct upper bound affine expr
        aff = isl_aff_var_on_domain(isl_local_space_copy(ls), isl_dim_set, j);
        val = isl_multi_val_get_val(data->sizes, j);
        val = isl_val_sub_ui(val, 1);
        aff = isl_aff_add_constant_val(aff, val);
        sub = isl_aff_var_on_domain(ls, isl_dim_set, j + dim);
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
        struct maxmin_data vdata = { vlist, j };
        isl_map_foreach_basic_map(candidate, &obtain_maxmin_in_bmap, &vdata);

        coeff = isl_val_list_get_val(vdata.list, 0);
        coeff = isl_val_sub(coeff, isl_val_list_get_val(vdata.list, 1));
        isl_val_list_free(vdata.list);

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
        val = isl_multi_val_get_val(data->sizes, j);
        val = isl_val_sub_ui(val, 1);
        rev = isl_val_int_from_si(ctx, -1);
        sub = isl_aff_scale_val(sub, rev);
        sub = isl_aff_add_constant_val(sub, val);
        sub = isl_aff_scale_val(sub, coeff);
        aff = isl_aff_var_on_domain(isl_local_space_copy(ls), isl_dim_set, j);
        sub = isl_aff_sub(aff, sub);

        // then construct lower bound constraint from affine expr
        aff = isl_aff_var_on_domain(ls, isl_dim_set, j + dim);
        aff = isl_aff_sub(aff, sub);
        c = isl_inequality_from_aff(aff);
        set = isl_set_add_constraint(set, c);

        map = isl_set_unwrap(set);
    }
    
    data->result = isl_union_map_add_map(data->result, map);

    return isl_stat_ok;
}

/* Update the identity mapping of "expansion".
 * "expansion" is a union_map. The first step is to remove all equality constraints
 * along the space dimensions that overlapped tiling is to be applied.
 * 
 * For each space dimension that overlapped tiling to be applied, the original bounding
 * constraints are introduced to guarantee the expanded points should not exceed the
 * original iteration domain.
 * 
 * The starting point of an expansion mapping can be an arbitrary point in a tile, and
 * we are free to choose a point on the left bounding face of a tile.
 * 
 * The final step is to construct the overlapped conditions according to dependences.
 */
static __isl_give isl_union_map* update_expansion(struct ppcg_scop *scop,
    __isl_take isl_union_map *expansion, __isl_take isl_union_set *domain,
    __isl_take isl_multi_union_pw_aff *mupa, __isl_take isl_multi_val *sizes, int multi_dim)
{
    int i, dim;
    isl_val *size;
    isl_space *space;
    isl_union_map *umap, *result, *dep;
    isl_union_pw_aff *upa;

    space = isl_union_map_get_space(expansion);
    umap = isl_union_map_empty(space);
    result = isl_union_map_copy(umap);
    struct multi_dim_data mdata = { umap, multi_dim };

    isl_union_map_foreach_map(expansion, &drop_space_dim_constraints, &mdata);
    isl_union_map_free(expansion);

    struct expansion_data data = { result, domain, multi_dim };
    isl_union_map_foreach_map(mdata.umap, &add_space_dim_bounds, &data);
    isl_union_map_free(mdata.umap);
    isl_union_set_free(data.domain);
    
    dim = isl_multi_union_pw_aff_dim(mupa, isl_dim_out);
    // construct starting point constraint for the first "multi_dim + 1" space dimensions
    for (i = 1; i <= multi_dim + 1; i++) {
        upa = isl_multi_union_pw_aff_get_union_pw_aff(mupa, i);
        data.expansion = construct_starting_point(data.expansion, 
            upa, isl_multi_val_copy(sizes), i);
    }
    isl_multi_union_pw_aff_free(mupa);

    // construct overlapped constraints
    space = isl_union_map_get_space(data.expansion);
    umap = isl_union_map_empty(space);
    dep = isl_union_map_copy(scop->dep_flow);
    dep = isl_union_map_coalesce(dep);
    dep = isl_union_map_gist_domain(dep, isl_union_set_copy(domain));
    dep = isl_union_map_gist_range(dep, isl_union_set_copy(domain));

    struct overlapped_data overlap = { data.expansion, umap, dep, sizes, multi_dim };
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

/* Return the domain name of "pa".
 */
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

/* Apply overlapped tiling on demand. "multi_dim" indicates whether multiple level
 * overlapped tiling should be performed. "after_mapping" is a flag indicating wheter
 * the expansion and contraction nodes should be inserted after gpu mapping. The
 * expansion node, together with its contraction node, can be inserted between the
 * tile band and point band when generating OpenMP code. On the other hand, it should
 * be inserted undernearth the point band of time dimension.
 * 
 * First check whether the tile sizes of those that to be applied overlapped tiling
 * are greater than the extents of these space dimensions. Return parallelogram
 * tiling if this is true.
 * 
 * Overlapped tiling is applied based on parallelogram tiling. In particular,
 * we first apply parallelogram tiling without shifting point loops, because we will
 * introduce additional points later.
 * 
 * The overlapped tile shapes are constructed by introducing an expansion node before
 * the band node of point loops. In addition, an empty contraction node is also
 * introduced to be used together with an expansion, due to the syntax of schedule
 * trees.
 */
__isl_give isl_schedule_node *overlapped_tile(__isl_take isl_schedule_node *node,
		struct ppcg_scop *scop, __isl_take isl_multi_val *sizes, int multi_dim,
        int after_mapping)
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
    // first construct an identity mapping and then update the expansion
    universe = isl_union_set_universe(isl_union_set_copy(domain));
    expansion = isl_union_set_identity(universe);
    expansion = update_expansion(scop, expansion, domain, mupa, sizes, multi_dim);
    
    // insert expension node
    child = isl_schedule_node_get_child(node, 0);
    isl_schedule_node_free(node);
    if (after_mapping) {
        child = isl_schedule_node_band_split(child, 1);
        node = isl_schedule_node_get_child(child, 0);
        isl_schedule_node_free(child);
        node = isl_schedule_node_insert_expansion(node, contraction, expansion);
        node = isl_schedule_node_parent(node);
        node = isl_schedule_node_parent(node);
    }
    else {
        child = isl_schedule_node_insert_expansion(child, contraction, expansion);
        node = isl_schedule_node_parent(child);
    }

    return node;
}

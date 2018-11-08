/*
 * Copyright 2018 INRIA Paris-Rocquencourt
 * Copyright 2018 Ecole Normale Superieure
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
#include "split_tiling.h"
#include "util.h"

/* Obtain the lexicographically minimum tile of the iteration domain of "node".
 * The input node "node" should have applied parallelogram tiling.
 */
static __isl_give isl_point *split_tile_obtain_source_tile(__isl_keep isl_schedule_node *node)
{
	int n;
	isl_point *result;
	isl_set *params;
	isl_union_set *domain, *tile;
	isl_union_map *schedule;

	if(!node || isl_schedule_node_get_type(node) != isl_schedule_node_band)
		return NULL;

	domain = isl_schedule_node_get_domain(node);
	schedule = isl_schedule_node_band_get_partial_schedule_union_map(node);

	tile = isl_union_set_apply(domain, schedule);
	tile = isl_union_set_lexmin(tile);

	params = isl_union_set_params(isl_union_set_copy(tile));
	tile = isl_union_set_gist_params(tile, params);
	
	//TODO: non-continuous parameter dims
	n = isl_union_set_dim(tile, isl_dim_param);
	tile = isl_union_set_project_out(tile, isl_dim_param, 0, n);

	result = isl_union_set_sample_point(tile);

	return result;
}

/* Obtain the lexicographically minimum point of those covered by
 * all parallelogram tiles. Such points should be a superset of
 * the original iteration domain. The result should be a point
 * that may be lexicographically smaller than the minimum point
 * of the original iteration domain.
 */
static __isl_give isl_point *split_tile_obtain_source_point(__isl_keep isl_schedule_node *node)
{
	int n;
	isl_set *params;
	isl_point *result;
	isl_union_set *domain, *tile, *points;
	isl_union_map *schedule;

	if(!node || isl_schedule_node_get_type(node) != isl_schedule_node_band)
		return NULL;

	domain = isl_schedule_node_get_domain(node);
	schedule = isl_schedule_node_band_get_partial_schedule_union_map(node);

	tile = isl_union_set_apply(domain, isl_union_map_copy(schedule));
	tile = isl_union_set_lexmin(tile);
	
	//obtain the lexmin point of the lexmin tile
	schedule = isl_union_map_reverse(schedule);
	points = isl_union_set_apply(tile, schedule);
	points = isl_union_set_lexmin(points);

	params = isl_union_set_params(isl_union_set_copy(points));
	points = isl_union_set_gist_params(points, params);
	
	//TODO: non-continuous parameter dims
	n = isl_union_set_dim(points, isl_dim_param);
	points = isl_union_set_project_out(points, isl_dim_param, 0, n);

	result = isl_union_set_sample_point(points);
	
	return result;
}

/* Obtain the time dimension size of input "domain". This size
 * should be compared with parallelogram tiling size. In case
 * the parallelogram tiling size is greater than this size,
 * this size should be used to compute the power of flow dep.  
 */
static int obtain_time_dim_size(__isl_keep isl_union_set *domain)
{
	int bound;
	isl_val *ub, *lb;
	isl_point *max, *min;
	isl_union_set *points;

	if(!domain)
		return -1;

	points = isl_union_set_lexmax(isl_union_set_copy(domain));
	max = isl_union_set_sample_point(points);
	ub = isl_point_get_coordinate_val(max, isl_dim_set, 0);

	points = isl_union_set_lexmin(isl_union_set_copy(domain));
	min = isl_union_set_sample_point(points);
	lb = isl_point_get_coordinate_val(min, isl_dim_set, 0);

	bound = isl_val_get_num_si(ub) - isl_val_get_num_si(lb);
	
	isl_point_free(max);
	isl_point_free(min);
	isl_val_free(ub);
	isl_val_free(lb);

	if(!bound)
		return INT32_MAX;

	return bound + 1;
}

/* Compute the dependence along time dimension for one iteration within
 * a stencil in the case of multiple statements. In this case, we abstract
 * the multiple statements as a macro statement and compute dependence
 * for this macro statement. As each original statement should be
 * homogeneous with a macro statement, we are allowed to consider only one
 * statement.
 * 
 * For the input statement, we can first compute all the dependence relations
 * that take this statement as domain, then we can compute all the dependence
 * relations that take this statement as range. The dependence of the macro
 * statement should be those that join these two relations.
 */
static __isl_give isl_union_map *compute_whole_iteration_dependence(__isl_take isl_union_map *dependence,
	__isl_take isl_union_set *source)
{
	isl_union_map *in_domain, *in_range;

	in_domain = isl_union_map_intersect_domain(isl_union_map_copy(dependence), isl_union_set_copy(source));
	in_range = isl_union_map_subtract_domain(dependence, source);
	dependence = isl_union_map_apply_range(in_domain, in_range);
	printf("####################dependence####################\n");
	isl_union_map_dump(dependence);

	return dependence;
}

/* Compute the transitive closure of flow dependence. The time dimension of
 * such space for the transitive closure should be the smaller one between 
 * parallelogram tiling size and time dimension size.
 * 
 * In other words, we need to first determine whether the parallelogram
 * tiling only produces partial tiles or include both full and partial tiles.
 * The result should be gist for both domain and range because
 * partial tiles may cover points that are not included by the original
 * iteration domain.
 */
static int split_tile_compute_dependence(__isl_keep isl_schedule_node *node,
	__isl_keep isl_point *point, struct ppcg_scop *scop)
{
	int n_stmt, n, size, factor;
	isl_val *val, *val0, *val1, *val2, *val3;
	isl_set *params;
	isl_point *pnt0, *pnt1;
	isl_union_set *domain, *source, *sink, *universe;
	isl_union_map *dependence;

	domain = isl_schedule_node_get_domain(node);
	source = isl_union_set_from_point(isl_point_copy(point));
	n_stmt = isl_union_set_n_set(domain);

	dependence = isl_union_map_copy(scop->dep_flow);

	if(n_stmt > 1){
		universe = isl_union_set_universe(isl_union_set_copy(source));
		dependence = compute_whole_iteration_dependence(dependence, universe);
	}
	
	params = isl_union_map_params(isl_union_map_copy(dependence));
	dependence = isl_union_map_gist_params(dependence, params);
	dependence = isl_union_map_gist_domain(dependence, isl_union_set_copy(domain));
	dependence = isl_union_map_gist_range(dependence, domain);
	
	pnt0 = isl_union_set_sample_point(isl_union_set_copy(source));
	sink = isl_union_set_apply(source, dependence);
	sink = isl_union_set_lexmax(sink);
	//TODO: non-continuous parameter dims
	n = isl_union_set_dim(sink, isl_dim_param);
	sink = isl_union_set_project_out(sink, isl_dim_param, 0, n);
	pnt1 = isl_union_set_sample_point(sink);

	val0 = isl_point_get_coordinate_val(pnt0, isl_dim_set, 1);
	val1 = isl_point_get_coordinate_val(pnt1, isl_dim_set, 1);
	factor = isl_val_get_num_si(val1) - isl_val_get_num_si(val0);

	val2 = isl_point_get_coordinate_val(pnt0, isl_dim_set, 0);
	val3 = isl_point_get_coordinate_val(pnt1, isl_dim_set, 0);
	factor = factor/(isl_val_get_num_si(val3) - isl_val_get_num_si(val2));

	isl_val_free(val0);
	isl_val_free(val1);
	isl_val_free(val2);
	isl_val_free(val3);
	isl_point_free(pnt0);
	isl_point_free(pnt1);

	return factor;
}

/*
 *
 */
static isl_stat check_space_dependence(__isl_take isl_map *map, void *user)
{
	int n, m;
	isl_basic_map_list *list, **result = user;

	if(!*result)
		return isl_stat_error;

	list = isl_map_get_basic_map_list(map);
	n = isl_basic_map_list_n_basic_map(list);
	for (int i=0; i<n; i++){
		isl_basic_map *bmap = isl_basic_map_list_get_basic_map(list, i);
		//isl_basic_map_dump(bmap);
		isl_constraint_list *constraints = isl_basic_map_get_constraint_list(bmap);
		m = isl_constraint_list_n_constraint(constraints);
		for (int j=0; j<m; j++){
			isl_constraint *constraint = isl_constraint_list_get_constraint(constraints, j);
			if(isl_constraint_involves_dims(constraint, isl_dim_set, 0, 1)){
				isl_val *const_val;
				const_val = isl_constraint_get_constant_val(constraint);
				if(isl_val_is_zero(const_val)){
					isl_basic_map *copy = isl_basic_map_copy(bmap);
					*result = isl_basic_map_list_add(*result, copy);
					//isl_basic_map_list_dump(*result);
				}
				isl_val_free(const_val);
			}
			isl_constraint_free(constraint);
		}
		isl_basic_map_free(bmap);
		isl_constraint_list_free(constraints);
	}

	isl_map_free(map);
	isl_basic_map_list_free(list);

	return isl_stat_ok;
}

/* Compute the transitive closure of flow dependence. The time dimension of
 * such space for the transitive closure should be the smaller one between 
 * parallelogram tiling size and time dimension size.
 * 
 * In other words, we need to first determine whether the parallelogram
 * tiling only produces partial tiles or include both full and partial tiles.
 * The result should be gist for both domain and range because
 * partial tiles may cover points that are not included by the original
 * iteration domain.
 */
static int split_tile_compute_space_dependence(__isl_keep isl_schedule_node *node,
	__isl_keep isl_point *point, struct ppcg_scop *scop)
{
	int n_stmt, n, m, shift;
	isl_ctx *ctx;
	isl_set *params;
	isl_union_set *domain;
	isl_union_map *dependence;
	isl_basic_map_list *space_dep_bmap_list;

	//TODO: handle other cases

	domain = isl_schedule_node_get_domain(node);
	n_stmt = isl_union_set_n_set(domain);
	shift = 0;

	if(n_stmt == 1)
		return shift;

	dependence = isl_union_map_copy(scop->dep_flow);
	params = isl_union_map_params(isl_union_map_copy(dependence));
	dependence = isl_union_map_gist_params(dependence, params);
	dependence = isl_union_map_gist_domain(dependence, isl_union_set_copy(domain));
	dependence = isl_union_map_gist_range(dependence, domain);

	ctx = isl_union_map_get_ctx(dependence);
	space_dep_bmap_list = isl_basic_map_list_alloc(ctx, 0);
	isl_union_map_foreach_map(dependence, &check_space_dependence, &space_dep_bmap_list);
	isl_basic_map_list_dump(space_dep_bmap_list);

	n = isl_basic_map_list_n_basic_map(space_dep_bmap_list);
	for (int i=0; i<n; i++){
		isl_basic_map *bmap = isl_basic_map_list_get_basic_map(space_dep_bmap_list, i);
		isl_constraint_list *constraints = isl_basic_map_get_constraint_list(bmap);
		m = isl_constraint_list_n_constraint(constraints);
		for (int j=0; j<m; j++){
			isl_constraint *constraint = isl_constraint_list_get_constraint(constraints, j);
			if(isl_constraint_involves_dims(constraint, isl_dim_set, 1, 1)){
				int value;
				isl_val *const_val;
				const_val = isl_constraint_get_constant_val(constraint);
				value = isl_val_get_num_si(const_val);
				if(shift > -value)
					shift = -value;
				isl_val_free(const_val);
			}
			isl_constraint_free(constraint);
		}
		isl_basic_map_free(bmap);
		isl_constraint_list_free(constraints);
	}

	isl_union_map_free(dependence);
	isl_basic_map_list_free(space_dep_bmap_list);

	return shift;
}

/* Obtain the lexicographically maximum tile of the input "dependence".
 */
static __isl_give isl_point *split_tile_obtain_sink_tile(__isl_keep isl_schedule_node *node,
	__isl_keep isl_point *point)
{
	int n;
	isl_point *result;
	isl_union_set *tile;
	isl_union_map *schedule;

	if(!node || !point || isl_schedule_node_get_type(node) != isl_schedule_node_band)
		return NULL;

	schedule = isl_schedule_node_band_get_partial_schedule_union_map(node);
	tile = isl_union_set_from_point(isl_point_copy(point));
	tile = isl_union_set_apply(tile, schedule);
	//TODO: non-continuous parameter dims
	n = isl_union_set_dim(tile, isl_dim_param);
	tile = isl_union_set_project_out(tile, isl_dim_param, 0, n);
	result = isl_union_set_sample_point(tile);

	return result;
}

/* Obtain the lexicographically maximum point of the input "dependence".
 * The input node "node" should have applied parallelogram tiling.
 */
static __isl_give isl_point *split_tile_obtain_sink_point(__isl_take isl_point *point,
	int delta, int factor)
{
	int t, s;
	isl_ctx *ctx;
	isl_val *val0, *val1, *val2, *val3;

	ctx = isl_point_get_ctx(point);

	val0 = isl_point_get_coordinate_val(point, isl_dim_set, 0);
	t = isl_val_get_num_si(val0);
	val1 = isl_point_get_coordinate_val(point, isl_dim_set, 1);
	s = isl_val_get_num_si(val1);

	t = t + delta;
	val2 = isl_val_int_from_si(ctx, t);
	point = isl_point_set_coordinate_val(point, isl_dim_set, 0, val2);
	s = s + factor * delta;
	val3 = isl_val_int_from_si(ctx, s);
	point = isl_point_set_coordinate_val(point, isl_dim_set, 1, val3);
	isl_point_dump(point);

	isl_val_free(val0);
	isl_val_free(val1);

	return point;
}

/* Given the source tile "source" and sink tile "sink", compute the
 * number of tiles crossed by the fixed power of flow dependence.
 * The dependence guarantees that such tiles lie in the same time
 * tile band. The parallelogram tiling size "size" is used to switch
 * between scale or unscale tile band.
 */
static int split_tile_n_dependent_tiles(__isl_keep isl_point *source, 
	__isl_keep isl_point *sink, __isl_keep isl_multi_val *sizes)
{
	int n, scale, size;
	isl_ctx *ctx;
	isl_val *val, *val0, *val1;

	val0 = isl_point_get_coordinate_val(source, isl_dim_set, 1);
	val1 = isl_point_get_coordinate_val(sink, isl_dim_set, 1);
	n = isl_val_get_num_si(val1) - isl_val_get_num_si(val0);

	ctx = isl_point_get_ctx(source);
	scale = isl_options_get_tile_scale_tile_loops(ctx);

	val = isl_multi_val_get_val(sizes, 1);
	size = isl_val_get_num_si(val);

	n = scale ? (n / size) : n;

	isl_val_free(val);
	isl_val_free(val0);
	isl_val_free(val1);

	return n + 1;
}

/* Given the source point "source" and sink point "sink", compute the
 * slope between these two points. We first compute the shift along
 * time dimension, then divide the result by the shift along the first
 * dimension of space.
 */
static isl_val *split_tile_compute_slope(isl_point *source, isl_point *sink)
{
	int delta;
	isl_val *val0, *val1, *val2, *val3;
	isl_val *slope;

	val0 = isl_point_get_coordinate_val(source, isl_dim_set, 0);
	val1 = isl_point_get_coordinate_val(sink, isl_dim_set, 0);
	delta = isl_val_get_num_si(val1) - isl_val_get_num_si(val0);

	val2 = isl_point_get_coordinate_val(source, isl_dim_set, 1);
	val3 = isl_point_get_coordinate_val(sink, isl_dim_set, 1);
	delta = delta/(isl_val_get_num_si(val3) - isl_val_get_num_si(val2));

	slope = isl_val_int_from_si(isl_point_get_ctx(source), delta);

	isl_val_free(val0);
	isl_val_free(val1);
	isl_val_free(val2);
	isl_val_free(val3);

	return slope;
}

/* Drop "str" from "name".
 */
static char *drop_str(char *str, char *sub)
{
    char *p;
	
	p = str;

    while (*p && *sub){
        char *s1 = p;
        char *s2 = sub;
        while (*s1 && *s2 && !(*s1 - *s2)){
            s1++;
            s2++;
        }
        if (!*s2){
            while ((*p++=*s1++));
            p = str;  
        }
        p++;
    }

    return str;
}

/* Drop braces from "name". In particular, this function also delete
 * all blanks from "name".
 */
static char *drop_braces(char *name)
{
	char *str;

	str = "{";
	name = drop_str(name, str);

	str = "}";
	name = drop_str(name, str);

	str = " ";
	name = drop_str(name, str);

	return name;
}

/* Drop brackets from "name".
 */
static char *drop_brackets(char *name)
{
	char *str;

	str = "[";
	name = drop_str(name, str);

	str = "]";
	name = drop_str(name, str);

	return name;
}

/* Drop "->" from "name".
 */
static char *drop_parameters_and_to(char *name)
{
	int n;

	n = strlen(name);

	for (int i=0; i<n-4; i++)
		if ( name[i] == '-' && name[i+1] == '>'){
			printf("%s\n",&name[i+2]);
        	return &name[i+2];
		}

	return name;
}

/* Insert "sub" to "str" at "loc".
 */
static char *insert_str(char *s1, char *s2, int f)
{
    char *cp, *tcp;
 
    tcp = s1 + strlen(s1);
    cp = s1 + f;
 
    if(tcp < cp)
        return s1;
 
    while(tcp >= cp)
    {
        *(tcp + strlen(s2)) = *tcp;
        tcp--;
    }
 
    while(*s2 != '\0')
    {
        *cp = *s2;
        cp++;
        s2++;
    }
	
	return s1;
}

/* Add parentheses to "name".
 */
static char *add_parentheses(char *name)
{
	char *str;

	str = "(";
	name = insert_str(name, str, 0);

	str = ")";
	name = strcat(name, str);

	return name;
}

/* Add braces to "name".
 */
static char *add_braces(char *name)
{
	char *str;

	str = "{ ";
	name = insert_str(name, str, 0);

	str = " }";
	name = strcat(name, str);

	return name;
}

/* Construct the internal data structure for split tiling.
 * In particular, collect all statements that are scheduled
 * by the original parallelogram tiling. In other words,
 * construct "list", "n_list" and "stmt" of "data".
 */
static void *collect_stmts(__isl_keep isl_union_set *uset,
	struct split_tile_phases_data *data)
{
	int n;
	isl_ctx *ctx;
	isl_set_list *list;

	list = isl_union_set_get_set_list(uset);
	n = isl_set_list_n_set(list);

	data->list = list;
	data->n_stmt = n;

	data->stmt = (char **) calloc(data->n_stmt, sizeof(char *));

	for (int i=0; i<n; i++){
		isl_set *set;
		
		set = isl_set_list_get_set(list, i);
		data->stmt[i] = drop_braces(isl_set_to_str(set));
		printf("name %d is %s\n", i, data->stmt[i]);

		char *to = "->";
		if(strstr(data->stmt[i], to))
			data->stmt[i] = drop_parameters_and_to(data->stmt[i]);
		
		printf("name %d is %s\n", i, data->stmt[i]);

		isl_set_free(set);
	}

	return NULL;
}

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

/* Construct the internal data structure for split tiling.
 * In particular, construct "expr" of "data".
 */
static void *construct_expr(__isl_keep isl_multi_union_pw_aff *mupa, 
	struct split_tile_phases_data *data, struct ppcg_scop *scop)
{
	//n: the number of union_pw_aff. In other words, the number of band dimensions
	//m: the number of pw_aff. In ohter words, the number of statements.
	int n, m;
	char *expr;
	isl_ctx *ctx;
	isl_map *map;
	isl_multi_union_pw_aff *copy;

	copy = isl_multi_union_pw_aff_copy(mupa);

	data->constant = (int *) calloc(data->n_stmt, sizeof(int));
	data->no_constraints = (int *) calloc(data->n_stmt, sizeof(int));
	data->expr = (char **) calloc(data->n_stmt, sizeof(char *));
	for (int i=0; i<data->n_stmt; i++)
		data->expr[i] = (char *) calloc(256, sizeof(char));
	expr = (char *) calloc(256, sizeof(char));
	
	n = isl_multi_union_pw_aff_dim(copy, isl_dim_set);

	ctx = isl_multi_union_pw_aff_get_ctx(copy);
	
	for (int i=0; i<n; i++){
		isl_union_pw_aff *upa = isl_multi_union_pw_aff_get_union_pw_aff(copy, i);
		printf("####################upa####################\n");
		isl_union_pw_aff_dump(upa);

		isl_pw_aff_list *list = isl_union_pw_aff_get_pw_aff_list(upa);
		m = isl_pw_aff_list_n_pw_aff(list);
		for (int j=0; j<m; j++){
			isl_pw_aff *pa = isl_pw_aff_list_get_pw_aff(list, j);
			printf("####################pa####################\n");
			isl_pw_aff_dump(pa);

			isl_aff *aff;
			isl_pw_aff_foreach_piece(pa, &extract_single_piece, &aff);
			printf("####################aff####################\n");
			isl_aff_dump(aff);
			
			isl_val *val;
			val = isl_aff_get_constant_val(aff);
			data->constant[j] = isl_val_get_num_si(val);
			printf("####################constaint=%d####################\n", data->constant[j]);
			//isl_val_dump(isl_aff_get_constant_val(aff));
			isl_val_free(val);
			isl_aff_free(aff);

			expr = drop_braces(isl_pw_aff_to_str(pa));
			isl_set *set = isl_pw_aff_domain(pa);
			char *str = drop_braces(isl_set_to_str(set));
			expr = drop_str(expr, str);
			
			char *to = "->";
			expr = drop_str(expr, to);
			expr = drop_brackets(expr);

			if(i){
				//TODO: coefficient != 1
				if(!strcmp(expr, data->expr[j]))
					data->no_constraints[j] = 1;
				else{
					expr = strcat(expr, "-");
					expr = strcat(expr, data->expr[j]);
				}
			}
			
			strcpy(data->expr[j], expr);
			printf("############expr=%s###################\n", data->expr[j]);
			isl_set_free(set);
		}
		isl_union_pw_aff_free(upa);
		isl_pw_aff_list_free(list);
	}

	isl_multi_union_pw_aff_free(copy);

	return NULL;
}

/* Construct the internal data structure for split tiling.
 * In particular, construct "bound" of "data".
 */
static void *construct_bound(__isl_keep isl_multi_val *sizes,
	__isl_keep isl_val *slope, int space_shift,
	struct split_tile_phases_data *data)
{
	int n, size;
	char *bound, *expr;

	n = isl_set_list_n_set(data->list);

	data->bound = (char **) calloc(n, sizeof(char *));
	data->time_dim_name = (char **) calloc(n, sizeof(char *));
	data->space_dim_name = (char **) calloc(n, sizeof(char *));
	for (int i=0; i<n; i++){
		data->bound[i] = (char *) calloc(256, sizeof(char));
		data->time_dim_name[i] = (char *) calloc(256, sizeof(char));
		data->space_dim_name[i] = (char *) calloc(256, sizeof(char));
	}
	bound = (char *) calloc(256, sizeof(char));
	expr = (char *) calloc(256, sizeof(char));

	for (int i=0; i<n; i++){

		isl_set *set = isl_set_list_get_set(data->list, i);
		isl_val *val = isl_multi_val_get_val(sizes, 1);
		size = isl_val_get_num_si(val);

		strcpy(bound, isl_set_get_dim_name(set, isl_dim_set, 1));
		strcpy(data->space_dim_name[i], isl_set_get_dim_name(set, isl_dim_set, 1));
		bound = strcat(bound, "-");
		if(isl_val_get_num_si(slope) != 1){
			sprintf(bound, "%s%ld", bound, isl_val_get_num_si(slope));
			bound = strcat(bound, "*");
		}
		bound = strcat(bound, isl_set_get_dim_name(set, isl_dim_set, 0));

		if (data->constant[i] != 0){
			if (data->constant[i] > 0)
				bound = strcat(bound, "+");
			sprintf(bound, "%s%d", bound, data->constant[i]);
		}

		strcpy(expr, bound);
		if (data->constant[i] != 0){
			char str[20];
			sprintf(str, "%d", data->constant[i]);
			expr = drop_str(expr, str);
			if(data->constant[i] > 0)
				expr = drop_str(expr, "+");
		}
		if(space_shift != 0){
			if(space_shift > 0)
				expr = strcat(expr, "+");
			sprintf(expr, "%s%d", expr, space_shift);
		}
		
		expr = add_parentheses(expr);

		bound = add_parentheses(bound);

		strcpy(data->time_dim_name[i], isl_set_get_dim_name(set, isl_dim_set, 0));

		bound = strcat(bound, "-");
		sprintf(bound, "%s%d", bound, size);
		bound = strcat(bound, "*floor");

		expr = strcat(expr, "/");
		sprintf(expr, "%s%d", expr, size);
		expr = add_parentheses(expr);

		bound = strcat(bound, expr);
		bound =  add_parentheses(bound);
		strncpy(data->bound[i], bound, strlen(bound) + 1);

		printf("############bound[%d]=%s###################\n", i, data->bound[i]);
		
		isl_set_free(set);
		isl_val_free(val);
	}
	
	return NULL;
}

/* Construct the expression of each phase for split tiling.
 * The constraints of each phase should be expressed in the form of
 * 
 * 		lb <= expr < ub
 * 
 * "expr" is the union of each element of "data->expr". In case of multiple
 * statements, "expr" should be united by ";". "lb" and "ub" are the lower
 * and upper bounds of "expr", both extraced from "data->bound". There shoule
 * be a shift of "order * size - t_name" between "lb" and "ub" where "size"
 * is the parallelogram tiling size, "order" for the order of phases and
 * "t_name" for the name of time dimension.
 * 
 * "lb" or "ub" may be absent in some cases but at least one should be present.
 * In particular, "ub" should be absent for the first phase, while "lb" can be
 * taken off from the last phase.
 * 
 */
__isl_give isl_union_set *construct_phase(__isl_keep isl_multi_val *sizes,
	struct split_tile_phases_data *data, int order)
{	
	int n, t_size, s_size;
	char *phase_string, *shift;
	char **lb, **ub, **constraints, **tail;
	isl_ctx *ctx;
	isl_val *val0, *val1;
	isl_union_set *phase;

	n = isl_set_list_n_set(data->list);

	phase_string = (char *) calloc( n * 256, sizeof(char));
	shift = (char *) calloc(256, sizeof(char));

	constraints = (char **) calloc(n, sizeof(char *));
	lb = (char **) calloc(n, sizeof(char *));
	ub = (char **) calloc(n, sizeof(char *));
	tail = (char **) calloc(n, sizeof(char *));
	for (int i=0; i<n; i++){
		constraints[i] = (char *) calloc(256, sizeof(char));
		lb[i] = (char *) calloc(256, sizeof(char));
		ub[i] = (char *) calloc(256, sizeof(char));
		tail[i] = (char *) calloc(256, sizeof(char));
	}

	val0 = isl_multi_val_get_val(sizes, 0);
	t_size = isl_val_get_num_si(val0);
	val1 = isl_multi_val_get_val(sizes, 1);
	s_size = isl_val_get_num_si(val1);

	for (int i=0; i<n; i++){

		if(i == 0)
			strcpy(constraints[i], data->stmt[i]);
		else{
			phase_string = strcat(phase_string, "; ");
			constraints[i] = strcat(constraints[i], data->stmt[i]);
		}

		if(!data->no_constraints[i]){
			constraints[i] = strcat(constraints[i], " : ");

			if(order < data->n_phase - 1){
				strcpy(lb[i], data->bound[i]);
				if(order > 0){
					lb[i] = strcat(lb[i], "-");
					sprintf(shift, "%d", order * s_size);
					lb[i] = strcat(lb[i], shift);

					lb[i] = strcat(lb[i], "+");

					strcpy(tail[i], data->time_dim_name[i]);
					tail[i] = strcat(tail[i], "-");
					sprintf(tail[i], "%s%d", tail[i], s_size);
					tail[i] = strcat(tail[i], "*floor");

					char *floor_str = (char *) calloc(256, sizeof(char));
					strcpy(floor_str, data->time_dim_name[i]);
					floor_str = add_parentheses(floor_str);
					floor_str = strcat(floor_str, "/");
					sprintf(floor_str, "%s%d", floor_str, t_size);
					floor_str = add_parentheses(floor_str);

					tail[i] = strcat(tail[i], floor_str);
					lb[i] = strcat(lb[i], tail[i]);

				}
				printf("############lb%d[%d]=%s###################\n", order, i, lb[i]);
				constraints[i] = strcat(constraints[i], lb[i]);
				constraints[i] = strcat(constraints[i], "<=");
			}

			constraints[i] = strcat(constraints[i], data->expr[i]);
			
			if(order > 0){
				strcpy(ub[i], data->bound[i]);
				if(order > 1){
					ub[i] = strcat(ub[i], "-");
					sprintf(shift, "%d", (order - 1) * s_size);
					ub[i] = strcat(ub[i], shift);

					ub[i] = strcat(ub[i], "+");

					strcpy(tail[i], data->time_dim_name[i]);
					tail[i] = strcat(tail[i], "-");
					sprintf(tail[i], "%s%d", tail[i], s_size);
					tail[i] = strcat(tail[i], "*floor");

					char *floor_str = (char *) calloc(256, sizeof(char));
					strcpy(floor_str, data->time_dim_name[i]);
					floor_str = add_parentheses(floor_str);
					floor_str = strcat(floor_str, "/");
					sprintf(floor_str, "%s%d", floor_str, t_size);
					floor_str = add_parentheses(floor_str);

					tail[i] = strcat(tail[i], floor_str);
					ub[i] = strcat(ub[i], tail[i]);
				}
				printf("############ub%d[%d]=%s###################\n", order, i, ub[i]);
				constraints[i] = strcat(constraints[i], "<");
				constraints[i] = strcat(constraints[i], ub[i]);
			}
		}

		phase_string = strcat(phase_string, constraints[i]);
	}
	phase_string = add_braces(phase_string);
	printf("############phase%d=%s###################\n", order, phase_string);

	ctx = isl_set_list_get_ctx(data->list);
	phase = isl_union_set_read_from_str(ctx, phase_string);
	phase = isl_union_set_coalesce(phase);

	isl_val_free(val0);
	isl_val_free(val1);

	return phase;
}

/* Construct the phases for split tiling. The internal data
 * structure is first constructed, which in turn is represented
 * by its members. "data->stmt" is constructed by dropping the
 * constraints of the domain of "node"; "data->expr" is driven by
 * the multi_union_pw_aff of "node"; "data->bound" is built on top
 * of "slope". 
 * 
 * Each phase is constructed independently by padding the
 * constraints with statement names.
 */
static void *split_tile_construct_phases(__isl_keep isl_union_set_list *phases,
	__isl_keep isl_schedule_node *node,  struct ppcg_scop *scop,
	__isl_take isl_multi_val *sizes, __isl_keep isl_val *slope,
	int space_shift, int n_list, int splitted)
{
	int n, m, scale, shift, dim;
	isl_ctx *ctx;
	isl_val *val;
	isl_union_set *uset, *phase;
	isl_schedule_node *copy;
	isl_multi_val *copy_sizes;
	isl_multi_union_pw_aff *tile_mupa, *mupa;
	struct split_tile_phases_data *data;

	if(!phases || !node)
		return NULL;

	n = isl_union_set_list_n_union_set(phases);
	if(n < 0)
		return NULL;

	ctx = isl_union_set_list_get_ctx(phases);
	data = isl_alloc_type(ctx, struct split_tile_phases_data);

	if(!data)
		return NULL;

	scale = isl_options_get_tile_scale_tile_loops(ctx);
	shift = isl_options_get_tile_shift_point_loops(ctx);
	
	copy = isl_schedule_node_copy(node);
	uset = isl_schedule_node_get_domain(copy);
	uset = isl_union_set_universe(uset);

	if(!shift){
		tile_mupa = isl_schedule_node_band_get_partial_schedule(copy);
		if(!scale){
			copy_sizes = isl_multi_val_copy(sizes);
			dim = isl_multi_val_dim(copy_sizes, isl_dim_set);

			copy_sizes = isl_multi_val_drop_dims(
				copy_sizes, isl_dim_set, 2, dim - 2);
			tile_mupa = isl_multi_union_pw_aff_scale_multi_val(
				tile_mupa, copy_sizes);
		}
	}

	copy = isl_schedule_node_child(copy, 0);
	if(splitted){
		copy = isl_schedule_node_child(copy, 0);
		copy = isl_schedule_node_band_split(copy, 2);
	}

	mupa = isl_schedule_node_band_get_partial_schedule(copy);
	if(!shift)
		mupa = isl_multi_union_pw_aff_sub(mupa, tile_mupa);
	printf("####################mupa####################\n");
	isl_multi_union_pw_aff_dump(mupa);

	data->n_phase = n_list;

	collect_stmts(uset, data);

	construct_expr(mupa, data, scop);

	construct_bound(sizes, slope, space_shift, data);

	for(int i=0; i<n_list; i++){
		phase = construct_phase(sizes, data, i);
		printf("####################phase%d####################\n", i);
		isl_union_set_dump(phase);
		phases = isl_union_set_list_add(phases, phase);
	}

	printf("####################split tile phases####################\n");
	isl_union_set_list_dump(phases);

	isl_union_set_free(uset);
	isl_multi_val_free(sizes);
	isl_schedule_node_free(copy);
	isl_multi_union_pw_aff_free(mupa);
	split_tile_phases_data_free(data);

	return phases;
}

/* Split tiling. We first apply parallelogram tiling on the band node,
 * followed by constructing the fixed power of flow dependence, slope of
 * dependence across tiles along the same time tile band, and introduced
 * independent phases that can be executed in parallel. 
 * 
 * The phases are inserted underneath the time tile dimension.
 */
__isl_give isl_schedule_node *split_tile(__isl_take isl_schedule_node *node,
		struct ppcg_scop *scop, __isl_take isl_multi_val *sizes)
{
	int n, n_stmt, n_list, bound, delta, factor, shift, splitted, mini_sync;
	isl_ctx *ctx;
	isl_union_map *dependence;
	isl_union_set *domain;
	isl_point *source_tile, *sink_tile, *source_point, *sink_point;
	isl_val *v0, *v1, *slope;
	isl_multi_val *copy_sizes;
	isl_union_set_list *phases;
	
	if(!node || !scop || !sizes || 
			isl_schedule_node_get_type(node) != isl_schedule_node_band )
		return NULL;

	n = isl_schedule_node_band_n_member(node);
	ctx = isl_schedule_node_get_ctx(node);
	domain = isl_schedule_node_get_domain(node);
	n_stmt = isl_union_set_n_set(domain);

	//compute the bound of time dimension
	bound = obtain_time_dim_size(domain);
	printf("####################time bound=%d####################\n", bound);

	//compute size along time dimension
	v0 = isl_multi_val_get_val(sizes, 0);
	delta = isl_val_get_num_si(v0) - 1;

	//minimize synchronization by enlarging the time tiling 
	if(scop->options->min_sync && bound != INT32_MAX){
		delta = bound - 1;
		v1 = isl_val_int_from_si(ctx, bound);
		sizes = isl_multi_val_set_val(sizes, 0, v1);
	}
	printf("####################delta=%d####################\n", delta);

	copy_sizes = isl_multi_val_copy(sizes);
	
	//1. apply parallelogram tiling
	node = isl_schedule_node_band_tile(node, copy_sizes);
	
	//. obtain the lexmin tile
	source_tile = split_tile_obtain_source_tile(node);
	printf("####################source_tile####################\n");
	isl_point_dump(source_tile);
	
	//obtain the lexmin point of the lexmin tile
	source_point = split_tile_obtain_source_point(node);
	printf("####################source point####################\n");
	isl_point_dump(source_point);
	
	//compute the size-th power
	factor = split_tile_compute_dependence(node, source_point, scop);
	printf("####################factor=%d####################\n", factor);

	//compute the shift due to the dependence of space dimension if needed
	shift = 0;
	if(n_stmt > 1)
		shift = split_tile_compute_space_dependence(node, source_point, scop);
	printf("####################shift=%d####################\n", shift);
	
	//compute the lexmax sink of size-th power
	sink_point = split_tile_obtain_sink_point(isl_point_copy(source_point), delta, factor);
	printf("####################sink point####################\n");
	isl_point_dump(sink_point);

	//obtain the tile of the lexmax sink
	sink_tile = split_tile_obtain_sink_tile(node, sink_point);
	printf("####################sink_tile####################\n");
	isl_point_dump(sink_tile);
	
	//compute the number of tiles crossed by dep
	n_list = split_tile_n_dependent_tiles(source_tile, sink_tile, sizes) + 1;
	printf("####################n_list = %d####################\n", n_list);
	
	//compute the slope
	slope = split_tile_compute_slope(source_point, sink_point);
	printf("####################slope####################\n");
	isl_val_dump(slope);

	//. split the band for multi-dimensional cases
	if(n > 2){
		node = isl_schedule_node_band_split(node, 2);
		splitted = 1;
	}
	else
		splitted = 0;

	//construct phases
	phases = isl_union_set_list_alloc(ctx, 0);
	phases = split_tile_construct_phases(phases, node, scop, sizes, slope, shift, n_list, splitted);
	
	//insert a sequence node with phases
	node = isl_schedule_node_band_split(node, 1);
	node = isl_schedule_node_child(node, 0);
	node = isl_schedule_node_insert_sequence(node, phases);
	node = isl_schedule_node_parent(node);
	//isl_schedule_node_dump(node);

	isl_union_set_free(domain);
	isl_val_free(v0);
	isl_point_free(source_tile);
	isl_point_free(sink_tile);
	isl_point_free(source_point);
	isl_point_free(sink_point);
	isl_val_free(slope);

	return node;
}

/* Given a singleton set, extract the first (at most *len) elements
 * of the single integer tuple into *sizes and update *len if needed.
 *
 * If "set" is NULL, then the "sizes" array is not updated.
 */
static isl_stat read_sizes_from_set(__isl_take isl_set *set, int *sizes,
	int *len)
{
	int i;
	int dim;

	if (!set)
		return isl_stat_ok;

	dim = isl_set_dim(set, isl_dim_set);
	if (dim < *len)
		*len = dim;

	for (i = 0; i < *len; ++i) {
		isl_val *v;

		v = isl_set_plain_get_val_if_fixed(set, isl_dim_set, i);
		if (!v)
			goto error;
		sizes[i] = isl_val_get_num_si(v);
		isl_val_free(v);
	}

	isl_set_free(set);
	return isl_stat_ok;
error:
	isl_set_free(set);
	return isl_stat_error;
}

__isl_give isl_multi_val *split_tile_read_tile_sizes(__isl_keep isl_schedule_node *node,
	struct ppcg_scop *scop, int *tile_len)
{
	int n;
	int *tile_size;
	isl_ctx *ctx;
	isl_set *size;
	isl_space *space;
	isl_multi_val *mv;
	isl_val_list *list;

	if(!node || isl_schedule_node_get_type(node) != isl_schedule_node_band)
		return NULL;

	space = isl_schedule_node_band_get_space(node);
	ctx = isl_space_get_ctx(space);

	tile_size = isl_alloc_array(ctx, int, *tile_len);
	if (!tile_size)
		return NULL;

	for (int i = 0; i < *tile_len; i++)
		tile_size[i] = scop->options->tile_size;

	size = isl_set_read_from_str(ctx, scop->options->tile_sizes);
	isl_set_dump(size);

	if (read_sizes_from_set(size, tile_size, tile_len) < 0)
		goto error;
	
	//TODO: set for debug
	//set_used_sizes(gen, "tile", gen->kernel_id, tile_size, *tile_len);

	mv = ppcg_multi_val_from_int_list(space, tile_size);

	return mv;
error:
	free(tile_size);
	return NULL;
}

static void *split_tile_phases_data_free(struct split_tile_phases_data *data)
{
	if (!data)
		return NULL;
	
	isl_set_list_free(data->list);

	free(data);

	return NULL;
}

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
#include <isl/schedule_node.h>

#include "ppcg.h"
#include "ppcg_options.h"
#include "split_tiling.h"

/* Obtain the lexicographically minimum tile of the iteration domain of "node".
 * The input node "node" should have applied parallelogram tiling.
 */
static __isl_give isl_point *split_tile_obtain_source_tile(__isl_keep isl_schedule_node *node)
{
	isl_point *result;
	isl_union_set *domain, *tile;
	isl_union_map *schedule;

	if(!node || isl_schedule_node_get_type(node) != isl_schedule_node_band)
		return NULL;

	domain = isl_schedule_node_get_domain(node);
	schedule = isl_schedule_node_band_get_partial_schedule_union_map(node);

	tile = isl_union_set_apply(domain, schedule);
	tile = isl_union_set_lexmin(tile);
	result = isl_union_set_sample_point(tile);

	return result;
}

/* Obtain the lexicographically minimum point of those covered by
 * all parallelogram tiles. Such points should be a superset of
 * the original iteration domain. The result should be a point
 * that may be lexicographically smaller than the minimum point
 * of the original iteration domain.
 */
static __isl_give isl_union_set *split_tile_obtain_source_point(__isl_keep isl_schedule_node *node)
{
	isl_union_set *domain, *tile, *points;
	isl_union_map *schedule;

	if(!node || isl_schedule_node_get_type(node) != isl_schedule_node_band)
		return NULL;

	domain = isl_schedule_node_get_domain(node);
	schedule = isl_schedule_node_band_get_partial_schedule_union_map(node);

	tile = isl_union_set_apply(domain, isl_union_map_copy(schedule));
	tile = isl_union_set_lexmin(tile);
	
	//obtain the lexmax point of the lexmax tile
	schedule = isl_union_map_reverse(schedule);
	points = isl_union_set_apply(tile, schedule);
	points = isl_union_set_lexmin(points);
	
	return points;
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

	bound = isl_val_get_num_si(ub) - isl_val_get_num_si(lb) + 1;
	
	isl_point_free(max);
	isl_point_free(min);
	isl_val_free(ub);
	isl_val_free(lb);

	return bound;
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
static __isl_give isl_union_map *split_tile_compute_dependence(__isl_keep isl_schedule_node *node,
	__isl_keep isl_point *tile_point, struct ppcg_scop *scop, int bound)
{
	int n, m, exact, is_full_tile;
	isl_ctx *ctx;
	isl_val *power;
	isl_union_set *domain, *tile, *points, *range;
	isl_union_map *schedule, *dependence;
	isl_basic_set *bset;
	isl_basic_set_list *list;

	domain = isl_schedule_node_get_domain(node);
	schedule = isl_schedule_node_band_get_partial_schedule_union_map(node);
	schedule = isl_union_map_reverse(schedule);

	tile = isl_union_set_from_point(isl_point_copy(tile_point));
	points = isl_union_set_apply(tile, schedule);

	is_full_tile = scop->options->tile_size < bound ? 1 : 0;

	if(!is_full_tile)
		points = isl_union_set_intersect(points, isl_union_set_copy(domain));

	list = isl_union_set_get_basic_set_list(points);
	n = isl_basic_set_list_n_basic_set(list);
	for (int i=0; i<n; i++){
		bset = isl_basic_set_list_get_basic_set(list, i);
		m = isl_basic_set_n_dim(bset);
		bset = isl_basic_set_drop_constraints_involving_dims(bset, isl_dim_set, 1, m-1);
		if(i==0)
			range = isl_union_set_from_basic_set(bset);
		else{
			isl_union_set *uset = isl_union_set_from_basic_set(bset);
			range = isl_union_set_union(range, uset);
		}
	}

	dependence = isl_union_map_copy(scop->dep_flow);
	dependence = isl_union_map_gist_domain(dependence, isl_union_set_copy(domain));
	dependence = isl_union_map_gist_range(dependence, domain);
	dependence = isl_union_map_intersect_domain(dependence, isl_union_set_copy(range));
	dependence = isl_union_map_intersect_range(dependence, range);
	
	exact = 0;
	dependence = isl_union_map_transitive_closure(dependence, &exact);
	//TODO: handle inexact case
	if(!exact)
		printf("inexact!\n");

	isl_union_set_free(points);
	isl_basic_set_list_free(list);

	return dependence;
}

/* Obtain the lexicographically maximum tile of the input "dependence".
 */
static __isl_give isl_point *split_tile_obtain_sink_tile(__isl_keep isl_schedule_node *node,
	__isl_keep isl_union_map *dependence)
{
	isl_point *result;
	isl_union_set *tile;
	isl_union_map *schedule;

	if(!node || !dependence || isl_schedule_node_get_type(node) != isl_schedule_node_band)
		return NULL;

	tile = isl_union_map_range(isl_union_map_copy(dependence));
	tile = isl_union_set_lexmax(tile);

	schedule = isl_schedule_node_band_get_partial_schedule_union_map(node);
	tile = isl_union_set_apply(tile, schedule);
	result = isl_union_set_sample_point(tile);

	return result;
}

/* Obtain the lexicographically maximum point of the input "dependence".
 * The input node "node" should have applied parallelogram tiling.
 */
static __isl_give isl_point *split_tile_obtain_sink_point(__isl_take isl_union_map *dependence)
{
	isl_point *sink;
	isl_union_set *points;

	points = isl_union_map_range(dependence);
	points = isl_union_set_lexmax(points);
	sink = isl_union_set_sample_point(points);

	return sink;
}

/* Given the source tile "source" and sink tile "sink", compute the
 * number of tiles crossed by the fixed power of flow dependence.
 * The dependence guarantees that such tiles lie in the same time
 * tile band. The parallelogram tiling size "size" is used to switch
 * between scale or unscale tile band.
 */
static int split_tile_n_dependent_tiles(isl_point *source, isl_point *sink,
	int size)
{
	int n, scale;
	isl_ctx *ctx;
	isl_val *val0, *val1;

	val0 = isl_point_get_coordinate_val(source, isl_dim_set, 1);
	val1 = isl_point_get_coordinate_val(sink, isl_dim_set, 1);
	n = isl_val_get_num_si(val1) - isl_val_get_num_si(val0);

	ctx = isl_point_get_ctx(source);
	scale = isl_options_get_tile_scale_tile_loops(ctx);

	n = scale ? (n / size) : n;

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

		isl_set_free(set);
	}

	return NULL;
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

	data->expr = (char **) calloc(data->n_stmt, sizeof(char *));
	for (int i=0; i<data->n_stmt; i++)
		data->expr[i] = (char *) calloc(256, sizeof(char));
	expr = (char *) calloc(256, sizeof(char));
	
	n = isl_multi_union_pw_aff_dim(copy, isl_dim_set);

	ctx = isl_multi_union_pw_aff_get_ctx(mupa);
	
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

			expr = drop_braces(isl_pw_aff_to_str(pa));
			isl_set *set = isl_pw_aff_domain(pa);
			char *str = drop_braces(isl_set_to_str(set));
			expr = drop_str(expr, str);
			
			char *to = "->";
			expr = drop_str(expr, to);
			expr = drop_brackets(expr);

			if(i){
				//TODO: coefficient != 1
				expr = strcat(expr, "-");
				expr = strcat(expr, data->expr[j]);
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
static void *construct_bound(struct ppcg_scop *scop,
	__isl_keep isl_val *slope,
	struct split_tile_phases_data *data)
{
	int n, dim;
	char *bound, *expr;
	isl_set *set;
	isl_set_list *list;

	data->bound = (char *) calloc(256, sizeof(char));
	bound = (char *) calloc(256, sizeof(char));
	expr = (char *) calloc(256, sizeof(char));

	list = isl_set_list_copy(data->list);
	n = isl_set_list_n_set(list);

	for (int i=0; i<n; i++){
		set = isl_set_list_get_set(list, i);
		dim = isl_set_n_dim(set);
		if(dim >= 2)
			break;
	}

	strcpy(bound, isl_set_get_dim_name(set, isl_dim_set, 1));
	bound = strcat(bound, "-");
	if(isl_val_get_num_si(slope) > 1){
		sprintf(bound, "%s%ld", bound, isl_val_get_num_si(slope));
		bound = strcat(bound, "*");
	}
	bound = strcat(bound, isl_set_get_dim_name(set, isl_dim_set, 0));
	bound = add_parentheses(bound);

	strcpy(data->bound, bound);
	strcpy(expr, bound);

	bound = strcat(bound, "-");
	sprintf(bound, "%s%d", bound, scop->options->tile_size);
	bound = strcat(bound, "*floor");

	expr = strcat(expr, "/");
	sprintf(expr, "%s%d", expr, scop->options->tile_size);
	expr = add_parentheses(expr);

	data->bound = strcat(bound, expr);
	data->bound = add_parentheses(data->bound);

	printf("############bound=%s###################\n", data->bound);

	isl_set_free(set);
	isl_set_list_free(list);
	
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
 * be a shift of "size" between "lb" and "ub" where "size" is the parallelogram
 * tiling size.
 * 
 * "lb" or "ub" may be absent in some cases but at least one should be present.
 * In particular, "ub" should be absent for the first phase, while "lb" can be
 * taken off from the last phase.
 * 
 */
__isl_give isl_union_set *construct_phase(struct split_tile_phases_data *data,
	struct ppcg_scop *scop,
	int order)
{	
	int n, shift;
	isl_ctx *ctx;
	char *phase_string, *lb, *ub;
	char **constraints;
	isl_union_set *phase;

	n = data->n_stmt;

	phase_string = (char *) calloc(data->n_stmt * 256, sizeof(char));
	lb = (char *) calloc(data->n_stmt * 256, sizeof(char));
	ub = (char *) calloc(data->n_stmt * 256, sizeof(char));

	constraints = (char **) calloc(data->n_stmt, sizeof(char *));
	for (int i=0; i<n; i++)
		constraints[i] = (char *) calloc(256, sizeof(char));

	if(order < data->n_phase - 1){
		strcpy(lb, data->bound);
		if(order > 0){
			lb = strcat(lb, "-");
			shift = order * scop->options->tile_size;
			sprintf(lb, "%s%d", lb, shift);
		}
	}
	
	if(order > 0){
		strcpy(ub, data->bound);
		if(order > 1){
			ub = strcat(ub, "-");
			shift = (order - 1) * scop->options->tile_size;
			sprintf(ub, "%s%d", ub, shift);
		}
	}
	printf("############lb%d=%s###################\n", order, lb);
	printf("############ub%d=%s###################\n", order, ub);

	for (int i=0; i<n; i++){
		if(i == 0)
			strcpy(constraints[i], data->stmt[i]);
		else{
			phase_string = strcat(phase_string, "; ");
			constraints[i] = strcat(constraints[i], data->stmt[i]);;
		}
		constraints[i] = strcat(constraints[i], " : ");

		if(*lb!='\0'){
			constraints[i] = strcat(constraints[i], lb);
			constraints[i] = strcat(constraints[i], "<=");
		}

		constraints[i] = strcat(constraints[i], data->expr[i]);
		
		if(*ub!='\0'){
			constraints[i] = strcat(constraints[i], "<");
			constraints[i] = strcat(constraints[i], ub);
		}

		phase_string = strcat(phase_string, constraints[i]);
	}
	phase_string = add_braces(phase_string);
	printf("############phase%d=%s###################\n", order, phase_string);

	ctx = isl_set_list_get_ctx(data->list);
	phase = isl_union_set_read_from_str(ctx, phase_string);
	phase = isl_union_set_coalesce(phase);

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
static void *split_tile_construct_phases(isl_union_set_list *phases,
	isl_schedule_node *node,  struct ppcg_scop *scop,
	isl_val *slope, int n_list, int splitted)
{
	int n, m, scale, shift;
	isl_ctx *ctx;
	isl_val *val;
	isl_union_set *uset, *phase;
	isl_schedule_node *copy;
	isl_multi_union_pw_aff *tile_mupa, *mupa;
	isl_union_pw_multi_aff *upma;
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
			val = isl_val_int_from_si(ctx, scop->options->tile_size);
			tile_mupa = isl_multi_union_pw_aff_scale_val(tile_mupa, val);
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

	construct_bound(scop, slope, data);

	for(int i=0; i<n_list; i++){
		phase = construct_phase(data, scop, i);
		printf("####################phase%d####################\n", i);
		isl_union_set_dump(phase);
		phases = isl_union_set_list_add(phases, phase);
	}

	printf("####################split tile phases####################\n");
	isl_union_set_list_dump(phases);

	isl_union_set_free(uset);
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
	int n, n_list, bound, splitted;
	isl_ctx *ctx;
	isl_union_map *dependence;
	isl_union_set *domain, *point;
	isl_point *source_tile, *sink_tile, *source_point, *sink_point;
	isl_val *slope;
	isl_union_set_list *phases;
	
	if(!node || !scop || !sizes || 
			isl_schedule_node_get_type(node) != isl_schedule_node_band )
		return NULL;
	
	//1. apply parallelogram tiling
	node = isl_schedule_node_band_tile(node, sizes);
	domain = isl_schedule_node_get_domain(node);
	
	//. obtain the lexmin tile
	source_tile = split_tile_obtain_source_tile(node);
	printf("####################source_tile####################\n");
	isl_point_dump(source_tile);
	
	//obtain the lexmin point of the lexmin tile
	point = split_tile_obtain_source_point(node);
	printf("####################point####################\n");
	isl_union_set_dump(point);

	source_point = isl_union_set_sample_point(isl_union_set_copy(point));
	printf("####################source point####################\n");
	isl_point_dump(source_point);

	//compute the bound of time dimension
	bound = obtain_time_dim_size(domain);
	
	//compute the size-th power
	dependence = split_tile_compute_dependence(node, source_tile, scop, bound);
	dependence = isl_union_map_intersect_domain(dependence, point);
	printf("####################dependence####################\n");
	isl_union_map_dump(dependence);

	//obtain the tile of the lexmax sink
	sink_tile = split_tile_obtain_sink_tile(node, dependence);
	printf("####################sink_tile####################\n");
	isl_point_dump(sink_tile);
	
	//compute the lexmax sink of size-th power
	sink_point = split_tile_obtain_sink_point(dependence);
	printf("####################sink point####################\n");
	isl_point_dump(sink_point);
	
	//compute the number of tiles crossed by dep
	n_list = split_tile_n_dependent_tiles(source_tile, sink_tile, scop->options->tile_size) + 1;
	printf("n_list = %d\n", n_list);
	
	//compute the slope
	slope = split_tile_compute_slope(source_point, sink_point);
	printf("####################slope####################\n");
	isl_val_dump(slope);

	//. split the band for multi-dimensional cases
	n = isl_schedule_node_band_n_member(node);
	if(n > 2){
		node = isl_schedule_node_band_split(node, 2);
		splitted = 1;
	}
	else
		splitted = 0;

	//construct phases
	ctx = isl_schedule_node_get_ctx(node);
	phases = isl_union_set_list_alloc(ctx, 0);
	phases = split_tile_construct_phases(phases, node, scop, slope, n_list, splitted);
	
	//insert a sequence node with phases
	node = isl_schedule_node_band_split(node, 1);
	node = isl_schedule_node_child(node, 0);
	node = isl_schedule_node_insert_sequence(node, phases);
	node = isl_schedule_node_parent(node);
	//isl_schedule_node_dump(node);

	isl_union_set_free(domain);
	isl_point_free(source_tile);
	isl_point_free(sink_tile);
	isl_point_free(source_point);
	isl_point_free(sink_point);
	isl_val_free(slope);

	return node;
}

static void *split_tile_phases_data_free(struct split_tile_phases_data *data)
{
	if (!data)
		return NULL;
	
	isl_set_list_free(data->list);

	free(data);

	return NULL;
}

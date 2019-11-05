/*
 * Copyright 2012      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl/space.h>
#include <isl/local_space.h>

#include <isl_multi_macro.h>

/* Create a multi expression in the given space that maps each
 * input dimension to the corresponding output dimension.
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),identity)(__isl_take isl_space *space)
{
	int i;
	isl_size n_in, n_out;
	isl_local_space *ls;
	MULTI(BASE) *multi;

	if (!space)
		return NULL;

	if (isl_space_is_set(space))
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"expecting map space", goto error);

	n_in = isl_space_dim(space, isl_dim_in);
	n_out = isl_space_dim(space, isl_dim_out);
	if (n_in < 0 || n_out < 0)
		goto error;
	if (n_in != n_out)
		isl_die(isl_space_get_ctx(space), isl_error_invalid,
			"number of input and output dimensions needs to be "
			"the same", goto error);

	multi = FN(MULTI(BASE),alloc)(isl_space_copy(space));

	if (!n_out) {
		isl_space_free(space);
		return multi;
	}

	space = isl_space_domain(space);
	ls = isl_local_space_from_space(space);

	for (i = 0; i < n_out; ++i) {
		EL *el;
		el = FN(EL,var_on_domain)(isl_local_space_copy(ls),
						isl_dim_set, i);
		multi = FN(FN(MULTI(BASE),set),BASE)(multi, i, el);
	}

	isl_local_space_free(ls);

	return multi;
error:
	isl_space_free(space);
	return NULL;
}

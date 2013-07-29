/****************************************************************
 *
 * force_tersoff.c: Routines used for calculating Tersoff forces/energies
 *
 ****************************************************************
 *
 * Copyright 2002-2013
 *	Institute for Theoretical and Applied Physics
 *	University of Stuttgart, D-70550 Stuttgart, Germany
 *	http://potfit.sourceforge.net/
 *
 ****************************************************************
 *
 *   This file is part of potfit.
 *
 *   potfit is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   potfit is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with potfit; if not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************/

#ifdef TERSOFF

#include "potfit.h"

#include "functions.h"
#include "potential.h"
#include "splines.h"
#include "utils.h"

/****************************************************************
 *
 *  compute forces using pair potentials with spline interpolation
 *
 *  returns sum of squares of differences between calculated and reference
 *     values
 *
 *  arguments: *xi - pointer to potential
 *             *forces - pointer to forces calculated from potential
 *             flag - used for special tasks
 *
 * When using the mpi-parallelized version of potfit, all processes but the
 * root process jump into this function immediately after initialization and
 * stay in here for an infinite loop, to exit only when a certain flag value
 * is passed from process 0. When a set of forces needs to be calculated,
 * the root process enters the function with a flag value of 0, broadcasts
 * the current potential table xi and the flag value to the other processes,
 * thus initiating a force calculation. Whereas the root process returns with
 * the result, the other processes stay in the loop. If the root process is
 * called with flag value 1, all processes exit the function without
 * calculating the forces.
 * If anything changes about the potential beyond the values of the parameters,
 * e.g. the location of the sampling points, these changes have to be broadcast
 * from rank 0 process to the higher ranked processes. This is done when the
 * root process is called with flag value 2. Then a potsync function call is
 * initiated by all processes to get the new potential from root.
 *
 * xi_opt is the array storing the potential parameters (usually it is the
 *     opt_pot.table - part of the struct opt_pot, but it can also be
 *     modified from the current potential.
 *
 * forces is the array storing the deviations from the reference data, not
 *     only for forces, but also for energies, stresses or dummy constraints
 *     (if applicable).
 *
 * flag is an integer controlling the behaviour of calc_forces_pair.
 *    flag == 1 will cause all processes to exit calc_forces_pair after
 *             calculation of forces.
 *    flag == 2 will cause all processes to perform a potsync (i.e. broadcast
 *             any changed potential parameters from process 0 to the others)
 *             before calculation of forces
 *    all other values will cause a set of forces to be calculated. The root
 *             process will return with the sum of squares of the forces,
 *             while all other processes remain in the function, waiting for
 *             the next communication initiating another force calculation
 *             loop
 *
 ****************************************************************/

double calc_forces_tersoff(double *xi_opt, double *forces, int flag)
{
  int   col, i;
  double tmpsum = 0.0, sum = 0.0;
  const tersoff_t *tersoff = &apot_table.tersoff;

#ifndef MPI
  myconf = nconf;
#endif /* !MPI */

  /* This is the start of an infinite loop */
  while (1) {
    tmpsum = 0.;		/* sum of squares of local process */

#ifndef MPI
    apot_check_params(xi_opt);
#endif /* !MPI */

#ifdef MPI
    MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (flag == 1)
      break;			/* Exception: flag 1 means clean up */

    if (myid == 0)
      apot_check_params(xi_opt);
    MPI_Bcast(xi_opt, ndimtot, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif /* MPI */

    update_tersoff_pointers(xi_opt);

    /* region containing loop over configurations */
    {
      atom_t *atom;
      int   h, j, k, l;
      int   self, uf;
#ifdef STRESS
      int   us, stresses;
#endif /* STRESS */

      neigh_t *neigh_j;

      /* pair variables */
      double phi_val, phi_grad;
      double cut_tmp, cut_tmp_j;
      vector tmp_force;

      /* loop over configurations */
      for (h = firstconf; h < firstconf + myconf; h++) {
	uf = conf_uf[h - firstconf];
#ifdef STRESS
	us = conf_us[h - firstconf];
#endif /* STRESS */
	/* reset energies and stresses */
	forces[energy_p + h] = 0.;
#ifdef STRESS
	for (i = 0; i < 6; i++)
	  forces[stress_p + 6 * h + i] = 0.;
#endif /* STRESS */

	/* first loop over atoms: reset forces, densities */
	for (i = 0; i < inconf[h]; i++) {
	  if (uf) {
	    k = 3 * (cnfstart[h] + i);
	    forces[k] = -force_0[k];
	    forces[k + 1] = -force_0[k + 1];
	    forces[k + 2] = -force_0[k + 2];
	  } else {
	    k = 3 * (cnfstart[h] + i);
	    forces[k] = 0.;
	    forces[k + 1] = 0.;
	    forces[k + 2] = 0.;
	  }
	}
	/* end first loop */

	/* 2nd loop: calculate cutoff function f_c for all neighbors */
	for (i = 0; i < inconf[h]; i++) {
	  atom = conf_atoms + i + cnfstart[h] - firstatom;
	  k = 3 * (cnfstart[h] + i);
	  /* loop over neighbors */
	  for (j = 0; j < atom->n_neigh; j++) {
	    neigh_j = atom->neigh + j;
	    col = neigh_j->col[0];

	    cut_tmp = M_PI / (*(tersoff->S[col]) - *(tersoff->R[col]));
	    cut_tmp_j = cut_tmp * (neigh_j->r - *(tersoff->R[col]));
	    if (neigh_j->r < *(tersoff->R[col])) {
	      neigh_j->f = 1.0;
	      neigh_j->df = 0.0;
	    } else if (neigh_j->r > *(tersoff->S[col])) {
	      neigh_j->f = 0.0;
	      neigh_j->df = 0.0;
	    } else {
	      neigh_j->f = 0.5 * (1.0 + cos(cut_tmp_j));
	      neigh_j->df = -0.5 * cut_tmp * sin(cut_tmp_j);
	    }
	  }			/* loop over neighbors */

	  for (j = 0; j < atom->n_neigh; j++) {
	    neigh_j = atom->neigh + j;
	    col = neigh_j->col[0];

	    if (0.0 == *(tersoff->B[col]))
	      continue;
	  }

/*then we can calculate contribution of forces right away */
	  if (uf) {
#ifdef FWEIGHT
	    /* Weigh by absolute value of force */
	    forces[k] /= FORCE_EPS + atom->absforce;
	    forces[k + 1] /= FORCE_EPS + atom->absforce;
	    forces[k + 2] /= FORCE_EPS + atom->absforce;
#endif /* FWEIGHT */
	    /* sum up forces */
#ifdef CONTRIB
	    if (atom->contrib)
#endif /* CONTRIB */
	      tmpsum +=
		conf_weight[h] * (dsquare(forces[k]) + dsquare(forces[k + 1]) + dsquare(forces[k + 2]));
	  }			/* second loop over atoms */
	}

	/* energy contributions */
	forces[energy_p + h] /= (double)inconf[h];
	forces[energy_p + h] -= force_0[energy_p + h];
	tmpsum += conf_weight[h] * eweight * dsquare(forces[energy_p + h]);
#ifdef STRESS
	/* stress contributions */
	if (uf && us) {
	  for (i = 0; i < 6; i++) {
	    forces[stress_p + 6 * h + i] /= conf_vol[h - firstconf];
	    forces[stress_p + 6 * h + i] -= force_0[stress_p + 6 * h + i];
	    tmpsum += conf_weight[h] * sweight * dsquare(forces[stress_p + 6 * h + i]);
	  }
	}
#endif /* STRESS */
	/* limiting constraints per configuration */
      }				/* loop over configurations */
    }				/* parallel region */

    /* dummy constraints (global) */
#ifdef APOT
    /* add punishment for out of bounds (mostly for powell_lsq) */
    if (myid == 0) {
      tmpsum += apot_punish(xi_opt, forces);
    }
#endif /* APOT */

    sum = tmpsum;		/* global sum = local sum  */
#ifdef MPI
    /* reduce global sum */
    sum = 0.;
    MPI_Reduce(&tmpsum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    /* gather forces, energies, stresses */
    if (myid == 0) {		/* root node already has data in place */
      /* forces */
      MPI_Gatherv(MPI_IN_PLACE, myatoms, MPI_VECTOR, forces, atom_len,
	atom_dist, MPI_VECTOR, 0, MPI_COMM_WORLD);
      /* energies */
      MPI_Gatherv(MPI_IN_PLACE, myconf, MPI_DOUBLE, forces + natoms * 3,
	conf_len, conf_dist, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      /* stresses */
      MPI_Gatherv(MPI_IN_PLACE, myconf, MPI_STENS, forces + natoms * 3 + nconf,
	conf_len, conf_dist, MPI_STENS, 0, MPI_COMM_WORLD);
    } else {
      /* forces */
      MPI_Gatherv(forces + firstatom * 3, myatoms, MPI_VECTOR, forces, atom_len,
	atom_dist, MPI_VECTOR, 0, MPI_COMM_WORLD);
      /* energies */
      MPI_Gatherv(forces + natoms * 3 + firstconf, myconf, MPI_DOUBLE,
	forces + natoms * 3, conf_len, conf_dist, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      /* stresses */
      MPI_Gatherv(forces + natoms * 3 + nconf + 6 * firstconf, myconf, MPI_STENS,
	forces + natoms * 3 + nconf, conf_len, conf_dist, MPI_STENS, 0, MPI_COMM_WORLD);
    }
#endif /* MPI */

    /* root process exits this function now */
    if (myid == 0) {
      fcalls++;			/* Increase function call counter */
      if (isnan(sum)) {
#ifdef DEBUG
	printf("\n--> Force is nan! <--\n\n");
#endif /* DEBUG */
	return 10e10;
      } else
	return sum;
    }

  }

  /* once a non-root process arrives here, all is done. */
  return -1.;
}

void update_tersoff_pointers(double *xi)
{
  int   i, j, k;
  int   index = 2;
  tersoff_t *tersoff = &apot_table.tersoff;

  /* allocate if this has not been done */
  if (0 == tersoff->init) {
    tersoff->A = (double **)malloc(paircol * sizeof(double *));
    tersoff->B = (double **)malloc(paircol * sizeof(double *));
    tersoff->lambda = (double **)malloc(paircol * sizeof(double *));
    tersoff->mu = (double **)malloc(paircol * sizeof(double *));
    tersoff->gamma = (double **)malloc(paircol * sizeof(double *));
    tersoff->n = (double **)malloc(paircol * sizeof(double *));
    tersoff->c = (double **)malloc(paircol * sizeof(double *));
    tersoff->d = (double **)malloc(paircol * sizeof(double *));
    tersoff->h = (double **)malloc(paircol * sizeof(double *));
    tersoff->S = (double **)malloc(paircol * sizeof(double *));
    tersoff->R = (double **)malloc(paircol * sizeof(double *));
    tersoff->chi = (double **)malloc(paircol * sizeof(double *));
    tersoff->omega = (double **)malloc(paircol * sizeof(double *));
    for (i = 0; i < paircol; i++) {
      tersoff->A[i] = NULL;
      tersoff->B[i] = NULL;
      tersoff->lambda[i] = NULL;
      tersoff->mu[i] = NULL;
      tersoff->gamma[i] = NULL;
      tersoff->n[i] = NULL;
      tersoff->c[i] = NULL;
      tersoff->d[i] = NULL;
      tersoff->h[i] = NULL;
      tersoff->S[i] = NULL;
      tersoff->R[i] = NULL;
      tersoff->chi[i] = NULL;
      tersoff->omega[i] = NULL;
    }
    tersoff->init = 1;
    tersoff->one = 1.0;
  }

  /* update only if the address has changed */
  if (tersoff->A[0] != xi + index) {
    /* set the pair parameters */
    for (i = 0; i < paircol; i++) {
      tersoff->A[i] = xi + index++;
      tersoff->B[i] = xi + index++;
      tersoff->lambda[i] = xi + index++;
      tersoff->mu[i] = xi + index++;
      tersoff->gamma[i] = xi + index++;
      tersoff->n[i] = xi + index++;
      tersoff->c[i] = xi + index++;
      tersoff->d[i] = xi + index++;
      tersoff->h[i] = xi + index++;
      tersoff->S[i] = xi + index++;
      tersoff->R[i] = xi + index++;
      index += 2;
    }
    for (i = 0; i < paircol; i++) {
      if (0 == (i % ntypes)) {
	tersoff->chi[i] = &tersoff->one;
	tersoff->omega[i] = &tersoff->one;
      } else {
	tersoff->chi[i] = xi + index++;
	tersoff->omega[i] = xi + index++;
	index += 2;
      }
    }
  }

  return;
}

#endif /* TERSOFF */
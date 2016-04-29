/****************************************************************
 *
 * force_adp.c: Routine used for calculating adp forces/energies
 *
 ****************************************************************
 *
 * Copyright 2002-2014
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

#include "potfit.h"

#ifdef ADP

#include "functions.h"
#include "potential.h"
#include "splines.h"
#include "utils.h"

/****************************************************************
 *
 *  compute forces using adp potentials with spline interpolation
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
 * flag is an integer controlling the behaviour of calc_forces_adp.
 *    flag == 1 will cause all processes to exit calc_forces_adp after
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

double calc_forces(double *xi_opt, double *forces, int flag)
{
  int   first, col, i = flag;
  double *xi = NULL;

  /* Some useful temp variables */
  double tmpsum = 0.0, sum = 0.0;
  double rho_sum_loc = 0.0, rho_sum = 0.0;

  /* Temp variables */
  atom_t *atom;
  int   h, j;
  int   n_i, n_j;
  int   self;
  int   uf;
#ifdef STRESS
  int   us, stresses;
#endif /* STRESS */

#ifdef APOT
  double temp_eng;
#endif /* APOT */

  /* pointer for neighbor table */
  neigh_t *neigh;

  /* pair variables */
  double phi_val, phi_grad;
  vector tmp_force;

  /* EAM variables */
  int   col_F;
  double eam_force;
  double rho_val, rho_grad, rho_grad_j;

  /* ADP variables */
  double eng_store;
  double f1, f2;
  double nu;
  double tmp, trace;
  vector tmp_vect;
  sym_tens w_force;
  vector u_force;

  switch (format) {
      case 0:
	xi = calc_pot.table;
	break;
      case 3:			/* fall through */
      case 4:
	xi = xi_opt;		/* calc-table is opt-table */
	break;
      case 5:
	xi = calc_pot.table;	/* we need to update the calc-table */
  }

  /* This is the start of an infinite loop */
  while (1) {
    /* Reset tmpsum and rho_sum_loc
       tmpsum = Sum of all the forces, energies and constraints
       rho_sum_loc = Sum of density, rho, for all atoms */
    tmpsum = 0.0;
    rho_sum_loc = 0.0;

#if defined APOT && !defined MPI
    if (0 == format) {
      apot_check_params(xi_opt);
      update_calc_table(xi_opt, xi, 0);
    }
#endif /* APOT && !MPI */

#ifdef MPI
    /* exchange potential and flag value */
#ifndef APOT
    MPI_Bcast(xi, calc_pot.len, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif /* APOT */
    MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (1 == flag)
      break;			/* Exception: flag 1 means clean up */

#ifdef APOT
    if (0 == myid)
      apot_check_params(xi_opt);
    MPI_Bcast(xi_opt, ndimtot, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    update_calc_table(xi_opt, xi, 0);
#else
    /* if flag==2 then the potential parameters have changed -> sync */
    if (2 == flag)
      potsync();
#endif /* APOT */
#endif /* MPI */

    /* init second derivatives for splines */
    /* [0, ...,  paircol - 1] = pair potentials */
    /* [paircol, ..., paircol + ntypes - 1] = transfer function */
    /* [paircol + ntypes, ..., paircol + 2 * ntypes - 1] = embedding function */
    /* [paircol + 2 * ntypes, ..., 2 * paircol + 2 * ntypes - 1] = dipole function */
    /* [2 * paircol + 2 * ntypes, ..., 3 * paircol + 2 * ntypes - 1] = quadrupole function */
    for (col = 0; col < 3 * paircol + 2 * ntypes; col++) {
      first = calc_pot.first[col];
      if (format == 0 || format == 3)
	spline_ed(calc_pot.step[col], xi + first,
	  calc_pot.last[col] - first + 1, *(xi + first - 2), 0.0, calc_pot.d2tab + first);
      else			/* format >= 4 ! */
	spline_ne(calc_pot.xcoord + first, xi + first,
	  calc_pot.last[col] - first + 1, *(xi + first - 2), 0.0, calc_pot.d2tab + first);
    }

#ifndef MPI
    myconf = nconf;
#endif /* MPI */

    /* region containing loop over configurations */
    {
      /* loop over configurations */
      for (h = firstconf; h < firstconf + myconf; h++) {
	uf = conf_uf[h - firstconf];
#ifdef STRESS
	us = conf_us[h - firstconf];
#endif /* STRESS */
	/* reset energies and stresses */
	forces[energy_p + h] = 0.0;
#ifdef STRESS
	stresses = stress_p + 6 * h;
	for (i = 0; i < 6; i++)
	  forces[stresses + i] = 0.0;
#endif /* STRESS */

	/* set limiting constraints */
	forces[limit_p + h] = -force_0[limit_p + h];

	/* first loop over atoms: reset forces, densities */
	for (i = 0; i < inconf[h]; i++) {
	  n_i = 3 * (cnfstart[h] + i);
	  if (uf) {
	    forces[n_i + 0] = -force_0[n_i + 0];
	    forces[n_i + 1] = -force_0[n_i + 1];
	    forces[n_i + 2] = -force_0[n_i + 2];
	  } else {
	    forces[n_i + 0] = 0.0;
	    forces[n_i + 1] = 0.0;
	    forces[n_i + 2] = 0.0;
	  }
	  /* reset atomic density, dipole and quadrupol distortions */
	  conf_atoms[cnfstart[h] - firstatom + i].rho = 0.0;
	  conf_atoms[cnfstart[h] - firstatom + i].mu.x = 0.0;
	  conf_atoms[cnfstart[h] - firstatom + i].mu.y = 0.0;
	  conf_atoms[cnfstart[h] - firstatom + i].mu.z = 0.0;
	  conf_atoms[cnfstart[h] - firstatom + i].lambda.xx = 0.0;
	  conf_atoms[cnfstart[h] - firstatom + i].lambda.yy = 0.0;
	  conf_atoms[cnfstart[h] - firstatom + i].lambda.zz = 0.0;
	  conf_atoms[cnfstart[h] - firstatom + i].lambda.xy = 0.0;
	  conf_atoms[cnfstart[h] - firstatom + i].lambda.yz = 0.0;
	  conf_atoms[cnfstart[h] - firstatom + i].lambda.zx = 0.0;
	}
	/* end first loop */

	/* 2nd loop: calculate pair forces and energies, atomic densities. */
	for (i = 0; i < inconf[h]; i++) {
	  atom = conf_atoms + i + cnfstart[h] - firstatom;
	  n_i = 3 * (cnfstart[h] + i);
	  /* loop over neighbors */
	  for (j = 0; j < atom->num_neigh; j++) {
	    neigh = atom->neigh + j;
	    /* In small cells, an atom might interact with itself */
	    self = (neigh->nr == i + cnfstart[h]) ? 1 : 0;

	    /* pair potential part */
	    if (neigh->r < calc_pot.end[neigh->col[0]]) {
	      /* fn value and grad are calculated in the same step */
	      if (uf)
		phi_val = splint_comb_dir(&calc_pot, xi, neigh->slot[0],
		  neigh->shift[0], neigh->step[0], &phi_grad);
	      else
		phi_val = splint_dir(&calc_pot, xi, neigh->slot[0], neigh->shift[0], neigh->step[0]);

	      /* avoid double counting if atom is interacting with a copy of itself */
	      if (self) {
		phi_val *= 0.5;
		phi_grad *= 0.5;
	      }

	      /* add cohesive energy */
	      forces[energy_p + h] += phi_val;

	      /* calculate forces */
	      if (uf) {
		tmp_force.x = neigh->dist_r.x * phi_grad;
		tmp_force.y = neigh->dist_r.y * phi_grad;
		tmp_force.z = neigh->dist_r.z * phi_grad;
		forces[n_i + 0] += tmp_force.x;
		forces[n_i + 1] += tmp_force.y;
		forces[n_i + 2] += tmp_force.z;
		/* actio = reactio */
		n_j = 3 * neigh->nr;
		forces[n_j + 0] -= tmp_force.x;
		forces[n_j + 1] -= tmp_force.y;
		forces[n_j + 2] -= tmp_force.z;
#ifdef STRESS
		/* also calculate pair stresses */
		if (us) {
		  forces[stresses + 0] -= neigh->dist.x * tmp_force.x;
		  forces[stresses + 1] -= neigh->dist.y * tmp_force.y;
		  forces[stresses + 2] -= neigh->dist.z * tmp_force.z;
		  forces[stresses + 3] -= neigh->dist.x * tmp_force.y;
		  forces[stresses + 4] -= neigh->dist.y * tmp_force.z;
		  forces[stresses + 5] -= neigh->dist.z * tmp_force.x;
		}
#endif /* STRESS */
	      }
	    }

	    /* dipole distortion part */
	    if (neigh->r < calc_pot.end[neigh->col[2]]) {
	      /* fn value and grad are calculated in the same step */
	      if (uf)
		neigh->u_val =
		  splint_comb_dir(&calc_pot, xi, neigh->slot[2], neigh->shift[2], neigh->step[2],
		  &neigh->u_grad);
	      else
		neigh->u_val = splint_dir(&calc_pot, xi, neigh->slot[2], neigh->shift[2], neigh->step[2]);

	      /* avoid double counting if atom is interacting with a copy of itself */
	      if (self) {
		neigh->u_val *= 0.5;
		neigh->u_grad *= 0.5;
	      }

	      /* sum up contribution for mu */
	      tmp = neigh->u_val * neigh->dist.x;
	      atom->mu.x += tmp;
	      conf_atoms[neigh->nr - firstatom].mu.x -= tmp;
	      tmp = neigh->u_val * neigh->dist.y;
	      atom->mu.y += tmp;
	      conf_atoms[neigh->nr - firstatom].mu.y -= tmp;
	      tmp = neigh->u_val * neigh->dist.z;
	      atom->mu.z += tmp;
	      conf_atoms[neigh->nr - firstatom].mu.z -= tmp;
	    }

	    /* quadrupole distortion part */
	    if (neigh->r < calc_pot.end[neigh->col[3]]) {
	      /* fn value and grad are calculated in the same step */
	      if (uf)
		neigh->w_val =
		  splint_comb_dir(&calc_pot, xi, neigh->slot[3], neigh->shift[3], neigh->step[3],
		  &neigh->w_grad);
	      else
		neigh->w_val = splint_dir(&calc_pot, xi, neigh->slot[3], neigh->shift[3], neigh->step[3]);

	      /* avoid double counting if atom is interacting with a copy of itself */
	      if (self) {
		neigh->w_val *= 0.5;
		neigh->w_grad *= 0.5;
	      }

	      /* sum up contribution for lambda */
	      /* diagonal elements */
	      tmp = neigh->w_val * neigh->sqrdist.xx;
	      atom->lambda.xx += tmp;
	      conf_atoms[neigh->nr - firstatom].lambda.xx += tmp;
	      tmp = neigh->w_val * neigh->sqrdist.yy;
	      atom->lambda.yy += tmp;
	      conf_atoms[neigh->nr - firstatom].lambda.yy += tmp;
	      tmp = neigh->w_val * neigh->sqrdist.zz;
	      atom->lambda.zz += tmp;
	      conf_atoms[neigh->nr - firstatom].lambda.zz += tmp;
	      /* offdiagonal elements */
	      tmp = neigh->w_val * neigh->sqrdist.yz;
	      atom->lambda.yz += tmp;
	      conf_atoms[neigh->nr - firstatom].lambda.yz += tmp;
	      tmp = neigh->w_val * neigh->sqrdist.zx;
	      atom->lambda.zx += tmp;
	      conf_atoms[neigh->nr - firstatom].lambda.zx += tmp;
	      tmp = neigh->w_val * neigh->sqrdist.xy;
	      atom->lambda.xy += tmp;
	      conf_atoms[neigh->nr - firstatom].lambda.xy += tmp;
	    }

	    /* calculate atomic densities */
	    if (atom->type == neigh->type) {
	      /* then transfer(a->b)==transfer(b->a) */
	      if (neigh->r < calc_pot.end[neigh->col[1]]) {
		rho_val = splint_dir(&calc_pot, xi, neigh->slot[1], neigh->shift[1], neigh->step[1]);
		atom->rho += rho_val;
		/* avoid double counting if atom is interacting with a copy of itself */
		if (!self) {
		  conf_atoms[neigh->nr - firstatom].rho += rho_val;
		}
	      }
	    } else {
	      /* transfer(a->b)!=transfer(b->a) */
	      if (neigh->r < calc_pot.end[neigh->col[1]]) {
		atom->rho += splint_dir(&calc_pot, xi, neigh->slot[1], neigh->shift[1], neigh->step[1]);
	      }
	      /* cannot use slot/shift to access splines */
	      if (neigh->r < calc_pot.end[paircol + atom->type])
		conf_atoms[neigh->nr - firstatom].rho +=
		  splint(&calc_pot, xi, paircol + atom->type, neigh->r);
	    }
	  }			/* loop over neighbors */

	  col_F = paircol + ntypes + atom->type;	/* column of F */
#ifdef RESCALE
	  if (atom->rho > calc_pot.end[col_F]) {
	    /* then punish target function -> bad potential */
	    forces[limit_p + h] += DUMMY_WEIGHT * 10.0 * dsquare(atom->rho - calc_pot.end[col_F]);
	    atom->rho = calc_pot.end[col_F];
	  }

	  if (atom->rho < calc_pot.begin[col_F]) {
	    /* then punish target function -> bad potential */
	    forces[limit_p + h] += DUMMY_WEIGHT * 10.0 * dsquare(calc_pot.begin[col_F] - atom->rho);
	  }
#endif /* RESCALE */

	  /* embedding energy, embedding gradient */
	  /* contribution to cohesive energy is F(n) */

#ifndef RESCALE
	  if (atom->rho < calc_pot.begin[col_F]) {
#ifdef APOT
	    /* calculate analytic value explicitly */
	    apot_table.fvalue[col_F] (atom->rho, xi_opt + apot_table.idxpot[col_F], &temp_eng);
	    atom->gradF = apot_grad(atom->rho, xi_opt + opt_pot.first[col_F], apot_table.fvalue[col_F]);
	    forces[energy_p + h] += temp_eng;
#else
	    /* linear extrapolation left */
	    rho_val = splint_comb(&calc_pot, xi, col_F, calc_pot.begin[col_F], &atom->gradF);
	    forces[energy_p + h] += rho_val + (atom->rho - calc_pot.begin[col_F]) * atom->gradF;
#endif /* APOT */
	  } else if (atom->rho > calc_pot.end[col_F]) {
#ifdef APOT
	    /* calculate analytic value explicitly */
	    apot_table.fvalue[col_F] (atom->rho, xi_opt + apot_table.idxpot[col_F], &temp_eng);
	    atom->gradF = apot_grad(atom->rho, xi_opt + opt_pot.first[col_F], apot_table.fvalue[col_F]);
	    forces[energy_p + h] += temp_eng;
#else
	    /* and right */
	    rho_val =
	      splint_comb(&calc_pot, xi, col_F, calc_pot.end[col_F] - 0.5 * calc_pot.step[col_F],
	      &atom->gradF);
	    forces[energy_p + h] += rho_val + (atom->rho - calc_pot.end[col_F]) * atom->gradF;
#endif /* APOT */
	  }
	  /* and in-between */
	  else {
#ifdef APOT
	    /* calculate small values directly */
	    if (atom->rho < 0.1) {
	      apot_table.fvalue[col_F] (atom->rho, xi_opt + apot_table.idxpot[col_F], &temp_eng);
	      atom->gradF = apot_grad(atom->rho, xi_opt + opt_pot.first[col_F], apot_table.fvalue[col_F]);
	      forces[energy_p + h] += temp_eng;
	    } else
#endif /* APOT */
	      forces[energy_p + h] += splint_comb(&calc_pot, xi, col_F, atom->rho, &atom->gradF);
	  }
#else /* !RESCALE */
	  forces[energy_p + h] += splint_comb(&calc_pot, xi, col_F, atom->rho, &atom->gradF);
#endif /* !RESCALE */
	  /* sum up rho */
	  rho_sum_loc += atom->rho;

	  eng_store = 0.0;
	  /* calculate ADP energy for atom i */
	  eng_store += dsquare(atom->mu.x);
	  eng_store += dsquare(atom->mu.y);
	  eng_store += dsquare(atom->mu.z);
	  atom->nu = atom->lambda.xx + atom->lambda.yy + atom->lambda.zz;
	  trace = atom->nu / 3.0;
	  eng_store += dsquare(atom->lambda.xx - trace);
	  eng_store += dsquare(atom->lambda.yy - trace);
	  eng_store += dsquare(atom->lambda.zz - trace);
	  eng_store += dsquare(atom->lambda.xy) * 2.0;
	  eng_store += dsquare(atom->lambda.yz) * 2.0;
	  eng_store += dsquare(atom->lambda.zx) * 2.0;
	  eng_store *= 0.5;
	  forces[energy_p + h] += eng_store;
	}			/* second loop over atoms */

	/* 3rd loop over atom: ADP forces */
	if (uf) {		/* only required if we calc forces */
	  for (i = 0; i < inconf[h]; i++) {
	    atom = conf_atoms + i + cnfstart[h] - firstatom;
	    n_i = 3 * (cnfstart[h] + i);
	    for (j = 0; j < atom->num_neigh; j++) {
	      /* loop over neighbors */
	      neigh = atom->neigh + j;
	      /* In small cells, an atom might interact with itself */
	      self = (neigh->nr == i + cnfstart[h]) ? 1 : 0;
	      col_F = paircol + ntypes + atom->type;	/* column of F */

	      /* are we within reach? */
	      if ((neigh->r < calc_pot.end[neigh->col[1]]) || (neigh->r < calc_pot.end[col_F - ntypes])) {
		rho_grad = (neigh->r < calc_pot.end[neigh->col[1]]) ? splint_grad_dir(&calc_pot,
		  xi, neigh->slot[1], neigh->shift[1], neigh->step[1]) : 0.0;
		if (atom->type == neigh->type)	/* use actio = reactio */
		  rho_grad_j = rho_grad;
		else
		  rho_grad_j = (neigh->r < calc_pot.end[col_F - ntypes])
		    ? splint_grad(&calc_pot, xi, col_F - ntypes, neigh->r) : 0.0;
		/* now we know everything - calculate forces */
		eam_force = (rho_grad * atom->gradF + rho_grad_j * conf_atoms[(neigh->nr) - firstatom].gradF);
		/* avoid double counting if atom is interacting with a
		   copy of itself */
		if (self)
		  eam_force *= 0.5;
		tmp_force.x = neigh->dist_r.x * eam_force;
		tmp_force.y = neigh->dist_r.y * eam_force;
		tmp_force.z = neigh->dist_r.z * eam_force;
		forces[n_i + 0] += tmp_force.x;
		forces[n_i + 1] += tmp_force.y;
		forces[n_i + 2] += tmp_force.z;
		/* actio = reactio */
		n_j = 3 * neigh->nr;
		forces[n_j + 0] -= tmp_force.x;
		forces[n_j + 1] -= tmp_force.y;
		forces[n_j + 2] -= tmp_force.z;
#ifdef STRESS
		/* and stresses */
		if (us) {
		  forces[stresses + 0] -= neigh->dist.x * tmp_force.x;
		  forces[stresses + 1] -= neigh->dist.y * tmp_force.y;
		  forces[stresses + 2] -= neigh->dist.z * tmp_force.z;
		  forces[stresses + 3] -= neigh->dist.x * tmp_force.y;
		  forces[stresses + 4] -= neigh->dist.y * tmp_force.z;
		  forces[stresses + 5] -= neigh->dist.z * tmp_force.x;
		}
#endif /* STRESS */
	      }			/* within reach */
	      if (neigh->r < calc_pot.end[neigh->col[2]]) {
		u_force.x = (atom->mu.x - conf_atoms[(neigh->nr) - firstatom].mu.x);
		u_force.y = (atom->mu.y - conf_atoms[(neigh->nr) - firstatom].mu.y);
		u_force.z = (atom->mu.z - conf_atoms[(neigh->nr) - firstatom].mu.z);
		/* avoid double counting if atom is interacting with a
		   copy of itself */
		if (self) {
		  u_force.x *= 0.5;
		  u_force.y *= 0.5;
		  u_force.z *= 0.5;
		}
		tmp = SPROD(u_force, neigh->dist) * neigh->u_grad;
		tmp_force.x = u_force.x * neigh->u_val + tmp * neigh->dist_r.x;
		tmp_force.y = u_force.y * neigh->u_val + tmp * neigh->dist_r.y;
		tmp_force.z = u_force.z * neigh->u_val + tmp * neigh->dist_r.z;
		forces[n_i + 0] += tmp_force.x;
		forces[n_i + 1] += tmp_force.y;
		forces[n_i + 2] += tmp_force.z;
		/* actio = rectio */
		n_j = 3 * neigh->nr;
		forces[n_j + 0] -= tmp_force.x;
		forces[n_j + 1] -= tmp_force.y;
		forces[n_j + 2] -= tmp_force.z;
#ifdef STRESS
		/* and stresses */
		if (us) {
		  forces[stresses + 0] -= neigh->dist.x * tmp_force.x;
		  forces[stresses + 1] -= neigh->dist.y * tmp_force.y;
		  forces[stresses + 2] -= neigh->dist.z * tmp_force.z;
		  forces[stresses + 3] -= neigh->dist.x * tmp_force.y;
		  forces[stresses + 4] -= neigh->dist.y * tmp_force.z;
		  forces[stresses + 5] -= neigh->dist.z * tmp_force.x;
		}
#endif /* STRESS */
	      }
	      if (neigh->r < calc_pot.end[neigh->col[3]]) {
		w_force.xx = (atom->lambda.xx + conf_atoms[(neigh->nr) - firstatom].lambda.xx);
		w_force.yy = (atom->lambda.yy + conf_atoms[(neigh->nr) - firstatom].lambda.yy);
		w_force.zz = (atom->lambda.zz + conf_atoms[(neigh->nr) - firstatom].lambda.zz);
		w_force.yz = (atom->lambda.yz + conf_atoms[(neigh->nr) - firstatom].lambda.yz);
		w_force.zx = (atom->lambda.zx + conf_atoms[(neigh->nr) - firstatom].lambda.zx);
		w_force.xy = (atom->lambda.xy + conf_atoms[(neigh->nr) - firstatom].lambda.xy);
		/* avoid double counting if atom is interacting with a
		   copy of itself */
		if (self) {
		  w_force.xx *= 0.5;
		  w_force.yy *= 0.5;
		  w_force.zz *= 0.5;
		  w_force.yz *= 0.5;
		  w_force.zx *= 0.5;
		  w_force.xy *= 0.5;
		}
		tmp_vect.x =
		  w_force.xx * neigh->dist.x + w_force.xy * neigh->dist.y + w_force.zx * neigh->dist.z;
		tmp_vect.y =
		  w_force.xy * neigh->dist.x + w_force.yy * neigh->dist.y + w_force.yz * neigh->dist.z;
		tmp_vect.z =
		  w_force.zx * neigh->dist.x + w_force.yz * neigh->dist.y + w_force.zz * neigh->dist.z;
		nu = (atom->nu + conf_atoms[(neigh->nr) - firstatom].nu) / 3.0;
		f1 = 2.0 * neigh->w_val;
		f2 = (SPROD(tmp_vect, neigh->dist) - nu * neigh->r * neigh->r) *
		  neigh->w_grad - nu * f1 * neigh->r;
		tmp_force.x = f1 * tmp_vect.x + f2 * neigh->dist_r.x;
		tmp_force.y = f1 * tmp_vect.y + f2 * neigh->dist_r.y;
		tmp_force.z = f1 * tmp_vect.z + f2 * neigh->dist_r.z;
		forces[n_i + 0] += tmp_force.x;
		forces[n_i + 1] += tmp_force.y;
		forces[n_i + 2] += tmp_force.z;
		/* actio = reactio */
		n_j = 3 * neigh->nr;
		forces[n_j + 0] -= tmp_force.x;
		forces[n_j + 1] -= tmp_force.y;
		forces[n_j + 2] -= tmp_force.z;
#ifdef STRESS
		/* and stresses */
		if (us) {
		  forces[stresses + 0] -= neigh->dist.x * tmp_force.x;
		  forces[stresses + 1] -= neigh->dist.y * tmp_force.y;
		  forces[stresses + 2] -= neigh->dist.z * tmp_force.z;
		  forces[stresses + 3] -= neigh->dist.x * tmp_force.y;
		  forces[stresses + 4] -= neigh->dist.y * tmp_force.z;
		  forces[stresses + 5] -= neigh->dist.z * tmp_force.x;
		}
#endif /* STRESS */
	      }
	    }			/* loop over neighbors */

#ifdef FWEIGHT
	    /* Weigh by absolute value of force */
	    forces[n_i + 0] /= FORCE_EPS + atom->absforce;
	    forces[n_i + 1] /= FORCE_EPS + atom->absforce;
	    forces[n_i + 2] /= FORCE_EPS + atom->absforce;
#endif /* FWEIGHT */
	    /* sum up forces  */
#ifdef CONTRIB
	    if (atom->contrib)
#endif /* CONTRIB */
	      tmpsum += conf_weight[h] *
		(dsquare(forces[n_i + 0]) + dsquare(forces[n_i + 1]) + dsquare(forces[n_i + 2]));
	  }			/* third loop over atoms */
	}

	/* energy contributions */
	forces[energy_p + h] /= (double)inconf[h];
	forces[energy_p + h] -= force_0[energy_p + h];
	tmpsum += conf_weight[h] * eweight * dsquare(forces[energy_p + h]);

#ifdef STRESS
	/* stress contributions */
	if (uf && us) {
	  for (i = 0; i < 6; i++) {
	    forces[stresses + i] /= conf_vol[h - firstconf];
	    forces[stresses + i] -= force_0[stresses + i];
	    tmpsum += conf_weight[h] * sweight * dsquare(forces[stresses + i]);
	  }
	}
#endif /* STRESS */

	/* limiting constraints per configuration */
	tmpsum += conf_weight[h] * dsquare(forces[limit_p + h]);

      }				/* loop over configurations */
    }				/* parallel region */

#ifdef MPI
    /* Reduce rho_sum */
    MPI_Reduce(&rho_sum_loc, &rho_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#else /* MPI */
    rho_sum = rho_sum_loc;
#endif /* MPI */

    /* dummy constraints (global) */
#ifdef APOT
    /* add punishment for out of bounds (mostly for powell_lsq) */
    if (myid == 0) {
      tmpsum += apot_punish(xi_opt, forces);
    }
#endif /* APOT */

#ifndef NOPUNISH
    if (myid == 0) {
      int   g;
      for (g = 0; g < ntypes; g++) {
#ifndef RESCALE
	/* clear field */
	forces[dummy_p + ntypes + g] = 0.0;	/* Free end... */
	/* NEW: Constraint on U': U'(1.0)=0.0; */
	forces[dummy_p + g] = DUMMY_WEIGHT * splint_grad(&calc_pot, xi, paircol + ntypes + g, 1.0);
#else /* !RESCALE */
	forces[dummy_p + ntypes + g] = 0.0;	/* Free end... */
	/* constraints on U`(n) */
	forces[dummy_p + g] =
	  DUMMY_WEIGHT * splint_grad(&calc_pot, xi, paircol + ntypes + g,
	  0.5 * (calc_pot.begin[paircol + ntypes + g] + calc_pot.end[paircol + ntypes + g]))
	  - force_0[dummy_p + g];
#endif /* !RESCALE */

	/* add punishments to total error sum */
	tmpsum += dsquare(forces[dummy_p + g]);
	tmpsum += dsquare(forces[dummy_p + ntypes + g]);
      }				/* loop over types */

#ifndef RESCALE
      /* NEW: Constraint on n: <n>=1.0 ONE CONSTRAINT ONLY */
      /* Calculate averages */
      rho_sum /= (double)natoms;
      /* ATTN: if there are invariant potentials, things might be problematic */
      forces[dummy_p + ntypes] = DUMMY_WEIGHT * (rho_sum - 1.0);
      tmpsum += dsquare(forces[dummy_p + ntypes]);
#endif /* !RESCALE */
    }				/* only root process */
#endif /* !NOPUNISH */


#ifdef MPI
    /* reduce global sum */
    sum = 0.0;
    MPI_Reduce(&tmpsum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    /* gather forces, energies, stresses */
    if (myid == 0) {		/* root node already has data in place */
      /* forces */
      MPI_Gatherv(MPI_IN_PLACE, myatoms, MPI_VECTOR, forces,
	atom_len, atom_dist, MPI_VECTOR, 0, MPI_COMM_WORLD);
      /* energies */
      MPI_Gatherv(MPI_IN_PLACE, myconf, MPI_DOUBLE, forces + energy_p,
	conf_len, conf_dist, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#ifdef STRESS
      /* stresses */
      MPI_Gatherv(MPI_IN_PLACE, myconf, MPI_STENS, forces + stress_p,
	conf_len, conf_dist, MPI_STENS, 0, MPI_COMM_WORLD);
#endif /* STRESS */
#ifdef RESCALE
      /* punishment constraints */
      MPI_Gatherv(MPI_IN_PLACE, myconf, MPI_DOUBLE, forces + limit_p,
	conf_len, conf_dist, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif /* RESCALE */
    } else {
      /* forces */
      MPI_Gatherv(forces + firstatom * 3, myatoms, MPI_VECTOR,
	forces, atom_len, atom_dist, MPI_VECTOR, 0, MPI_COMM_WORLD);
      /* energies */
      MPI_Gatherv(forces + energy_p + firstconf, myconf, MPI_DOUBLE,
	forces + energy_p, conf_len, conf_dist, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#ifdef STRESS
      /* stresses */
      MPI_Gatherv(forces + stress_p + 6 * firstconf, myconf, MPI_STENS,
	forces + stress_p, conf_len, conf_dist, MPI_STENS, 0, MPI_COMM_WORLD);
#endif /* STRESS */
#ifndef RESCALE
      /* punishment constraints */
      MPI_Gatherv(forces + limit_p + firstconf, myconf, MPI_DOUBLE,
	forces + limit_p, conf_len, conf_dist, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif /* RESCALE */
    }
    /* no need to pick up dummy constraints - they are already @ root */
#else
    sum = tmpsum;		/* global sum = local sum  */
#endif /* MPI */

    /* root process exits this function now */
    if (myid == 0) {
      fcalls++;			/* increase function call counter */
      if (isnan(sum)) {
#ifdef DEBUG
	printf("\n--> Force is nan! <--\n\n");
#endif /* DEBUG */
	return 10e10;
      } else
	return sum;
    }
  }				/* end of infinite loop */

  /* once a non-root process arrives here, all is done. */
  return -1.0;
}

#endif /* ADP */

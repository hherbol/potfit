/****************************************************************
 *
 * force_mix.c: Routines used for mixing pair + other
 * forces/energies. (SMRFF)
 *
 ****************************************************************
 *
 * This file is part of potfit.
 *
 * potfit is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * potfit is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with potfit; if not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************/

#if !defined(MIX)
#error force_mix.c compiled without MIX support
#endif

/****************************************************************
* When we mix, we re-define function calls to pair and the mixed
* potential.  This allows the force call to go to force_mix.c
* instead, which takes care of mixing.
****************************************************************/

#include "potfit.h"

#include "chempot.h"
#if defined(MPI)
#include "mpi_utils.h"
#endif
#include "memory.h"
#include "force.h"
#include "functions.h"
#include "potential_input.h"
#include "splines.h"
#include "utils.h"

/****************************************************************
  init_force
    called after all parameters and potentials are read
    additional assignments and initializations can be performed here
****************************************************************/

void init_force(int is_worker)
{
  /* Currently we will leave this blank, but if we needed we could
  *  get the init_force call of pair and other (where other is the
  *  close range potential we chose) via:
  *
  *  double calc_mix_pair_force(double* xi_opt, double* forces, int flag)
  *  void init_mix_pair_force(int is_worker)
  *
  *  double calc_mix_pot_force(double* xi_opt, double* forces, int flag)
  *  void init_mix_pot_force(int is_worker)
  */
}

/****************************************************************
 *
 *  compute forces using pair + other potentials with spline interpolation
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
 *     g_pot.opt_pot.table - part of the struct g_pot.opt_pot, but it can also
 *     be modified from the current potential.
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

double calc_forces(double* xi_opt, double* forces, int flag)
{
  double* xi = NULL;
  // First, let's generate our "total forces" variable.  We'll then
  // later loop through the "sub" forces and add them here
  double* total_forces = (double*)Malloc(g_calc.mdim * sizeof(double));

  /* mdim is the dimension of the force vector:
   - 3*natoms forces
   - nconf cohesive energies,
   - 6*nconf stress tensor components */
  for (int f_index = 0; f_index < g_calc.mdim; f_index++){
    total_forces[f_index] = -g_config.force_0[f_index];
  }


  // Start an infinite loop
  // TODO - Is this necessary? I see all the other force_X.c codes doing it, but
  // it doesn't make sense...
  while(1){
    double error_sum = 0.0;
    /* If we have an analytical potential with NO MPI, then we
       can directly call these functions*/
  #if defined(APOT) && !defined(MPI)
    // If it's an analytic function
    if (g_pot.format_type == POTENTIAL_FORMAT_ANALYTIC) {
      // Check if the given parameters are valid.  Ex, is R > S?
      apot_check_params(xi_opt);

      // Now, update g_pot.calc_pot.table from g_pot.opt_pot.table, including globals
      update_calc_table(xi_opt, xi, 0);
    }
  #endif  // APOT && !MPI

    /*Now, if instead we have MPI but NOT ANALYTICAL POTENTIAL, we just
      need to broadcast g_pot.calc_pot across the processors.*/
  #if defined(MPI)
  #if !defined(APOT)
    // exchange potential and flag value
    MPI_Bcast(xi, g_pot.calc_pot.len, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  #endif  // !APOT

    // Broadcast the flag across processors
    MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (flag == 1)
      break; // Exception: flag 1 means clean up

    /* HOWEVER!  If we DO have an analytical potential, then we need to verify the parameters
       are correct (only once, so we do if to mpi id == 0), but we need to broadcast the values
       to all the MPI_Bcast to update.  That is, we first check if xi_opt are okay (if not, fix)
       in the apot_check_params function, then we update the values across the other processors.*/
  #if defined(APOT)
    if (g_mpi.myid == 0)
      apot_check_params(xi_opt);
    MPI_Bcast(xi_opt, g_calc.ndimtot, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    update_calc_table(xi_opt, xi, 0);
  #else   // APOT

    // This is what takes care of the broadcasting.
    // if flag == 2 then the potential parameters have changed -> sync
    if (flag == 2)
      potsync();
  #endif  // APOT
  #endif  // MPI

    // Now, we get our long range, Lennard Jones, forces
    error_sum = calc_mix_pair_force(xi_opt, forces, flag, CUTOFF_1, CUTOFF_2);
    for (int atom_idx = 0; atom_idx < g_config.natoms; atom_idx++){
      total_forces[3 * atom_idx + 0] += forces[3 * atom_idx + 0];
      total_forces[3 * atom_idx + 1] += forces[3 * atom_idx + 1];
      total_forces[3 * atom_idx + 2] += forces[3 * atom_idx + 2];
    }

    // Next, we get our short range, Reactive, forces
    error_sum = calc_mix_pot_force(xi_opt, forces, flag, CUTOFF_1);
    for (int atom_idx = 0; atom_idx < g_config.natoms; atom_idx++){
      total_forces[3 * atom_idx + 0] += forces[3 * atom_idx + 0];
      total_forces[3 * atom_idx + 1] += forces[3 * atom_idx + 1];
      total_forces[3 * atom_idx + 2] += forces[3 * atom_idx + 2];
    }


    // TODO - SMOOTHING GOES HERE


    // Now, we can set forces, and free total_forces
    for (int atom_idx = 0; atom_idx < g_config.natoms; atom_idx++){
      forces[3 * atom_idx + 0] = total_forces[3 * atom_idx + 0];
      forces[3 * atom_idx + 1] = total_forces[3 * atom_idx + 1];
      forces[3 * atom_idx + 2] = total_forces[3 * atom_idx + 2];
    }
    free(total_forces);

    // Here, we can calculate our error_sum
    // Loop over configurations
    for (int config_idx = g_mpi.firstconf; config_idx < g_mpi.firstconf + g_mpi.myconf; config_idx++) {

      // Force Error
      for (int atom_idx = 0; atom_idx < g_config.inconf[config_idx]; atom_idx++) {
        int n_i = 3*(g_config.cnfstart[config_idx] + atom_idx);
        // At this point, forces can be found via the following
        error_sum += g_config.conf_weight[config_idx] * (dsquare(forces[n_i + 0]) + dsquare(forces[n_i + 1]) + dsquare(forces[n_i + 2]));
      }

      // Energy Error
      error_sum += g_config.conf_weight[config_idx] * g_param.eweight * dsquare(forces[g_calc.energy_p + config_idx]);
    }

    // A convenient way to combine error_sum and forces across processors if MPI was used
    gather_forces(&error_sum, forces);

    // root process exits this function now
    if (g_mpi.myid == 0) {
      // Increase function call counter
      g_calc.fcalls++;
      if (isnan(error_sum)) {
  #if defined(DEBUG)
        printf("\n--> Force is nan! <--\n\n");
  #endif  // DEBUG
        return 10e10;
      } else
        return error_sum;
    }

  }

}

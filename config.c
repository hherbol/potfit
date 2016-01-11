/****************************************************************
 *
 * config.c: Reads atomic configurations and forces.
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
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with potfit; if not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************/

#include "potfit.h"

#include "config.h"
#include "utils.h"

/* added */ 
#ifdef KIM
#include "kim/kim.h"
#endif 
/* added ends */

/****************************************************************
 *
 *  read the configurations
 *
 ****************************************************************/

void read_config(char *filename)
{
  atom_t *atom;
  char  msg[255], buffer[1024];
  char *res, *ptr;
  char *tmp, *res_tmp;
  int   count;
  int   i, j, k, ix, iy, iz;
  int   type1, type2, col, slot, klo, khi;
  int   cell_scale[3];
  int   fixed_elements;
  int   h_stress = 0, h_eng = 0, h_boxx = 0, h_boxy = 0, h_boxz = 0, use_force;
  int   have_small_box = 0;
#ifdef CONTRIB
  int   have_contrib = 0;
#endif /* CONTRIB */
  int   line = 0;
  int   max_type = 0;
  int   sh_dist = 0;		/* short distance flag */
  int   str_len;
  int   tag_format = 0;
  int   w_force = 0, w_stress = 0;
#ifdef APOT
  int   index;
#endif /* APOT */
  FILE *infile;
  fpos_t filepos;
  double r, rr, istep, shift, step;
  double *mindist;
  sym_tens *stresses;
  vector d, dd, iheight;
#ifdef THREEBODY
  int   ijk;
  int   nnn;
  double ccos;
#endif /* THREEBODY */

  /* initialize elements array */
  elements = (char **)malloc(ntypes * sizeof(char *));
  if (NULL == elements)
    error(1, "Cannot allocate memory for element names.");
  reg_for_free(elements, "elements");
  for (i = 0; i < ntypes; i++) {
    elements[i] = (char *)malloc(3 * sizeof(char));
    if (NULL == elements[i])
      error(1, "Cannot allocate memory for %d. element name.\n", i + 1);
    reg_for_free(elements[i], "elements[%d]", i);
    snprintf(elements[i], 3, "%d", i);
  }

  /* initialize minimum distance array */
  mindist = (double *)malloc(ntypes * ntypes * sizeof(double));
  if (NULL == mindist)
    error(1, "Cannot allocate memory for minimal distance.");

  /* set maximum cutoff distance as starting value for mindist */
  for (i = 0; i < ntypes * ntypes; i++)
    mindist[i] = 99.9;
  for (i = 0; i < ntypes; i++)
    for (j = 0; j < ntypes; j++) {
      k = (i <= j) ? i * ntypes + j - ((i * (i + 1)) / 2) : j * ntypes + i - ((j * (j + 1)) / 2);
      mindist[k] = MAX(rcut[i * ntypes + j], mindist[i * ntypes + j]);
    }

  nconf = 0;

  /* open file */
  infile = fopen(filename, "r");
  if (NULL == infile)
    error(1, "Could not open file %s\n", filename);

  printf("Reading the config file >> %s << and calculating neighbor lists ...\n", filename);
  fflush(stdout);

  /* read configurations until the end of the file */
  do {
    res = fgets(buffer, 1024, infile);
    line++;
    if (NULL == res)
      error(1, "Unexpected end of file in %s", filename);
    if (res[0] == '#') {	/* new file format (with tags) */
      tag_format = 1;
      h_eng = h_stress = h_boxx = h_boxy = h_boxz = 0;
      if (res[1] == 'N') {	/* Atom number line */
	if (sscanf(res + 3, "%d %d", &count, &use_force) < 2)
	  error(1, "%s: Error in atom number specification on line %d\n", filename, line);
      } else
	error(1, "%s: Number of atoms missing on line %d\n", filename, line);
    } else {
      /* number of atoms in this configuration */
      tag_format = 0;
      use_force = 1;
      if (1 > sscanf(buffer, "%d", &count))
	error(1, "Unexpected end of file in %s", filename);
    }

    /* check if there are enough atoms, 2 for pair and 3 for manybody potentials */
#ifndef THREEBODY
    if (2 > count)
      error(1, "The configuration %d (starting on line %d) has not enough atoms. Please remove it.",
	nconf + 1, line);
#else
    if (3 > count)
      error(1, "The configuration %d (starting on line %d) has not enough atoms. Please remove it.",
	nconf + 1, line);
#endif /* THREEBODY */

    /* increase memory for this many additional atoms */
    atoms = (atom_t *)realloc(atoms, (natoms + count) * sizeof(atom_t));
    if (NULL == atoms)
      error(1, "Cannot allocate memory for atoms");
    for (i = 0; i < count; i++) {
      atoms[natoms + i].neigh = (neigh_t *)malloc(sizeof(neigh_t));
      reg_for_free(atoms[natoms + i].neigh, "test neigh");
    }
    coheng = (double *)realloc(coheng, (nconf + 1) * sizeof(double));
    if (NULL == coheng)
      error(1, "Cannot allocate memory for cohesive energy");
    conf_weight = (double *)realloc(conf_weight, (nconf + 1) * sizeof(double));
    if (NULL == conf_weight)
      error(1, "Cannot allocate memory for configuration weights");
    else
      conf_weight[nconf] = 1.0;
    volume = (double *)realloc(volume, (nconf + 1) * sizeof(double));
    if (NULL == volume)
      error(1, "Cannot allocate memory for volume");
#ifdef STRESS
    stress = (sym_tens *)realloc(stress, (nconf + 1) * sizeof(sym_tens));
    if (NULL == stress)
      error(1, "Cannot allocate memory for stress");
#endif /* STRESS */
    inconf = (int *)realloc(inconf, (nconf + 1) * sizeof(int));
    if (NULL == inconf)
      error(1, "Cannot allocate memory for atoms in conf");
    cnfstart = (int *)realloc(cnfstart, (nconf + 1) * sizeof(int));
    if (NULL == cnfstart)
      error(1, "Cannot allocate memory for start of conf");
    useforce = (int *)realloc(useforce, (nconf + 1) * sizeof(int));
    if (NULL == useforce)
      error(1, "Cannot allocate memory for useforce");
#ifdef STRESS
    usestress = (int *)realloc(usestress, (nconf + 1) * sizeof(int));
    if (NULL == usestress)
      error(1, "Cannot allocate memory for usestress");
#endif /* STRESS */
    na_type = (int **)realloc(na_type, (nconf + 2) * sizeof(int *));
    if (NULL == na_type)
      error(1, "Cannot allocate memory for na_type");
    na_type[nconf] = (int *)malloc(ntypes * sizeof(int));
    reg_for_free(na_type[nconf], "na_type[%d]", nconf);
    if (NULL == na_type[nconf])
      error(1, "Cannot allocate memory for na_type");

    for (i = natoms; i < natoms + count; i++)
      init_atom(atoms + i);

    for (i = 0; i < ntypes; i++)
      na_type[nconf][i] = 0;

    inconf[nconf] = count;
    cnfstart[nconf] = natoms;
    useforce[nconf] = use_force;
#ifdef STRESS
    stresses = stress + nconf;
#endif /* STRESS */
#ifdef CONTRIB
    have_contrib = 0;
    have_contrib_box = 0;
#endif /* CONTRIB */


/* added */
#ifdef KIM
    if (strcmp(NBC_method, "MI_OPBC_F") == 0 || strcmp(NBC_method, "MI_OPBC_H") == 0) {
      /* realloc memory for box_side_len */
      box_side_len = (double *)realloc(box_side_len, 3*(nconf+1)*sizeof(double));
      if(NULL == box_side_len) 
        error(1, "Cannot allocate memory for box_side_len");
    }
#endif /* KIM */
/* added ends */


    if (tag_format) {
      do {
	res = fgets(buffer, 1024, infile);
	if ((ptr = strchr(res, '\n')) != NULL)
	  *ptr = '\0';
	line++;

	/* read the box vectors: x */
	if (res[1] == 'X') {
	  if (sscanf(res + 3, "%lf %lf %lf\n", &box_x.x, &box_x.y, &box_x.z) == 3) {
	    h_boxx++;
	    if (global_cell_scale != 1.0) {
	      box_x.x *= global_cell_scale;
	      box_x.y *= global_cell_scale;
	      box_x.z *= global_cell_scale;
	    }
	  } else
	    error(1, "%s: Error in box vector x, line %d\n", filename, line);
	}

	/* read the box vectors: y */
	else if (res[1] == 'Y') {
	  if (sscanf(res + 3, "%lf %lf %lf\n", &box_y.x, &box_y.y, &box_y.z) == 3) {
	    h_boxy++;
	    if (global_cell_scale != 1.0) {
	      box_y.x *= global_cell_scale;
	      box_y.y *= global_cell_scale;
	      box_y.z *= global_cell_scale;
	    }
	  } else
	    error(1, "%s: Error in box vector y, line %d\n", filename, line);
	}

	/* read the box vectors: z */
	else if (res[1] == 'Z') {
	  if (sscanf(res + 3, "%lf %lf %lf\n", &box_z.x, &box_z.y, &box_z.z) == 3) {
	    h_boxz++;
	    if (global_cell_scale != 1.0) {
	      box_z.x *= global_cell_scale;
	      box_z.y *= global_cell_scale;
	      box_z.z *= global_cell_scale;
	    }
	  } else
	    error(1, "%s: Error in box vector z, line %d\n", filename, line);
	}
#ifdef CONTRIB
	/* box of contributing particles: origin */
	else if (strncmp(res + 1, "B_O", 3) == 0) {
	  if (1 == have_contrib_box) {
	    error(0, "There can only be one box of contributing atoms\n");
	    error(1, "This occured in %s on line %d", filename, line);
	  }
	  if (sscanf(res + 5, "%lf %lf %lf\n", &cbox_o.x, &cbox_o.y, &cbox_o.z) == 3) {
	    have_contrib_box = 1;
	    have_contrib++;
	    if (global_cell_scale != 1.0) {
	      cbox_o.x *= global_cell_scale;
	      cbox_o.y *= global_cell_scale;
	      cbox_o.z *= global_cell_scale;
	    }
	  } else
	    error(1, "%s: Error in box of contributing atoms, line %d\n", filename, line);
	}

	/* box of contributing particles: a */
	else if (strncmp(res + 1, "B_A", 3) == 0) {
	  if (sscanf(res + 5, "%lf %lf %lf\n", &cbox_a.x, &cbox_a.y, &cbox_a.z) == 3) {
	    have_contrib++;
	    if (global_cell_scale != 1.0) {
	      cbox_a.x *= global_cell_scale;
	      cbox_a.y *= global_cell_scale;
	      cbox_a.z *= global_cell_scale;
	    }
	  } else
	    error(1, "%s: Error in box of contributing atoms, line %d\n", filename, line);
	}

	/* box of contributing particles: b */
	else if (strncmp(res + 1, "B_B", 3) == 0) {
	  if (sscanf(res + 5, "%lf %lf %lf\n", &cbox_b.x, &cbox_b.y, &cbox_b.z) == 3) {
	    have_contrib++;
	    if (global_cell_scale != 1.0) {
	      cbox_b.x *= global_cell_scale;
	      cbox_b.y *= global_cell_scale;
	      cbox_b.z *= global_cell_scale;
	    }
	  } else
	    error(1, "%s: Error in box of contributing atoms, line %d\n", filename, line);
	}

	/* box of contributing particles: c */
	else if (strncmp(res + 1, "B_C", 3) == 0) {
	  if (sscanf(res + 5, "%lf %lf %lf\n", &cbox_c.x, &cbox_c.y, &cbox_c.z) == 3) {
	    have_contrib++;
	    if (global_cell_scale != 1.0) {
	      cbox_c.x *= global_cell_scale;
	      cbox_c.y *= global_cell_scale;
	      cbox_c.z *= global_cell_scale;
	    }
	  } else
	    error(1, "%s: Error in box of contributing atoms, line %d\n", filename, line);
	}

	/* sphere of contributing particles */
	else if (strncmp(res + 1, "B_S", 3) == 0) {
	  sphere_centers = (vector *)realloc(sphere_centers, (n_spheres + 1) * sizeof(vector));
	  r_spheres = (double *)realloc(r_spheres, (n_spheres + 1) * sizeof(double));
	  if (sscanf(res + 5, "%lf %lf %lf %lf\n", &sphere_centers[n_spheres].x,
	      &sphere_centers[n_spheres].y, &sphere_centers[n_spheres].z, &r_spheres[n_spheres]) == 4) {
	    n_spheres++;
	    if (global_cell_scale != 1.0) {
	      sphere_centers[n_spheres].x *= global_cell_scale;
	      sphere_centers[n_spheres].y *= global_cell_scale;
	      sphere_centers[n_spheres].z *= global_cell_scale;
	      r_spheres[n_spheres] *= global_cell_scale;
	    }
	  } else
	    error(1, "%s: Error in sphere of contributing atoms, line %d\n", filename, line);
	}
#endif /* CONTRIB */

	/* cohesive energy */
	else if (res[1] == 'E') {
	  if (sscanf(res + 3, "%lf\n", &(coheng[nconf])) == 1)
	    h_eng = 1;
	  else
	    error(1, "%s: Error in energy on line %d\n", filename, line);
	}

	/* configuration weight */
	else if (res[1] == 'W') {
	  if (sscanf(res + 3, "%lf\n", &(conf_weight[nconf])) != 1)
	    error(1, "%s: Error in configuration weight on line %d\n", filename, line);
	  if (conf_weight[nconf] < 0.0)
	    error(1, "%s: The configuration weight is negative on line %d\n", filename, line);
	}

	/* chemical elements */
	else if (res[1] == 'C') {
	  fgetpos(infile, &filepos);
	  if (!have_elements) {
	    i = 0;
	    for (j = 0; j < ntypes; j++) {
	      res_tmp = res + 3 + i;
	      if (strchr(res_tmp, ' ') != NULL && strlen(res_tmp) > 0) {
		tmp = strchr(res_tmp, ' ');
		str_len = tmp - res_tmp + 1;
		strncpy(elements[j], res_tmp, str_len - 1);
		elements[j][str_len - 1] = '\0';
		i += str_len;
	      } else if (strlen(res_tmp) >= 1) {
		if ((ptr = strchr(res_tmp, '\n')) != NULL)
		  *ptr = '\0';
		strcpy(elements[j], res_tmp);
		i += strlen(res_tmp);
		fixed_elements = j;
	      } else
		break;
	    }
	    have_elements = 1;
	  } else {
	    i = 0;
	    for (j = 0; j < ntypes; j++) {
	      res_tmp = res + 3 + i;
	      if (strchr(res_tmp, ' ') != NULL && strlen(res_tmp) > 0) {
		/* more than one element left */
		tmp = strchr(res_tmp, ' ');
		str_len = tmp - res_tmp + 1;
		strncpy(msg, res_tmp, str_len - 1);
		msg[str_len - 1] = '\0';
		i += str_len;
		if (strcmp(msg, elements[j]) != 0) {
		  if (atoi(elements[j]) == j && j > fixed_elements) {
		    strcpy(elements[j], msg);
		    fixed_elements++;
		  } else {
		    /* Fix newline at the end of a string */
		    if ((ptr = strchr(msg, '\n')) != NULL)
		      *ptr = '\0';
		    error(0, "Mismatch found in configuration %d, line %d.\n", nconf, line);
		    error(0, "Expected element >> %s << but found element >> %s <<.\n", elements[j], msg);
		    error(0, "You can use list_config to identify that configuration.\n");
		    error(1, "Please check your configuration files!\n");
		  }
		}
	      } else if (strlen(res_tmp) > 1) {
		strcpy(msg, res_tmp);
		if ((ptr = strchr(msg, '\n')) != NULL)
		  *ptr = '\0';
		i += strlen(msg);
		if (strcmp(msg, elements[j]) != 0) {
		  if (atoi(elements[j]) == j && j > fixed_elements) {
		    strcpy(elements[j], msg);
		    fixed_elements++;
		  } else {
		    /* Fix newline at the end of a string */
		    if ((ptr = strchr(msg, '\n')) != NULL)
		      *ptr = '\0';
		    error(0, "Mismatch found in configuration %d on line %d.\n", nconf, line);
		    error(0, "Expected element >> %s << but found element >> %s <<.\n", elements[j], msg);
		    error(0, "You can use list_config to identify that configuration.\n");
		    error(1, "Please check your configuration files!");
		  }
		}
	      } else
		break;
	    }
	  }
	  fsetpos(infile, &filepos);
	}
#ifdef STRESS
	/* read stress */
	else if (res[1] == 'S') {
	  if (sscanf(res + 3, "%lf %lf %lf %lf %lf %lf\n", &(stresses->xx),
	      &(stresses->yy), &(stresses->zz), &(stresses->xy), &(stresses->yz), &(stresses->zx)) == 6)
	    h_stress = 1;
	  else
	    error(1, "Error in stress tensor on line %d\n", line);
	}
#endif /* STRESS */
	else if (res[1] != '#' && res[1] != 'F') {
	  warning("Unknown header line in %s detected:\n", filename);
	  warning("Line %d : %s\n", line, res);
	}

      } while (res[1] != 'F');
      if (0 == h_eng)
	error(1, "%s: missing energy in configuration %d!", filename, nconf);
      if (!(h_boxx && h_boxy && h_boxz))
	error(1, "Incomplete box vectors for config %d!", nconf);
#ifdef CONTRIB
      if (have_contrib_box && have_contrib != 4)
	error(1, "Incomplete box of contributing atoms for config %d!", nconf);
#endif /* CONTRIB */
#ifdef STRESS
      usestress[nconf] = h_stress;	/* no stress tensor available */
#endif /* STRESS */
    } else {
      /* read the box vectors */
      fscanf(infile, "%lf %lf %lf\n", &box_x.x, &box_x.y, &box_x.z);
      fscanf(infile, "%lf %lf %lf\n", &box_y.x, &box_y.y, &box_y.z);
      fscanf(infile, "%lf %lf %lf\n", &box_z.x, &box_z.y, &box_z.z);
      line += 3;
      if (global_cell_scale != 1.0) {
	box_x.x *= global_cell_scale;
	box_x.y *= global_cell_scale;
	box_x.z *= global_cell_scale;
	box_y.x *= global_cell_scale;
	box_y.y *= global_cell_scale;
	box_y.z *= global_cell_scale;
	box_z.x *= global_cell_scale;
	box_z.y *= global_cell_scale;
	box_z.z *= global_cell_scale;
      }

      /* read cohesive energy */
      if (1 != fscanf(infile, "%lf\n", &(coheng[nconf])))
	error(1, "Configuration file without cohesive energy -- old format!");
      line++;

#ifdef STRESS
      /* read stress tensor */
      if (6 != fscanf(infile, "%lf %lf %lf %lf %lf %lf\n", &(stresses->xx),
	  &(stresses->yy), &(stresses->zz), &(stresses->xy), &(stresses->yz), &(stresses->zx)))
	error(1, "No stresses given -- old format");
      usestress[nconf] = 1;
      line++;
#endif /* STRESS */
    }

#ifdef STRESS
    if (usestress[nconf])
      w_stress++;
#endif /* STRESS */
    if (useforce[nconf])
      w_force++;

    volume[nconf] = make_box();


/* added */
#ifdef KIM 
    if (strcmp(NBC_method, "MI_OPBC_F") == 0 || strcmp(NBC_method, "MI_OPBC_H") == 0) {
      double small_value = 1e-8;
      if(   box_x.y > small_value || box_x.z > small_value 
          || box_y.z > small_value || box_y.x > small_value 
          || box_z.x > small_value || box_z.y > small_value){
        error(1,"KIM: simulation box of configuration %d is not orthogonal. Try to use 'NEIGH_RVEC' "
                "instead of 'MI_OPBC'.\n", nconf);

      } else {
        /* store the box size info in box_side_len */
        box_side_len[3*nconf + 0] = box_x.x;
        box_side_len[3*nconf + 1] = box_y.y;
        box_side_len[3*nconf + 2] = box_z.z;
      }
    }
#endif 
/* added ends */

    /* read the atoms */
    for (i = 0; i < count; i++) {
      atom = atoms + natoms + i;
      if (7 > fscanf(infile, "%d %lf %lf %lf %lf %lf %lf\n", &(atom->type),
	  &(atom->pos.x), &(atom->pos.y), &(atom->pos.z), &(atom->force.x), &(atom->force.y),
	  &(atom->force.z)))
	error(1, "Corrupt configuration file on line %d\n", line + 1);
      line++;
      if (global_cell_scale != 1.0) {
	atom->pos.x *= global_cell_scale;
	atom->pos.y *= global_cell_scale;
	atom->pos.z *= global_cell_scale;
      }
      if (atom->type >= ntypes || atom->type < 0)
	error(1, "Corrupt configuration file on line %d: Incorrect atom type (%d)\n", line, atom->type);
      atom->absforce = sqrt(dsquare(atom->force.x) + dsquare(atom->force.y) + dsquare(atom->force.z));
      atom->conf = nconf;
#ifdef CONTRIB
      if (have_contrib_box || n_spheres != 0)
	atom->contrib = does_contribute(atom->pos);
      else
	atom->contrib = 1;
#endif
      na_type[nconf][atom->type] += 1;
      max_type = MAX(max_type, atom->type);
    }

    /* check cell size */
    /* inverse height in direction */
    iheight.x = sqrt(SPROD(tbox_x, tbox_x));
    iheight.y = sqrt(SPROD(tbox_y, tbox_y));
    iheight.z = sqrt(SPROD(tbox_z, tbox_z));

    if ((ceil(rcutmax * iheight.x) > 30000)
      || (ceil(rcutmax * iheight.y) > 30000)
      || (ceil(rcutmax * iheight.z) > 30000))
      error(1, "Very bizarre small cell size - aborting");

    cell_scale[0] = (int)ceil(rcutmax * iheight.x);
    cell_scale[1] = (int)ceil(rcutmax * iheight.y);
    cell_scale[2] = (int)ceil(rcutmax * iheight.z);

    if (cell_scale[0] > 1 || cell_scale[1] > 1 || cell_scale[2] > 1)
      have_small_box = 1;

#ifdef DEBUG
    fprintf(stderr, "\nChecking cell size for configuration %d:\n", nconf);
    fprintf(stderr, "Box dimensions:\n");
    fprintf(stderr, "     %10.6f %10.6f %10.6f\n", box_x.x, box_x.y, box_x.z);
    fprintf(stderr, "     %10.6f %10.6f %10.6f\n", box_y.x, box_y.y, box_y.z);
    fprintf(stderr, "     %10.6f %10.6f %10.6f\n", box_z.x, box_z.y, box_z.z);
    fprintf(stderr, "Box normals:\n");
    fprintf(stderr, "     %10.6f %10.6f %10.6f\n", tbox_x.x, tbox_x.y, tbox_x.z);
    fprintf(stderr, "     %10.6f %10.6f %10.6f\n", tbox_y.x, tbox_y.y, tbox_y.z);
    fprintf(stderr, "     %10.6f %10.6f %10.6f\n", tbox_z.x, tbox_z.y, tbox_z.z);
    fprintf(stderr, "Box heights:\n");
    fprintf(stderr, "     %10.6f %10.6f %10.6f\n", 1.0 / iheight.x, 1.0 / iheight.y, 1.0 / iheight.z);
    fprintf(stderr, "Potential range:  %f\n", rcutmax);
    fprintf(stderr, "Periodic images needed: %d %d %d\n\n",
      2 * cell_scale[0] + 1, 2 * cell_scale[1] + 1, 2 * cell_scale[2] + 1);
#endif /* DEBUG */

    /* compute the neighbor table */
    for (i = natoms; i < natoms + count; i++) {
      atoms[i].num_neigh = 0;

/* added */
#ifndef KIM       
      /* loop over all atoms for threebody interactions */
#ifdef THREEBODY
      for (j = natoms; j < natoms + count; j++) {
#else
      for (j = i; j < natoms + count; j++) {
#endif /* THREEBODY */

#else /* !KIM */
      int j_start; 
      if (is_half_neighbors == 1) {  /* half neighbor list */
        j_start = i;
      } else {                       /* full neighbor list */
        j_start = natoms;
      }

      for (j = j_start; j < natoms + count; j++) {
#endif /* !KIM */
/* added ends */


	d.x = atoms[j].pos.x - atoms[i].pos.x;
	d.y = atoms[j].pos.y - atoms[i].pos.y;
	d.z = atoms[j].pos.z - atoms[i].pos.z;
	for (ix = -cell_scale[0]; ix <= cell_scale[0]; ix++) {
	  for (iy = -cell_scale[1]; iy <= cell_scale[1]; iy++) {
	    for (iz = -cell_scale[2]; iz <= cell_scale[2]; iz++) {
	      if ((i == j) && (ix == 0) && (iy == 0) && (iz == 0))
		continue;
	      dd.x = d.x + ix * box_x.x + iy * box_y.x + iz * box_z.x;
	      dd.y = d.y + ix * box_x.y + iy * box_y.y + iz * box_z.y;
	      dd.z = d.z + ix * box_x.z + iy * box_y.z + iz * box_z.z;
	      r = sqrt(SPROD(dd, dd));
	      type1 = atoms[i].type;
	      type2 = atoms[j].type;
	      if (r <= rcut[type1 * ntypes + type2]) {
		if (r <= rmin[type1 * ntypes + type2]) {
		  sh_dist = nconf;
		  fprintf(stderr, "Configuration %d: Distance %f\n", nconf, r);
		  fprintf(stderr, "atom %d (type %d) at pos: %f %f %f\n",
		    i - natoms, type1, atoms[i].pos.x, atoms[i].pos.y, atoms[i].pos.z);
		  fprintf(stderr, "atom %d (type %d) at pos: %f %f %f\n", j - natoms, type2, dd.x, dd.y,
		    dd.z);
		}
		atoms[i].neigh =
		  (neigh_t *)realloc(atoms[i].neigh, (atoms[i].num_neigh + 1) * sizeof(neigh_t));
		dd.x /= r;
		dd.y /= r;
		dd.z /= r;
		k = atoms[i].num_neigh++;
		init_neigh(atoms[i].neigh + k);
		atoms[i].neigh[k].type = type2;
		atoms[i].neigh[k].nr = j;
		atoms[i].neigh[k].r = r;
		atoms[i].neigh[k].r2 = r * r;
		atoms[i].neigh[k].inv_r = 1.0 / r;
		atoms[i].neigh[k].dist_r = dd;
		atoms[i].neigh[k].dist.x = dd.x * r;
		atoms[i].neigh[k].dist.y = dd.y * r;
		atoms[i].neigh[k].dist.z = dd.z * r;


/* added */ 
#ifndef KIM 

#ifdef ADP
		atoms[i].neigh[k].sqrdist.xx = dd.x * dd.x * r * r;
		atoms[i].neigh[k].sqrdist.yy = dd.y * dd.y * r * r;
		atoms[i].neigh[k].sqrdist.zz = dd.z * dd.z * r * r;
		atoms[i].neigh[k].sqrdist.yz = dd.y * dd.z * r * r;
		atoms[i].neigh[k].sqrdist.zx = dd.z * dd.x * r * r;
		atoms[i].neigh[k].sqrdist.xy = dd.x * dd.y * r * r;
#endif /* ADP */

		col = (type1 <= type2) ? type1 * ntypes + type2 - ((type1 * (type1 + 1)) / 2)
		  : type2 * ntypes + type1 - ((type2 * (type2 + 1)) / 2);
		atoms[i].neigh[k].col[0] = col;
		mindist[col] = MIN(mindist[col], r);

		/* pre-compute index and shift into potential table */

		/* pair potential */
		if (!sh_dist) {
		  if (format == 0 || format == 3) {
		    rr = r - calc_pot.begin[col];
		    if (rr < 0) {
		      fprintf(stderr, "The distance %f is smaller than the beginning\n", r);
		      fprintf(stderr, "of the potential #%d (r_begin=%f).\n", col, calc_pot.begin[col]);
		      fflush(stdout);
		      error(1, "Short distance!");
		    }
		    istep = calc_pot.invstep[col];
		    slot = (int)(rr * istep);
		    shift = (rr - slot * calc_pot.step[col]) * istep;
		    slot += calc_pot.first[col];
		    step = calc_pot.step[col];
		  } else {	/* format == 4 ! */
		    klo = calc_pot.first[col];
		    khi = calc_pot.last[col];
		    /* bisection */
		    while (khi - klo > 1) {
		      slot = (khi + klo) >> 1;
		      if (calc_pot.xcoord[slot] > r)
			khi = slot;
		      else
			klo = slot;
		    }
		    slot = klo;
		    step = calc_pot.xcoord[khi] - calc_pot.xcoord[klo];
		    shift = (r - calc_pot.xcoord[klo]) / step;

		  }
		  /* independent of format - we should be left of last index */
		  if (slot >= calc_pot.last[col]) {
		    slot--;
		    shift += 1.0;
		  }
		  atoms[i].neigh[k].shift[0] = shift;
		  atoms[i].neigh[k].slot[0] = slot;
		  atoms[i].neigh[k].step[0] = step;

#if defined EAM || defined ADP || defined MEAM
		  /* transfer function */
		  col = paircol + type2;
		  atoms[i].neigh[k].col[1] = col;
		  if (format == 0 || format == 3) {
		    rr = r - calc_pot.begin[col];
		    if (rr < 0) {
		      fprintf(stderr, "The distance %f is smaller than the beginning\n", r);
		      fprintf(stderr, "of the potential #%d (r_begin=%f).\n", col, calc_pot.begin[col]);
		      fflush(stdout);
		      error(1, "short distance in config.c!");
		    }
		    istep = calc_pot.invstep[col];
		    slot = (int)(rr * istep);
		    shift = (rr - slot * calc_pot.step[col]) * istep;
		    slot += calc_pot.first[col];
		    step = calc_pot.step[col];
		  } else {	/* format == 4 ! */
		    klo = calc_pot.first[col];
		    khi = calc_pot.last[col];
		    /* bisection */
		    while (khi - klo > 1) {
		      slot = (khi + klo) >> 1;
		      if (calc_pot.xcoord[slot] > r)
			khi = slot;
		      else
			klo = slot;
		    }
		    slot = klo;
		    step = calc_pot.xcoord[khi] - calc_pot.xcoord[klo];
		    shift = (r - calc_pot.xcoord[klo]) / step;

		  }
		  /* Check if we are at the last index */
		  if (slot >= calc_pot.last[col]) {
		    slot--;
		    shift += 1.0;
		  }
		  atoms[i].neigh[k].shift[1] = shift;
		  atoms[i].neigh[k].slot[1] = slot;
		  atoms[i].neigh[k].step[1] = step;

#ifdef TBEAM
		  /* transfer function - d band */
		  col = paircol + 2 * ntypes + type2;
		  atoms[i].neigh[k].col[2] = col;
		  if (format == 0 || format == 3) {
		    rr = r - calc_pot.begin[col];
		    if (rr < 0) {
		      fprintf(stderr, "The distance %f is smaller than the beginning\n", r);
		      fprintf(stderr, "of the potential #%d (r_begin=%f).\n", col, calc_pot.begin[col]);
		      fflush(stdout);
		      error(1, "short distance in config.c!");
		    }
		    istep = calc_pot.invstep[col];
		    slot = (int)(rr * istep);
		    shift = (rr - slot * calc_pot.step[col]) * istep;
		    slot += calc_pot.first[col];
		    step = calc_pot.step[col];
		  } else {	/* format == 4 ! */
		    klo = calc_pot.first[col];
		    khi = calc_pot.last[col];
		    /* bisection */
		    while (khi - klo > 1) {
		      slot = (khi + klo) >> 1;
		      if (calc_pot.xcoord[slot] > r)
			khi = slot;
		      else
			klo = slot;
		    }
		    slot = klo;
		    step = calc_pot.xcoord[khi] - calc_pot.xcoord[klo];
		    shift = (r - calc_pot.xcoord[klo]) / step;

		  }
		  /* Check if we are at the last index */
		  if (slot >= calc_pot.last[col]) {
		    slot--;
		    shift += 1.0;
		  }
		  atoms[i].neigh[k].shift[2] = shift;
		  atoms[i].neigh[k].slot[2] = slot;
		  atoms[i].neigh[k].step[2] = step;
#endif /* TBEAM */

#endif /* EAM || ADP || MEAM */

#ifdef MEAM
		  /* Store slots and stuff for f(r_ij) */
		  col = paircol + 2 * ntypes + atoms[i].neigh[k].col[0];
		  atoms[i].neigh[k].col[2] = col;
		  if (0 == format || 3 == format) {
		    rr = r - calc_pot.begin[col];
		    if (rr < 0) {
		      fprintf(stderr, "The distance %f is smaller than the beginning\n", r);
		      fprintf(stderr, "of the potential #%d (r_begin=%f).\n", col, calc_pot.begin[col]);
		      fflush(stdout);
		      error(1, "short distance in config.c!");
		    }
		    istep = calc_pot.invstep[col];
		    slot = (int)(rr * istep);
		    shift = (rr - slot * calc_pot.step[col]) * istep;
		    slot += calc_pot.first[col];
		    step = calc_pot.step[col];
		  } else {	/* format == 4 ! */
		    klo = calc_pot.first[col];
		    khi = calc_pot.last[col];
		    /* bisection */
		    while (khi - klo > 1) {
		      slot = (khi + klo) >> 1;
		      if (calc_pot.xcoord[slot] > r)
			khi = slot;
		      else
			klo = slot;
		    }
		    slot = klo;
		    step = calc_pot.xcoord[khi] - calc_pot.xcoord[klo];
		    shift = (r - calc_pot.xcoord[klo]) / step;

		  }
		  /* Check if we are at the last index */
		  if (slot >= calc_pot.last[col]) {
		    slot--;
		    shift += 1.0;
		  }
		  atoms[i].neigh[k].shift[2] = shift;
		  atoms[i].neigh[k].slot[2] = slot;
		  atoms[i].neigh[k].step[2] = step;
#endif /* MEAM */

#ifdef ADP
		  /* dipole part */
		  col = paircol + 2 * ntypes + atoms[i].neigh[k].col[0];
		  atoms[i].neigh[k].col[2] = col;
		  if (format == 0 || format == 3) {
		    rr = r - calc_pot.begin[col];
		    if (rr < 0) {
		      fprintf(stderr, "The distance %f is smaller than the beginning\n", r);
		      fprintf(stderr, "of the potential #%d (r_begin=%f).\n", col, calc_pot.begin[col]);
		      fflush(stdout);
		      error(1, "short distance in config.c!");
		    }
		    istep = calc_pot.invstep[col];
		    slot = (int)(rr * istep);
		    shift = (rr - slot * calc_pot.step[col]) * istep;
		    slot += calc_pot.first[col];
		    step = calc_pot.step[col];
		  } else {	/* format == 4 ! */
		    klo = calc_pot.first[col];
		    khi = calc_pot.last[col];
		    /* bisection */
		    while (khi - klo > 1) {
		      slot = (khi + klo) >> 1;
		      if (calc_pot.xcoord[slot] > r)
			khi = slot;
		      else
			klo = slot;
		    }
		    slot = klo;
		    step = calc_pot.xcoord[khi] - calc_pot.xcoord[klo];
		    shift = (r - calc_pot.xcoord[klo]) / step;

		  }
		  /* Check if we are at the last index */
		  if (slot >= calc_pot.last[col]) {
		    slot--;
		    shift += 1.0;
		  }
		  atoms[i].neigh[k].shift[2] = shift;
		  atoms[i].neigh[k].slot[2] = slot;
		  atoms[i].neigh[k].step[2] = step;

		  /* quadrupole part */
		  col = 2 * paircol + 2 * ntypes + atoms[i].neigh[k].col[0];
		  atoms[i].neigh[k].col[3] = col;
		  if (format == 0 || format == 3) {
		    rr = r - calc_pot.begin[col];
		    if (rr < 0) {
		      fprintf(stderr, "The distance %f is smaller than the beginning\n", r);
		      fprintf(stderr, "of the potential #%d (r_begin=%f).\n", col, calc_pot.begin[col]);
		      fflush(stdout);
		      error(1, "short distance in config.c!");
		    }
		    istep = calc_pot.invstep[col];
		    slot = (int)(rr * istep);
		    shift = (rr - slot * calc_pot.step[col]) * istep;
		    slot += calc_pot.first[col];
		    step = calc_pot.step[col];
		  } else {	/* format == 4 ! */
		    klo = calc_pot.first[col];
		    khi = calc_pot.last[col];
		    /* bisection */
		    while (khi - klo > 1) {
		      slot = (khi + klo) >> 1;
		      if (calc_pot.xcoord[slot] > r)
			khi = slot;
		      else
			klo = slot;
		    }
		    slot = klo;
		    step = calc_pot.xcoord[khi] - calc_pot.xcoord[klo];
		    shift = (r - calc_pot.xcoord[klo]) / step;

		  }
		  /* Check if we are at the last index */
		  if (slot >= calc_pot.last[col]) {
		    slot--;
		    shift += 1.0;
		  }
		  atoms[i].neigh[k].shift[3] = shift;
		  atoms[i].neigh[k].slot[3] = slot;
		  atoms[i].neigh[k].step[3] = step;
#endif /* ADP */

#ifdef STIWEB
		  /* Store slots and stuff for exp. function */
		  col = paircol + atoms[i].neigh[k].col[0];
		  atoms[i].neigh[k].col[1] = col;
		  if (0 == format || 3 == format) {
		    rr = r - calc_pot.begin[col];
		    if (rr < 0) {
		      fprintf(stderr, "The distance %f is smaller than the beginning\n", r);
		      fprintf(stderr, "of the potential #%d (r_begin=%f).\n", col, calc_pot.begin[col]);
		      fflush(stdout);
		      error(1, "short distance in config.c!");
		    }
		    istep = calc_pot.invstep[col];
		    slot = (int)(rr * istep);
		    shift = (rr - slot * calc_pot.step[col]) * istep;
		    slot += calc_pot.first[col];
		    step = calc_pot.step[col];
		  } else {	/* format == 4 ! */
		    klo = calc_pot.first[col];
		    khi = calc_pot.last[col];
		    /* bisection */
		    while (khi - klo > 1) {
		      slot = (khi + klo) >> 1;
		      if (calc_pot.xcoord[slot] > r)
			khi = slot;
		      else
			klo = slot;
		    }
		    slot = klo;
		    step = calc_pot.xcoord[khi] - calc_pot.xcoord[klo];
		    shift = (r - calc_pot.xcoord[klo]) / step;

		  }
		  /* Check if we are at the last index */
		  if (slot >= calc_pot.last[col]) {
		    slot--;
		    shift += 1.0;
		  }
		  atoms[i].neigh[k].shift[1] = shift;
		  atoms[i].neigh[k].slot[1] = slot;
		  atoms[i].neigh[k].step[1] = step;
#endif /* STIWEB */

		}		/* !sh_dist */

#endif /* !KIM */
/* added ends */

	      }			/* r < r_cut */
	    }			/* loop over images in z direction */
	  }			/* loop over images in y direction */
	}			/* loop over images in x direction */
      }				/* second loop over atoms (neighbors) */

      reg_for_free(atoms[i].neigh, "neighbor table atom %d", i);
    }				/* first loop over atoms */

/* added */
#ifndef KIM 

    /* compute the angular part */
    /* For TERSOFF we create a full neighbor list, for all other potentials only a half list */
#ifdef THREEBODY
    for (i = natoms; i < natoms + count; i++) {
      nnn = atoms[i].num_neigh;
      ijk = 0;
      atoms[i].angle_part = (angle_t *) malloc(sizeof(angle_t));
#ifdef TERSOFF
      for (j = 0; j < nnn; j++) {
#else
      for (j = 0; j < nnn - 1; j++) {
#endif /* TERSOFF */
	atoms[i].neigh[j].ijk_start = ijk;
#ifdef TERSOFF
	for (k = 0; k < nnn; k++) {
	  if (j == k)
	    continue;
#else
	for (k = j + 1; k < nnn; k++) {
#endif /* TERSOFF */
	  atoms[i].angle_part = (angle_t *) realloc(atoms[i].angle_part, (ijk + 1) * sizeof(angle_t));
	  init_angle(atoms[i].angle_part + ijk);
	  ccos =
	    atoms[i].neigh[j].dist_r.x * atoms[i].neigh[k].dist_r.x +
	    atoms[i].neigh[j].dist_r.y * atoms[i].neigh[k].dist_r.y +
	    atoms[i].neigh[j].dist_r.z * atoms[i].neigh[k].dist_r.z;

	  atoms[i].angle_part[ijk].cos = ccos;

	  col = 2 * paircol + 2 * ntypes + atoms[i].type;
	  if (0 == format || 3 == format) {
	    if ((fabs(ccos) - 1.0) > 1e-10) {
	      printf("%.20f %f %d %d %d\n", ccos, calc_pot.begin[col], col, type1, type2);
	      fflush(stdout);
	      error(1, "cos out of range, it is strange!");
	    }
#ifdef MEAM
	    istep = calc_pot.invstep[col];
	    slot = (int)((ccos + 1) * istep);
	    shift = ((ccos + 1) - slot * calc_pot.step[col]) * istep;
	    slot += calc_pot.first[col];
	    step = calc_pot.step[col];

	    /* Don't want lower bound spline knot to be final knot or upper
	       bound knot will cause trouble since it goes beyond the array */
	    if (slot >= calc_pot.last[col]) {
	      slot--;
	      shift += 1.0;
	    }
#endif /* !MEAM */
	  }
#ifdef MEAM
	  atoms[i].angle_part[ijk].shift = shift;
	  atoms[i].angle_part[ijk].slot = slot;
	  atoms[i].angle_part[ijk].step = step;
#endif /* MEAM */
	  ijk++;
	}			/* third loop over atoms */
      }				/* second loop over atoms */
      atoms[i].num_angles = ijk;
      reg_for_free(atoms[i].angle_part, "angular part atom %d", i);
    }				/* first loop over atoms */
#endif /* THREEBODY */

#endif /* !KIM */
/* added ends */

    /* increment natoms and configuration number */
    natoms += count;
    nconf++;

  } while (!feof(infile));

  /* close config file */
  fclose(infile);

  /* the calculation of the neighbor lists is now complete */
  printf("done\n");

  /* calculate the total number of the atom types */
  na_type = (int **)realloc(na_type, (nconf + 1) * sizeof(int *));
  reg_for_free(na_type, "na_type");
  if (NULL == na_type)
    error(1, "Cannot allocate memory for na_type");
  na_type[nconf] = (int *)malloc(ntypes * sizeof(int));
  reg_for_free(na_type[nconf], "na_type[%d]", nconf);
  for (i = 0; i < ntypes; i++)
    na_type[nconf][i] = 0;

  for (i = 0; i < nconf; i++)
    for (j = 0; j < ntypes; j++)
      na_type[nconf][j] += na_type[i][j];

  /* print diagnostic message */
  printf("\nRead %d configurations (%d with forces, %d with stresses)\n", nconf, w_force, w_stress);
  printf("with a total of %d atoms (", natoms);


/* added */
/* elements info has to be given */
#ifdef KIM 
  if(!have_elements) {
    error(1, "Elements info cannot be found. You need to include the info beginning "
    "with '#C' in file: %s.", filename);
  }
#endif /* KIM */
/*added ends*/


  for (i = 0; i < ntypes; i++) {
    if (have_elements)
      printf("%d %s (%.2f%%)", na_type[nconf][i], elements[i], 100.0 * na_type[nconf][i] / natoms);
    else
      printf("%d type %d (%.2f%%)", na_type[nconf][i], i, 100.0 * na_type[nconf][i] / natoms);
    if (i != (ntypes - 1))
      printf(", ");
  }
  printf(").\n");

  /* be pedantic about too large ntypes */
  if ((max_type + 1) < ntypes) {
    error(0, "There are less than %d atom types in your configurations!\n", ntypes);
    error(1, "Please adjust \"ntypes\" in your parameter file.", ntypes);
  }

  if (0 != have_small_box) {
    warning("The box size of at least one configuration is smaller than the cutoff distance.\n");
    warning("Using additional periodic images for energy and force calculations.\n");
  }

  reg_for_free(atoms, "atoms");
  reg_for_free(coheng, "coheng");
  reg_for_free(conf_weight, "conf_weight");
  reg_for_free(volume, "volume");
#ifdef STRESS
  reg_for_free(stress, "stress");
#endif /* STRESS */
  reg_for_free(inconf, "inconf");
  reg_for_free(cnfstart, "cnfstart");
  reg_for_free(useforce, "useforce");
#ifdef STRESS
  reg_for_free(usestress, "usestress");
#endif /* STRESS */
#ifdef CONTRIB
  if (n_spheres > 0) {
    reg_for_free(r_spheres, "sphere radii");
    reg_for_free(sphere_centers, "sphere centers");
  }
#endif /* CONTRIB */

/* added */
#ifdef KIM
  if (strcmp(NBC_method, "MI_OPBC_F") == 0 || strcmp(NBC_method, "MI_OPBC_H") == 0) {
  	reg_for_free(box_side_len, "box side lengths");
  }
#endif 
/* added ends*/



  /* mdim is the dimension of the force vector:
     - 3*natoms forces
     - nconf cohesive energies,
     - 6*nconf stress tensor components */
  mdim = 3 * natoms + nconf;
#ifdef STRESS
  mdim += 6 * nconf;
#endif /* STRESS */

  /* mdim has additional components for EAM-like potentials */
#if defined EAM || defined ADP || defined MEAM
  mdim += nconf;		/* nconf limiting constraints */
  mdim += 2 * ntypes;		/* ntypes dummy constraints */
#ifdef TBEAM
  mdim += 2 * ntypes;		/* additional dummy constraints for s-band */
#endif /* TBEAM */
#endif /* EAM || ADP || MEAM */

  /* mdim has additional components for analytic potentials */
#ifdef APOT
  /* 1 slot for each analytic parameter -> punishment */
  mdim += opt_pot.idxlen;
  /* 1 slot for each analytic potential -> punishment */
  mdim += apot_table.number + 1;
#endif /* APOT */

  /* copy forces into single vector */
  if (NULL == (force_0 = (double *)malloc(mdim * sizeof(double))))
    error(1, "Cannot allocate force vector");
  reg_for_free(force_0, "force_0");

  k = 0;
  for (i = 0; i < natoms; i++) {	/* first forces */
    force_0[k++] = atoms[i].force.x;
    force_0[k++] = atoms[i].force.y;
    force_0[k++] = atoms[i].force.z;
  }
  for (i = 0; i < nconf; i++) {	/* then cohesive energies */
    force_0[k++] = coheng[i];
  }

#ifdef STRESS
  for (i = 0; i < nconf; i++) {	/* then stresses */
    if (usestress[i]) {
      force_0[k++] = stress[i].xx;
      force_0[k++] = stress[i].yy;
      force_0[k++] = stress[i].zz;
      force_0[k++] = stress[i].xy;
      force_0[k++] = stress[i].yz;
      force_0[k++] = stress[i].zx;
    } else {
      for (j = 0; j < 6; j++)
	force_0[k++] = 0.0;
    }
  }
#endif /* STRESS */

#if defined EAM || defined ADP || defined MEAM
  for (i = 0; i < nconf; i++)
    force_0[k++] = 0.0;		/* punishment rho out of bounds */
  for (i = 0; i < 2 * ntypes; i++)
    force_0[k++] = 0.0;		/* constraint on U(n=0):=0 */
#ifdef TBEAM
  for (i = 0; i < 2 * ntypes; i++)
    force_0[k++] = 0.0;		/* constraint on U(n=0):=0 for s-band */
#endif /* TBEAM */
#endif /* EAM || ADP || MEAM */

#ifdef APOT
  for (i = 0; i < opt_pot.idxlen; i++)
    force_0[k++] = 0.0;		/* punishment for individual parameters */
  for (i = 0; i <= apot_table.number; i++)
    force_0[k++] = 0.0;		/* punishment for potential functions */
#endif /* APOT */

  /* write pair distribution file */
  if (1 == write_pair) {
    char  pairname[255];
    FILE *pairfile;
#ifdef APOT
    int   pair_steps = APOT_STEPS / 2;
#else
    int   pair_steps = 500;
#endif /* APOT */
    double pair_table[paircol * pair_steps];
    double pair_dist[paircol];
    int   pos, max_count = 0;

    strcpy(pairname, config);
    strcat(pairname, ".pair");
    pairfile = fopen(pairname, "w");
    fprintf(pairfile, "# radial distribution file for %d potential(s)\n", paircol);

    for (i = 0; i < paircol * pair_steps; i++)
      pair_table[i] = 0.0;

    for (i = 0; i < ntypes; i++)
      for (k = 0; k < ntypes; k++)
	pair_dist[(i <= k) ? i * ntypes + k - (i * (i + 1) / 2) : k * ntypes + i - (k * (k + 1) / 2)] =
	  rcut[i * ntypes + k] / pair_steps;

    for (k = 0; k < paircol; k++) {
      for (i = 0; i < natoms; i++) {
	type1 = atoms[i].type;
	for (j = 0; j < atoms[i].num_neigh; j++) {
	  type2 = atoms[i].neigh[j].type;
	  col = atoms[i].neigh[j].col[0];
	  if (col == k) {
	    pos = (int)(atoms[i].neigh[j].r / pair_dist[k]);
#ifdef DEBUG
	    if (atoms[i].neigh[j].r <= 1) {
	      fprintf(stderr, "Short distance (%f) found.\n", atoms[i].neigh[j].r);
	      fprintf(stderr, "\tatom=%d neighbor=%d\n", i, j);
	    }
#endif /* DEBUG */
	    pair_table[k * pair_steps + pos]++;
	    if ((int)pair_table[k * pair_steps + pos] > max_count)
	      max_count = (int)pair_table[k * pair_steps + pos];
	  }
	}
      }
    }

    for (k = 0; k < paircol; k++) {
      for (i = 0; i < pair_steps; i++) {
	pair_table[k * pair_steps + i] /= max_count;
	fprintf(pairfile, "%f %f\n", i * pair_dist[k], pair_table[k * pair_steps + i]);
      }
      if (k != (paircol - 1))
	fprintf(pairfile, "\n\n");
    }
    fclose(pairfile);
  }

/* added */
#ifndef KIM

  /* assign correct distances to different tables */
#ifdef APOT
  double min = 10.0;

  /* pair potentials */
  for (i = 0; i < ntypes; i++) {
    for (j = 0; j < ntypes; j++) {
      k = (i <= j) ? i * ntypes + j - ((i * (i + 1)) / 2) : j * ntypes + i - ((j * (j + 1)) / 2);
      if (mindist[k] >= 99.9)
	mindist[k] = 2.5;
      rmin[i * ntypes + j] = mindist[k];
      apot_table.begin[k] = mindist[k] * 0.95;
      opt_pot.begin[k] = mindist[k] * 0.95;
      calc_pot.begin[k] = mindist[k] * 0.95;
      min = MIN(min, mindist[k]);
    }
  }

  /* transfer functions */
#if defined EAM || defined ADP || defined MEAM
  for (i = paircol; i < paircol + ntypes; i++) {
    apot_table.begin[i] = min * 0.95;
    opt_pot.begin[i] = min * 0.95;
    calc_pot.begin[i] = min * 0.95;
  }
#ifdef TBEAM
  for (i = paircol + 2 * ntypes; i < paircol + 3 * ntypes; i++) {
    apot_table.begin[i] = min * 0.95;
    opt_pot.begin[i] = min * 0.95;
    calc_pot.begin[i] = min * 0.95;
  }
#endif /* TBEAM */
#endif /* EAM || ADP || MEAM */

  /* dipole and quadrupole functions */
#ifdef ADP
  for (i = 0; i < paircol; i++) {
    j = paircol + 2 * ntypes + i;
    apot_table.begin[j] = min * 0.95;
    opt_pot.begin[j] = min * 0.95;
    calc_pot.begin[j] = min * 0.95;
    j = 2 * paircol + 2 * ntypes + i;
    apot_table.begin[j] = min * 0.95;
    opt_pot.begin[j] = min * 0.95;
    calc_pot.begin[j] = min * 0.95;
  }
#endif /* ADP */

#ifdef MEAM
  /* f_ij */
  for (i = 0; i < paircol; i++) {
    j = paircol + 2 * ntypes + i;
    apot_table.begin[j] = min * 0.95;
    opt_pot.begin[j] = min * 0.95;
    calc_pot.begin[j] = min * 0.95;
  }
  /* g_i */
  /* g_i takes cos(theta) as an argument, so we need to tabulate it only
     in the range of [-1:1]. Actually we use [-1.1:1.1] to be safe. */
  for (i = 0; i < ntypes; i++) {
    j = 2 * paircol + 2 * ntypes + i;
    apot_table.begin[j] = -1.1;
    opt_pot.begin[j] = -1.1;
    calc_pot.begin[j] = -1.1;
    apot_table.end[j] = 1.1;
    opt_pot.end[j] = 1.1;
    calc_pot.end[j] = 1.1;
  }
#endif /* MEAM */

  /* recalculate step, invstep and xcoord for new tables */
  for (i = 0; i < calc_pot.ncols; i++) {
    calc_pot.step[i] = (calc_pot.end[i] - calc_pot.begin[i]) / (APOT_STEPS - 1);
    calc_pot.invstep[i] = 1.0 / calc_pot.step[i];
    for (j = 0; j < APOT_STEPS; j++) {
      index = i * APOT_STEPS + (i + 1) * 2 + j;
      calc_pot.xcoord[index] = calc_pot.begin[i] + j * calc_pot.step[i];
    }
  }

  update_slots();
#endif /* APOT */

  /* print minimal distance matrix */
  printf("\nMinimal Distances Matrix:\n");
  printf("Atom\t");
  for (i = 0; i < ntypes; i++)
    printf("%8s\t", elements[i]);
  printf("with\n");
  for (i = 0; i < ntypes; i++) {
    printf("%s\t", elements[i]);
    for (j = 0; j < ntypes; j++) {
      k = (i <= j) ? i * ntypes + j - ((i * (i + 1)) / 2) : j * ntypes + i - ((j * (j + 1)) / 2);
      printf("%f\t", mindist[k]);

    }
    printf("\n");
  }
  printf("\n");

#endif /* !KIM */
/* added ends */

  free(mindist);

  if (sh_dist)
    error(1, "Distances too short, last occurence conf %d, see above for details\n", sh_dist);

  return;
}

/****************************************************************
 *
 *  compute box transformation matrix
 *
 ****************************************************************/

double make_box(void)
{
  double volume;

  /* compute tbox_j such that SPROD(box_i,tbox_j) == delta_ij */
  /* first unnormalized */
  tbox_x = vec_prod(box_y, box_z);
  tbox_y = vec_prod(box_z, box_x);
  tbox_z = vec_prod(box_x, box_y);

  /* volume */
  volume = SPROD(box_x, tbox_x);
  if (0.0 == volume)
    error(1, "Box edges are parallel\n");

  /* normalization */
  tbox_x.x /= volume;
  tbox_x.y /= volume;
  tbox_x.z /= volume;
  tbox_y.x /= volume;
  tbox_y.y /= volume;
  tbox_y.z /= volume;
  tbox_z.x /= volume;
  tbox_z.y /= volume;
  tbox_z.z /= volume;

  return volume;
}

#ifdef CONTRIB

/****************************************************************
 *
 *  check if the atom does contribute to the error sum
 *
 ****************************************************************/

int does_contribute(vector pos)
{
  int   i;
  double n_a, n_b, n_c, r;
  vector dist;

  if (have_contrib_box) {
    dist.x = pos.x - cbox_o.x;
    dist.y = pos.y - cbox_o.y;
    dist.z = pos.z - cbox_o.z;
    n_a = SPROD(dist, cbox_a) / SPROD(cbox_a, cbox_a);
    n_b = SPROD(dist, cbox_b) / SPROD(cbox_b, cbox_b);
    n_c = SPROD(dist, cbox_c) / SPROD(cbox_c, cbox_c);
    if (n_a >= 0 && n_a <= 1)
      if (n_b >= 0 && n_b <= 1)
	if (n_c >= 0 && n_c <= 1)
	  return 1;
  }

  for (i = 0; i < n_spheres; i++) {
    dist.x = (pos.x - sphere_centers[i].x);
    dist.y = (pos.y - sphere_centers[i].y);
    dist.z = (pos.z - sphere_centers[i].z);
    r = SPROD(dist, dist);
    r = sqrt(r);
    if (r < r_spheres[i])
      return 1;
  }

  return 0;
}

#endif /* CONTRIB */

#ifdef APOT

/****************************************************************
 *
 * recalculate the slots of the atoms for analytic potential
 *
 ****************************************************************/

void update_slots(void)
{
  int   col, i, j;
  double r, rr;

  for (i = 0; i < natoms; i++) {
    for (j = 0; j < atoms[i].num_neigh; j++) {
      r = atoms[i].neigh[j].r;

      /* update slots for pair potential part, slot 0 */
      col = atoms[i].neigh[j].col[0];
      if (r < calc_pot.end[col]) {
	rr = r - calc_pot.begin[col];
	atoms[i].neigh[j].slot[0] = (int)(rr * calc_pot.invstep[col]);
	atoms[i].neigh[j].step[0] = calc_pot.step[col];
	atoms[i].neigh[j].shift[0] =
	  (rr - atoms[i].neigh[j].slot[0] * calc_pot.step[col]) * calc_pot.invstep[col];
	/* move slot to the right potential */
	atoms[i].neigh[j].slot[0] += calc_pot.first[col];
      }
#if defined EAM || defined ADP || defined MEAM
      /* update slots for eam transfer functions, slot 1 */
      col = atoms[i].neigh[j].col[1];
      if (r < calc_pot.end[col]) {
	rr = r - calc_pot.begin[col];
	atoms[i].neigh[j].slot[1] = (int)(rr * calc_pot.invstep[col]);
	atoms[i].neigh[j].step[1] = calc_pot.step[col];
	atoms[i].neigh[j].shift[1] =
	  (rr - atoms[i].neigh[j].slot[1] * calc_pot.step[col]) * calc_pot.invstep[col];
	/* move slot to the right potential */
	atoms[i].neigh[j].slot[1] += calc_pot.first[col];
      }
#ifdef TBEAM
      /* update slots for tbeam transfer functions, s-band, slot 2 */
      col = atoms[i].neigh[j].col[2];
      if (r < calc_pot.end[col]) {
	rr = r - calc_pot.begin[col];
	atoms[i].neigh[j].slot[2] = (int)(rr * calc_pot.invstep[col]);
	atoms[i].neigh[j].step[2] = calc_pot.step[col];
	atoms[i].neigh[j].shift[2] =
	  (rr - atoms[i].neigh[j].slot[2] * calc_pot.step[col]) * calc_pot.invstep[col];
	/* move slot to the right potential */
	atoms[i].neigh[j].slot[2] += calc_pot.first[col];
      }
#endif /* TBEAM */
#endif /* EAM || ADP || MEAM */

#ifdef MEAM
      /* update slots for MEAM f functions, slot 2 */
      col = atoms[i].neigh[j].col[2];
      if (r < calc_pot.end[col]) {
	rr = r - calc_pot.begin[col];
	atoms[i].neigh[j].slot[2] = (int)(rr * calc_pot.invstep[col]);
	atoms[i].neigh[j].step[2] = calc_pot.step[col];
	atoms[i].neigh[j].shift[2] =
	  (rr - atoms[i].neigh[j].slot[2] * calc_pot.step[col]) * calc_pot.invstep[col];
	/* move slot to the right potential */
	atoms[i].neigh[j].slot[2] += calc_pot.first[col];
      }
#endif /* MEAM */

#ifdef ADP
      /* update slots for adp dipole functions, slot 2 */
      col = atoms[i].neigh[j].col[2];
      if (r < calc_pot.end[col]) {
	rr = r - calc_pot.begin[col];
	atoms[i].neigh[j].slot[2] = (int)(rr * calc_pot.invstep[col]);
	atoms[i].neigh[j].step[2] = calc_pot.step[col];
	atoms[i].neigh[j].shift[2] =
	  (rr - atoms[i].neigh[j].slot[2] * calc_pot.step[col]) * calc_pot.invstep[col];
	/* move slot to the right potential */
	atoms[i].neigh[j].slot[2] += calc_pot.first[col];
      }

      /* update slots for adp quadrupole functions, slot 3 */
      col = atoms[i].neigh[j].col[3];
      if (r < calc_pot.end[col]) {
	rr = r - calc_pot.begin[col];
	atoms[i].neigh[j].slot[3] = (int)(rr * calc_pot.invstep[col]);
	atoms[i].neigh[j].step[3] = calc_pot.step[col];
	atoms[i].neigh[j].shift[3] =
	  (rr - atoms[i].neigh[j].slot[3] * calc_pot.step[col]) * calc_pot.invstep[col];
	/* move slot to the right potential */
	atoms[i].neigh[j].slot[3] += calc_pot.first[col];
      }
#endif /* ADP */

    }				/* end loop over all neighbors */
  }				/* end loop over all atoms */

#ifdef THREEBODY
  /* update angular slots */
  for (i = 0; i < natoms; i++) {
    for (j = 0; j < atoms[i].num_angles; j++) {
      rr = atoms[i].angle_part[j].cos + 1.1;
#ifdef MEAM
      col = 2 * paircol + 2 * ntypes + atoms[i].type;
      atoms[i].angle_part[j].slot = (int)(rr * calc_pot.invstep[col]);
      atoms[i].angle_part[j].step = calc_pot.step[col];
      atoms[i].angle_part[j].shift =
	(rr - atoms[i].angle_part[j].slot * calc_pot.step[col]) * calc_pot.invstep[col];
      /* move slot to the right potential */
      atoms[i].angle_part[j].slot += calc_pot.first[col];
#endif /* MEAM */
    }
  }
#endif /* THREEBODY */

#ifdef STIWEB
  apot_table.sw.init = 0;
#endif /* STIWEB */

#ifdef TERSOFF
  apot_table.tersoff.init = 0;
#endif /* TERSOFF */

  return;
}

#endif /* APOT */

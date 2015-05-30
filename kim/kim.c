
/*******************************************************************************
*
*
*
*******************************************************************************/




/*******************************************************************************
*
* kim.c 
*
* All functions all defined in this file
*
*******************************************************************************/

#define KIM_MAIN
#include "kim.h"
#undef KIM_MAIN

#include "../potfit.h" 


/*******************************************************************************
*
* Create KIM objects, each object for a reference configuration.
*
* potfit global variables:
* nconf
* name_opt_param
* num_opt_param 
* rcutmax
* 
*******************************************************************************/

void InitKIM() 
{
  printf("\nInitializing KIM ... started\n");

  /* create KIM objects and do the necessary initialization */
  InitObject();
  
  /* create free parameter data sturct and nest optimizable parameter */
  InitOptimizableParam();

  printf("Initializing KIM ... done\n");
}


/*******************************************************************************
*
* Initialize KIM objects
*
*
*******************************************************************************/

void InitObject()
{
  /* local variables */
  int status;
  int i;

  /* Allocate memory for KIM objects */
  pkimObj = (void**) malloc(nconf * sizeof(void *));
  if (NULL == pkimObj) {
    KIM_API_report_error(__LINE__, __FILE__,"malloc unsuccessful", -1);
    exit(1);
  }
    

  /* Checking whether .kim files in test and model are compatible or not */
  for (i = 0; i < nconf; i++) {         
    status = KIM_API_file_init(&pkimObj[i], "descriptor.kim", kim_model_name);
    if (KIM_STATUS_OK > status)
    {
      KIM_API_report_error(__LINE__, __FILE__, "KIM_API_file_init", status);
      exit(1);
    }
  }

  /* QUESTION: does each config has the same number of species? e.g. there are
  two types of species, but a specific config may one have one species). If
  not, then ntypes below need to be modified from config to config */

  /*  Initialize KIM objects */
  for (i = 0; i < nconf; i++) { 
    status = CreateKIMObj(pkimObj[i], inconf[i], ntypes, cnfstart[i]);
    if (KIM_STATUS_OK > status) {
      KIM_API_report_error(__LINE__, __FILE__,
                            "KIM: initializing objects failed", status);
      exit(1);
    }
  }
}

/*******************************************************************************
*
* Initialize optimizable parameter 
*
*
*******************************************************************************/

void InitOptimizableParam()
{
  /* local vars */
  int i;

  /* allocate memory */
  FreeParamAllConfig = (FreeParamType*) malloc(nconf*sizeof(FreeParamType));
  if (NULL == FreeParamAllConfig) {
    KIM_API_report_error(__LINE__, __FILE__,"malloc unsuccessful", -1);
    exit(1);
  }

  for (i = 0; i <nconf; i++) {
    /* nest optimizable parameters */
    get_FreeParamDouble(pkimObj[i], &FreeParamAllConfig[i]);  
    nest_OptimizableParamValue(pkimObj[i], &FreeParamAllConfig[i], 
                               name_opt_param, num_opt_param);
 
    /* Publish cutoff (only needs to be published once, so here) */
    PublishCutoff(pkimObj[i], rcutmax);
  }
}


/***************************************************************************
*
* Get free parameters of type type
*
* put all free parameters of type `double' into the data struct (only type
* `double' parameters are optimizable. `int' ones may be some flag.)
*
* parameter names are nested into `name'
* parameter values pointer are nested into `value'
* parameter ranks are nested into `rank'
* parameter shapes are nested into `shape'
*
* `PARAM_FREE_cutoff' will also be included in ParamType->name.
* Although it is not an optimiazble parameter, but we may want to copy it to
* potfit to calculate the neighbor list. 
***************************************************************************/
int get_FreeParamDouble(void* pkim, FreeParamType* FreeParam) {

  /*local vars*/
  int status;
  int NumFreeParam;
  int maxStringLength;
  char* pstr;
  char buffer[128];
  char name[64];
  char type[16];
  int tmp_size;
  int NumFreeParamDouble = 0;
  int i, j, k;

  /* get the number of free parameters */
  status = KIM_API_get_num_free_params(pkim, &NumFreeParam, &maxStringLength);
  if (KIM_STATUS_OK > status) {
    KIM_API_report_error(__LINE__, __FILE__, "KIM_API_get_num_free_params", status);
    return(status);
  }

  /* get the descriptor file, pointed by pstr. the type of data will be phrased from pstr*/  
  status = KIM_API_get_model_kim_str(kim_model_name, &pstr);
  if (KIM_STATUS_OK > status) {
    KIM_API_report_error(__LINE__, __FILE__, "KIM_API_get_model_kim_str", status);
    return(status);
  }

  /* initialize name */
  FreeParam->name = (char**) malloc(sizeof(char*));
  if (NULL==FreeParam->name) {
    KIM_API_report_error(__LINE__, __FILE__,"malloc unsuccessful", -1);
    exit(1);
  }

  /* infinite loop to find PARAM_FREE_* of type double, only they are optimizable.
     PARAM_FREE_cutoff will also be included. */
  /* It's safe to do pstr = strstr(pstr+1,"PARAM_FREE") because the ``PARAM_FREE''
    will never ever occur at the beginning of the ``descriptor.kim'' file */  
  while (1) {
    pstr = strstr(pstr+1,"PARAM_FREE");
    if (pstr == NULL) {
      break;
    } else {
      snprintf(buffer, sizeof(buffer), "%s", pstr);
      sscanf(buffer, "%s%s", name, type);
      if (strcmp(type, "double") == 0) {
        NumFreeParamDouble++;          
        if (NumFreeParamDouble > 1) {
          FreeParam->name = (char**) realloc(FreeParam->name, (NumFreeParamDouble)*sizeof(char*));
          if (NULL==FreeParam->name) {
            KIM_API_report_error(__LINE__, __FILE__,"malloc unsuccessful", -1);
            exit(1);
          }
        }
        FreeParam->name[NumFreeParamDouble - 1] =  /*maxStringLength+1 to hold the `\0' at end*/
                              (char*) malloc((maxStringLength+1)*sizeof(char));
        if (NULL==FreeParam->name[NumFreeParamDouble - 1]) {
          KIM_API_report_error(__LINE__, __FILE__,"malloc unsuccessful", -1);
          exit(1);
        }               
        strcpy(FreeParam->name[NumFreeParamDouble - 1], name);
      }
    }
  }

  FreeParam->Nparam = NumFreeParamDouble;

  /* allocate memory for value */
  FreeParam->value = (double**) malloc(FreeParam->Nparam * sizeof(double*));
  if (NULL==FreeParam->value) {
    KIM_API_report_error(__LINE__, __FILE__,"malloc unsuccessful", -1);
    exit(1);
  } 

  /* get the pointer to parameter */
  for(i = 0; i < FreeParam->Nparam; i++ ) {
    FreeParam->value[i] = KIM_API_get_data(pkim, FreeParam->name[i], &status);
  }
  if (KIM_STATUS_OK > status) {
    KIM_API_report_error(__LINE__, __FILE__, "KIM_API_get_data", status);
    return(status);
  }

  /* allocate memory for rank */
  FreeParam->rank = (int*) malloc(FreeParam->Nparam * sizeof(int));
  if (NULL==FreeParam->rank) {
    KIM_API_report_error(__LINE__, __FILE__,"malloc unsuccessful", -1);
    exit(1);
  } 

  /* get rank */
  for(i = 0; i < FreeParam->Nparam; i++) {
    FreeParam->rank[i] = KIM_API_get_rank(pkim, FreeParam->name[i], &status);
  }
  if (KIM_STATUS_OK > status) {
    KIM_API_report_error(__LINE__, __FILE__, "KIM_API_get_rank", status);
    return(status);
  }

  /* allocate memory for shape */
  FreeParam->shape = (int**) malloc(FreeParam->Nparam * sizeof(int*));
  if (NULL==FreeParam->shape) {
    KIM_API_report_error(__LINE__, __FILE__,"malloc unsuccessful", -1);
    exit(1);
  } 
  for (i = 0; i < FreeParam->Nparam; i++) {
    FreeParam->shape[i] = (int*) malloc(FreeParam->rank[i] * sizeof(int));
    /* should not do the check, since rank may be zero. Then in some implementation
      malloc zero would return NULL */  
    /*  if (NULL==FreeParam->shape[i]) {
    KIM_API_report_error(__LINE__, __FILE__,"malloc unsuccessful", -1);
    exit(1);
    }*/ 
  }

  /* get shape */
  for(i = 0; i < FreeParam->Nparam; i++) {
    KIM_API_get_shape(pkim, FreeParam->name[i], FreeParam->shape[i], &status);
  }
  if (KIM_STATUS_OK > status) {
    KIM_API_report_error(__LINE__, __FILE__, "KIM_API_get_shape", status);
    return(status);
  }
  
  /* nestedvalue is not allocated here, give NULL pointer to it */
  FreeParam->nestedvalue = NULL;


  return KIM_STATUS_OK;
}




/*****************************************************************************
* nest optimizable parameters values
* nest the values (pointer) obtained from get_OptParamInfo() to nestedvalue.
* nestedvalue would be equal to value if all parameters have rank 0. 
* include_cutoff: flag, whether cutoff will be incldued in the nested list
* return: the length of nestedvalue

* After calling get_OptParamInfo, the PARAM_FREE_cutoff will be in the last slot
* of name list. 
* this function will nest the FREE_PARAM_cutoff into nestedvaule if
* inlcude_cutoff is true, otherwise, do not incldue it. 
*****************************************************************************/
nest_OptimizableParamValue(void* pkim, FreeParamType* FreeParam,
                            char** input_param_name, int input_param_num)
{
  /* local variables */
  int tmp_size;
  int total_size;           /* the size of the nested values*/
  int have_name;            /* flag, is the name in the data struct? */
  int idx[input_param_num]; /* the index of input_param_num in FreeParam */
  int i, j, k;
  /* nest values  */
  /*first compute the total size of the optimizable parameters */
  total_size = 0;
  for (i = 0; i < input_param_num; i++) { 
    have_name = 0;
    for (j = 0; j< FreeParam->Nparam; j++) {
      if (strcmp(FreeParam->name[j], input_param_name[i]) == 0) {
        idx[i] = j;
        have_name = 1;
        break;
      }
    }
    if (have_name) {
      tmp_size = 1;
      for (j = 0; j < FreeParam->rank[idx[i]]; j++) {
        tmp_size *= FreeParam->shape[idx[i]][j];
      }
      /* store tmp_size for later use in potential outpot */
      size_opt_param[i] = tmp_size; 
      total_size += tmp_size;
    } else {
      error(1, "The parameter `%s' is not optimizable, check spelling.\n",
          input_param_name[i]);
    }
  }
  
  
  /* allocate memory for nestedvalue*/
  FreeParam->nestedvalue = (double**) malloc((total_size) * sizeof(double*));

  
  /*copy the values (pointers) to nestedvalue */
  k = 0;
  for (i = 0; i < input_param_num; i++ ) {
    tmp_size = 1; 
    for (j = 0; j < FreeParam->rank[idx[i]]; j++) {
      tmp_size *= FreeParam->shape[idx[i]][j];
    }
   for (j = 0; j <tmp_size; j++) {
      FreeParam->nestedvalue[k] = FreeParam->value[idx[i]]+j;
      k++;
    }
  }


  FreeParam->Nnestedvalue = total_size;

  return total_size;
}





/******************************************************************************
* get_OptimizableParam
* inquire KIM to get the number of optimizable parameters 
* and their names, which will be used later to nest the ParameterList

* input_name: the name list of the parameters read in from input
*******************************************************************************/

int get_OptimizableParamSize(FreeParamType* FreeParam, 
                              char** input_param_name, int input_param_num) 
{
  /*local variables */
  void* pkim;
  int status;
  int size;     /* This would be equal to Nparam if all parameters have rank zero.*/  
  int NumFreeParamNoDouble = 0;  /*number of FREE_PARAM_* with type other than double*/
  char* pstr;
  char buffer[128];
  char name[64];
  char type[16];
  int tmp_size;
  int i, j;

  /* create a temporary KIM objects, in order to inquire KIM model for 
   PARAM_FREE_* parameters info, delete at the end of the function */
  status = KIM_API_file_init(&pkim, "descriptor.kim", kim_model_name);
  if (KIM_STATUS_OK > status) {
    KIM_API_report_error(__LINE__, __FILE__, "KIM_API_file_init", status);
    exit(1);
  }
  
  /* Allocate memory via the KIM system */
  KIM_API_allocate(pkim, 2, 1, &status);
  if (KIM_STATUS_OK > status)
  {
    KIM_API_report_error(__LINE__, __FILE__, "KIM_API_allocate", status);
    return(status);
  }

  /* call Model's init routine */
  status = KIM_API_model_init(pkim);
  if (KIM_STATUS_OK > status)
  {
    KIM_API_report_error(__LINE__, __FILE__, "KIM_API_model_init", status);
    return(status);   
  }
  
  /*get the info of the Optimizable parameters */
  get_FreeParamDouble(pkim, FreeParam);

  /* compute the total size of the optimizable */
  size = nest_OptimizableParamValue(pkim, FreeParam, input_param_name,
                                    input_param_num);


  /*number of free parameters of type other than double */
    /* get the descriptor file, pointed by pstr, the type will be phrased from pstr */  
  status = KIM_API_get_model_kim_str(kim_model_name, &pstr);
  if (KIM_STATUS_OK > status) {
    KIM_API_report_error(__LINE__, __FILE__, "KIM_API_get_model_kim_str", status);
    return(status);
  }
    /* infinite loop to find PARAM_FREE_* of type other than double*/ 
  while (1) {
    pstr = strstr(pstr+1,"PARAM_FREE");
    if (pstr == NULL) {
      break;
    } else {
      snprintf(buffer, sizeof(buffer), "%s", pstr);
      sscanf(buffer, "%s%s", name, type);
      if ( strcmp(type, "double") != 0) {
        NumFreeParamNoDouble++;
      }
    }
  }

  if (NumFreeParamNoDouble != 0) {
    printf("-  There is (are) %d `PARAM_FREE_*' parameter(s) of type other than "
            "`double'.\n", NumFreeParamNoDouble);
  }


  /* deallocate */  
/*  
  status = KIM_API_model_destroy(pkim);
  if (KIM_STATUS_OK > status){ 
    KIM_API_report_error(__LINE__, __FILE__,"destroy", status);
    return status;
  }
*/

/*
  KIM_API_free(pkim, &status);
  if (KIM_STATUS_OK > status){ 
    KIM_API_report_error(__LINE__, __FILE__,"destroy", status);
    return status;
  }
*/

  /* return value */
  return size;
}













































/***************************************************************************
*
* Function used to publsih cutoff
*
***************************************************************************/
PublishCutoff(void* pkim, double cutoff) {

  /* local variable */
  int status;
  double* pcutoff;
  
  pcutoff = KIM_API_get_data(pkim, "PARAM_FREE_cutoff", &status);
  if (KIM_STATUS_OK > status) {
    KIM_API_report_error(__LINE__, __FILE__, "KIM_API_get_data", status);
    return(status);
  }

  *pcutoff = cutoff;
  
  return 0;
}




/* added */ 
/***************************************************************************
* 
* Create KIM object  
*
* transferred varialbes:
* pkim: KIM object pointer
* Natoms: number of atoms in this configuration  
* Nspecies: number of atom species in this configuraiton 
* start: index of the first atom of this configuration in potfit atom array
*
* golbal potfit varialbes, used but not transferred: 
* atoms 
* elements
* box_side_len  (this is defined by us, not potfit) 
***************************************************************************/

int CreateKIMObj(void* pkim, int Natoms, int Nspecies, int start)
{
  /* model inputs */
  int* numberOfParticles;
  int* numberOfSpecies;
  int* particleSpecies;
  double* coords; 
  int* numberContrib;
  NeighObjectType* NeighObject;
  const char* NBCstr;
  int NBC;
  /* local variables */
  int status; 
  int species_code; 
  int halfflag;
  int Ncontrib; /* number of contriburting particles */ 
  int i, j, k;
  /* We have to allocate additional memory for neighbor list, since potfit 
    assigns the neighbors of each particle to atoms[i].neigh[k].nr (nr is
    the index neighbor k of atom i.). The memory is not continuous.
    The following variable (neighListLength), together with `BeginIdx' in
    the NeighObject, are used to gather the neighbor info to continuous memory. 
  */  
  int neighListLength; /* total length of neighList */


  /* set value */
  Ncontrib = Natoms;

  /* allocate memory for NeighObject ( simulator should be in charge of
  allocating memory for neighbor objects, not kim model ) */
  NeighObject = (NeighObjectType*) malloc(sizeof(NeighObjectType));
  if (NULL == NeighObject) {
    KIM_API_report_error(__LINE__, __FILE__,"malloc unsuccessful", -1);
    exit(1);
  } 

  /* Allocate memory via the KIM system */
  KIM_API_allocate(pkim, Natoms, Nspecies, &status);
  if (KIM_STATUS_OK > status)
  {
    KIM_API_report_error(__LINE__, __FILE__, "KIM_API_allocate", status);
    return(status);
  }

  /* register for neighObject */
  KIM_API_setm_data(pkim, &status, 1*4,
                     "neighObject",     1,   NeighObject,   1);
  if (KIM_STATUS_OK > status) {
   KIM_API_report_error(__LINE__, __FILE__,"KIM_API_setm_data",status);
    return(status);   
   }

  /* register for get_neigh */
  status = KIM_API_set_method(pkim, "get_neigh", 1, (func_ptr) &get_neigh);
  if (KIM_STATUS_OK > status) {
    KIM_API_report_error(__LINE__, __FILE__,"KIM_API_set_method",status);
    return(status);       
  } 
  /* call Model's init routine */
  status = KIM_API_model_init(pkim);
  if (KIM_STATUS_OK > status)
  {
    KIM_API_report_error(__LINE__, __FILE__, "KIM_API_model_init", status);
    return(status);   
  }

/* Determine which neighbor list type to use */
  halfflag = KIM_API_is_half_neighbors(pkim, &status);
  if (KIM_STATUS_OK > status) {
    KIM_API_report_error(__LINE__, __FILE__,"is_half_neighbors", status);
    return(status);   
  }

  /* Unpack data from KIM object */
  KIM_API_getm_data(pkim, &status, 5*3,
                    "numberOfParticles",   &numberOfParticles,   1,
                    "numberOfSpecies",     &numberOfSpecies,     1,
                    "particleSpecies",     &particleSpecies,     1,
                    "coordinates",         &coords,              1,
           "numberContributingParticles",  &numberContrib, (1==halfflag) );
  if (KIM_STATUS_OK > status) {
    KIM_API_report_error(__LINE__, __FILE__, "KIM_API_getm_data", status);
    return status;
  }

  /* Set values */
  *numberOfParticles = Natoms;
  *numberOfSpecies   = Nspecies;  
  *numberContrib     = Ncontrib;


  /* set boxSideLengths if MI_OPBC is used */
  /* determine which NBC is used */
  status = KIM_API_get_NBC_method(pkim, &NBCstr);
  if (KIM_STATUS_OK > status) {
    KIM_API_report_error(__LINE__, __FILE__, "KIM_API_get_NBC_method", status);
    return status; }
  if ((!strcmp("NEIGH_RVEC_H",NBCstr)) || (!strcmp("NEIGH_RVEC_F",NBCstr))) {
    NBC = 0; }
  else if ((!strcmp("NEIGH_PURE_H",NBCstr)) || (!strcmp("NEIGH_PURE_F",NBCstr))) {
    NBC = 1; }
  else if ((!strcmp("MI_OPBC_H",NBCstr)) || (!strcmp("MI_OPBC_F",NBCstr))) {
    NBC = 2; }
  else if (!strcmp("CLUSTER",NBCstr)) {
    NBC = 3; }
  else {
    status = KIM_STATUS_FAIL;
    KIM_API_report_error(__LINE__, __FILE__, "Unknown NBC method", status);
    return status; }

  if (NBC == 2) {
    /* define local varialbe */
    double* boxSideLen;
    int which_conf;   /* which config we are in? */
  
    which_conf = atoms[start].conf; 

    /* Unpack data from KIM object */
    KIM_API_getm_data(pkim, &status, 1*3,
                    "boxSideLengths",      &boxSideLen,     1 );
    if (KIM_STATUS_OK > status)
    {
      KIM_API_report_error(__LINE__, __FILE__, "KIM_API_getm_data", status);
      return status;
    }
 
    /* Set values */
    boxSideLen[0] = box_side_len[DIM*which_conf + 0];
    boxSideLen[1] = box_side_len[DIM*which_conf + 1];
    boxSideLen[2] = box_side_len[DIM*which_conf + 2];
  }   


  /* set the species types */
  /* to use this, the #C Header line in configuration file has to be included.
    Also the order of elements name after #C should be in accordane with the
    species code in the first column of configuration data, ranging from 0 to
    (ntypes-1).  
    Here, assume potfit ensures this.
    e.g. if we have `#C Ar Ne', then the code for Ar and Ne should be 0 and 1,
    respectively. */
    
  /* QUESTION: in KIM_API_get_species_code, doest the second argument has to
     to be totally the same as that kim descriptor file? e.g. Upper case or lower
     case matters or not? */

  for (i = 0; i < *numberOfParticles; i++) { 
    /* in potfit, atom types range from 0 to (ntype-1) */
    if (atoms[start+i].type < 0 || atoms[start+i].type >= *numberOfSpecies) {
      status = KIM_STATUS_FAIL;     
      KIM_API_report_error(__LINE__, __FILE__, "Unexpected species detected, "
       "the element(s) in potfit config file is not supported by the KIM model "
       " you specified.", status);
      return status;
    }
    else {
      j = atoms[start+i].type;
      species_code =  KIM_API_get_species_code(pkim, elements[j], &status); 
      if (KIM_STATUS_OK > status)
      {
        KIM_API_report_error(__LINE__, __FILE__, "KIM_API_get_species_code," 
                            "the species names need to be exactly the same"
                            "as that in KIM standard file", status);
        return status;
      }
      particleSpecies[i] = species_code;
    }
  }

  /* set coords values */
  for (i = 0; i < *numberOfParticles; i++) {
    coords[DIM*i]   = atoms[start+i].pos.x;
    coords[DIM*i+1] = atoms[start+i].pos.y;
    coords[DIM*i+2] = atoms[start+i].pos.z;
  }

  /* calcualte the length of neighborList */
  neighListLength = 0;
  for (i = 0; i < *numberOfParticles; i++) {
    neighListLength += atoms[start+i].num_neigh;  
  }

  /* allocate memory for NeighObject members */
  NeighObject->NNeighbors = (int*) malloc((*numberOfParticles) * sizeof(int));
  if (NULL==NeighObject->NNeighbors) {
    KIM_API_report_error(__LINE__, __FILE__,"malloc unsuccessful", -1);
    exit(1);
  } 
  NeighObject->neighborList = (int*) malloc(neighListLength * sizeof(int));
  if (NULL==NeighObject->neighborList) {
    KIM_API_report_error(__LINE__, __FILE__,"malloc unsuccessful", -1);
    exit(1);    
  } 
 NeighObject->RijList = (double*) malloc((DIM*neighListLength) * sizeof(double));
  if (NULL==NeighObject->RijList) {
    KIM_API_report_error(__LINE__, __FILE__,"malloc unsuccessful", -1);
    exit(1);    
  } 
  NeighObject->BeginIdx = (int*) malloc((*numberOfParticles) * sizeof(int));
  if (NULL==NeighObject->BeginIdx) {
    KIM_API_report_error(__LINE__,__FILE__,"malloc unsuccessful", -1);
    exit(1);    
  }

  /* copy the number of neighbors to NeighObject.NNeighbors */
  status = KIM_STATUS_FAIL; /* assume staus fails */
  for (i = 0; i < *numberOfParticles; i++) {
    NeighObject->NNeighbors[i] = atoms[start+i].num_neigh;  
  }
  if (i == *numberOfParticles){
    status = KIM_STATUS_OK;
  }
  if (KIM_STATUS_OK > status) {
    KIM_API_report_error(__LINE__, __FILE__,
                        "copy number of neighbors to NNeighbors failed", status);
  return status;
  }

  /* copy neighborlist from distributed memory locations to 
    continuous ones.
    copy atoms[i].neigh[j].nr to NeighObject.neighborList,
    copy atoms[i].neigh[k].dist.x ... to  NeighObject.Rij[DIM*k] ... */ 
  status = KIM_STATUS_FAIL; /* assume staus fails */  
  k = 0;
  for (i = 0; i < *numberOfParticles; i++) {
    NeighObject->BeginIdx[i] = k;
    for (j = 0; j < NeighObject->NNeighbors[i]; j++) {
      NeighObject->neighborList[k]  = atoms[start+i].neigh[j].nr - start;
      NeighObject->RijList[DIM*k]   = atoms[start+i].neigh[j].dist.x;
      NeighObject->RijList[DIM*k+1] = atoms[start+i].neigh[j].dist.y;
      NeighObject->RijList[DIM*k+2] = atoms[start+i].neigh[j].dist.z;
      k++;

/* unittest */
/* used together with to see that the stuff in the neighbor lista are correct
  set  start == 0 , will check for the first config. We set j==0, and j==last
  atom in the neighbor list of an atom, since we don't want that verbose info.
*/
/*
if (start != 0 && (j== 0 || j == atoms[start+i].num_neigh-1 )) {
printf("last neighbor: %d\n", atoms[start+ *numberOfParticles-1].num_neigh);
printf("which atom: %d\n",i);
printf("%d %f %f %f\n", atoms[start+i].neigh[j].nr,
                        atoms[start+i].neigh[j].dist.x,
                        atoms[start+i].neigh[j].dist.y,
                        atoms[start+i].neigh[j].dist.z );
}
*/
/* unittest ends*/
    }
  }
  
  if (i == *numberOfParticles && k == neighListLength){
    status = KIM_STATUS_OK;
  }
  if (KIM_STATUS_OK > status) {
    KIM_API_report_error(__LINE__, __FILE__,
                        "copy neighbor list failed", status);
    return status; } 

  /* If the number of neighbors of an atom is zero, set the BeginIdx to the 
     last position in neghbor list. Actually, the main purpose is to ensure 
     that the BeginIdx of the last atom will not go beyond limit of 
     neighborlist length.
     e.g. there are 128 atoms in the config, and we use half neighbor list,
     then the 128th atom will have no neighbors. Then the the begin index
     for the last atom, BeginIdx[127] will go beyond the limit of Neighbor
     list, which may result in segfault. So we need to do something to avoid
     this.
     I'm sure, there are better ways to do this.
  */  
  for (i = 0; i < *numberOfParticles; i++) {
    if( NeighObject->NNeighbors[i] == 0) {
      NeighObject->BeginIdx[i] = k-1;
    }
  }

  return KIM_STATUS_OK;
}
/* added ends */


/***************************************************************************
* 
* get_neigh 
*
***************************************************************************/

int get_neigh(void* kimmdl, int *mode, int *request, int* part,
                       int* numnei, int** nei1part, double** Rij)
{
   /* local variables */
  intptr_t* pkim = *((intptr_t**) kimmdl);
  int partToReturn;
  int status;
  int* numberOfParticles;
  int idx; /* index of first neighbor of each particle*/
  NeighObjectType* nl;

  /* initialize numnei */
  *numnei = 0;

  /* unpack neighbor list object */
  numberOfParticles = (int*) KIM_API_get_data(pkim, "numberOfParticles", &status);
  if (KIM_STATUS_OK > status) {
  KIM_API_report_error(__LINE__, __FILE__,"get_data", status);
  }

  nl = (NeighObjectType*) KIM_API_get_data(pkim, "neighObject", &status);
  if (KIM_STATUS_OK > status) {
    KIM_API_report_error(__LINE__, __FILE__,"get_data", status);
  }

  /* check mode and request */
  if (0 == *mode) /* iterator mode */
  {
    if (0 == *request) /* reset iterator */
    {
      (*nl).iteratorId = -1;
      return KIM_STATUS_NEIGH_ITER_INIT_OK;
    }
    else if (1 == *request) /* increment iterator */
    {
      (*nl).iteratorId++;
      if ((*nl).iteratorId >= *numberOfParticles)
      {
        return KIM_STATUS_NEIGH_ITER_PAST_END;
      }
      else
      {
        partToReturn = (*nl).iteratorId;
      }
    }
    else /* invalid request value */
    {
      KIM_API_report_error(__LINE__, __FILE__,"Invalid request in get_periodic_neigh",
                           KIM_STATUS_NEIGH_INVALID_REQUEST);
      return KIM_STATUS_NEIGH_INVALID_REQUEST;
    }
  }
  else if (1 == *mode) /* locator mode */
  {
    if ((*request >= *numberOfParticles) || (*request < 0)) /* invalid id */
    {
      KIM_API_report_error(__LINE__, __FILE__,"Invalid part ID in get_periodic_neigh",
                          KIM_STATUS_PARTICLE_INVALID_ID);
      return KIM_STATUS_PARTICLE_INVALID_ID;
    }
    else
    {
      partToReturn = *request;
    }
  }
  else /* invalid mode */
  {
    KIM_API_report_error(__LINE__, __FILE__,"Invalid mode in get_periodic_neigh",
                          KIM_STATUS_NEIGH_INVALID_MODE);
    return KIM_STATUS_NEIGH_INVALID_MODE;
  }

  /* index of the first neigh of each particle */
  idx = (*nl).BeginIdx[partToReturn];

  /* set the returned part */
  *part = partToReturn;

  /* set the returned number of neighbors for the returned part */
  *numnei = (*nl).NNeighbors[partToReturn];

  /* set the location for the returned neighbor list */
  *nei1part = &((*nl).neighborList[idx]);

  /* set the pointer to Rij to appropriate value */
  *Rij = &((*nl).RijList[DIM*idx]);

  return KIM_STATUS_OK;
}

/* added ends */





/* added */
/***************************************************************************
* 
* Calculate force from KIM (general force, including force, virial, energy)
*
* transferred variaves 
* Pkim: KIM object
* the following three are also model output 
* energy;
* force;
* virial;
*
***************************************************************************/

int CalcForce(void* pkim, double** energy, double** force, double** virial,
              int useforce, int usestress)
{ 
  /* local variables */
  int status;
  /* Access to free parameters, also unpack other variables as needed */
  /* potfit will always compute energy, so is here */
  KIM_API_getm_data(pkim, &status, 1*3,
                    "energy",              energy,              1);
  if (useforce)
    KIM_API_getm_data(pkim, &status, 1*3,
                    "forces",              force,               1);
  if(usestress)
    KIM_API_getm_data(pkim, &status, 1*3,
                    "virial",              virial,             1 );


  if (KIM_STATUS_OK > status)
  {
    KIM_API_report_error(__LINE__, __FILE__, "KIM_API_getm_data", status);
    return status;
  }

  /* Call model compute */
  status = KIM_API_model_compute(pkim);
  if (KIM_STATUS_OK > status)
  {
    KIM_API_report_error(__LINE__, __FILE__, "KIM_API_model_compute", status);
    return status;
  }

  return KIM_STATUS_OK;
}
/* added ends */








/***************************************************************************
*
* Publish KIM parameter 
*
* transferred varialbes:
* pkkm: KIM ojbect
* PotTable: potential table where the potential paramenters are stored
*
***************************************************************************/
int PublishParam(void* pkim, FreeParamType* FreeParam, double* PotTable)
{ 
  /*local variables*/  
  int status;
  int i;

  /*publish parameters */ 
  for (i = 0; i < FreeParam->Nnestedvalue; i++) {
    *FreeParam->nestedvalue[i] = PotTable[i]; 
  }

  status = KIM_API_model_reinit(pkim);
  if (KIM_STATUS_OK > status) {
    KIM_API_report_error(__LINE__, __FILE__, "KIM_API_model_reinit", status);
    return status;
  }
  return KIM_STATUS_OK;
}










/***************************************************************************
*
* Read the keywords (`type', `cutoff', etc) and parameter names (beginning 
* with `PARAM_FREE') in the potential input file
*
* In this function, the memory of three global variables: `num_opt_param',
* `name_opt_param', `size_opt_param' will be allocated and the first two will be
* initialized here (based on the infomation from the input), `size_opt_param' will
* be initialized in nest_OptimizableParamValue. 
*
* pt: potential table
* filename: potential input file name 
* infile: potential input file
* FreeParam: data struct that contains the free parameters info of the KIM Model
* 
***************************************************************************/

int ReadPotentialKeywords(pot_table_t* pt, char* filename, FILE* infile,
													FreeParamType* FreeParam)
{
	/* local variables */
  int   i, j, k, ret_val;
  char  buffer[255], name[255];
  fpos_t filepos, startpos; 
  
  /* save starting position */
  fgetpos(infile, &startpos);

  /* scan for "type" keyword */
  buffer[0] = '\0';
  name[0] = '\0';
  do {
    fgets(buffer, 255, infile);
    sscanf(buffer, "%s", name);
  } while (strncmp(name, "type", 4) != 0 && !feof(infile));
  if (strncmp(name, "type", 4) != 0) {
    error(1, "Keyword `type' is missing in file: %s.", filename);
  }
  if (1 > sscanf(buffer, "%*s %s", name))
    error(1, "Cannot read KIM Model name file: %s.", filename);
  /* copy name*/
  strcpy(kim_model_name, name);
  printf("\nKIM Model being used: %s.\n\n", kim_model_name);
 

  /* find `check_kim_opt_param' or `num_opt_param'. The two keywords are mutually
	 * exculsive, which comes first will be read, and the other one will be ignored. */
  fsetpos(infile, &startpos);
  do {
    fgets(buffer, 255, infile);
    sscanf(buffer, "%s", name);
  } while (strcmp(name, "check_kim_opt_param") != 0 
						&& strcmp(name, "num_opt_param") != 0 && !feof(infile));
  
	if (strcmp(name, "check_kim_opt_param") != 0 && strcmp(name, "num_opt_param") != 0){
    error(1, "Cannot find keyword `num_opt_param' in file: %s.", filename);
  }
  /* read `check_kim_opt_param' or `num_opt_param' */
  if (strncmp(buffer,"check_kim_opt_param", 19) == 0) {
    /* We do not need to get the `nestedvalue', so `NULL' and `0' works well here. */
    get_OptimizableParamSize(FreeParam, NULL, 0); 

    printf(" - The following potential parameters are available to fit. Include the "
        "name(s) (and the initial value(s) and lower and upper boudaries if "
        "`NOLIMITS' is not enabled while compilation) that you want to optimize "
        "in file: %s.\n",filename);
    printf("         param name                 param extent\n");
    printf("        ############               ##############\n");
    for(k = 0; k < FreeParam->Nparam; k++ ) {
      if (strncmp(FreeParam->name[k], "PARAM_FREE_cutoff", 17) == 0){
        continue;
      }
      printf("     %-35s[ ", FreeParam->name[k]);
      for(j = 0; j < FreeParam->rank[k]; j++) {
        printf("%d ", FreeParam->shape[k][j]);
      }
      printf("]\n");
   }
    printf("\n - Note that KIM array parameter is row based, while listing the "
        "initial values of such parameter, you should ensure that the sequence is "
        "correct. For example, if the extent of a parameter `PARAM_FREE_A' is "
        "[ 2 2 ], then you should list the initial values like: A[0 0], A[0 1], "
        "A[1 0], A[1 1].\n");
   exit(1); 
  } else if (strncmp(buffer,"num_opt_param", 13) == 0) {
    if(1 != sscanf(buffer, "%*s%d", &num_opt_param)) {
      error(1, "Cannot read `num_opt_param' in file: %s.", filename);  
    }
  }


  /* allocate memory */ 
  name_opt_param = (char**)malloc(num_opt_param*sizeof(char*)); 
  size_opt_param = (int*)malloc(num_opt_param*sizeof(int)); 
  reg_for_free(name_opt_param, "name_opt_param");
  reg_for_free(size_opt_param, "size_opt_param");
  if (NULL == name_opt_param || NULL == size_opt_param) {
    error(1, "Error in allocating memory for parameter name or size");
  }
  for (i = 0; i < num_opt_param; i++) {
    name_opt_param[i] = (char *)malloc(255 * sizeof(char));
    reg_for_free(name_opt_param[i], "name_opt_param[%d]", i);
    if (NULL == name_opt_param[i]) {
      error(1, "Error in allocating memory for parameter name");
    }
  }

	/* find parameter names beginning with `PARAM_FREE_*' */
	fsetpos(infile, &startpos);
	for (j = 0; j < num_opt_param; j++) {
		buffer[0] = '\0';
		name[0] = '\0';
		do {
			fgets(buffer, 255, infile);
			ret_val = sscanf(buffer, "%s", name);
		} while (strncmp(name, "PARAM_FREE", 10) != 0 && !feof(infile));
		if (feof(infile) ) {
			error(0, "Not enough parameter(s) `PARAM_FREE_*' in file %s!\n", filename);
			error(1, "You listed %d parameter(s), but required are %d.\n", j, num_opt_param);
		}
		if (ret_val == 1) {
			strcpy(name_opt_param[j], name); 
		} else {
			error(1, "Could not read parameter #%d in file %s.", j + 1, filename);
		}
	}

	return 0;
}





/****************************************************************
 *
 *  write potential table (format 5)
 *
 ****************************************************************/

void write_pot_table5(pot_table_t *pt, char *filename)
{
  FILE *outfile = NULL;
  int i, j, k;

  /* open file */
  outfile = fopen(filename, "w");
  if (NULL == outfile)
    error(1, "Could not open file %s\n", filename);

  /* write header */
  fprintf(outfile, "#F 5 1");
  if (have_elements) {
    fprintf(outfile, "\n#C");
    for (i = 0; i < ntypes; i++)
      fprintf(outfile, " %s", elements[i]);
    fprintf(outfile, "\n##");
    for (i = 0; i < ntypes; i++)
      for (j = i; j < ntypes; j++)
        fprintf(outfile, " %s-%s", elements[i], elements[j]);
  }
  fprintf(outfile, "\n#E");

  /* write KIM Model name */
  fprintf(outfile, "\n\n# KIM Model name");
  fprintf(outfile, "\ntype  %s", kim_model_name);

  /* write cutoff */
  fprintf(outfile, "\n\n# cutoff");
  fprintf(outfile, "\ncutoff  %24.16e", rcutmax);

  /* check KIM optimizable params */
  fprintf(outfile, "\n\n# uncomment the following line to check the optimizable "
                   "parameters of the KIM Model");
  fprintf(outfile, "\n#check_kim_opt_param");

  /* number of opt params */
  fprintf(outfile, "\n\n# the number of optimizable parameters that will be listed below");
  fprintf(outfile, "\nnum_opt_param  %d", num_opt_param);
  
  /* write data */
  k = 0; 
  fprintf(outfile, "\n\n# parameters");
  for (i = 0; i < num_opt_param; i++) {
    fprintf(outfile, "\n%s", name_opt_param[i]);
#ifndef NOLIMITS
    for (j = 0; j < size_opt_param[i]; j++) {
      fprintf(outfile, "\n%24.16e %24.16e %24.16e", pt->table[k], apot_table.pmin[0][k],
              apot_table.pmax[0][k]); 
      k++;
    }
    fprintf(outfile, "\n");
#endif  
  }

  fclose(outfile);
}

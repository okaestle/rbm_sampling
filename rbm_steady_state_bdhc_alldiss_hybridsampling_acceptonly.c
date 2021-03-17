/**************************************
* Copyright (C) 2020 by Oliver Kästle *
***************************************/

/******************************************
* Required compile commands: -lm -fopenmp *
*******************************************/

/**********************************************************
* Restricted Boltzmann machine (RBM) implementation		  *
* for the stationary state of the						  *
* isotropic 3D boundary-driven Heisenberg chain			  *
* with incoherent dissipation and excitation on all sites *
* using accept-only and hybrid sampling strategies		  *
* [arXiv:2012.10990 (2020)]								  *
***********************************************************/

// OUTPUT FILE:
// | iteration step | spin_1 Gr.St. | spin_1 Ex.St. | ... | spin_N Gr.St. | spin_N Ex.St. |

#include "rbm_steady_state_bdhc_alldiss_hybridsampling_acceptonly.h"

// ######################### PARAMETERS ###########################

// number of samples drawn per iteration
#define N_SAMPLES (40)
	
// learning rate
#define LRATE (0.05)

// N visible units, M hidden (auxiliary) units for RBM,
// K hidden (ancillary) units to purify the density matrix
#define N_VISIBLE (10)
#define M_HIDDEN  (10)
#define K_MIXING  (10)

// Lindblad decay/driving rate
#define GAMMA_OUT (0.2) // mu
#define GAMMA_IN  (GAMMA_OUT*1.05) // Gamma

// site coupling J = J_x = J_y = J_z
#define J_ALL (GAMMA_IN/2.)

// additional Lindblad dephasing -- switched off
#define DEPHASING (0.0)

// magnetic field -- switched off
#define BMAG (0.0)

// Number of iterations
#define N_TIMES (5000)

// #################################################################
// ############################## MAIN #############################
// #################################################################

int main(){
	
	int N_samples = N_SAMPLES;
	int N_times = N_TIMES;
	int N_vis = N_VISIBLE;
	int M_hid = M_HIDDEN;
	int K_mix = K_MIXING;
	double lrate = LRATE;
	
	// calculate and store powers of 2
	int *POW_2 = calloc((N_vis+N_vis+2),sizeof(int));
	POW_2[0] = 1;
	for(int n0=1;n0<(N_vis+N_vis+2);n0++){
		POW_2[n0] = 2*POW_2[n0-1];
	}
	
	// total number of variational coefficients (split up in Re and Im):
	// 2N + 2M + K + 2MN + 2KN = 2[ N + M + (M+K)N ] + K
	int numcoeff = 2*( N_vis + M_hid + (M_hid + K_mix)*N_vis ) + K_mix;
	
	char FILE_NAME[2048+1024];
	char FILE_NAME_T[2048+1024];

	time_t curtime;
	struct tm *loctime; // structure to write the system time to
	curtime = time(NULL); // Connect curtime with the function time
	loctime = localtime (&curtime); // write the system time in the structure format
	
	// get current hour, minute, second 
	int hour = loctime -> tm_hour;
	int minute = loctime -> tm_min;
	int second = loctime -> tm_sec;
	
	// get current day, month, year
	int year = 1900+loctime -> tm_year;
	int month = loctime -> tm_mon; month = month +1;
	int day  = loctime -> tm_mday;
	
	#ifndef RUN_ON_CLUSTER
	printf("Date: %d.%d.%.d -- Time: %d:%d:%d  \n",day,month,year,hour,minute,second);  
	#endif

	// for the stopwatch
	time_t end_t;
	double diff_t;
	
    // initialize the random generator with system time
	srand( time(NULL) );
	srand48( time(NULL) );

// #################################################################
// ######################### CREATE FILE ###########################
// #################################################################

	FILE *f_dat;
	snprintf(FILE_NAME,2048+1024,"%02d_%02d_%02d_%02d_%d_RBM_BDHC_steadystate_hybrid_acconly_N%d_M%d_K%d_lr%.5f_J%.5f_gin%.5f_gout%.5f_Ns%d_Nt%d.dat",
				day,month,hour,minute,second, N_vis,M_hid,K_mix,lrate,J_ALL,GAMMA_IN,GAMMA_OUT,N_samples,N_times);		
	f_dat=fopen(FILE_NAME,"w");
	fprintf(f_dat,"# Date: %02d.%02d.%.d -- Time: %02d:%02d:%02d  \n",day,month,year,hour,minute,second);
	
// #################################################################
// ####################### INITIALIZE ARRAYS #######################
// #################################################################
	
	double gam_in = GAMMA_IN;
	double gam_out = GAMMA_OUT;
	double Jx = J_ALL;
	double Jy = Jx;
	double Jz = Jx;
	double Bmag = BMAG;
	double deph = DEPHASING;
	
	// system parameters
	double *mparameters = calloc(7,sizeof(double));
	mparameters[0] = gam_in;
	mparameters[1] = gam_out;
	mparameters[2] = Jx;
	mparameters[3] = Jy;
	mparameters[4] = Jz;
	mparameters[5] = Bmag;
	mparameters[6] = deph;
	
	complex double hamilton;
	complex double dissipator;
	
	complex double hamilton_grad;
	complex double dissipator_grad;
	
	complex double meanliouv;
	complex double *liouvillian = calloc(2*N_samples, sizeof(double));
	
	complex double **liouv_grad = calloc(2*N_samples, sizeof(double));
	
	for(int n=0;n<N_samples;n++){
		liouv_grad[n] = calloc(2*numcoeff, sizeof(double));
		
		for(int p=0;p<numcoeff;p++){
			liouv_grad[n][p] = 0.;
		}
	}
	
	// store mean gradients
	complex double *meangrad = calloc(2*numcoeff, sizeof(double));

	// store cost function gradient for stochastic gradient descent
	double *Fvec = calloc(numcoeff, sizeof(double));
	complex double Ftemp;
	
	
	// visible spins: spin_left [0,...,N-1], spin_right [N,...,2N-1]
	// possible spin/neuron configurations: +1/-1
	double *current_qsample  = calloc(2*N_vis, sizeof(double));
	double *current_psample  = calloc(2*N_vis, sizeof(double));
	double *proposed_qsample = calloc(2*N_vis, sizeof(double));
	double *proposed_psample = calloc(2*N_vis, sizeof(double));
	
	// probabilities
	double current_pprob, proposed_pprob;
	double current_qprob, proposed_qprob;
	
	double *pprob = calloc(N_samples, sizeof(double));
	double *qprob = calloc(N_samples, sizeof(double));
	
	double ppartition;
	double qpartition;
	
	double accept_q, accept_p;
	double rnum_p, rnum_q;
	
	int acc_pcount = 0;
	int acc_qcount = 0;
	
	int sploop = 0;
	int sqloop = 0;
	
	// store ground/excited state occupations
	double *occu_gs  = calloc(2*N_vis, sizeof(double));
	double *occu_es  = calloc(2*N_vis, sizeof(double));
	
// #################################################################
// ################## INITIALIZE COEFFICIENTS ######################
// #################################################################
	
	// the complex coefficients of the RBM consist of weights and biases.
	// all coefficients are split up into real and imaginary parts
	// and treated independently from each other
	
	// we have to store all indices of all coefficients (Re and Im) in a single vector
	
	// coeff[p] =
	// areal[0 ... N-1], aimag[N ... 2N-1],
	// breal[2N ... 2N+M-1], bimag[2N+M ... 2N+2M-1],
	// Wreal[2N+2M ... 2N+2M + MN], Wimag[2N+2M+MN ... 2N+2M + 2MN],
	// creal[2N+2M+2MN ... 2N+2M+2MN+K-1],
	// Ureal[2N+2M+2MN+K ... 2N+2M+2MN+K + KN], Uimag[2N+2M+2MN+K+KN ... 2N+2M+2MN+K + 2KN]
	
	// access e.g. Wreal[m][i] = coeff[ 2N+2M +m*N + i ]
	// 			   Uimag[k][i] = coeff[ 2N+2M+2MN+K+KN + k*N + i ]
	
	double *coeff = calloc(numcoeff, sizeof(double));
	double vz;
	
	// initialize variational coefficients with random values in [-0.01,0.01]\{0}
	for(int p=0;p<numcoeff;p++){
		vz = RANDOM_NUMBER(1);
		coeff[p] = RANDOM_NUMBER(0);
		if(vz < 0.5){
			coeff[p] *= -1.;
		}
	}
	
	// define constants to access coeff/grad elements easily
	int startWr = 2*N_vis + 2*M_hid;
	int startWi = startWr + M_hid*N_vis;
	int startcr = startWi + M_hid*N_vis;
	int startUr = startcr + K_mix;
	int startUi = startUr + K_mix*N_vis;
	
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// %%%%%%%%%%%%%%% START TRAINING ITERATION LOOP %%%%%%%%%%%%%%%%%%%
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	printf("\n");
	printf("RBM calculation of the isotropic 3D Heisenberg model:\n");
	printf("Number of neurons: N=%d, M=%d, K=%d\n", N_vis,M_hid,K_mix);
	printf("Number of coefficients: %d\n",numcoeff);
	printf("Samples per iteration: %d\n", N_samples);
	printf("Learning rate: %.5f\n", lrate);
	printf("\n");
	
	printf("Start neural network training ...\n");
	
	for(int t=0;t<N_times;t++){
	
// #################################################################
// ##################### INITIALIZE SAMPLES ########################
// #################################################################
	
// two probabilities are sampled independently from each other:
// 1) q(l,l) = rho(l,l) for the calculation of diagonal observables
// 2) p(l,r) = |rho(l,r)|^2 for the training of the RBM

		// sample vector: spin_left [0,...,N-1] ,spin_right [N,...,2N-1]

		// initial q-sample (spin configuration)
		// sample for the probability q = rho(l,l), i.e. only densities l=r allowed
		for(int i=1;i<N_vis;i++){
			rnum_q = RANDOM_NUMBER(1);
			if(rnum_q < 0.5){
				current_qsample[i] = -1.;
				current_qsample[N_vis+i] = -1.;
			}
			else{
				current_qsample[i] = 1.;
				current_qsample[N_vis+i] = 1.;
			}
		}
		
		// initial p-sample (spin configuration)
		// sample for the probability p = |rho(l,r)|^2, no restrictions
		for(int i=0;i<2*N_vis;i++){
			rnum_p = RANDOM_NUMBER(1);
			if(rnum_p < 0.5){
				current_psample[i] = -1.;
			}
			else{
				current_psample[i] = 1.;
			}
		}
		
		// reset mean gradients for training loop
		for(int p=0;p<numcoeff;p++){
			meangrad[p] = 0.;
		}
		// reset mean Liouvillian for training loop
		meanliouv = 0.;
	
		// reset observables for training loop
		for(int i=0;i<N_vis;i++){
			occu_gs[i] = 0.;
			occu_es[i] = 0.;
		}
		
		// reset partition functions for training loop
		ppartition = 0.;
		qpartition = 0.;
		
		sploop = 0;
		sqloop = 0;
		
		
// ################################################################
// ##################### SAMPLING ALGORITHM #######################
// ########### HYBRID SAMPLING, ACCEPT ONLY STRATEGY ##############
// ################################################################
	
		// start sample loop
		for(int n=0;n<N_samples;n++){
			
			// reset accept counters
			acc_pcount = 0;
			acc_qcount = 0;
			
			// reset counters for possible edge configurations
			if(sploop==16) sploop = 0;
			if(sqloop==4)  sqloop = 0;
					
// #################### START HYBRID PSAMPLING ####################
			
			// STEP 1: calculate mean current prob for fixed bulk configuration,
			// average over all 16 possible edge configurations
			
			current_pprob = 0.;
			
			for(int edge=0;edge<16;edge++){
				for(int i=0;i<2;i++){
					if(edge % POW_2[i+1] < POW_2[i]){
						
						if(i==0) current_psample[0] = -1.;
						else	 current_psample[N_vis-1] = -1.;
					}
					else{
						if(i==0) current_psample[0] = 1.;
						else	 current_psample[N_vis-1] = 1.;
					}
				
					if(edge % POW_2[2+i+1] < POW_2[2+i]){
						if(i==0) current_psample[N_vis+0] = -1.;
						else	 current_psample[N_vis+N_vis-1] = -1.;
					}
					else{
						if(i==0) current_psample[N_vis+0] = 1.;
						else	 current_psample[N_vis+N_vis-1] = 1.;
					}
				}
				
				current_pprob  += NDO(coeff,current_psample,N_vis,M_hid,K_mix)
							*conj(NDO(coeff,current_psample,N_vis,M_hid,K_mix));
			}
			current_pprob *= (1./16.);

			// STEP 2: draw a new sample
			while(acc_pcount==0){

				proposed_psample = psampler(current_psample,N_vis);
				proposed_pprob = 0.;
				
				// STEP 2a: calculate mean proposed probability for fixed bulk configuration,
				// average over all 16 possible edge configurations
				for(int edge=0;edge<16;edge++){
					for(int i=0;i<2;i++){
						if(edge % POW_2[i+1] < POW_2[i]){
							
							if(i==0) proposed_psample[0] = -1.;
							else	 proposed_psample[N_vis-1] = -1.;
						}
						else{
							if(i==0) proposed_psample[0] = 1.;
							else	 proposed_psample[N_vis-1] = 1.;
						}
					
						if(edge % POW_2[2+i+1] < POW_2[2+i]){
							if(i==0) proposed_psample[N_vis+0] = -1.;
							else	 proposed_psample[N_vis+N_vis-1] = -1.;
						}
						else{
							if(i==0) proposed_psample[N_vis+0] = 1.;
							else	 proposed_psample[N_vis+N_vis-1] = 1.;
						}
					}
					
					proposed_pprob  += NDO(coeff,proposed_psample,N_vis,M_hid,K_mix)
								*conj(NDO(coeff,proposed_psample,N_vis,M_hid,K_mix));
				}
				proposed_pprob *= (1./16.);
				
				
				// STEP 2b: accept or reject proposed sample -- ACCEPT ONLY AGENDA
				accept_p = acceptance(proposed_pprob,current_pprob);
				rnum_p = RANDOM_NUMBER(1);
				
				// reject if bad sample
				if(isnan(creal(NDO(coeff,proposed_psample,N_vis,M_hid,K_mix))) || isnan(cimag(NDO(coeff,proposed_psample,N_vis,M_hid,K_mix)))){
					acc_pcount = 0; // otherwise repeat
				}
				else{
					if(rnum_p <= accept_p){
						for(int i=0;i<2*N_vis;i++){
							current_psample[i] = proposed_psample[i]; // accept the proposed sample
						}
						acc_pcount++;
					}
					else{
						acc_pcount = 0; // otherwise repeat procedure until a sample is accepted
					}
				}
			}
			
			// STEP 3: HYBRID SAMPLING -- vary edge configuration, repeats every 16 samples
			
			// we start at -1,...,-1 in left and right side
			for(int i=0;i<2;i++){
				if(sploop % POW_2[i+1] < POW_2[i]){
					
					if(i==0) current_psample[0] = -1.;
					else	 current_psample[N_vis-1] = -1.;
				}
				else{
					if(i==0) current_psample[0] = 1.;
					else	 current_psample[N_vis-1] = 1.;
				}
			
				if(sploop % POW_2[2+i+1] < POW_2[2+i]){
					if(i==0) current_psample[N_vis+0] = -1.;
					else	 current_psample[N_vis+N_vis-1] = -1.;
				}
				else{
					if(i==0) current_psample[N_vis+0] = 1.;
					else	 current_psample[N_vis+N_vis-1] = 1.;
				}
			}
			sploop++;
			
			// calculate p-probability of new sample
			pprob[n] = NDO(coeff,current_psample,N_vis,M_hid,K_mix)
					*conj(NDO(coeff,current_psample,N_vis,M_hid,K_mix));
					
// ##################### END HYBRID PSAMPLING #####################
// ################################################################
// #################### START HYBRID QSAMPLING ####################
			
			// STEP 1: calculate mean current prob for fixed bulk configuration,
			// average over all 4 possible edge configurations
			current_qprob = 0.;
			
			for(int edge=0;edge<4;edge++){
				if(edge==0){
					current_qsample[0] = -1;
					current_qsample[N_vis-1] = -1.;
					current_qsample[N_vis+0] = -1.;
					current_qsample[N_vis+N_vis-1] = -1.;
				}
				else if(edge==1){
					current_qsample[0] = -1;
					current_qsample[N_vis-1] = 1.;
					current_qsample[N_vis+0] = -1.;
					current_qsample[N_vis+N_vis-1] = 1.;
				}
				else if(edge==2){
					current_qsample[0] = 1;
					current_qsample[N_vis-1] = -1.;
					current_qsample[N_vis+0] = 1.;
					current_qsample[N_vis+N_vis-1] = -1.;
				}
				else if(edge==3){
					current_qsample[0] = 1;
					current_qsample[N_vis-1] = 1.;
					current_qsample[N_vis+0] = 1.;
					current_qsample[N_vis+N_vis-1] = 1.;
				}
				current_qprob  += NDO(coeff,current_qsample,N_vis,M_hid,K_mix);
			}

			current_qprob *= (1./4.);


			// STEP 2: draw a new sample
			while(acc_qcount==0){
				
				// STEP 2a: calculate mean proposed sample for fixed bulk configuration,
				// average over all 4 possible edge configurations
				proposed_qsample = qsampler(current_qsample,N_vis);
				proposed_qprob = 0.;
			
				for(int edge=0;edge<4;edge++){
					if(edge==0){
						proposed_qsample[0] = -1;
						proposed_qsample[N_vis-1] = -1.;
						proposed_qsample[N_vis+0] = -1.;
						proposed_qsample[N_vis+N_vis-1] = -1.;
					}
					else if(edge==1){
						proposed_qsample[0] = -1;
						proposed_qsample[N_vis-1] = 1.;
						proposed_qsample[N_vis+0] = -1.;
						proposed_qsample[N_vis+N_vis-1] = 1.;
					}
					else if(edge==2){
						proposed_qsample[0] = 1;
						proposed_qsample[N_vis-1] = -1.;
						proposed_qsample[N_vis+0] = 1.;
						proposed_qsample[N_vis+N_vis-1] = -1.;
					}
					else if(edge==3){
						proposed_qsample[0] = 1;
						proposed_qsample[N_vis-1] = 1.;
						proposed_qsample[N_vis+0] = 1.;
						proposed_qsample[N_vis+N_vis-1] = 1.;
					}
					proposed_qprob  += NDO(coeff,proposed_qsample,N_vis,M_hid,K_mix);
				}
	
				proposed_qprob *= (1./4.);
				
				// STEP 2b: accept or reject proposed sample -- ACCEPT ONLY AGENDA
				accept_q = acceptance(proposed_qprob,current_qprob);
				rnum_q = RANDOM_NUMBER(1);
				
				// reject if bad sample
				if(isnan(creal(NDO(coeff,proposed_qsample,N_vis,M_hid,K_mix))) || isnan(cimag(NDO(coeff,proposed_qsample,N_vis,M_hid,K_mix)))){
					acc_qcount = 0; // otherwise repeat
				}
				else{
					if(rnum_q <= accept_q){
						for(int i=0;i<2*N_vis;i++){
							current_qsample[i] = proposed_qsample[i]; // accept the proposed sample
						}
						acc_qcount++;
					}
					else{
						acc_qcount = 0; // otherwise repeat procedure until a sample is accepted
					}
				}
			}
			
			// STEP 3: HYBRID SAMPLING -- vary edge configuration -> repeats every 4 samples
			if(sqloop==0){
				current_qsample[0] = -1;
				current_qsample[N_vis-1] = -1.;
				current_qsample[N_vis+0] = -1.;
				current_qsample[N_vis+N_vis-1] = -1.;
			}
			else if(sqloop==1){
				current_qsample[0] = -1;
				current_qsample[N_vis-1] = 1.;
				current_qsample[N_vis+0] = -1.;
				current_qsample[N_vis+N_vis-1] = 1.;
			}
			else if(sqloop==2){
				current_qsample[0] = 1;
				current_qsample[N_vis-1] = -1.;
				current_qsample[N_vis+0] = 1.;
				current_qsample[N_vis+N_vis-1] = -1.;
			}
			else if(sqloop==3){
				current_qsample[0] = 1;
				current_qsample[N_vis-1] = 1.;
				current_qsample[N_vis+0] = 1.;
				current_qsample[N_vis+N_vis-1] = 1.;
			}
			sqloop++;
			
			qprob[n] = NDO(coeff,current_qsample,N_vis,M_hid,K_mix);
			
// ##################### END HYBRID QSAMPLING #####################

			// right now, rho/pprob/qprob are unnormalized - we need the partition functions
			// they are approximated using all of the drawn samples
			ppartition += pprob[n];
			qpartition += qprob[n];
	
// ################################################################
// ################## CALCULATE MEAN GRADIENT #####################
// ############### FOR ALL VARIATIONAL PARAMETERS #################
// ################################################################
	
#pragma omp parallel for
			for(int p=0;p<numcoeff;p++){
				meangrad[p] += cost_gradient_p(coeff,current_psample,p,N_vis,M_hid,K_mix) * pprob[n];
			}
	
// ################################################################
// ################### CALCULATE OBSERVABLES ######################
// ################################################################

			for(int i=0;i<N_vis;i++){
				if( (current_qsample[i]==-1.) && (current_qsample[N_vis+i]==-1.) ){
					occu_gs[i] += qprob[n];
				}
				if( (current_qsample[i]==1.) && (current_qsample[N_vis+i]==1.) ){
					occu_es[i] += qprob[n];
				}
			}

// ################################################################
// ################### CALCULATE LIOUVILLIAN ######################
// ################################################################

			hamilton = 0.;
			dissipator = 0.;
			
			hamilton = get_hamilton(coeff,current_psample,mparameters,N_vis,M_hid,K_mix);
			dissipator = get_dissipator(coeff,current_psample,mparameters,N_vis,M_hid,K_mix);
			
			// L(l1,r1,l2,r2) * rho(l2,r2) / rho(l1,r1)
			liouvillian[n] = hamilton + dissipator;
			
			meanliouv += liouvillian[n] * conj(liouvillian[n]) * pprob[n];
			
			if(isnan(cabs(hamilton))) {
				printf("hamilton is nan!\n"); 
				exit(0);
			}

// ################################################################
// ########## LIOUVILLIAN*GRADIENT FOR NEW COST GRADIENT ##########
// ################################################################

			hamilton_grad = 0.;
			dissipator_grad = 0.;
			
#pragma omp parallel for private(hamilton_grad,dissipator_grad)

			// L(l1,r1,l2,r2) * rho(l2,r2) * grad(l2,r2) / rho(l1,r1)
			for(int p=0;p<numcoeff;p++){
				hamilton_grad = get_hamilton_grad(coeff,current_psample,p,mparameters,N_vis,M_hid,K_mix);
				dissipator_grad = get_dissipator_grad(coeff,current_psample,p,mparameters,N_vis,M_hid,K_mix);
				
				liouv_grad[n][p] = hamilton_grad + dissipator_grad;
			}

		}
		// END OF SAMPLE LOOP
		
		
		// normalize everything containing pprob/qprob using the partition functions
		// otherwise pprob/qprob are not probabilities!
		for(int i=0;i<N_vis;i++){
			occu_gs[i] *= (1./qpartition);
			occu_es[i] *= (1./qpartition);
		}

		meanliouv *= (1./ppartition);
		for(int n=0;n<N_samples;n++){
			pprob[n] *= (1./ppartition);
		}
		for(int p=0;p<numcoeff;p++){
			meangrad[p] *= (1./ppartition);
		}
	
// ################################################################
// ############## CALCULATE COST FUNCTION GRADIENT ################
// ############### FOR STOCHASTIC GRADIENT DESCENT ################
// ################################################################

#pragma omp parallel for private(Ftemp)

		for(int p=0;p<numcoeff;p++){
			Ftemp = 0.;
			Fvec[p] = 0.;
			for(int n=0;n<N_samples;n++){
				Ftemp += pprob[n] * conj(liouvillian[n]) * liouv_grad[n][p];
			}
			Ftemp += -meangrad[p] * meanliouv;
			
			Fvec[p] = -2.*creal(Ftemp);
		}
	
// ################################################################
// #################### UPDATE COEFFICIENTS #######################
// ################################################################
		
#pragma omp parallel for
		
		// update iteration t -> t+1 via stochastic gradient descent:
		// coeff(t+1) = coeff(t) - lrate * Fvec
		for(int p=0;p<numcoeff;p++){
			coeff[p] += lrate * Fvec[p];
		}
		printf("Training finished for iteration %d out of %d\n",t,N_times);
		
// ######################## WRITE TO FILE #########################

		fprintf(f_dat,"%d \t",t);
		for(int i=0;i<N_vis;i++){
			fprintf(f_dat,"%.10f \t %.10f ",occu_gs[i],occu_es[i]);
			
			if(i<(N_vis-1)){
				fprintf(f_dat,"\t");
			}
			else{
				fprintf(f_dat,"\n");
			}
		}
	}
	// END OF TRAINING ITERATION LOOP
	
	
	// free all pointer arrays
	free(mparameters);
	free(liouvillian);
	free(coeff);
	free(current_psample);
	free(current_qsample);
	free(pprob);
	free(qprob);
	free(meangrad);
	free(Fvec);
	for(int n=0;n<N_samples;n++){
		free(liouv_grad[n]);
	}
	free(liouv_grad);
	free(occu_gs);
	free(occu_es);

	time(&end_t);
	diff_t = difftime(end_t, curtime);
	fprintf(f_dat,"# Execution time = %f in Seconds \n", diff_t);
	curtime = time(NULL); loctime = localtime (&curtime);
	hour = loctime -> tm_hour;minute = loctime -> tm_min;second = loctime -> tm_sec;
	year = 1900+loctime -> tm_year;month = loctime -> tm_mon;day  = loctime -> tm_mday;
	fprintf(f_dat,"# Date: %02d.%02d.%.d -- Time: %02d:%02d:%02d  \n",day,month,year,hour,minute,second);

	fclose(f_dat);

	return 0;
}

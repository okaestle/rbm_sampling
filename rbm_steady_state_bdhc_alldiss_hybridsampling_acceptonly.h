#include <math.h>
#include <stdlib.h>
#include <complex.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

// Random Number Generator (RNG)
// define max. initial value of variational coefficients
#define RandMax  (0.01)

// ########################## STOPWATCH ###########################

void show_remaining_time(int k,double seconds_diff,int N){
	double hour,min,sek;
	double rest,v;
   
	if(k>0) v = k/seconds_diff;
	else v = 1;
	rest = (N - k)/ v;
	hour = floor(rest/3600.);
	rest = rest - 3600.*floor(hour);
	min  = floor(rest/60.);
	sek  = rest - 60. * floor(min);
	printf("Time left: %02.0f:%02.0f:%02.0f -- ",hour,min,sek);
	return;
}

// #################################################################
// #################### RANDOM NUMBER GENERATOR ####################
// #################################################################

double RANDOM_NUMBER(int prob){
	
	// drand48() returns (double) random number in [0,1]
	// rand()%X returns (int) random number in [0,X-1]
	double temp;
	
	// random value in [0,RandMax] (even distribution)
	if(prob==0){
		temp = drand48()*RandMax;
		if(temp==0.) temp += 0.01;
	}
	// random value in [0,1] (even distribution)
	else if(prob==1){
		temp = drand48();
	}
	else{
		printf("Invalid RNG input! Choose 0/1.\n");
		exit(0);
	}
	return temp;
}


// #################################################################
// #################### METROPOLIS ALGORITHM: ######################
// ##################### SAMPLING FUNCTIONS ########################
// #################################################################

// sampler for probability q(l,l) = rho(l,l) (diagonal elements only)
double *qsampler(
	double *current_sample,
	int N_vis)
{	
	double rnum_a,rnum_b,rnum_c;
	double *proposed_sample = calloc(2*N_vis, sizeof(double));
	
	rnum_a = RANDOM_NUMBER(1);
	
	// option 1: flip each spin with 50% probability
	if(rnum_a < 0.99){

		for(int i=0;i<2*N_vis;i++){
			proposed_sample[i] = current_sample[i];
		}
		
		for(int i=0;i<N_vis;i++){
			rnum_b = RANDOM_NUMBER(1);
			if(rnum_b<0.5){
				proposed_sample[i] *= -1.;
				proposed_sample[N_vis+i] *= -1.;
			}
		}
	}

	// option 2: draw a new uniform random configuration, 1% probability
	else{
		for(int i=0;i<N_vis;i++){
			rnum_c = RANDOM_NUMBER(1);
			if(rnum_c < 0.5){
				proposed_sample[i] = -1.;
				proposed_sample[N_vis + i] = -1.;
			}
			else{
				proposed_sample[i] = 1.;
				proposed_sample[N_vis + i] = 1.;
			}
		}
		
	}
	return proposed_sample;
}

// sampler for probability p(l,r) = |rho(l,r)|^2
double *psampler(
	double *current_sample,
	int N_vis)
{
	double rnum_a,rnum_b,rnum_c;
	double *proposed_sample = calloc(2*N_vis, sizeof(double));
	
	rnum_a = RANDOM_NUMBER(1);
	
	// option 1: flip each spin with 50% probability
	if(rnum_a < 0.99){

		for(int i=0;i<2*N_vis;i++){
			proposed_sample[i] = current_sample[i];
		}
		
		for(int i=0;i<2*N_vis;i++){
			rnum_b = RANDOM_NUMBER(1);
			if(rnum_b<0.5){
				proposed_sample[i] *= -1.;
			}
		}
	}
	
	// option 2: draw a new uniform random configuration, 1% probability
	else{
		for(int i=0;i<2*N_vis;i++){
			rnum_c = RANDOM_NUMBER(1);
			if(rnum_c < 0.5){
				proposed_sample[i] = -1.;
			}
			else{
				proposed_sample[i] = 1.;
			}
		}
		
	}
	return proposed_sample;
}

// Metropolis acceptance probability function
double acceptance(
	complex double proposed_prob,
	complex double current_prob)
{
	complex double temp_accept;
	
	//temp_accept = exp(-current_prob/proposed_prob);
	temp_accept = proposed_prob/current_prob;
	
	if(cabs(temp_accept)<1.){
		return cabs(temp_accept);
	}
	else{
		return 1.;
	}
}


// #################################################################
// ################### COST FUNCTION GRADIENTS #####################
// #################################################################

// calculate the cost gradient for coefficient p = 0,...,numcoeff-1
complex double cost_gradient_p(
	double *coeff,
	double *spin,
	int p,
	int N_vis,
	int M_hid,
	int K_mix)
{
	int firstindex;
	int secondindex;
	int coefficient;
	
	// DEFINE CONSTANTS TO ACCESS COEFF/GRAD ELEMENTS EASILY
	int startWr = 2*N_vis + 2*M_hid;
	int startWi = startWr + M_hid*N_vis;
	int startcr = startWi + M_hid*N_vis;
	int startUr = startcr + K_mix;
	int startUi = startUr + K_mix*N_vis;
	
	//a_real
	if(p<N_vis){
		firstindex = p;
		secondindex = 0;
		coefficient = 0;	
	}
	//a_imag
	else if((p>=N_vis) && (p<2*N_vis)){
		firstindex = p-N_vis;
		secondindex = 0;
		coefficient = 1;
	}
	//b_real
	else if((p>=2*N_vis) && (p<2*N_vis+M_hid)){
		firstindex = p-2*N_vis;
		secondindex = 0;
		coefficient = 2;
	}
	//b_imag
	else if((p>=2*N_vis+M_hid) && (p<startWr)){
		firstindex = p-2*N_vis-M_hid;
		secondindex = 0;
		coefficient = 3;
	}
	//W_real and W_imag
	else if((p>=startWr) && (p<startcr)){
		for(int m=0;m<M_hid;m++){
			for(int i=0;i<N_vis;i++){
				if(p==startWr+m*N_vis+i){
					firstindex = m;
					secondindex = i;
					coefficient = 4;
					break;
				}
				if(p==startWi+m*N_vis+i){
					firstindex = m;
					secondindex = i;
					coefficient = 5;
					break;
				}
			}
		}
	}
	//c_real
	else if((p>=startcr) && (p<startUr)){
		firstindex = p-startcr;
		secondindex = 0;
		coefficient = 6;
	}
	//U_real and U_imag
	else if(p>=startUr){
		for(int k=0;k<K_mix;k++){
			for(int i=0;i<N_vis;i++){
				if(p==startUr+k*N_vis+i){
					firstindex = k;
					secondindex = i;
					coefficient = 7;
					break;
				}
				if(p==startUi+k*N_vis+i){
					firstindex = k;
					secondindex = i;
					coefficient = 8;
					break;
				}
			}
		}
	}
	
	
	complex double gradient = 0.;
	
	complex double Wleft = 0.;
	complex double Wright = 0.;
	complex double Uleft = 0.;
	complex double Uright = 0.;
	
	complex double Xi_b_Wl = 0.;
	complex double Xi_b_Wr = 0.;
	complex double Xi_c_Ulr = 0.;
	
	if((coefficient >= 2) && (coefficient <= 5)){
		for(int j=0;j<N_vis;j++){
			Wleft  += (coeff[startWr + firstindex*N_vis + j] + I*coeff[startWi + firstindex*N_vis + j]) * spin[j];
			Wright += (coeff[startWr + firstindex*N_vis + j] - I*coeff[startWi + firstindex*N_vis + j]) * spin[N_vis + j];
		}
		Xi_b_Wl  = ctanh( (coeff[2*N_vis + firstindex] + I*coeff[2*N_vis + M_hid + firstindex]) + Wleft );
		Xi_b_Wr  = ctanh( (coeff[2*N_vis + firstindex] - I*coeff[2*N_vis + M_hid + firstindex]) + Wright );
	}
	if(coefficient >= 6){
		for(int j=0;j<N_vis;j++){
			Uleft  += (coeff[startUr + firstindex*N_vis + j] + I*coeff[startUi + firstindex*N_vis + j]) * spin[j];
			Uright += (coeff[startUr + firstindex*N_vis + j] - I*coeff[startUi + firstindex*N_vis + j]) * spin[N_vis + j];
		}
		Xi_c_Ulr = ctanh( coeff[startcr + firstindex] + coeff[startcr + firstindex] + Uleft + Uright );
	}
	

	// calculate cost gradients of all coefficients
	// a_real
	if(coefficient==0){
		gradient = spin[firstindex] + spin[N_vis + firstindex];
	}
	// a_imag
	else if(coefficient==1){
		gradient = (spin[firstindex] - spin[N_vis + firstindex])*I; //*I
	}
	// b_real
	else if(coefficient==2){
		gradient = Xi_b_Wl + Xi_b_Wr;
	}
	// b_imag
	else if(coefficient==3){
		gradient = (Xi_b_Wl - Xi_b_Wr)*I; //*I
	}
	// W_real
	else if(coefficient==4){
		gradient = Xi_b_Wl * spin[secondindex] + Xi_b_Wr * spin[N_vis + secondindex];
	}
	// W_imag
	else if(coefficient==5){
		gradient = ( Xi_b_Wl * spin[secondindex] - Xi_b_Wr * spin[N_vis + secondindex])*I; //*I
	}
	// c_real (coefficient c_imag is not required)
	else if(coefficient==6){
		gradient = (Xi_c_Ulr + Xi_c_Ulr);
	}
	// U_real
	else if(coefficient==7){
		gradient = Xi_c_Ulr * (spin[secondindex] + spin[N_vis + secondindex]);
	}
	// U_imag
	else if(coefficient==8){
		gradient = Xi_c_Ulr * (spin[secondindex] - spin[N_vis + secondindex])*I; //*I
	}
	
	if(isnan(creal(gradient))) printf("gradient (creal) is nan!\n");
	if(isnan(cimag(gradient))) printf("gradient (cimag) is nan!\n");
	
	return gradient;
}

// #################################################################
// ################## NEURAL DENSITY OPERATOR ######################
// #################################################################

// here we calculate the neural density operator (NDO) -- the RBM estimation of the density matrix
// for a given spin (sample) configuration
// ATTENTION: unnormalized! partition function must be sampled for normalization

complex double NDO(
	double *coeff,
	double *spin,
	int N_vis,
	int M_hid,
	int K_mix)
{
	// DEFINE CONSTANTS TO ACCESS COEFF/GRAD ELEMENTS EASILY
	int startWr = 2*N_vis + 2*M_hid;
	int startWi = startWr + M_hid*N_vis;
	int startcr = startWi + M_hid*N_vis;
	int startUr = startcr + K_mix;
	int startUi = startUr + K_mix*N_vis;
	
	complex double *temp_Wleft = calloc(2*M_hid,sizeof(double));
	complex double *temp_Wright = calloc(2*M_hid,sizeof(double));
	complex double *temp_Ulr= calloc(2*K_mix,sizeof(double));
	
	complex double temp_alr = 0.;
	complex double hidden_X = 1.;
	complex double ancill_Y = 1.;
	complex double ndo_output = 0.;
	
	// calculate prod_m X_m
	for(int m=0;m<M_hid;m++){
		temp_Wleft[m] = 0.;
		temp_Wright[m] = 0.;
		
		for(int i=0;i<N_vis;i++){
			temp_Wleft[m] += ( coeff[startWr + m*N_vis + i] + I*coeff[startWi + m*N_vis + i] ) * spin[i];
			temp_Wright[m] += ( coeff[startWr + m*N_vis + i] - I*coeff[startWi + m*N_vis + i] ) * spin[N_vis + i];
		}
	
		hidden_X *=  ccosh( (coeff[2*N_vis + m] + I*coeff[2*N_vis + M_hid + m]) + temp_Wleft[m] )
					*ccosh( (coeff[2*N_vis + m] - I*coeff[2*N_vis + M_hid + m]) + temp_Wright[m] );
	}
	
	// calculate prod_k Y_k
	for(int k=0;k<K_mix;k++){
		temp_Ulr[k] = 0.;
		
		for(int i=0;i<N_vis;i++){
			temp_Ulr[k] += ( coeff[startUr + k*N_vis + i] + I*coeff[startUi + k*N_vis + i] ) * spin[i]
						 + ( coeff[startUr + k*N_vis + i] - I*coeff[startUi + k*N_vis + i] ) * spin[N_vis + i];
		}
		ancill_Y *=  ccosh( coeff[startcr + k] + coeff[startcr + k] + temp_Ulr[k] );
	}
	
	// calculate exponential factor
	for(int i=0;i<N_vis;i++){
		temp_alr += ( coeff[i] + I*coeff[N_vis + i] ) * spin[i]
				  + ( coeff[i] - I*coeff[N_vis + i] ) * spin[N_vis + i];
	}
	//printf("%.10f \t %.10f \t %.10f \n", cabs(temp_alr), cabs(hidden_X), cabs(ancill_Y));
	
	// full NDO
	ndo_output = 8.*cexp(temp_alr) * hidden_X * ancill_Y;
	
	free(temp_Wleft);
	free(temp_Wright);
	free(temp_Ulr);
	
	if(isnan(creal(ndo_output))) printf("target (creal) is nan!\n");
	if(isnan(cimag(ndo_output))) printf("target (cimag) is nan!\n");
	
	return ndo_output;
}

// #################################################################
// ######################## LIOUVILLIAN ############################
// ################# ISOTROPIC 3D HEISENBERG MODEL #################
// #################################################################

//contribution from -i[H,rho]
complex double get_hamilton(
	double *coeff,
	double *spin,
	double *mpara,
	int N_vis,
	int M_hid,
	int K_mix)
{
	
	complex double hamilton = 0.;
	
	// we need 3 samples of the NDO given a current sample (l,r)
	double *spin_l = calloc(2*N_vis,sizeof(double));
	double *spin_r = calloc(2*N_vis,sizeof(double));
	
	complex double rho = NDO(coeff,spin,N_vis,M_hid,K_mix); // NDO of current sample
	
	complex double rho_l = 0.; // NDO of current sample with sites l_i and l_(i+1) flipped
	complex double rho_r = 0.; // NDO of current sample with sites r_i and r_(i+1) flipped
	
	// loop over all sites
	for(int i=0;i<N_vis;i++){
		
		for(int j=0;j<2*N_vis;j++){
			spin_l[j] = spin[j];
			spin_r[j] = spin[j];
		}
		
		// flip sites
		if(i<(N_vis-1)){
			spin_l[i] *= -1.;
			spin_l[i+1] *= -1.;
			
			spin_r[N_vis+i] *= -1.;
			spin_r[N_vis+i+1] *= -1.;
		
			rho_l = NDO(coeff,spin_l,N_vis,M_hid,K_mix);
			rho_r = NDO(coeff,spin_r,N_vis,M_hid,K_mix);
		}
	
		
		// calculate the contribution of the Hamiltonian to the Liouvillian
		// mpara = [gamma_in, gamma_out, Jx, Jy, Jz, Bmag, dephasing]
	
		// 1. magnetic field terms
		if(spin[i]==1.){
			hamilton += -I*mpara[5];
		}
		if(spin[i]==-1.){
			hamilton += +I*mpara[5];
		}
		if(spin[N_vis+i]==1.){
			hamilton += +I*mpara[5];
		}
		if(spin[N_vis+i]==-1.){
			hamilton += -I*mpara[5];
		}
	
		// 2. J_x, J_y, J_z
		if(spin[i]==1.){
			if(i<(N_vis-1)){
				if(spin[i+1]==1.) hamilton += - I*mpara[2]*(rho_l/rho) //m_l-1, m_{l+1}-1
											  + I*mpara[3]*(rho_l/rho)
										      - I*mpara[4];
											
				if(spin[i+1]==-1.) hamilton += - I*mpara[2]*(rho_l/rho) //m_l-1, m_{l+1}+1
											   - I*mpara[3]*(rho_l/rho)
											   + I*mpara[4];
			}
		}
		if(spin[i]==-1.){
			if(i<(N_vis-1)){
				if(spin[i+1]==1.) hamilton += - I*mpara[2]*(rho_l/rho) //m_l+1, m_{l+1}-1
											  - I*mpara[3]*(rho_l/rho)
										      + I*mpara[4];
												
				if(spin[i+1]==-1.) hamilton += - I*mpara[2]*(rho_l/rho) //m_l+1, m_{l+1}+1
											   + I*mpara[3]*(rho_l/rho)
											   - I*mpara[4];
			}
		}
		if(spin[N_vis+i]==1.){
			if(i<(N_vis-1)){
				if(spin[N_vis+i+1]==1.) hamilton += + I*mpara[2]*(rho_r/rho) //n_l-1, n_{l+1}-1
													- I*mpara[3]*(rho_r/rho)
													+ I*mpara[4];
											
				if(spin[N_vis+i+1]==-1.) hamilton += + I*mpara[2]*(rho_r/rho) //n_l-1, n_{l+1}+1
													 + I*mpara[3]*(rho_r/rho)
													 - I*mpara[4];
			}
		}
		if(spin[N_vis+i]==-1.){
			if(i<(N_vis-1)){  
				if(spin[N_vis+i+1]==1.) hamilton += + I*mpara[2]*(rho_r/rho) //n_l+1, n_{l+1}-1
													+ I*mpara[3]*(rho_r/rho)
													- I*mpara[4];

				if(spin[N_vis+i+1]==-1.) hamilton += + I*mpara[2]*(rho_r/rho) //n_l+1, n_{l+1}+1
													 - I*mpara[3]*(rho_r/rho)
													 + I*mpara[4];
			}
		}
		rho_l = 0.;
		rho_r = 0.;
	}
	
	free(spin_l);
	free(spin_r);
	
	return hamilton;
}

// contribution from dissipators
complex double get_dissipator(
	double *coeff,
	double *spin,
	double *mpara,
	int N_vis,
	int M_hid,
	int K_mix)
{
	
	complex double dissipator = 0.;
	
	// we need 3 samples of the NDO given a current sample (l,r)
	double *spin_plus = calloc(2*N_vis,sizeof(double));
	double *spin_min = calloc(2*N_vis,sizeof(double));
	
	complex double rho = NDO(coeff,spin,N_vis,M_hid,K_mix); // NDO of current sample
	complex double rho_plus = 0.; // NDO of current sample with adjusted sites l_i=+1, r_i=+1
	complex double rho_min = 0.; // NDO of current sample with adjusted sites l_i=-1, r_i=-1
	
	double gamma_in =  mpara[0];
	double gamma_out = mpara[1];
	
	
	// calculate the contribution of the Dissipators to the Liouvillian
	// mpara = [gamma_in, gamma_out, Jx, Jy, Jz, Bmag, dephasing]

	for(int i=0;i<(N_vis);i++){
		
		if(i==0){
			gamma_in = mpara[0];
			gamma_out = mpara[1];
		}
		else if(i==(N_vis-1)){
			gamma_in = mpara[1];
			gamma_out = mpara[0];
		}
		else{
			gamma_in = mpara[1];
			gamma_out = mpara[1];
		}
		
		for(int j=0;j<2*N_vis;j++){
			spin_plus[j] = spin[j];
			spin_min[j] = spin[j];
		}
		
		// adjust sites
		spin_plus[i] = 1.;
		spin_plus[N_vis+i] = 1.;
		
		spin_min[i] = -1.;
		spin_min[N_vis+i] = -1.;
		
		rho_plus = NDO(coeff,spin_plus,N_vis,M_hid,K_mix);
		rho_min = NDO(coeff,spin_min,N_vis,M_hid,K_mix);
		
		
		// D[sigma_+] dissipator
		if(spin[i]==(1.)){
			if(spin[N_vis+i]==(1.)){
				dissipator += gamma_in * (rho_min/rho);
			}
		}
		if(spin[i]==-1.){
			dissipator += -gamma_in/2.;
		}
		if(spin[N_vis+i]==-1.){
			dissipator += -gamma_in/2.;
		}
		
		// D[sigma_-] dissipator
		if(spin[i]==(-1.)){
			if(spin[N_vis+i]==(-1.)){
				dissipator += gamma_out * (rho_plus/rho);
			}
		}
		if(spin[i]==1.){
			dissipator += -gamma_out/2.;
		}
		if(spin[N_vis+i]==1.){
			dissipator += -gamma_out/2.;
		}
	}
	
	// dephasing
	for(int i=1;i<(N_vis-1);i++){
		
		// D[sigma_-] dissipator
		if(spin[i]==(1.)){
			if(spin[N_vis+i]==(1.)){
				dissipator += mpara[6];
			}
		}
		if(spin[i]==1.){
			dissipator += -mpara[6]/2.;
		}
		if(spin[N_vis+i]==1.){
			dissipator += -mpara[6]/2.;
		}
	}

	free(spin_plus);
	free(spin_min);
	
	return dissipator;
}


//contribution from -i[H,rho]
complex double get_hamilton_grad(
	double *coeff,
	double *spin,
	int p,
	double *mpara,
	int N_vis,
	int M_hid,
	int K_mix)
{
	
	complex double hamilton = 0.;
	
	// we need 3 samples of the NDO given a current sample (l,r)
	double *spin_l = calloc(2*N_vis,sizeof(double));
	double *spin_r = calloc(2*N_vis,sizeof(double));
	
	complex double rho = NDO(coeff,spin,N_vis,M_hid,K_mix); // NDO of current sample
	complex double rho_l= 0.; // NDO of current sample with sites l_i and l_(i+1) flipped
	complex double rho_r = 0.; // NDO of current sample with sites r_i and r_(i+1) flipped
	
	// calculate gradients with respect to parameter p
	complex double grad = cost_gradient_p(coeff,spin,p,N_vis,M_hid,K_mix);
	complex double grad_l = 0.;
	complex double grad_r = 0.;
	
	// loop over all sites
	for(int i=0;i<N_vis;i++){
		
		for(int j=0;j<2*N_vis;j++){
			spin_l[j] = spin[j];
			spin_r[j] = spin[j];
		}
		
		// flip sites
		if(i<(N_vis-1)){
			spin_l[i] *= -1.;
			spin_l[i+1] *= -1.;
			
			spin_r[N_vis+i] *= -1.;
			spin_r[N_vis+i+1] *= -1.;
		
			rho_l = NDO(coeff,spin_l,N_vis,M_hid,K_mix);
			rho_r = NDO(coeff,spin_r,N_vis,M_hid,K_mix);
			
			// calculate gradients with respect to parameter p
			grad_l = cost_gradient_p(coeff,spin_l,p,N_vis,M_hid,K_mix);
			grad_r = cost_gradient_p(coeff,spin_r,p,N_vis,M_hid,K_mix);
		}
	
	
		// calculate the contribution of the Hamiltonian to the Liouvillian
		// mpara = [gamma_in, gamma_out, Jx, Jy, Jz, Bmag, dephasing]
	
		// 1. magnetic field terms
		if(spin[i]==1.){
			hamilton += -I*mpara[5]*grad;
		}
		if(spin[i]==-1.){
			hamilton += +I*mpara[5]*grad;
		}
		if(spin[N_vis+i]==1.){
			hamilton += +I*mpara[5]*grad;
		}
		if(spin[N_vis+i]==-1.){
			hamilton += -I*mpara[5]*grad;
		}
	
		// 2. J_x, J_y, J_z
		if(spin[i]==1.){
			if(i<(N_vis-1)){
				if(spin[i+1]==1.) hamilton += - I*mpara[2]*(rho_l/rho)*grad_l //m_l-1, m_{l+1}-1
											  + I*mpara[3]*(rho_l/rho)*grad_l
										      - I*mpara[4]*grad;
											
				if(spin[i+1]==-1.) hamilton += - I*mpara[2]*(rho_l/rho)*grad_l //m_l-1, m_{l+1}+1
											   - I*mpara[3]*(rho_l/rho)*grad_l
											   + I*mpara[4]*grad;
			}
		}
		if(spin[i]==-1.){
			if(i<(N_vis-1)){
				if(spin[i+1]==1.) hamilton += - I*mpara[2]*(rho_l/rho)*grad_l //m_l+1, m_{l+1}-1
											  - I*mpara[3]*(rho_l/rho)*grad_l
										      + I*mpara[4]*grad;
												
				if(spin[i+1]==-1.) hamilton += - I*mpara[2]*(rho_l/rho)*grad_l //m_l+1, m_{l+1}+1
											   + I*mpara[3]*(rho_l/rho)*grad_l
											   - I*mpara[4]*grad;
			}
		}
		if(spin[N_vis+i]==1.){
			if(i<(N_vis-1)){
				if(spin[N_vis+i+1]==1.) hamilton += + I*mpara[2]*(rho_r/rho)*grad_r //n_l-1, n_{l+1}-1
													- I*mpara[3]*(rho_r/rho)*grad_r
													+ I*mpara[4]*grad;
											
				if(spin[N_vis+i+1]==-1.) hamilton += + I*mpara[2]*(rho_r/rho)*grad_r //n_l-1, n_{l+1}+1
													 + I*mpara[3]*(rho_r/rho)*grad_r
													 - I*mpara[4]*grad;
			}
		}
		if(spin[N_vis+i]==-1.){
			if(i<(N_vis-1)){  
				if(spin[N_vis+i+1]==1.) hamilton += + I*mpara[2]*(rho_r/rho)*grad_r //n_l+1, n_{l+1}-1
													+ I*mpara[3]*(rho_r/rho)*grad_r
													- I*mpara[4]*grad;

				if(spin[N_vis+i+1]==-1.) hamilton += + I*mpara[2]*(rho_r/rho)*grad_r //n_l+1, n_{l+1}+1
													 - I*mpara[3]*(rho_r/rho)*grad_r
													 + I*mpara[4]*grad;
			}
		}
		rho_l = 0.;
		rho_r = 0.;
		grad_l = 0.;
		grad_r = 0.;
	}
	
	free(spin_l);
	free(spin_r);
	
	return hamilton;
}

// contribution from dissipator
complex double get_dissipator_grad(
	double *coeff,
	double *spin,
	int p,
	double *mpara,
	int N_vis,
	int M_hid,
	int K_mix)
{
	
	complex double dissipator = 0.;
	
	// we need 3 samples of the NDO given a current sample (l,r)
	double *spin_plus = calloc(2*N_vis,sizeof(double));
	double *spin_min = calloc(2*N_vis,sizeof(double));
	
	complex double rho = NDO(coeff,spin,N_vis,M_hid,K_mix); // NDO of current sample
	complex double rho_plus = 0.; // NDO of current sample with adjusted sites l_i=+1, r_i=+1
	complex double rho_min = 0.; // NDO of current sample with adjusted sites l_i=-1, r_i=-1
	
	// calculate gradients with respect to parameter p
	complex double grad = cost_gradient_p(coeff,spin,p,N_vis,M_hid,K_mix);
	complex double grad_plus = 0.;
	complex double grad_min = 0.;
	
	double gamma_in =  mpara[0];
	double gamma_out = mpara[1];
	
	
	// calculate the contribution of the Dissipators to the Liouvillian
	// mpara = [gamma_in, gamma_out, Jx, Jy, Jz, Bmag, dephasing]

	for(int i=0;i<(N_vis);i++){
		
		if(i==0){
			gamma_in = mpara[0];
			gamma_out = mpara[1];
		}
		else if(i==(N_vis-1)){
			gamma_in = mpara[1];
			gamma_out = mpara[0];
		}
		else{
			gamma_in = mpara[1];
			gamma_out = mpara[1];
		}
		
		
		for(int j=0;j<2*N_vis;j++){
			spin_plus[j] = spin[j];
			spin_min[j] = spin[j];
		}
		
		// adjust sites
		spin_plus[i] = 1.;
		spin_plus[N_vis+i] = 1.;
		
		spin_min[i] = -1.;
		spin_min[N_vis+i] = -1.;
		
		rho_plus = NDO(coeff,spin_plus,N_vis,M_hid,K_mix);
		rho_min = NDO(coeff,spin_min,N_vis,M_hid,K_mix);
		
		grad_plus = cost_gradient_p(coeff,spin_plus,p,N_vis,M_hid,K_mix);
		grad_min = cost_gradient_p(coeff,spin_min,p,N_vis,M_hid,K_mix);
		
		
		// D[sigma_+] dissipator
		if(spin[i]==(1.)){
			if(spin[N_vis+i]==(1.)){
				dissipator += gamma_in * (rho_min/rho) * grad_min;
			}
		}
		if(spin[i]==-1.){
			dissipator += -(gamma_in/2.) * grad;
		}
		if(spin[N_vis+i]==-1.){
			dissipator += -(gamma_in/2.) * grad;
		}
		
		// D[sigma_-] dissipator
		if(spin[i]==(-1.)){
			if(spin[N_vis+i]==(-1.)){
				dissipator += gamma_out * (rho_plus/rho) * grad_plus;
			}
		}
		if(spin[i]==1.){
			dissipator += -(gamma_out/2.) * grad;
		}
		if(spin[N_vis+i]==1.){
			dissipator += -(gamma_out/2.) * grad;
		}
	}
	
	// dephasing
	for(int i=1;i<(N_vis-1);i++){
		if(spin[i]==(1.)){
			if(spin[N_vis+i]==(1.)){
				dissipator += mpara[6] * grad;
			}
		}
		if(spin[i]==1.){
			dissipator += -(mpara[6]/2.) * grad;
		}
		if(spin[N_vis+i]==1.){
			dissipator += -(mpara[6]/2.) * grad;
		}
	}
	
	free(spin_plus);
	free(spin_min);
	
	return dissipator;
}

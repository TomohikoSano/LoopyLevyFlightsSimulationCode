#include "parameter.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>


/////////////Globally defined quantities///////
/////cut-off length at walls//
#define delta_x_cutoff 10*dt

/////Min. of Delta t////
#define TWINDOW_MIN 1.0

/////b*/////
#define b_star 1.0

////Stiffness of the tracer-core///
#define kex 0.0

////Cutoff of the tracer-core///
#define cutoff b_star

////Dipole strength////
#define dipole_p 1.0

///parameters for the distribution function//
#define bin_power 1.2
#define bin_time 10.0
#define twindow 10.0
#define COR_MAX 15
#define xmax_dt 20.0
#define TWINDOW_INT_MAX 7


///Integration time for each sample
#define DT_INT 16384000
#define encho 100
//// Total integration time = DT_INT * encho


////Total sample number
#define sample_num 10


//////////////////////////////////////////////////


int kk,kk_out,k_cor_bin, k_cor_data;

FILE *fp;


///////Position, director, velocity, force of the swimmers///////
double RX[NN],RY[NN],RZ[NN],Phi[NN],Theta[NN];
double RX_b[NN],RY_b[NN],RZ_b[NN],Phi_b[NN],Theta_b[NN];
double VX[NN],VY[NN],VZ[NN];
double FX[NN],FY[NN],FZ[NN];
double FX_b[NN],FY_b[NN],FZ_b[NN];
/////////////////////////////////////////////////////////////////

//////Position, velocity, force of the tracer//////
double X0,Y0,Z0,VX0,VY0,VZ0,X0_b,Y0_b,Z0_b,FX0,FY0,FZ0,FX0_b,FY0_b,FZ0_b;
double Xt0,Yt0,Zt0;
///////////////////////////////////////////////////


double output_time;

///////Viscosities//////
double gamma_vis  = 1.0;
double gamma_vis0 = 1.0;
////////////////////////


//////time step for integration/////
double dt = 1.0e-02;
////////////////////////////////////


/////bin for the time window////////
double DT_bin[COR_MAX];
int    DT_bin_int[COR_MAX];
////////////////////////////////////


/////Save array for position, force of the tracer/////
double RX_ANA[DT_INT],RY_ANA[DT_INT],RZ_ANA[DT_INT];
double FX_ANA[DT_INT],FY_ANA[DT_INT],FZ_ANA[DT_INT],FR_ANA[DT_INT];
//////////////////////////////////////////////////////



///////////////Save array for distribution function//////////////////
double lmax = 1.0e12;
double dx_bin_min;
int l_max_real;
double PDF_DX_i[200][TWINDOW_INT_MAX],PDF_DX_av[200][TWINDOW_INT_MAX];
double PDF_DX_sample[200][TWINDOW_INT_MAX][sample_num],PDF_DX_av_sum[200][TWINDOW_INT_MAX];
double PDF_DX_av_err[200][TWINDOW_INT_MAX];
//////////////////////////////////////////////////////////////////////




int main(int argc,char **argv){
    
    ///////// commands for MPI parallelization////////
    int rank,procs;
    MPI_Init(&argc,&argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    //////////////////////////////////////////////////
    
    /////////Main program////////////////
    //rank: seeds for random number//////
    main_loop(rank);
    /////////////////////////////////////
    
    //////////////////////////////////////////////////
    /////////End of the main simulation program///////
    //////////////////////////////////////////////////
    
    
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    
    ////////Followings are the average over different initial conditions///////
    
    ////////Settings of bin-width in power-law manner//////
    dx_bin_min = (xmax_dt/lmax);
    for(int l_p = 0; l_p < lmax; l_p++){
        double x_bin_max = pow(bin_power,l_p) * dx_bin_min;
        if(x_bin_max > xmax_dt){
            l_max_real = l_p;
            break;
        }
    }
    ///////////////////////////////////////////////////////
    
    
    ////////////////Calculation of average among different processes/////////////////////////
    int kp,l_t;
    for (l_t = 0; l_t < TWINDOW_INT_MAX; l_t++) {
        for(kp = 0; kp < l_max_real; kp++){
            MPI_Allreduce(&PDF_DX_av[kp][l_t], &PDF_DX_av_sum[kp][l_t], 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
            PDF_DX_av_sum[kp][l_t] = PDF_DX_av_sum[kp][l_t]/ (double)procs;
        }
    }
    /////////////////////////////////////////////////////////////////////////////////////////
    
    
    ////////////////Calculation of standard errors among different processes/////////////////////////
    double DIFF_FROM_MEAN;
    for (l_t = 0; l_t < TWINDOW_INT_MAX; l_t++) {
        for(kp = 0; kp < l_max_real; kp++){
            DIFF_FROM_MEAN = (PDF_DX_av[kp][l_t] - PDF_DX_av_sum[kp][l_t])*(PDF_DX_av[kp][l_t] - PDF_DX_av_sum[kp][l_t]);
            MPI_Allreduce(&DIFF_FROM_MEAN,&PDF_DX_av_err[kp][l_t], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
        ///Divide by # of proc
        for(kp = 0; kp < l_max_real; kp++){
            PDF_DX_av_err[kp][l_t] = sqrt(PDF_DX_av_err[kp][l_t]/ (double)procs);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    ////////////////////////////////////////////////
    
    
    ///////////////////////Saving all statistics into text file/////////////////////
    if (rank == 0) {
        double x_m,x_p;
        dx_bin_min = (xmax_dt/lmax);
        sprintf(fileoutput, "PDF_disp.txt");
        fp = fopen( fileoutput , "w");
        for(kp = 0; kp < l_max_real; kp++){
            x_m = pow(bin_power, kp)     * dx_bin_min;
            x_p = pow(bin_power, kp + 1) * dx_bin_min;
             
            fprintf(fp, "%e %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n",0.5*(x_m+x_p),
                     PDF_DX_av_sum[kp][0],PDF_DX_av_err[kp][0],
                     PDF_DX_av_sum[kp][1],PDF_DX_av_err[kp][1],
                     PDF_DX_av_sum[kp][2],PDF_DX_av_err[kp][2],
                     PDF_DX_av_sum[kp][3],PDF_DX_av_err[kp][3],
                     PDF_DX_av_sum[kp][4],PDF_DX_av_err[kp][4],
                     PDF_DX_av_sum[kp][5],PDF_DX_av_err[kp][5],
                     PDF_DX_av_sum[kp][6],PDF_DX_av_err[kp][6]);
        }
        fclose(fp);
    }
    ////////////////////////////////////////////////////////////////////////////
    MPI_Finalize();
}


void main_loop(int seed){
    
    /////calculation of max bin////
    dx_bin_min = (xmax_dt/lmax);
    for(int l_p = 0; l_p < lmax; l_p++){
        double x_bin_max = pow(bin_power,l_p) * dx_bin_min;
        if(x_bin_max > xmax_dt){
            l_max_real = l_p;
            break;
        }
    }
    ///////////////////////////////
    

    //////seed of random number/////
    init_genrand(seed);
    ////////////////////////////////

    
    /////Definition of bins for time window//////
    int countmax = DT_INT;
    DT_bin[0] = 0.0;
    for (k_cor_bin = 0; k_cor_bin < COR_MAX; k_cor_bin++) {
        DT_bin[k_cor_bin]     = pow(2.0, k_cor_bin) * TWINDOW_MIN;
        DT_bin_int[k_cor_bin] = (int) (DT_bin[k_cor_bin]/dt);
    }
    //////////////////////////////////////////////

    //////////Loops for different samples//////////
    int sample_int;
    for (sample_int = 0; sample_int < sample_num; sample_int++) {

        ///Resetting save array//
        for (k_cor_data = 0; k_cor_data < DT_INT; k_cor_data++) {
            FX_ANA[k_cor_data] = 0.0;
            FY_ANA[k_cor_data] = 0.0;
            FR_ANA[k_cor_data] = 0.0;
        }
        /////////////////////////
        
        /////Initial conditions/////
        initial();
        ////////////////////////////
        
        ////// Loops for a single sample//////
        output_time = 0.0;
        for (kk = 0; kk <= countmax * encho; kk++) {
			
            //////calculation of the force//////
            interaction();
            tracer_force();
            director();
            ///////////////////////////////////
            
            /////boundary treatment////////////
            boundary();
            ///////////////////////////////////
            
            /////updating the positions////////
            update_x();
            ///////////////////////////////////
            
            /////Saving protocols//////////////
            correlation_routine(kk);
            ///////////////////////////////////
            
            
            /////detection of errors/////////////
            if (fabs(X0) > LX || fabs(Y0) > LY || fabs(Z0) > LZ) {
                printf("tracer out of box %e %e %e, reset\n",X0,Y0,Z0);
                abort();
            }
            ///////////////////////////////////
        }
        
        
        ///if movie output is necessary////
        //output_movie();
        ///////////////////////////////////
        
        
        ////Calculation of statistics//////
        correlation_calc_sub_routine();
        ///////////////////////////////////
        
        ////Saving the data of statistics////////
        int l_t,bin_l;
        double x_m,x_p,x_i;
        
        for (l_t = 0; l_t < TWINDOW_INT_MAX; l_t++) {
            for(bin_l = 0; bin_l < l_max_real; bin_l++){
                PDF_DX_sample[bin_l][l_t][sample_int] = PDF_DX_i[bin_l][l_t];
            }
        }
        /////////////////////////////////////////
        
        
        ///////output the progress of the calculations////////
        sprintf(fileoutput, "progress.txt");
        fp = fopen( fileoutput , "a");
        fprintf(fp,"NODE = %d SAMPLE NUMBER: %d \n",seed,sample_int);
        fclose(fp);
        //////////////////////////////////////////////////////
        
    }
    
    ////////////////Calculation the average data over different samples///////////
    int l_t,bin_l,kp;
    
    for (sample_int = 0; sample_int < sample_num; sample_int++) {
        for (l_t = 0; l_t < TWINDOW_INT_MAX; l_t++) {
            for(bin_l = 0; bin_l < l_max_real; bin_l++){
                PDF_DX_av[bin_l][l_t] += PDF_DX_sample[bin_l][l_t][sample_int]/((double) sample_num);
            }
        }
    }
    //////////////////////////////////////////////////////////////////////////////
}


///////Save position and force trajectry every encho step///////
void correlation_routine(int k_correlation){
    
    if (k_correlation % encho == 0) {
        
      FX_ANA[((int)k_correlation/encho)] = FX0;
      FY_ANA[((int)k_correlation/encho)] = FY0;
      FZ_ANA[((int)k_correlation/encho)] = FZ0;
      FR_ANA[((int)k_correlation/encho)] = sqrt(FX0*FX0 + FY0*FY0 + FZ0*FZ0);
        
      RX_ANA[((int)k_correlation/encho)] = X0;
      RY_ANA[((int)k_correlation/encho)] = Y0;
      RZ_ANA[((int)k_correlation/encho)] = Z0;
    }
}
////////////////////////////////////////////////////////////////


//////If you need to make a movie, use this subroutine//////////
void output_movie(){
    
    int file_i,upto_i;
    int ini_neg = 100;
    int file_num;
    
    for(file_i = ini_neg; file_i < DT_INT; file_i++){
        
        if(file_i%1000 == 0){
        
            sprintf(fileoutput, "%07d.txt",file_i);
            fp = fopen( fileoutput , "w");
        
            for(upto_i = ini_neg; upto_i <= file_i; upto_i++){
                if(upto_i%10 == 0){
                    fprintf(fp,"%e %e %e %e %e %e %e\n",1.0*upto_i,
                            RX_ANA[upto_i],
                            RY_ANA[upto_i],
                            RZ_ANA[upto_i],
                            FX_ANA[upto_i],
                            FY_ANA[upto_i],
                            FZ_ANA[upto_i]);
                }
            }
            fclose(fp);
        }
    }
}
//////////////////////////////////////////////////////////////


/////Calculation of the displacement CDF/////////////
void correlation_calc_sub_routine(){
    int k_cor_routine,l_cor_routine;
    double DT_bin_int_sub;
    double data_num;
    double dx_msd,dy_msd,dz_msd;
    
    
    int l_pdf,bin_l;
    double x_m,x_p,x_i,y_i,z_i;
    
    int l_kankaku,l_t;
    for (l_t = 0; l_t < TWINDOW_INT_MAX; l_t++) {
        for(bin_l = 0; bin_l < l_max_real; bin_l++){
            PDF_DX_i[bin_l][l_t] = 0.0;
        }
    }
    dx_bin_min = (xmax_dt/lmax);
    for (l_t = 0; l_t < TWINDOW_INT_MAX; l_t++) {
        l_kankaku = (int)(pow(bin_time,l_t) * TWINDOW_MIN/(dt * ((double) encho)));
        
        for(l_pdf = 0;l_pdf < DT_INT - l_kankaku; l_pdf++){
            x_i = fabs(RX_ANA[l_pdf + l_kankaku] - RX_ANA[l_pdf])/(pow(bin_time,l_t) * TWINDOW_MIN);
            y_i = fabs(RY_ANA[l_pdf + l_kankaku] - RY_ANA[l_pdf])/(pow(bin_time,l_t) * TWINDOW_MIN);
            z_i = fabs(RZ_ANA[l_pdf + l_kankaku] - RZ_ANA[l_pdf])/(pow(bin_time,l_t) * TWINDOW_MIN);
            
            for(bin_l = 0; bin_l < l_max_real; bin_l++){
                x_m = 0.0;
                x_p = pow(bin_power, bin_l + 1) * dx_bin_min;
                if(x_i >= x_m && x_i < x_p){
                    PDF_DX_i[bin_l][l_t] = PDF_DX_i[bin_l][l_t] + 1.0;
                }
                if(y_i >= x_m && y_i < x_p){
                    PDF_DX_i[bin_l][l_t] = PDF_DX_i[bin_l][l_t] + 1.0;
                }
                if(z_i >= x_m && z_i < x_p){
                    PDF_DX_i[bin_l][l_t] = PDF_DX_i[bin_l][l_t] + 1.0;
                }
            }
        }
    }
}
/////////////////////////////////////////////////////


///////////Boundary conditions////////////////////////
void boundary(){
    int ii;
    double f_bound;
    for (ii = 0; ii < NN; ii++) {
        if (RX[ii] > LX - delta_x_cutoff) {
            Phi[ii] = M_PI + (M_PI * (genrand_real1()-0.5));
            Theta[ii] = acos(2.0*(genrand_real1()-0.5));
        }
        if (RX[ii] < -LX + delta_x_cutoff) {
            Phi[ii] = (M_PI * (genrand_real1()-0.5));
            Theta[ii] = acos(2.0*(genrand_real1()-0.5));
        }

        if (RY[ii] > LY - delta_x_cutoff) {
            Phi[ii] = 1.5*M_PI + (M_PI * (genrand_real1()-0.5));
            Theta[ii] = acos(2.0*(genrand_real1()-0.5));
        }
        if (RY[ii] < -LY + delta_x_cutoff) {
            Phi[ii] = 0.5*M_PI + (M_PI * (genrand_real1()-0.5));
            Theta[ii] = acos(2.0*(genrand_real1()-0.5));
        }
        if (RZ[ii] > LZ - delta_x_cutoff) {
            Phi[ii] = 2.0*(M_PI * (genrand_real1()-0.5));
            Theta[ii] = acos(-genrand_real1());
        }
        if (RZ[ii] < -LZ + delta_x_cutoff) {
            Phi[ii] = 2.0*(M_PI * (genrand_real1()-0.5));
            Theta[ii] = acos(genrand_real1());
        }
        
        if (fabs(RX[ii]) > LX || fabs(RY[ii]) > LY || fabs(RZ[ii]) > LZ) {
            printf("%d th particle out of box %e %e %e\n",ii,RX[ii],RY[ii],RZ[ii]);
            abort();
        }
    }
    f_bound = 0.0;
    if (X0 > LX - delta_x_cutoff) {
        f_bound = - lj_force_box(fabs(X0-LX));
        FX0 += f_bound;
    }
    if (X0 < -LX + delta_x_cutoff) {
        f_bound = lj_force_box(fabs(X0+LX));
        FX0 += f_bound;
    }
    if (Y0 > LY - delta_x_cutoff) {
        f_bound = - lj_force_box(fabs(Y0-LY));
        FY0 += f_bound;
    }
    if (Y0 < -LY + delta_x_cutoff) {
        f_bound = lj_force_box(fabs(Y0+LY));
        FY0 += f_bound;
    }
    if (Z0 > LZ - delta_x_cutoff) {
        f_bound = - lj_force_box(fabs(Z0-LZ));
        FZ0 += f_bound;
    }
    if (Z0 < -LZ + delta_x_cutoff) {
        f_bound = lj_force_box(fabs(Z0+LZ));
        FZ0 += f_bound;
    }
    if (fabs(X0) > LX || fabs(Y0) > LY || fabs(Z0) > LZ) {
        printf("tracer out of box\n");
        abort();
    }
}
//////////////////////////////////////////////////////////

////////Lenard-Jones potential/////////
double lj_force_box(double xx){
    return (e_LJ*pow(xx,-13));
}
///////////////////////////////////////


///////Initial conditions/////////////
void initial(){
    X0 = 0.0;
    Y0 = 0.0;
    Z0 = 0.0;
    
    X0_b = 0.0;
    Y0_b = 0.0;
    Z0_b = 0.0;
    
    VX0 = 0.0;
    VY0 = 0.0;
    VZ0 = 0.0;
    
    FX0 = 0.0;
    FY0 = 0.0;
    FZ0 = 0.0;
    
    FX0_b = 0.0;
    FY0_b = 0.0;
    FZ0_b = 0.0;
    
    int ii;
    for (ii = 0; ii < NN; ii++) {
        
        RX[ii] = -LX+delta_x_cutoff + (2.0 * (LX-delta_x_cutoff) * genrand_real1());
        RY[ii] = -LY+delta_x_cutoff + (2.0 * (LY-delta_x_cutoff) * genrand_real1());
        RZ[ii] = -LZ+delta_x_cutoff + (2.0 * (LZ-delta_x_cutoff) * genrand_real1());
        
        
        RX_b[ii] = RX[ii];
        RY_b[ii] = RY[ii];
        RZ_b[ii] = RZ[ii];
        
        VX[ii] = 0.0;
        VY[ii] = 0.0;
        VZ[ii] = 0.0;
        
        FX[ii] = 0.0;
        FY[ii] = 0.0;
        FZ[ii] = 0.0;
        
        FX_b[ii] = 0.0;
        FY_b[ii] = 0.0;
        FZ_b[ii] = 0.0;
        
        Phi[ii] = 2.0*M_PI * genrand_real1();
        Theta[ii] = M_PI * genrand_real1();
        
        if(sqrt(RX[ii]*RX[ii] + RY[ii]*RY[ii] + RZ[ii]*RZ[ii]) < cutoff){
            ii = ii - 1;
        }
    }
}
/////////////////////////////////////////////////////////


/////////////Subroutine for updating positions//////////
void update_x(){
	int ii;
	for (ii = 0; ii < NN; ii++) {
		RX_b[ii] = RX[ii];
		RY_b[ii] = RY[ii];
		RZ_b[ii] = RZ[ii];
		
		RX[ii] = RX[ii] + (dt * ((3.0 * FX[ii])-FX_b[ii])/(2.0*gamma_vis));
		RY[ii] = RY[ii] + (dt * ((3.0 * FY[ii])-FY_b[ii])/(2.0*gamma_vis));
		RZ[ii] = RZ[ii] + (dt * ((3.0 * FZ[ii])-FZ_b[ii])/(2.0*gamma_vis));
		
		VX[ii] = ((RX[ii] - RX_b[ii])/dt);
		VY[ii] = ((RY[ii] - RY_b[ii])/dt);
		VY[ii] = ((RY[ii] - RY_b[ii])/dt);
	}
	
	X0_b = X0;
	Y0_b = Y0;
	Z0_b = Z0;
	
	X0 = X0 + (dt * ((3.0 * FX0) - FX0_b)/(2.0*gamma_vis0));
	Y0 = Y0 + (dt * ((3.0 * FY0) - FY0_b)/(2.0*gamma_vis0));
	Z0 = Z0 + (dt * ((3.0 * FZ0) - FZ0_b)/(2.0*gamma_vis0));
	
	VX0 = (X0-X0_b)/dt;
	VY0 = (Y0-Y0_b)/dt;
	VZ0 = (Z0-Z0_b)/dt;
	
}
//////////////////////////////////////////////////////////


////////////Subroutine for the initialization //////////////
void interaction(){
	int ii,jj;
	double kyori,nx,ny,nz,njx,njy,njz;
	double fijx,fijy,fijz;
	double costhetaij;
	
	for (ii = 0; ii < NN; ii++) {
		FX_b[ii] = FX[ii];
		FY_b[ii] = FY[ii];
		FZ_b[ii] = FZ[ii];
		
		FX[ii] = 0.0;
		FY[ii] = 0.0;
		FZ[ii] = 0.0;
	}
}
////////////////////////////////////////////////////////

/////////Propulsion force///////////
void director(){
	int ii;
    
	for (ii = 0; ii < NN; ii++) {
		FX[ii] = FX[ii] + cos(Phi[ii])*sin(Theta[ii]);
		FY[ii] = FY[ii] + sin(Phi[ii])*sin(Theta[ii]);
		FZ[ii] = FZ[ii] + cos(Theta[ii]);
	}
}
/////////////////////////////////////


//////////Forces acting on the tracer//////////
void tracer_force(){
	int ii;
	double kyori,nx,ny,nz,nix,niy,niz;
	double costhetaij;
	double fix,fiy,fiz;
    
    double ovlap,fn;
	
	FX0_b = FX0;
	FY0_b = FY0;
	FZ0_b = FZ0;
	
	FX0 = 0.0;
	FY0 = 0.0;
	FZ0 = 0.0;
	
	for (ii = 0; ii < NN; ii++) {
		kyori = ((RX[ii] - X0) * (RX[ii] - X0)) + ((RY[ii] - Y0) * (RY[ii] - Y0)) + ((RZ[ii] - Z0) * (RZ[ii] - Z0));
		
		if (kyori > 0.0) {
			kyori = sqrt(kyori);
			
            if( kyori <= cutoff){
                ////inside core////
                nx = (X0 - RX[ii]) / (kyori);
                ny = (Y0 - RY[ii]) / (kyori);
                nz = (Z0 - RZ[ii]) / (kyori);
                
                ovlap = cutoff - kyori;
                
                fn = kex * ovlap;
                
                fix = fn * nx;
                fiy = fn * ny;
                fiz = fn * nz;
                
                FX0 += fix;
                FY0 += fiy;
                FZ0 += fiz;
                
                FX[ii] -= fix;
                FY[ii] -= fiy;
                FZ[ii] -= fiz;
                ///////////////////////
            }else{
                ////long-range force///
                fix = 0.0;
                fiy = 0.0;
                fiz = 0.0;
                
                nx = (X0 - RX[ii]) / (kyori);
                ny = (Y0 - RY[ii]) / (kyori);
                nz = (Z0 - RZ[ii]) / (kyori);
                
                nix = cos(Phi[ii]) * sin(Theta[ii]);
                niy = sin(Phi[ii]) * sin(Theta[ii]);
                niz = cos(Theta[ii]);
                
                costhetaij = (nx*nix + ny*niy + nz*niz);
                costhetaij = (3.0 * costhetaij * costhetaij) - 1.0;
                
                
                fix = dipole_p * costhetaij * nx/(kyori * kyori);
                fiy = dipole_p * costhetaij * ny/(kyori * kyori);
                fiz = dipole_p * costhetaij * nz/(kyori * kyori);
                
                
                FX0 += fix;
                FY0 += fiy;
                FZ0 += fiz;
                //////////////////////////
            }
		}
	}
}
//////////////////////////////////////////////






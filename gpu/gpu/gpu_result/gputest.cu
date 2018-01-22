//
//  main.cpp
//  DS
//
//  Created by Shubham Gupta on 31/03/17.
//  Copyright © 2017 Shubham Gupta. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <string.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <curand.h>
#include <curand_kernel.h>
#include <iomanip>
#include <ctime>

using namespace std;

#define PI 3.1415926535897932
#define DPI 6.283185307179586
#define SPI 1.772453850905516
#define BOLTZ 1.380658e-23
#define AVOG 6.022169e26


void ALLOCATE_GAS();
void HARD_SPHERE();
void ARGON();
void IDEAL_NITROGEN();
void REAL_OXYGEN();
void IDEAL_AIR();
void REAL_AIR();
void HELIUM_ARGON_XENON();
void OXYGEN_HYDROGEN();
void INITIALISE_SAMPLES();
void DERIVED_GAS_DATA();
void SET_INITIAL_STATE_1D();
void MOLECULES_ENTER_1D();
void FIND_CELL_1D(double &,int &,int &);
void FIND_CELL_MB_1D(double &,int &,int &,double &);
void RVELC(double &,double &,double &);
void SROT(int &,double &,double &);
void SVIB(int &,double &,int &, int&);
void SELE(int &,double &,double &);
void CQAX(double&,double &,double&);
void LBS(double,double,double&);
void REFLECT_1D(int&,int,double&);
void RBC(double &, double &, double & , double &, double &,double &);
void AIFX(double & ,double &, double & , double &, double &, double&, double &, double&);
void REMOVE_MOL(int &);
void INDEX_MOLS();
void SAMPLE_FLOW();
void ADAPT_CELLS_1D();
void EXTEND_MNM(double);
void DISSOCIATION();
void ENERGY(int ,double &);
void COLLISIONS();
void SETXT();
void READ_RESTART();
void WRITE_RESTART();
void READ_DATA();
void OUTPUT_RESULTS();
void MOLECULES_MOVE_1D();



class Managed 
{
public:
  void *operator new(size_t len) {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
    return ptr;
  }

  void operator delete(void *ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }
};

class CALC : public Managed
{
public:
    //declares the variables associated with the calculation
    int  NVER,MVER,IMEG,NREL,MOLSC,ISF,ISAD,ISECS,IGS,IREM,NNC,IMTS,ERROR,NLINE,ICLASS,NCLASS,NMCC,NMI,NMP,ICN;
    double FTIME,TLIM,FNUM,DTM,TREF,TSAMP,TOUT,SAMPRAT,OUTRAT,RANF,TOTCOLI,TOTMOVI,TENERGY,DTSAMP,DTOUT,TPOUT,FRACSAM,TOTMOV,TOTCOL,ENTMASS,ENTREM,CPDTM,TPDTM,TNORM,FNUMF;
    double *VNMAX,*TDISS,*TRECOMB,*ALOSS,*EME,*AJM,*COLL_TOTCOL;
    double **TCOL;

    void d_allocate(int x, double*&arr){
        cudaMallocManaged(&arr, x*sizeof(double));
    }
    void d_allocate(int x, int y, double**&arr){
        cudaMallocManaged(&arr, x*sizeof(double));
        for(int i =0; i< x; ++i)
            cudaMallocManaged(&arr[i], y*sizeof(double));
    }
    //NVER.MVER.NREL the version number
    //IMEG the initial number of megabytes to be used by the program
    //MOLSC the target number of molecules per sampling cell
    //FTIME the flow time
    //TLIM the time at which the calculation stops
    //FNUM the number of real molecules represented by each simulated molecule
    //CPDTM the maximum number of collisions per time step (standard 0.2)
    //TPDTM the maximum number of sampling cell transit times of the flow per time step
    //TOTMOV total molecule moves
    //TOTCOL total collisions
    //TDISS(L) dissociations of species L since sample reset
    //TRECOMB(L) recombinations of species L since sample reset
    //ENTMASS the current entry mass of which a fraction FREM is to be removed
    //ENTREM the remainder (always negative) after molecule removal
    //VNMAX(L) the maximum normal velocity component of species L
    //TCOL species dependent collision counter
    //ISF 0,1 for steady, unsteady flow sampling
    //ISAD 0,1 to not automatically adapt cells each output interval in unsteady sampling, 1 to automatically adapt
    //ISECS 0,1 for no secondary stream,a secondary stream that applies for positive values of x
    //IREM data item to set type of molecule removal
    //NNC 0 for normal collisions, 1 for nearest neighbor collisions
    //IMTS 0 for uniform move time steps, 1 for time steps that vary over the cells, 2 for fixed time steps
    //IGS 0 for initial gas, 1 for stream(s) or reference gas
    //ICLASS class of flow
    //NCLASS the dimension of PX for the class of flow
    //NMCC desired number of molecules in a collision cell
    //NMI the initial number of molecules
    //TNORM normalizing time (may vary e.g. mean collision time , or a transit time)
    //ALOSS(L) number of molecules of speciel L lost in the move rourine
    //EME(L) number of species L that enter the front boundary
    //AJM(L) the adjustment number to allow for negative downstream entry numbers
    //NMP the number of molecules at the start of the move routine
    //ICN 0 if molecules with ITYPE(2)=4 are not kept constant, 1 to keep molecule number constant
    //FNUMF adjustment factor that is applied to automatically generated value
};

class MOLECS : public Managed
{
    //declares the variables associated with the molecules
public:
    int *IPCELL,*IPSP,*ICREF,*IPCP;
    int **IPVIB;
    
    void i_allocate(int x, int *&arr){
        cudaMallocManaged(&arr, x*sizeof(int));
    }
    void i_allocate(int x, int y, int **&arr){
        cudaMallocManaged(&arr, x*sizeof(int));
        for(int i =0; i< x; ++i)
            cudaMallocManaged(&arr[i], y*sizeof(int));
    }
    
    double **PX,**PV;
    double *PTIM,*PROT,*PELE;
    
    void d_allocate(int x, double *&arr){
        cudaMallocManaged(&arr, x*sizeof(double));
        
    }
    void d_allocate(int x, int y, double **&arr){
        cudaMallocManaged(&arr, x*sizeof(double));
        for(int i =0; i< x; ++i){
            try{
                cudaMallocManaged(&arr[i], y*sizeof(double));
            }
            catch (std::bad_alloc& ba){
                std::cerr << "bad_alloc caught: " << ba.what() << '\n';
            }
        }
    }
    int NM,MNM;
    
    //PX(1,2 or 3,N) x,y,z position coordinates of molecule N
    //PTIM(N) molecule time
    //IPSP(N) the molecular species
    //IPCELL(N) the collision cell number
    //ICREF the cross-reference array (molecule numbers in order of collision cells)
    //IPCP(N) the code number of the last collision partner of molecule
    //PV(1-3,N) u,v,w velocity components
    //PROT(N) rotational energy
    //IPVIB(K,N) level of vibrational mode K of molecule N
    //PELE(N) electronic energy
    //NM number of molecules
    //MNM the maximum number of molecules
    
};

class GAS : public Managed
{
    
    //declares the variables associated with the molecular species and the stream definition
public:
    double RMAS,CXSS,RGFS,VMPM,FDEN,FPR,FMA,FPM,CTM;
    double FND[3],FTMP[3],FVTMP[3],VFX[3],VFY[3],TSURF[3],FSPEC[3],VSURF[3];
    double *ERS,*CR,*TNEX,*PSF,*SLER,*FP;
    double **FSP,**SP,**SPR,**SPV,**VMP;
    double ***SPM,***SPVM,***ENTR,***QELC,***SPRT;
    double ****SPEX,****SPRC,****SPRP;
    double *****SPREX;
    void d_allocate(int x, double *&arr){
        cudaMallocManaged(&arr, x*sizeof(double));
    }
    void d_allocate(int x, int y, double **&arr){
        cudaMallocManaged(&arr, x*sizeof(double));
        for(int i =0; i< x; ++i)
            cudaMallocManaged(&arr[i], y*sizeof(double));
    }
    void d_allocate(int x, int y, int z, double***&arr){
        cudaMallocManaged(&arr, x*sizeof(double));
        for (int i = 0; i < x; ++i)
        {
            cudaMallocManaged(&arr[i], y*sizeof(double));
            for (int j = 0; j < y; ++j)
                cudaMallocManaged(&arr[i][j], z*sizeof(double));
        }
        
    }
    void d_allocate(int x, int y, int z, int w, double ****&arr){
        cudaMallocManaged(&arr, x*sizeof(double));
        for (int i = 0; i < x; ++i)
        {
            cudaMallocManaged(&arr[i], y*sizeof(double));
            for (int j = 0; j < y; ++j)
            {
                cudaMallocManaged(&arr[i][j], z*sizeof(double));
                for(int k=0; k<z; ++k)
                    cudaMallocManaged(&arr[i][j][k], w*sizeof(double));
            }
        }
        
    }
    void d_allocate(int x, int y, int z, int w, int v, double*****&arr){
        cudaMallocManaged(&arr, x*sizeof(double));
        for (int i = 0; i < x; ++i)
        {
            cudaMallocManaged(&arr[i], y*sizeof(double));
            for (int j = 0; j < y; ++j)
            {
                cudaMallocManaged(&arr[i][j], z*sizeof(double));
                for(int k=0; k<z; ++k)
                {
                    cudaMallocManaged(&arr[i][j][k], w*sizeof(double));
                    for(int l=0; l<w; ++l)
                        cudaMallocManaged(&arr[i][j][k][l], v*sizeof(double));
                }
            }
        }
    }
    
    int MSP,MMVM,MMRM,MNSR,IGAS,MMEX,MEX,MELE,MVIBL;
    int *ISP,*ISPV,*NELL;
    int **ISPR,**LIS,**LRS,**ISRCD,**ISPRC,**ISPRK,**TREACG,**TREACL,**NSPEX,**NSLEV;
    int ***ISPVM,***NEX;
    int ****ISPEX;
    void i_allocate(int x, int *&arr){
        cudaMallocManaged(&arr, x);
    }
    void i_allocate(int x, int y, int **&arr){
        cudaMallocManaged(&arr, x*sizeof(int));
        for(int i =0; i< x; ++i)
            cudaMallocManaged(&arr[i], y*sizeof(int));
    }
    void i_allocate(int x, int y, int z, int ***&arr){
        cudaMallocManaged(&arr, x*sizeof(int));
        for (int i = 0; i < x; ++i)
        {
            cudaMallocManaged(&arr[i], y*sizeof(int));
            for (int j = 0; j < y; ++j)
                cudaMallocManaged(&arr[i][j], z*sizeof(int));
        }
        
    }
    void i_allocate(int x, int y, int z, int w, int ****&arr){
        cudaMallocManaged(&arr, x*sizeof(int));
        for (int i = 0; i < x; ++i)
        {
            cudaMallocManaged(&arr[i], y*sizeof(int));
            for (int j = 0; j < y; ++j)
            {
                cudaMallocManaged(&arr[i][j], z*sizeof(int));
                for(int k=0; k<z; ++k)
                    cudaMallocManaged(&arr[i][j][k], w*sizeof(int));
            }
        }
    }
    
    //MSP the number of molecular species
    //MMVM the maximum number of vibrational modes of any species
    //MEX number of exchange or chain reactions
    //MELE the maximum number of electronic states of any molecule
    //MVIBL the maximum number of vibrational levels for detailed balance lists
    //MMEX the maximum number of exchange reactions involving the same precollision pair of molecules
    //MMRM 0 if gass is completely monatomic, 1 if some species have rotation
    //MNSR the number oF surface reactions
    //SP(1,L) the reference diameter of species L
    //SP(2,L) the reference temperature of species L
    //SP(3,L) the viscosity-temperature power law of species L
    //SP(4,L) the reciprocal of the VSS scattering parameter
    //SP(5,L) molecular mass of species L
    //SP(6,L) the heat of formation at 273 K.
    //ISPR(1,L) number of rotational degrees of freedom of species L
    //ISPR(2,L) 0,1 for constant, polynomial rotational relaxation collision number
    //SPR(1,L) constant rotational relaxation collision number of species L
    //          or the constant in a second order polynomial in temperature
    //SPR(2,L) the coefficient of temperature in the polynomial
    //SPR(3,L) the coefficient of temperature squared in the polynomial
    //SPM(1,L,M) the reduced mass for species L,M
    //SPM(2,L,M) the reference collision cross-section for species L,M
    //SPM(3,L,M) the mean value of the viscosity-temperature power law
    //SPM(4,L,M) the reference diameter for L,M collisions
    //SPM(5,L,M) the reference temperature for species L,M
    //SPM(6,L,M) reciprocal of the gamma function of (5/2-w) for species L,M
    //SPM(7,L,M) rotational relaxation collision number for species L,M, or const in polynomial
    //SPM(8,L,M) reciprocal of VSS scattering parameter
    //ISPV(L) the number of vibrational modes
    //SPVM(1,K,L) the characteristic vibrational temperature
    //SPVM(2,K,L) constant Zv, or reference Zv for mode K
    //SPVM(3,K,L) -1. for constant Zv, or reference temperature
    //SPVM(4,K,L) the characteristic dissociation temperature
    //SPVM(5,K,L) the arbitrary rate reduction factor
    //ISPVM(1,K,L) the species code of the first dissociation product
    //ISPVM(2,K,L) the species code of the second dissociation product
    //NELL(L) the number of electronic levels of species L
    //QELC(N,M,L) for up to M levels of form g*exp(-a/T) in the electronic partition function for species L
    //            N=1 for the degeneracy g
    //            N=2 for the coefficient a
    //            N=3 for the ratio of the excitation cross-section to the elastic cross-section
    //ISPRC(L,M) the species of the recombined molecule from species L and M
    //ISPRK(L,M) the applicable vibrational mode of this species
    //SPRC(1,L,M,K) the constant a in the ternary collision volume
    //SPRC(2,L,M,K) the temperature exponent b in the ternary collision volume
    //SPRT(1,L,M) lower temperature value for SPRP
    //SPRT(2,L,M) higher temperature value for SPRP
    //SPRP(1,L,M,K) the cumulative dissociation distribution to level K for products L and M at the lower temperature
    //SPRP(2,L,M,K) ditto at higher temperature, for application to post-recombination molecule//
    //NSPEX(L,M) the number of exchange reactios with L,M as the pre-collision species
    //in the following variables, J is the reaction number (1 to NSPEX(L,M))
    //ISPEX(J,1,L,M) the species that splits in an exchange reaction
    //ISPEX(J,2,L,M) the other pre-reaction species (all ISPEX are set to 0 if no exchange reaction)
    //ISPEX(J,3,L,M) the post-reaction molecule that splits in the opposite reaction
    //ISPEX(J,4,L,M) the other post-reaction species
    //ISPEX(J,5,L,M) the vibrational mode of the molecule that splits
    //ISPEX(J,6,L,M) degeneracy of this reaction
    //ISPEX(J,7,L,M) the vibrational mode of the molecule that splits
    //SPEX(1,J,L,M) the constant a in the reaction probability for the reverse reaction
    //SPEX(2,J,L,M) the temperature exponent b in the reaction probability (reverse reaction only)
    //SPEX(3,J,L,M)  for the heat of reaction
    //SPEX(4,J,L,M)   the lower temperature for SPREX
    //SPEX(5,J,L,M)   the higher temperature for SPREX
    //SPEX(6,J,L,M)   the energy barrier
    //SPREX(1,J,L,M,K) at lower temperature, the Jth reverse exchange reaction of L,M cumulative level K viv. dist of post reac mol
    //SPREX(2,J,L,M,K) ditto at higher temperature
    //TNEX(N) total number of exchange reaction N
    //NEX(N,L,M) the code number of the Nth exchange or chain reaction in L,M collisions
    //RMAS reduced mass for single species case
    //CXSS reference cross-section for single species case
    //RGFS reciprocal of gamma function for single species case
    //for the following, J=1 for the reference gas and/or the minimum x boundary, J=2 for the secondary sream at maximum x boundary
    //FND(J) stream or reference gas number density
    //FTMP(J) stream temperature
    //FVTMP(J) the vibrational and any electronic temperature in the freestream
    //VFX(J)  the x velocity components of the stream
    //VFY(J) the y velocity component in the stream
    //FSP(N,J)) fraction of species N in the stream
    //FMA stream Mach number
    //VMP(N,J) most probable molecular velocity of species N at FTMP(J)
    //VMPM the maximum value of VMP in stream 1
    //ENTR(M,L,K) entry/removal information for species L at K=1 for 1, K=2 for XB(2)
    //    M=1 number per unut time
    //    M=2 remainder
    //    M=3 speed ratio
    //   M=4 first constant
    //    M=5 second constant
    //    M=6 the maxinum normal velocity component in the removal zone (> XREM)
    //LIS(1,N) the species code of the first incident molecule
    //LIS(2,N) the species code of the second incident molecule (0 if none)
    //LRS(1,N) the species code of the first reflected molecule
    //LRS(2,N) the species code of the second reflected molecule (0 if none)
    //LRS(3,N) the species code of the third reflected molecule (0 if none)
    //LRS(4,N) the species code of the fourth reflected molecule (0 if none)
    //LRS(5,N) the species code of the fifth reflected molecule (0 if none)
    //LRS(6,N) the species code of the sixth reflected molecule (0 if none)
    //ERS(N) the energy of the reaction (+ve for recombination, -ve for dissociation)
    //NSRSP(L) number of surface reactions that involve species L as incident molecule
    //ISRCD(N,L) code number of Nth surface reaction with species L as incident molecule
    //CTM mean collision time in stream
    //FPM mean free path in stream
    //FDEN stream 1 density
    //FPR stream 1 pressure
    //FMA stream 1 Mach number
    //RMAS reduced mass for single species case
    //CXSS reference cross-section for single species case
    //RGFS reciprocal of gamma function for single species case
    //CR(L) collision rate of species L
    //FP(L) mean free path of species L
    //TREACG(N,L) the total number of species L gained from reaction type N=1 for dissociation, 2 for recombination, 3 for forward exchange, 4 for reverse exchange
    //TREACL(N,L) the total number of species L lost from reaction type N=1 for dissociation, 2 for recombination, 3 for forward exchange, 4 for reverse exchange
    //NSLEV(2,L)  1 exo, 2 endo: vibrational levels to be made up for species L in detailed balance enforcement after reaction
    //SLER(L) rotational energy to be made up for species L in detailed balance enforcement after exothermic reaction
};
class OUTPUT : public Managed
{
public:
    //declares the variables associated with the sampling and output
    int NSAMP,NMISAMP,NOUT,NDISSOC,NRECOMB,NTSAMP;
    //int NDISSL[201];
    int *NDISSL;
    OUTPUT(){
        cudaMallocManaged(&NDISSL,201*sizeof(int));
    };
    double TISAMP,XVELS,YVELS,AVDTM;
    double *COLLS,*WCOLLS,*CLSEP,*SREAC,*STEMP,*TRANSTEMP,*ROTTEMP,*VIBTEMP,*ELTEMP;
    double **VAR,**VARS,**CSSS,**SUMVIB;
    double ***CS,***VARSP,***VIBFRAC;
    double ****CSS;
    void d_allocate(int x, double *&arr){
        cudaMallocManaged(&arr, x*sizeof(double));
    }
    void d_allocate(int x, int y, double **&arr){
        cudaMallocManaged(&arr, x*sizeof(double));
        for(int i =0; i< x; ++i)
            cudaMallocManaged(&arr[i], y*sizeof(double));
    }
    void d_allocate(int x, int y, int z, double ***&arr){
        cudaMallocManaged(&arr, x*sizeof(double));
        for (int i = 0; i < x; ++i)
        {
            cudaMallocManaged(&arr[i], y*sizeof(double));
            for (int j = 0; j < y; ++j)
                cudaMallocManaged(&arr[i][j], z*sizeof(double));
        }
    }
    void d_allocate(int x, int y, int z, int w, double ****&arr){
        cudaMallocManaged(&arr, x*sizeof(double));
        for (int i = 0; i < x; ++i)
        {
            cudaMallocManaged(&arr[i], y*sizeof(double));
            for (int j = 0; j < y; ++j)
            {
                cudaMallocManaged(&arr[i][j], z*sizeof(double));
                for(int k=0; k<z; ++k)
                    cudaMallocManaged(&arr[i][j][k], w*sizeof(double));
            }
        }
    }
    //NSAMP the number of samples
    //TISAMP the time at which the sampling was last reset
    //MNISAMP the number of molecules at the last reset
    //AVDTM the average value of DTM in the cells
    //NOUT the number of output intervals
    //COLLS(N) total number of collisions in sampling cell N
    //WCOLLS(N) total weighted collisins in N
    //CLSEP(N) sum of collision pair separation in cell N
    //CS(0,N,L) sampled number of species L in cell N
    //CS(1,N,L) sampled weighted number of species L in cell N
    //--all the following CS are weighted sums
    //CS(2,N,L), CS(3,N,L), CS(4,N,L) sampled sum of u, v, w
    //CS(5,N,L), CS(6,N,L), CS(7,N,L) sampled sum of u*u, v*v, w*w
    //CS(8,N,L) sampled sum of rotational energy of species L in cell N
    //CS(9,N,L) sampled sum of electronic energy of species L in cell N
    //CS(9+K,N,L) sampled sum of vibrational level of species L in cell N
    //              K is the mode
    //
    //in CSS, M=1 for incident molecules and M=2 for reflected molecules
    //J=1 for surface at x=XB(1), 2 for surface at x=XB(2)
    //
    //CSS(0,J,L,M) number sum of molecules of species L
    //CSS(1,J,L,M) weighted number sum of molecules of species L
    //--all the following CSS are weighted
    //CSS(2,J,L,M) normal momentum sum to surface
    //CSS(3,J,L,M) y momentum sum to surface
    //CSS(4,J,L,M) z momentum sum to surface
    //CSS(5,J,L,M) tranlational energy sum to surface
    //CSS(6,J,L,M) rotational energy sum to surface
    //CSS(7,J,L,M) vibrational energy sum to the surface
    //CSS(8,J,L,M) electronic energy sum to the surface
    //
    //CSSS(1,J) weighted sum (over incident AND reflected molecules) of 1/normal vel. component
    //--all the following CSSS are weighted
    //CSSS(2,J) similar sum of molecular mass / normal vel. component
    //CSSS(3,J) similar sum of molecular mass * parallel vel. component / normal vel. component
    //CSSS(4,J) similar sum of molecular mass * speed squared / normal vel. component
    //CSSS(5,J) similar sum of rotational energy / normal vel. component
    //CSSS(6,J) similar sum of rotational degrees of freedom /normal velocity component
    //
    //SREAC(N) the number of type N surface reactions
    //
    //VAR(M,N) the flowfield properties in cell N
    //M=1 the x coordinate
    //M=2 sample size
    //M=3 number density
    //M=4 density
    //M=5 u velocity component
    //M=6 v velocity component
    //M=7 w velocity component
    //M=8 translational temperature
    //M=9 rotational temperature
    //M=10 vibrational temperature
    //M=11 temperature
    //M=12 Mach number
    //M=13 molecules per cell
    //M=14 mean collision time / rate
    //M=15 mean free path
    //M=16 ratio (mean collisional separation) / (mean free path)
    //M=17 flow speed
    //M=18 scalar pressure nkT
    //M=19 x component of translational temperature TTX
    //M=20 y component of translational temperature TTY
    //M=21 z component of translational temperature TTZ
    //M=22 electronic temperature
    //
    //VARSP(M,N,L) the flowfield properties for species L in cell N
    //M=0 the sample size
    //M=1 the fraction
    //M=2 the temperature component in the x direction
    //M=3 the temperature component in the y direction
    //M=4 the temperature component in the z direction
    //M=5 the translational temperature
    //M=6 the rotational temperature
    //M=7 the vibrational temperature
    //M=8 the temperature
    //M=9 the x component of the diffusion velocity
    //M=10 the y component of the diffusion velocity
    //M=11 the z component of the diffusion velocity
    //M=12 the electronic temperature
    //
    //VARS(N,M) surface property N on interval L of surface M
    //
    //N=0 the unweighted sample (remainder of variables are weighted for cyl. and sph. flows)
    //N=1 the incident sample
    //N=2 the reflected sample
    //N=3 the incident number flux
    //N=4 the reflected number flux
    //N=5 the incident pressure
    //N=6 the reflected pressure
    //N=7 the incident parallel shear tress
    //N=8 the reflected parallel shear stress
    //N=9 the incident normal-to-plane shear stress
    //N=10 the reflected normal shear stress
    //N=11 the incident translational heat flux
    //N=12 the reflected translational heat fluc
    //N=13 the incident rotational heat flux
    //N=14 the reflected rotational heat flux
    //N=15 the incident vibrational heat flux
    //N=16 the reflected vibrational heat flux
    //N=17 the incident heat flux from surface reactions
    //N=18 the reflected heat flux from surface reactions
    //N=19 slip velocity
    //N=20 temperature slip
    //N=21 rotational temperature slip
    //N=22 the net pressure
    //N=23 the net parallel in-plane shear
    //N=24 the net parallel normal-to-plane shear
    //N=25 the net translational energy flux
    //N=26 the net rotational heat flux
    //N=27 the net vibrational heat flux
    //N=28 the heat flux from reactions
    //N=29 total incident heat transfer
    //N=30 total reflected heat transfer
    //N=31 net heat transfer
    //N=32 surface temperature   --not implemented
    //N=33 incident electronic energy
    //N=34 reflected electronic energy
    //N=35 net electronic energy
    //N=35+K the percentage of species K
    //
    //COLLS(N) the number of collisions in sampling cell N
    //WCOLLS(N) weighted number
    //CLSEP(N) the total collision partner separation distance in sampling cell N
    //
    //VIBFRAC(L,K,M) the sum of species L mode K in level M
    //SUMVIB(L,K) the total sample in VIBFRAC
    //
    //THE following variables apply in the sampling of distribution functions
    //(some are especially for the dissociation of oxygen
    //
    //NDISSOC the number of dissociations
    //NRECOMB the number of recombinations
    //NDISSL(L) the number of dissociations from level
    //NTSAMP the number of temperature samples
    //STEMP(L) the temperature of species L
    //TRANSTEMP(L) the translational temperature of species N
    //ROTTEMP(L) rotational temperature of species N
    //VIBTEMP(L) vibrational temperature of species N
    //ELTEMP(L) electronic temperature of species N
    //
};

class GEOM_1D : public Managed
{
public:
    //declares the variables associated with the flowfield geometry and cell structure
    //for homogeneous gas and one-dimensional flow studies
    int NCELLS,NCCELLS,NCIS,NDIV,MDIV,ILEVEL,IFX,JFX,IVB,IWF;
    //int ITYPE[3];
    int *ITYPE;
    int *ICELL;
    int ** ICCELL,**JDIV;
    void i_allocate(int x, int *&arr){
        cudaMallocManaged(&arr, x*sizeof(int));
    }
    void i_allocate(int x, int y, int **&arr){
        cudaMallocManaged(&arr, x*sizeof(int));
        for(int i =0; i< x; ++i)
            cudaMallocManaged(&arr[i], y*sizeof(int));
    }
    double DDIV,XS,VELOB,WFM,AWF,FREM,XREM;
    //double XB[3];
    double *XB;
    double **CELL,**CCELL;
    void d_allocate(int x, int y, double**&arr){
        cudaMallocManaged(&arr, x*sizeof(double));
        for(int i =0; i< x; ++i)
            cudaMallocManaged(&arr[i], y*sizeof(double));
    }
    GEOM_1D(){
        cudaMallocManaged(&ITYPE, 3*sizeof(int));
        cudaMallocManaged(&XB, 3*sizeof(double));   
    }
    //
    //XB(1), XB(2) the minimum, maximum x coordinate
    //DDIV the width of a division
    //ITYPE(K) the tpe of boundary at the minimum x (K=1) and maximum x (K=2) boundaries
    //          0 for a stream boundary
    //          1 for a plane of symmetry
    //          2 for a solid surface
    //          3 for a vacuum
    //NCELLS the number of sampling cells
    //NCCELLS the number of collision cells
    //NCIS the number of collision cells in a sampling cell
    //  MDIV the maximum number of sampling cell divisions at any level of subdivision
    //IVB 0,1 for stationary, moving outer boundary
    //IWF 0 for no radial weighting factors, 1 for radial weighting factors
    //WFM, set in data as the maximum weighting factor, then divided by the maximum radius
    //AWF overall ratio of real to weighted molecules
    //VELOB the speed of the outer boundary
    //ILEV level of subdivision in adaption (0 before adaption)
    //JDIV(N,M) (-cell number) or (start address -1 in JDIV(N+1,M), where M is MDIV
    //IFX 0 for plane flow, 1 for cylindrical flow, 3 for spherical flow
    //JFX  IFX+1
    //CELL(M,N) information on sampling cell N
    //    M=1 x coordinate
    //    M=2 minimum x coordinate
    //    M=3 maximum x cooedinate
    //    M=4 volume
    //ICELL(N) number of collision cells preceding those in sampling cell N
    //CCELL(M,N) information on collision cell N
    //    M=1 volume
    //    M=2 remainder in collision counting
    //    M=3 half the allowed time step
    //    M=4 maximum value of product of cross-section and relative velocity
    //    M=5 collision cell time
    //ICCELL(M,N) integer information on collision cell N
    //    M=1 the (start address -1) in ICREF of molecules in collision cell N
    //    M=2 the number of molecules in collision cell N
    //    M=3 the sampling cell in which the collision cell lies
    //FREM fraction of molecule removal
    //XREM the coordinate at which removal commences
    //
};

double colltime=0.0;
clock_t start;
fstream file_9;
fstream file_18;
CALC *calc = new CALC;
GAS *gas = new GAS;
MOLECS *molecs = new MOLECS;
GEOM_1D *geom = new GEOM_1D;
OUTPUT *output =new OUTPUT;


// __device__ double atomicAdd(double* address, double val)
// {
//     unsigned long long int* address_as_ull =
//                              (unsigned long long int*)address;
//     unsigned long long int old = *address_as_ull, assumed;
//     do {
//         assumed = old;
// old = atomicCAS(address_as_ull, assumed,
//                         __double_as_longlong(val +
//                                __longlong_as_double(assumed)));
//     } while (assumed != old);
//     return __longlong_as_double(old);
// }

__device__ float generate( curandState* globalState, int ind )
{
    //int ind = threadIdx.x;
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState;
    return RANDOM;
}

__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
}

__device__ void SROT(curandState* globalState, int &L,double &TEMP,double &ROTE, GAS* gas, CALC *calc)
{
    int I;
    double A,B,ERM;
    //
    if(gas->ISPR[1][L] == 2){
        // CALL RANDOM_NUMBER(RANF)
        calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
        ROTE=-logf(calc->RANF)*BOLTZ*TEMP;   //equation (4.8)
    }
    else{
        A=0.5e00*gas->ISPR[1][L]-1.e00;
        I=0;
        while(I == 0){
            // CALL RANDOM_NUMBER(RANF)
            calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
            ERM=calc->RANF*10.e00;
            //there is an energy cut-off at 10 kT
            B=(powf((ERM/A),A))*expf(A-ERM);      //equation (4.9)
            // CALL RANDOM_NUMBER(RANF)
            calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
            if(B > calc->RANF) I=1;
        }
        ROTE=ERM*BOLTZ*TEMP;
    }
    return;
}

__device__ void SVIB(curandState* globalState, int &L,double &TEMP,int &IVIB, int &K, GAS *gas, CALC *calc)
{
    //sets a typical vibrational state at temp. TEMP of mode K of species L
    //GAS gas;
    //CALC calc;
    //
    // IMPLICIT NONE
    //
    int N;
    //    double TEMP;
    //    int IVIB;
    //
    // CALL RANDOM_NUMBER(RANF)
    calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
    N=-logf(calc->RANF)*TEMP/gas->SPVM[1][K][L];                 //eqn(4.10)
    //the state is truncated to an integer
    IVIB=N;
}

__device__ void LBS(curandState* globalState, double XMA,double XMB,double &ERM)
{
    //selects a Larsen-Borgnakke energy ratio using eqn (11.9)
    //
    double PROB,RANF;
    int I,N;
    //
    //I is an indicator
    //PROB is a probability
    //ERM ratio of rotational to collision energy
    //XMA degrees of freedom under selection-1
    //XMB remaining degrees of freedom-1
    //
    I=0;
    while(I == 0){
        // CALL RANDOM_NUMBER(RANF)
        RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
        ERM=RANF;
        if((XMA < 1.e-6) || (XMB < 1.e-6)){
            //    IF (XMA < 1.E-6.AND.XMB < 1.E-6) RETURN
            //above can never occur if one mode is translational
            if(XMA < 1.e-6) PROB=powf((1.e00-ERM),XMB);
            if(XMB < 1.e-6) PROB=powf((1.e00-ERM),XMA);
        }
        else
            PROB=powf(((XMA+XMB)*ERM/XMA),XMA)*powf(((XMA+XMB)*(1.e00-ERM)/XMB),XMB);
        
        // CALL RANDOM_NUMBER(RANF)
        RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
        if(PROB > RANF) I=1;
    }
    //
    return;
}


int main()
{
    // //CALC calc;
    // //MOLECS molecs;
    // //GAS gas;
    // //OUTPUT output;
    // //GEOM_1D geom;
    //
    // IMPLICIT NONE\

    //


    int IRUN,ICONF,N,M,IADAPT,IRETREM,ISET;
    double A;
    //
    fstream file_7;
    
    calc->NVER=1;          //for major changes, e.g. to basic architecture
    calc->MVER=1 ;         //significant changes, but must change whenever the data in a DSnD.DAT file changes
    calc->NREL=1  ;        //the release number
    //
    //***********************
    //set constants
    // PI=3.1415926535897932D00
    // DPI=6.283185307179586D00
    // SPI=1.772453850905516D00
    // BOLTZ=1.380658D-23
    // AVOG=6.022169D26
    //***********************
    //
    //*************************************************
    //****  ADJUSTABLE COMPUTATIONAL PARAMETERS  ****
    //*************************************************
    //
    calc->NMCC=15;    //DEFAULT=15--desired number of simulated molecules in a collision cell
    //
    calc->CPDTM=0.2;   //DEFAULT=0.2--fraction of the local mean collision time that is the desired maximum time step
    //
    calc->TPDTM=0.5 ;  //DEFAULT=0.5--the fraction or multiple of a sampling cell transit time that is the desired maximum time step
    //
    calc->NNC=1;       //DEFAULT=0--0 to select collision partner randomly from collision cell, 1 for nearest-neighbor collisions
    //
    calc->SAMPRAT=5;   //DEFAULT=5--the number of time steps in a sampling interval
    //
    calc->OUTRAT=10;   //50   //DEFAULT=50--the number of flow samples in a file output interval
    //
    calc->FRACSAM=0.5; //0.5 //DEFAULT=0.5--fraction of the output interval interval over which a time-averaged sample is taken in an unsteady flow
    //
    calc->ISAD=0;      //DEFAULT=0--0,1 to not adapt, to adapt cells automatically at start of output interval in an unsteady flow (not yet implemented)
    //
    calc->IMTS=2;      //DEFAULT=0--0 to set the move time step to the instantaneous overall time step that changes with time
    //                         1 to use a cell dependent collision time
    //                         2 to keep the time step fixed at the initial value
    //
    calc->FNUMF=1;   //DEFAULT=1--adjustment factor to the automatically generated value for the number of real molecules
    //                                  that are represented by each simulated molecule.
    //          (The adjustment may be large because the automatic setting assumes that the whole flowfield is at the stream conditions.)
    //
    //automatic adjustments may be applied for some application classes (e.g homogeneous gas studies)
    //
    calc->TLIM=1.e-5;  //DEFAULT=1.D20 sets an indefinite run - set if a define STOP time is required
    //
    //************************************************
    //
    //open a diagnostic file and check whether an instance of the program is already running
    //
    //    fstream file_9;
    cout<<"DSMC PROGRAM"<<endl;
    file_9.open("DIAG.TXT", ios::trunc | ios::out);
    if(file_9.is_open()){
        file_9<<"File DIAG.TXT has been opened"<<endl;
        cout<<"File DIAG.TXT has been opened"<<endl;
    }
    else{
        cout<<"Stop the DS1.EXE that is already running and try again"<<endl;
        //return 0;
    }
    
    //    OPEN (9,FILE='DIAG.TXT',FORM='FORMATTED',STATUS='REPLACE')
    //    WRITE (9,*,IOSTAT=ERROR)
    //    IF (ERROR /= 0) THEN
    //    WRITE (*,*) 'Stop the DS1.EXE that is already running and try again'
    //    STOP
    //    ELSE
    //    WRITE (9,*) 'File DIAG.TXT has been opened'
    //    END IF
    
    //
    //open a molecule number history file
    //OPEN (13,FILE='MolNum.DAT',FORM='FORMATTED',STATUS='REPLACE')
    //
    //initialise run variables
    IRUN=0;
    geom->IVB=0;  //will be reset to 1 by restart program if there is a moving wall
    //
    while((IRUN < 1) || (IRUN > 2)){
        cout<< "DSMC Version" <<calc->NVER<<'.'<<calc->MVER<<'.'<<calc->NREL<<endl;
        cout<< "enter 1 to continue a current run"<<endl;
        cout<< "enter 2 to start a new run :-"<<endl;
        //
        cin>>  IRUN;
    }
    if(IRUN == 1) file_9<< "Continuing an existing run"<<endl;//WRITE (9,*) 'Continuing an existing run'
    if(IRUN == 2) {
        cout<< "Enter 1 to confirm, 0 to continue current run :-"<<endl;
        cin>> ICONF;
        if(ICONF == 1)
            file_9<<"Starting a new run"<<endl;//WRITE (9,*) 'Starting a new run'
        else{
            IRUN=1;
            file_9<<"Continuing an existing run"<<endl;
            //WRITE (9,*) 'Continuing an existing run'
        }
    }
    //
    if(IRUN == 2){          //new run
        cout<< "Enter 0 for a homogeneous gas, or"<<endl;
        cout<< "Enter 1 for a one-dimensional flow, or"<<endl;
        cout<< "Enter 2 for a two-dimensional plane flow, or"<<endl;
        cout<< "Enter 3 for a three dimensional flow, or"<<endl;
        cout<< "enter 4 for an axially-symmetric flow :-"<<endl;
        cin>> calc->ICLASS;
        calc->NCLASS=2;      //default 2D
        if(calc->ICLASS < 2) calc->NCLASS=1;   //0D or 1D
        if(calc->ICLASS == 3) calc->NCLASS=3;  //3D
        cout<<"Enter 0 for an eventually steady flow, or"<<endl;
        cout<<"enter 1 for a continuing unsteady flow :-"<<endl;
        cin>> calc->ISF;
        
        
        file_7.open("RUN_CLASS.TXT", ios::trunc |ios::out);
        if(file_7.is_open()){
            cout<<"RUN_CLASS.TXT is opened"<<endl;
        }
        else{
            cout<<"RUN_CLASS.TXT not opened"<<endl;
            cin.get();
        }
        file_7<<calc->ICLASS<<calc->ISF;
        file_7.close();
        //        OPEN (7,FILE='RUN_CLASS.TXT',FORM='FORMATTED',STATUS='REPLACE')
        //        WRITE (7,*) ICLASS,ISF
        //        CLOSE (7)
        file_9<<"Starting a new run with ICLASS, ISF "<<calc->ICLASS<<" "<<calc->ISF<<endl;
        //        WRITE (9,*) 'Starting a new run with ICLASS, ISF',ICLASS,ISF
        cout<<"Starting a new run with ICLASS, ISF "<<calc->ICLASS<<" "<<calc->ISF<<endl;
    }
    //
    if(IRUN == 1){       //continued run
        file_7.open("RUN_CLASS.TXT" , ios::in );
        if(file_7.is_open()){
            cout<<"RUN_CLASS.TXT is opened"<<endl;
        }
        else{
            cout<<"RUN_CLASS.TXT not opened"<<endl;
            cin.get();
        }
        file_7 >>calc->ICLASS>>calc->ISF;
        file_7.close();
        //        OPEN (7,FILE='RUN_CLASS.TXT',FORM='FORMATTED',STATUS='OLD')
        //        READ (7,*) ICLASS,ISF
        //        CLOSE(7)
        READ_RESTART();
        //
        calc->TSAMP=calc->FTIME+calc->DTSAMP;
        calc->TOUT=calc->FTIME+calc->DTOUT;
        if((gas->MEX > 0) && (calc->ISF == 1)){
            cout<<"Enter 0 to continue the reaction sample or"<<endl;
            cout<<"enter 1 to continue with a new reaction sample :-"<<endl;
            cin>> N;
            if(N == 1){
                //memset(gas->TNEX,0.e00,sizeof(*gas->TNEX));
                //memset(calc->TDISS,0.e00,sizeof(*calc->TDISS));
                //memset(calc->TRECOMB,0.e00,sizeof(*calc->TRECOMB));
                for(int i=0;i<gas->MEX+1;i++)
                    gas->TNEX[i]= 0.e00;
                for(int i=0;i<gas->MSP+1;i++)
                    calc->TDISS[i]=0.e00;
                for(int i=0;i<gas->MSP+1;i++)
                    calc->TRECOMB[i]=0.e00;
            }
        }
        //
        if((calc->ISAD == 0) && (calc->ISF == 0)){
            cout<<"Enter 0 to continue the current sample or"<<endl;
            cout<<"enter 1 to continue with a new sample :-"<<endl;
            cin>> N;
            if(N == 1){
                if((geom->ITYPE[2] == 4) && (calc->ICN == 0)){
                    cout<<"Enter 0 to continue to not enforce constant molecule number"<<endl;
                    cout<<"enter 1 to start to enforce constant molecule number :-"<<endl;
                    cin>> M;
                    if(M == 1) calc->ICN=1;
                }
                cout<<"Enter 1 to adapt the cells, or 0 to continue with current cells:-"<<endl;
                cin>>IADAPT;
                if(IADAPT == 1){
                    cout<<"Adapting cells"<<endl;
                    ADAPT_CELLS_1D() ;
                    INDEX_MOLS();
                    WRITE_RESTART();
                }
                else
                    cout<<"Continuing with existing cells"<<endl;
                //
                if(calc->IREM == 2){
                    cout<<"Enter 1 to reset the removal details, or 0 to continue with current details:-"<<endl;
                    cin>>IRETREM;
                    if(IRETREM == 1){
                        geom->FREM=-1.e00;
                        while((geom->FREM < -0.0001) || (geom->FREM > 5.0)){
                            cout<<"Enter the fraction of entering molecules that are removed:-"<<endl;
                            cin>>geom->FREM;
                            cout<<"The ratio of removed to entering mlecules is \t"<<geom->FREM<<endl;
                            //                            WRITE (*,999) FREM
                        }
                        file_9<<"The ratio of removed to entering mlecules is \t"<<geom->FREM<<endl;
                        //                        WRITE (9,999) FREM
                        //                        999       FORMAT (' The ratio of removed to entering molecules is ',G15.5)
                        if(geom->FREM > 1.e-10){
                            geom->XREM=geom->XB[1]-1.0;
                            while((geom->XREM < geom->XB[1]-0.0001) || (geom->XREM > geom->XB[2]+0.0001)){
                                cout<<"Enter x coordinate of the upstream removal limit:-"<<endl;
                                cin>>geom->XREM;
                                cout<<"The molecules are removed from \t"<<geom->XREM<<" to "<<geom->XB[2]<<endl; //988
                                //                                WRITE (*,998) XREM,XB(2)
                            }
                            file_9<<"The molecules are removed from \t"<<geom->XREM<<" to "<<geom->XB[2]<<endl;
                            //                            WRITE (9,998) XREM,XB(2)
                            //                            998         FORMAT (' The molecules are removed from ',G15.5,' to',G15.5)
                        }
                    }
                }
                //
                INITIALISE_SAMPLES();
            }
        }
    }
    //
    if(IRUN == 2){
        //
        READ_DATA();
        //
        if(calc->ICLASS < 2) SET_INITIAL_STATE_1D();
        //
        if(calc->ICLASS == 0) ENERGY(0,A);
        //
        WRITE_RESTART();
        //
    }
    //
    while(calc->FTIME < calc->TLIM){
        //
        //
        calc->FTIME=calc->FTIME+calc->DTM;
        //
        file_9<<"  TIME  "<<setw(20)<<setprecision(10)<<calc->FTIME<<"  NM  "<<molecs->NM<<"  COLLS  "<<std::left<<setw(20)<<setprecision(10)<<calc->TOTCOL<<"Collision_time : "<<colltime<<endl;
        //        WRITE (9,*) 'TIME',FTIME,' NM',NM,' COLLS',TOTCOL
        cout<< "  TIME   "<<setw(20)<<setprecision(10)<<calc->FTIME<<"  NM  "<<molecs->NM<<"  COLLS  "<<std::left<<setw(20)<<setprecision(10)<<calc->TOTCOL<<"Collision_time : "<<colltime<<endl;
        //
        //  WRITE (13,*) FTIME/TNORM,FLOAT(NM)/FLOAT(NMI)      //uncomment if a MOLFILE.DAT is to be generated
        //
        //  WRITE (*,*) 'MOVE'
        //cout<<"MOVE"<<endl;
        MOLECULES_MOVE_1D();
        //
        if((geom->ITYPE[1] == 0) || (geom->ITYPE[2] == 0) || (geom->ITYPE[2] == 4)) MOLECULES_ENTER_1D();
        //
        //  WRITE (*,*) 'INDEX'
        //ut<<"INDEX"<<endl;
        // cout<<calc->TOUT<<endl;
        // cin.get();
        INDEX_MOLS();
        //
         // WRITE (*,*) 'COLLISIONS'
        COLLISIONS();
        //
        // if(gas->MMVM > 0) {
        //     cout<<"DISSOCIATION"<<endl;
        //     DISSOCIATION();
        // }
        //
        if(calc->FTIME > calc->TSAMP){
            //    WRITE (*,*) 'SAMPLE'
            if(calc->ISF == 0) SAMPLE_FLOW();
            if((calc->ISF == 1) && (calc->FTIME < calc->TPOUT+(1.e00-calc->FRACSAM)*calc->DTOUT)){
                calc->TSAMP=calc->TSAMP+calc->DTSAMP;
                INITIALISE_SAMPLES();
            }
            if((calc->ISF == 1) && (calc->FTIME >= calc->TPOUT+(1.e00-calc->FRACSAM)*calc->DTOUT)) SAMPLE_FLOW();
        }
        //
        if(calc->FTIME > calc->TOUT){
            cout<<"writing OUTPUT"<<endl;
            //    WRITE (*,*) 'OUTPUT'
            WRITE_RESTART();
            //
            OUTPUT_RESULTS();
            calc->TPOUT=calc->FTIME;
        }
        //
    }
    return 0;
    //
}

template <typename T>
string to_string(T value)
{
    std::ostringstream os ;
    os << value ;
    return os.str() ;
}

void ALLOCATE_GAS()
{
    // //GAS gas;
    // //CALC calc;
    gas->d_allocate(gas->MSP+1,3,gas->FSP);
    gas->d_allocate(7,gas->MSP+1,gas->SP);
    gas->d_allocate(4,gas->MSP+1,gas->SPR);
    gas->d_allocate(9,gas->MSP+1,gas->MSP,gas->SPM);
    gas->i_allocate(3,gas->MSP+1,gas->ISPR);
    gas->i_allocate(gas->MSP+1,gas->ISPV);
    gas->d_allocate(7,gas->MSP+1,3,gas->ENTR);
    gas->d_allocate(gas->MSP+1,3,gas->VMP);
    calc->d_allocate(gas->MSP+1,calc->VNMAX);
    gas->d_allocate(gas->MSP+1,gas->CR);
    calc->d_allocate(gas->MSP+1,gas->MSP+1,calc->TCOL);
    gas->i_allocate(gas->MSP+1,gas->MSP+1,gas->ISPRC);
    gas->i_allocate(gas->MSP+1,gas->MSP+1,gas->ISPRK);
    gas->d_allocate(5,gas->MSP+1,gas->MSP+1,gas->MSP+1,gas->SPRC);
    gas->i_allocate(gas->MSP+1,gas->NELL);
    gas->d_allocate(4,gas->MELE+1,gas->MSP+1,gas->QELC);
    gas->d_allocate(3,gas->MSP+1,gas->MSP+1,gas->MVIBL+1,gas->SPRP);
    gas->d_allocate(3,gas->MSP+1,gas->MSP+1,gas->SPRT);
    calc->d_allocate(gas->MSP+1,calc->AJM);
    gas->d_allocate(gas->MSP+1,gas->FP);
    calc->d_allocate(gas->MSP+1,calc->ALOSS);
    calc->d_allocate(gas->MSP+1,calc->EME);
    
    /*ALLOCATE (FSP(MSP,2),SP(6,MSP),SPR(3,MSP),SPM(8,MSP,MSP),ISPR(2,MSP),ISPV(MSP),ENTR(6,MSP,2),      &
     VMP(MSP,2),VNMAX(MSP),CR(MSP),TCOL(MSP,MSP),ISPRC(MSP,MSP),ISPRK(MSP,MSP),SPRC(4,MSP,MSP,MSP),                        &
     NELL(MSP),QELC(3,MELE,MSP),SPRP(2,MSP,MSP,0:MVIBL),SPRT(2,MSP,MSP),AJM(MSP),FP(MSP),    &
     ALOSS(MSP),EME(MSP),STAT=ERROR)
     //
     IF (ERROR /= 0) THEN
     WRITE (*,*)'PROGRAM COULD NOT ALLOCATE SPECIES VARIABLES',ERROR
     END IF
     //*/
    gas->i_allocate(gas->MMEX+1,gas->MSP+1,gas->MSP+1,gas->NEX);
    gas->i_allocate(gas->MSP+1,gas->MSP+1,gas->NSPEX);
    gas->d_allocate(7,gas->MMEX+1,gas->MSP+1,gas->MSP+1,gas->SPEX);
    gas->i_allocate(gas->MMEX+1,8,gas->MSP+1,gas->MSP+1,gas->ISPEX);
    gas->i_allocate(5,gas->MSP+1,gas->TREACG);
    gas->d_allocate(gas->MMEX+1,gas->PSF);
    gas->i_allocate(5,gas->MSP+1,gas->TREACL);
    gas->d_allocate(gas->MEX+1,gas->TNEX);
    gas->d_allocate(3,gas->MMEX+1,gas->MSP+1,gas->MSP+1,gas->MVIBL+1,gas->SPREX);
    gas->i_allocate(3,gas->MSP+1,gas->NSLEV);
    gas->d_allocate(gas->MSP+1,gas->SLER);
    // ALLOCATE (NEX(MMEX,MSP,MSP),NSPEX(MSP,MSP),SPEX(6,MMEX,MSP,MSP),ISPEX(MMEX,7,MSP,MSP),TREACG(4,MSP),         &
    //           PSF(MMEX),TREACL(4,MSP),TNEX(MEX),SPREX(2,MMEX,MSP,MSP,0:MVIBL),NSLEV(2,MSP),SLER(MSP),STAT=ERROR)
    // //
    // IF (ERROR /= 0) THEN
    //   WRITE (*,*)'PROGRAM COULD NOT ALLOCATE Q-K REACTION VARIABLES',ERROR
    // END IF
    // //
    if(gas->MMVM >= 0){
        gas->d_allocate(6,gas->MMVM+1,gas->MSP+1,gas->SPVM);
        gas->i_allocate(3,gas->MMVM+1,gas->MSP+1,gas->ISPVM);
        calc->d_allocate(gas->MSP+1,calc->TDISS);
        calc->d_allocate(gas->MSP+1,calc->TRECOMB);
        //ALLOCATE (SPVM(5,MMVM,MSP),ISPVM(2,MMVM,MSP),TDISS(MSP),TRECOMB(MSP),STAT=ERROR)
        // IF (ERROR /= 0) THEN
        //   WRITE (*,*)'PROGRAM COULD NOT ALLOCATE VIBRATION VARIABLES',ERROR
    }
    
    //N.B. surface reactions are not yet implemented
    if(gas->MNSR > 0){
        gas->d_allocate(gas->MNSR+1,gas->ERS);
        gas->i_allocate(3,gas->MNSR+1,gas->LIS);
        gas->i_allocate(7,gas->MNSR+1,gas->LRS);
        gas->i_allocate(gas->MNSR+1,gas->MSP+1,gas->ISRCD);
        //ALLOCATE (ERS(MNSR),LIS(2,MNSR),LRS(6,MNSR),ISRCD(MNSR,MSP),STAT=ERROR)
        // IF (ERROR /= 0) THEN
        //   WRITE (*,*)'PROGRAM COULD NOT ALLOCATE SURFACE REACTION VARIABLES',ERROR
    }
     //calc->AJM=0.e00;
    //memset(calc->AJM,0.e00,sizeof(*calc->AJM));
    for(int i=0;i<gas->MSP+1;i++){
        calc->AJM[i]=0.e00;
    }
    return;
    
}

void HARD_SPHERE()
{
    ////GAS gas;
    ////CALC calc;
    cout<<"Reading HARD_SPHERE Data"<<endl;
    gas->MSP=1;
    gas->MMRM=0;
    gas->MMVM=0;
    gas->MNSR=0;
    gas->MEX=0;
    gas->MMEX=0;
    gas->MELE=1;
    gas->MVIBL=0;
    
    ALLOCATE_GAS();
    
    gas->SP[1][1]=4.0e-10;    //reference diameter
    gas->SP[2][1]=273.0;       //reference temperature
    gas->SP[3][1]=0.5;        //viscosity-temperature index
    gas->SP[4][1]=1.0;         //reciprocal of VSS scattering parameter (1 for VHS)
    gas->SP[5][1]=5.e-26;     //mass
    gas->ISPR[1][1]=0;        //number of rotational degrees of freedom
    cout<<"Hard Sphere data done"<<endl;
    return;
}


void ARGON()
{
    // //GAS gas;
    // //CALC calc;
    cout<<"Reading Argon Data"<<endl;
    gas->MSP=1;
    gas->MMRM=0;
    gas->MMVM=0;
    gas->MNSR=0;
    gas->MEX=0;
    gas->MMEX=0;
    gas->MELE=1;
    gas->MVIBL=0;
    ALLOCATE_GAS();
    gas->SP[1][1]=4.17e-10;
    gas->SP[2][1]=273.15;
    gas->SP[3][1]=0.81;
    gas->SP[4][1]=1.0;
    gas->SP[5][1]=6.63e-26;
    gas->ISPR[1][1]=0;
    gas->ISPR[2][1]=0;
    cout<<"Argon Data done"<<endl;
    return;
}
//
void IDEAL_NITROGEN()
{
    // //GAS gas;
    // //CALC calc;
    cout<<"Reading IDEAL_NITROGEN data"<<endl;
    gas->MSP=1;
    gas->MMRM=1;
    gas->MMVM=0;
    gas->MNSR=0;
    gas->MEX=0;
    gas->MMEX=0;
    gas->MELE=0;
    gas->MVIBL=0;
    
    ALLOCATE_GAS();
    gas->SP[1][1]=4.17e-10;
    gas->SP[2][1]=273.0;
    gas->SP[3][1]=0.74;
    gas->SP[4][1]=1.0;
    gas->SP[5][1]=4.65e-26;
    gas->ISPR[1][1]=2;
    gas->ISPR[2][1]=0;
    gas->SPR[1][1]=5.0;
    return;
}
//
void REAL_OXYGEN()
{
    //
    //GAS gas;
    //CALC calc;
    cout<<"Reading Real_Oxygen data"<<endl;
    gas->MSP=2;
    gas->MMRM=1;
    gas->MMVM=1;
    gas->MNSR=0;
    gas->MEX=0;
    gas->MMEX=0;
    gas->MELE=5;
    gas->MVIBL=26;
    ALLOCATE_GAS();
    gas->SP[1][1]=4.07e-10;
    gas->SP[2][1]=273.00;
    gas->SP[3][1]=0.77e00;
    gas->SP[4][1]=1.e00;
    gas->SP[5][1]=5.312e-26;
    gas->SP[6][1]=0.e00;
    gas->ISPR[1][1]=2;
    gas->ISPR[2][1]=0 ;            //0,1 for constant,polynomial rotational relaxation collision number
    gas->SPR[1][1]=5.0;             // the collision number or the coefficient of temperature in the polynomial (if a polynomial, the coeff. of T^2 is in spr_db(3  )
    
    gas->ISPV[1]=1   ;            // the number of vibrational modes
    gas->SPVM[1][1][1]=2256.e00  ;        // the characteristic vibrational temperature
    gas->SPVM[2][1][1]=90000.e00;        // a constant Zv, or the reference Zv
    gas->SPVM[3][1][1]=2256.e00;        // -1 for a constant Zv, or the reference temperature
    gas->SPVM[5][1][1]=1.0;            //arbitrary reduction factor
    gas->ISPVM[1][1][1]=2;
    gas->ISPVM[2][1][1]=2;
    gas->NELL[1]=3;
    if(gas->MELE > 1){
        //******
        gas->QELC[1][1][1]=3.0;
        gas->QELC[2][1][1]=0.0;
        gas->QELC[3][1][1]=50.0;  //500.
        gas->QELC[1][2][1]=2.0;
        gas->QELC[2][2][1]=11393.0;
        gas->QELC[3][2][1]=50.0;  //500         //for equipartition, the cross-section ratios must be the same for all levels
        gas->QELC[1][3][1]=1.0;
        gas->QELC[2][3][1]=18985.0;
        gas->QELC[3][3][1]=50.0;  //500.
    }
    //
    //species 2 is atomic oxygen
    gas->SP[1][2]=3.e-10;
    gas->SP[2][2]=273.e00;
    gas->SP[3][2]=0.8e00;
    gas->SP[4][2]=1.e00;
    gas->SP[5][2]=2.656e-26;
    gas->SP[6][2]=4.099e-19;
    gas->ISPR[1][2]=0;
    gas->ISPV[2]=0;     //must be set//
    //set electronic information
    if(gas->MELE > 1){
        gas->NELL[2]=5;
        gas->QELC[1][1][2]=5.0;
        gas->QELC[2][1][2]=0.0;
        gas->QELC[3][1][2]=50.0;
        gas->QELC[1][2][2]=3.0;
        gas->QELC[2][2][2]=228.9;
        gas->QELC[3][2][2]=50.0;
        gas->QELC[1][3][2]=1.0;
        gas->QELC[2][3][2]=325.9;
        gas->QELC[3][3][2]=50.0;
        gas->QELC[1][4][2]=5.0;
        gas->QELC[2][4][2]=22830.0;
        gas->QELC[3][4][2]=50.0;
        gas->QELC[1][5][2]=1.0;
        gas->QELC[2][5][2]=48621.0;
        gas->QELC[3][5][2]=50.0;
    }
    //set data needed for recombination
    //
    for(int i=0;i<gas->MSP+1;i++){
        for(int j=0;j<gas->MSP+1;j++){
            gas->ISPRC[i][j]=0;
            gas->ISPRK[i][j]=0;
        }
    }
    // gas->ISPRC=0;
    // gas->ISPRK=0;
    gas->ISPRC[2][2]=1;    //O+O -> O2  recombined species code for an O+O recombination
    gas->ISPRK[2][2]=1 ;     //the relevant vibrational mode of this species
    gas->SPRC[1][2][2][1]=0.04;
    gas->SPRC[2][2][2][1]=-1.3;
    gas->SPRC[1][2][2][2]=0.05;
    gas->SPRC[2][2][2][2]=-1.1;
    gas->SPRT[1][2][2]=5000.e00;
    gas->SPRT[2][2][2]=15000.e00;
    //
    //memset(gas->NSPEX,0,sizeof(**gas->NSPEX));
    //memset(gas->SPEX,0.e00,sizeof(****gas->SPEX));
    for(int i=0;i<gas->MSP+1;i++){
        for(int j=0;j<gas->MSP+1;j++){
            gas->NSPEX[i][j]=0;
        }
    }
    for(int i=0;i<7;i++){
        for(int j=0;j<gas->MMEX+1;j++){
            for(int k=0;k<gas->MSP+1;k++){
                for(int l=0;l<gas->MSP+1;l++)
                    gas->SPEX[i][j][k][l]=0.e00;
            }
        }
    }
    //gas->SPEX=0.e00;
    gas->ISPEX=0;
    //
    DERIVED_GAS_DATA();
    //
    cout<<"Real_Oxygen data done"<<endl;
    return;
}
//
void IDEAL_AIR()
{
    //GAS gas;
    //CALC calc;
    cout<<"Reading IDEAL_AIR data"<<endl;
    gas->MSP=2;
    gas->MMRM=1;
    gas->MMVM=0;
    gas->MNSR=0;
    gas->MEX=0;
    gas->MMEX=0;
    gas->MELE=1;
    gas->MVIBL=0;
    //
    ALLOCATE_GAS();
    //
    gas->SP[1][1]=4.07e-10;
    gas->SP[2][1]=273.0;
    gas->SP[3][1]=0.77;
    gas->SP[4][1]=1.0;
    gas->SP[5][1]=5.312e-26;
    gas->ISPR[1][1]=2;
    gas->ISPR[2][1]=0;
    gas->SPR[1][1]=5.0;
    gas->SP[1][2]=4.17e-10;
    gas->SP[2][2]=273.0;
    gas->SP[3][2]=0.74;
    gas->SP[4][2]=1.0;
    gas->SP[5][2]=4.65e-26;
    gas->ISPR[1][2]=2;
    gas->ISPR[2][2]=0;
    gas->SPR[1][2]=5.0;
    cout<<"IDEAL_AIR data done"<<endl;
    return;
}
//
void REAL_AIR()
{
    //GAS gas;
    //CALC calc;
    cout<<"REAL_AIR data done"<<endl;
    gas->MSP=5;
    gas->MMRM=1;
    gas->MMVM=1;
    gas->MELE=5;
    gas->MVIBL=40;  //?
    //
    gas->MEX=4;
    gas->MMEX=1;
    //
    gas->MNSR=0;
    ALLOCATE_GAS();
    //species 1 is oxygen
    gas->SP[1][1]=4.07e-10;
    gas->SP[2][1]=273.e00;
    gas->SP[3][1]=0.77e00;
    gas->SP[4][1]=1.e00;
    gas->SP[5][1]=5.312e-26;
    gas->SP[6][1]=0.e00;
    gas->ISPR[1][1]=2;
    gas->ISPR[2][1]=0;
    gas->SPR[1][1]=5.e00;
    gas->ISPV[1]=1;               // the number of vibrational modes
    gas->SPVM[1][1][1]=2256.e00;          // the characteristic vibrational temperature
    gas->SPVM[2][1][1]=18000.e00;  //90000.D00        // a constant Zv, or the reference Zv
    gas->SPVM[3][1][1]=2256.e00;       // -1 for a constant Zv, or the reference temperature
    gas->SPVM[5][1][1]=1.0;
    gas->ISPVM[1][1][1]=3;
    gas->ISPVM[2][1][1]=3;
    gas->NELL[1]=3;
    gas->QELC[1][1][1]=3.0;
    gas->QELC[2][1][1]=0.0;
    gas->QELC[3][1][1]=50.0;
    gas->QELC[1][2][1]=2.0;
    gas->QELC[2][2][1]=11393.0;
    gas->QELC[3][2][1]=50.0;
    gas->QELC[1][3][1]=1.0;
    gas->QELC[2][3][1]=18985.0;
    gas->QELC[3][3][1]=50.0;
    //species 2 is nitrogen
    gas->SP[1][2]=4.17e-10;
    gas->SP[2][2]=273.e00;
    gas->SP[3][2]=0.74e00;
    gas->SP[4][2]=1.e00;
    gas->SP[5][2]=4.65e-26;
    gas->SP[6][2]=0.e00;
    gas->ISPR[1][2]=2;
    gas->ISPR[2][2]=0;
    gas->SPR[1][2]=5.e00;
    gas->ISPV[2]=1;
    gas->SPVM[1][1][2]=3371.e00;
    gas->SPVM[2][1][2]=52000.e00;     //260000.D00
    gas->SPVM[3][1][2]=3371.e00;
    gas->SPVM[5][1][2]=0.3;
    gas->ISPVM[1][1][2]=4;
    gas->ISPVM[2][1][2]=4;
    gas->NELL[2]=1;
    gas->QELC[1][1][2]=1.0;
    gas->QELC[2][1][2]=0.0;
    gas->QELC[3][1][2]=100.0;
    //species 3 is atomic oxygen
    gas->SP[1][3]=3.e-10;
    gas->SP[2][3]=273.e00;
    gas->SP[3][3]=0.8e00;
    gas->SP[4][3]=1.e00;
    gas->SP[5][3]=2.656e-26;
    gas->SP[6][3]=4.099e-19;
    gas->ISPR[1][3]=0;
    gas->ISPV[3]=0;
    gas->NELL[3]=5;
    gas->QELC[1][1][3]=5.0;
    gas->QELC[2][1][3]=0.0;
    gas->QELC[3][1][3]=50.0;
    gas->QELC[1][2][3]=3.0;
    gas->QELC[2][2][3]=228.9;
    gas->QELC[3][2][3]=50.0;
    gas->QELC[1][3][3]=1.0;
    gas->QELC[2][3][3]=325.9;
    gas->QELC[3][3][3]=50.0;
    gas->QELC[1][4][3]=5.0;
    gas->QELC[2][4][3]=22830.0;
    gas->QELC[3][4][3]=50.0;
    gas->QELC[1][5][3]=1.0;
    gas->QELC[2][5][3]=48621.0;
    gas->QELC[3][5][3]=50.0;
    //species 4 is atomic nitrogen
    gas->SP[1][4]=3.e-10;
    gas->SP[2][4]=273.e00;
    gas->SP[3][4]=0.8e00;
    gas->SP[4][4]=1.0e00;
    gas->SP[5][4]=2.325e-26;
    gas->SP[6][4]=7.849e-19;
    gas->ISPR[1][4]=0;
    gas->ISPV[4]=0;
    gas->NELL[4]=3;
    gas->QELC[1][1][4]=4.0;
    gas->QELC[2][1][4]=0.0;
    gas->QELC[3][1][4]=50.0;
    gas->QELC[1][2][4]=10.0;
    gas->QELC[2][2][4]=27658.0;
    gas->QELC[3][2][4]=50.0;
    gas->QELC[1][3][4]=6.0;
    gas->QELC[2][3][4]=41495.0;
    gas->QELC[3][3][4]=50.0;
    //species 5 is NO
    gas->SP[1][5]=4.2e-10;
    gas->SP[2][5]=273.e00;
    gas->SP[3][5]=0.79e00;
    gas->SP[4][5]=1.0e00;
    gas->SP[5][5]=4.98e-26;
    gas->SP[6][5]=1.512e-19;
    gas->ISPR[1][5]=2;
    gas->ISPR[2][5]=0;
    gas->SPR[1][5]=5.e00;
    gas->ISPV[5]=1;
    gas->SPVM[1][1][5]=2719.e00;
    gas->SPVM[2][1][5]=14000.e00;   //70000.D00
    gas->SPVM[3][1][5]=2719.e00;
    gas->SPVM[5][1][5]=0.2;
    gas->ISPVM[1][1][5]=3;
    gas->ISPVM[2][1][5]=4;
    gas->NELL[5]=2;
    gas->QELC[1][1][5]=2.0;
    gas->QELC[2][1][5]=0.0;
    gas->QELC[3][1][5]=50.0;
    gas->QELC[1][2][5]=2.0;
    gas->QELC[2][2][5]=174.2;
    gas->QELC[3][2][5]=50.0;
    //set the recombination data for the molecule pairs
    //memset(gas->ISPRC,0,sizeof(**gas->ISPRC));//gas->ISPRC=0;    //data os zero unless explicitly set
    //memset(gas->ISPRK,0,sizeof(**gas->ISPRK));//gas->ISPRK=0;
    //memset(gas->SPRC,0,sizeof(****gas->SPRC));//gas->SPRC=0.e00;
    for(int i=0;i<gas->MSP+1;i++){
        for(int j=0;j<gas->MSP+1;j++){
            gas->ISPRC[i][j]=0;
        }
    }
    for(int i=0;i<gas->MSP+1;i++){
        for(int j=0;j<gas->MSP+1;j++){
            gas->ISPRK[i][j]=0;
        }
    }
    for(int i=0;i<5;i++){
        for(int j=0;j<gas->MSP+1;j++){
            for(int k=0;k<gas->MSP+1;k++){
                for(int l=0;l<gas->MSP+1;l++)
                    gas->SPEX[i][j][k][l]=0.e00;
            }
        }
    }
    gas->ISPRC[3][3]=1; //O+O -> O2  recombined species code for an O+O recombination
    gas->ISPRK[3][3]=1;
    gas->SPRC[1][3][3][1]=0.04e00;
    gas->SPRC[2][3][3][1]=-1.3e00;
    gas->SPRC[1][3][3][2]=0.07e00;
    gas->SPRC[2][3][3][2]=-1.2e00;
    gas->SPRC[1][3][3][3]=0.08e00;
    gas->SPRC[2][3][3][3]=-1.2e00;
    gas->SPRC[1][3][3][4]=0.09e00;
    gas->SPRC[2][3][3][4]=-1.2e00;
    gas->SPRC[1][3][3][5]=0.065e00;
    gas->SPRC[2][3][3][5]=-1.2e00;
    gas->SPRT[1][3][3]=5000.e00;
    gas->SPRT[2][3][3]=15000.e00;
    gas->ISPRC[4][4]=2;  //N+N -> N2
    gas->ISPRK[4][4]=1;
    gas->SPRC[1][4][4][1]=0.15e00;
    gas->SPRC[2][4][4][1]=-2.05e00;
    gas->SPRC[1][4][4][2]=0.09e00;
    gas->SPRC[2][4][4][2]=-2.1e00;
    gas->SPRC[1][4][4][3]=0.16e00;
    gas->SPRC[2][4][4][3]=-2.0e00;
    gas->SPRC[1][4][4][4]=0.17e00;
    gas->SPRC[2][4][4][4]=-2.0e00;
    gas->SPRC[1][4][4][5]=0.17e00;
    gas->SPRC[2][4][4][5]=-2.1e00;
    gas->SPRT[1][4][4]=5000.e00;
    gas->SPRT[2][4][4]=15000.e00;
    gas->ISPRC[3][4]=5;
    gas->ISPRK[3][4]=1;
    gas->SPRC[1][3][4][1]=0.3e00;
    gas->SPRC[2][3][4][1]=-1.9e00;
    gas->SPRC[1][3][4][2]=0.4e00;
    gas->SPRC[2][3][4][2]=-2.0e00;
    gas->SPRC[1][3][4][3]=0.3e00;
    gas->SPRC[2][3][4][3]=-1.75e00;
    gas->SPRC[1][3][4][4]=0.3e00;
    gas->SPRC[2][3][4][4]=-1.75e00;
    gas->SPRC[1][3][4][5]=0.15e00;
    gas->SPRC[2][3][4][5]=-1.9e00;
    gas->SPRT[1][3][4]=5000.e00;
    gas->SPRT[2][3][4]=15000.e00;
    //set the exchange reaction data
    //memset(gas->SPEX,0,sizeof(****gas->SPEX));//gas->SPEX=0.e00;
    for(int i=0;i<7;i++){
        for(int j=0;j<gas->MMEX+1;j++){
            for(int k=0;k<gas->MSP+1;k++){
                for(int l=0;l<gas->MSP+1;l++)
                    gas->SPEX[i][j][k][l]=0.e00;
            }
        }
    }
    gas->ISPEX=0;
    gas->NSPEX=0;
    gas->NSPEX[2][3]=1;
    gas->NSPEX[4][5]=1;
    gas->NSPEX[3][5]=1;
    gas->NSPEX[1][4]=1;
    //N2+O->NO+N
    gas->ISPEX[1][1][2][3]=2;
    gas->ISPEX[1][2][2][3]=3;
    gas->ISPEX[1][3][2][3]=5;
    gas->ISPEX[1][4][2][3]=4;
    gas->ISPEX[1][5][2][3]=1;
    gas->ISPEX[1][6][2][3]=1;
    gas->SPEX[6][1][2][3]=0.e00;
    gas->NEX[1][2][3]=1;
    //NO+N->N2+0
    gas->ISPEX[1][1][4][5]=5;
    gas->ISPEX[1][2][4][5]=4;
    gas->ISPEX[1][3][4][5]=2;
    gas->ISPEX[1][4][4][5]=3;
    gas->ISPEX[1][5][4][5]=1;
    gas->ISPEX[1][6][4][5]=1;
    gas->ISPEX[1][7][4][5]=1;
    gas->SPEX[1][1][4][5]=0.8e00;
    gas->SPEX[2][1][4][5]=-0.75e00;
    gas->SPEX[4][1][4][5]=5000.e00;
    gas->SPEX[5][1][4][5]=15000.e00;
    gas->SPEX[6][1][4][5]=0.e00;
    gas->NEX[1][4][5]=2;
    //NO+O->O2+N
    gas->ISPEX[1][1][3][5]=5;
    gas->ISPEX[1][2][3][5]=3;
    gas->ISPEX[1][3][3][5]=1;
    gas->ISPEX[1][4][3][5]=4;
    gas->ISPEX[1][5][3][5]=1;
    gas->ISPEX[1][6][3][5]=1;
    gas->SPEX[6][1][3][5]=2.e-19;
    gas->NEX[1][3][5]=3;
    //O2+N->NO+O
    gas->ISPEX[1][1][1][4]=1;
    gas->ISPEX[1][2][1][4]=4;
    gas->ISPEX[1][3][1][4]=5;
    gas->ISPEX[1][4][1][4]=3;
    gas->ISPEX[1][5][1][4]=1;
    gas->ISPEX[1][6][1][4]=1;
    gas->ISPEX[1][7][1][4]=1 ;
    gas->SPEX[1][1][1][4]=7.e00;
    gas->SPEX[2][1][1][4]=-0.85e00;
    gas->SPEX[4][1][1][4]=5000.e00;
    gas->SPEX[5][1][1][4]=15000.e00;
    gas->SPEX[6][1][1][4]=0.e00;
    gas->NEX[1][1][4]=4;
    
    DERIVED_GAS_DATA();
    cout<<"REAL_AIR data done"<<endl;
    return;
}
//
void HELIUM_ARGON_XENON()
{
    //GAS gas;
    //CALC calc;
    cout<<"Reading HELIUM_ARGON_XENON data"<<endl;
    gas->MSP=3;
    gas->MMRM=0;
    gas->MMVM=0;
    gas->MNSR=0;
    gas->MEX=0;
    gas->MMEX=0;
    gas->MELE=1;
    gas->MVIBL=0;
    
    ALLOCATE_GAS();
    
    gas->SP[1][1]=2.30e-10;   //2.33D-10
    gas->SP[2][1]=273.0;
    gas->SP[3][1]=0.66;
    gas->SP[4][1]=0.794;   //1.
    gas->SP[5][1]=6.65e-27;
    gas->ISPR[1][1]=0;
    gas->ISPR[2][1]=0;
    //
    gas->SP[1][2]=4.11e-10;   //4.17D-10
    gas->SP[2][2]=273.15;
    gas->SP[3][2]=0.81;
    gas->SP[4][2]=0.714;    //1.
    gas->SP[5][2]=6.63e-26;
    gas->ISPR[1][2]=0;
    gas->ISPR[2][2]=0;
    //
    gas->SP[1][3]=5.65e-10;   //5.74D-10
    gas->SP[2][3]=273.0;
    gas->SP[3][3]=0.85;
    gas->SP[4][3]=0.694;   //1.
    gas->SP[5][3]=21.8e-26;
    gas->ISPR[1][3]=0;
    gas->ISPR[2][3]=0;
    cout<<"HELIUM_ARGON_XENON data done"<<endl;
    return;
}
//
void OXYGEN_HYDROGEN()
{
    //
    //GAS gas;
    //CALC calc;
    cout<<"Reading OXYGEN_HYDROGEN data"<<endl;
    gas->MSP=8;
    gas->MMRM=3;
    gas->MMVM=3;
    gas->MELE=1;
    gas->MVIBL=40;  //the maximum number of vibrational levels before a cumulative level reaches 1
    //
    gas->MEX=16;
    gas->MMEX=3;
    //
    gas->MNSR=0;
    //
    ALLOCATE_GAS();
    //
    //species 1 is hydrogen H2
    gas->SP[1][1]=2.92e-10;
    gas->SP[2][1]=273.e00;
    gas->SP[3][1]=0.67e00;
    gas->SP[4][1]=1.e00;
    gas->SP[5][1]=3.34e-27;
    gas->SP[6][1]=0.e00;
    gas->ISPR[1][1]=2;
    gas->ISPR[2][1]=0;
    gas->SPR[1][1]=5.e00;
    gas->ISPV[1]=1;         // the number of vibrational modes
    gas->SPVM[1][1][1]=6159.e00;          // the characteristic vibrational temperature
    gas->SPVM[2][1][1]=20000.e00;  //estimate
    gas->SPVM[3][1][1]=2000.e00; //estimate
    gas->SPVM[5][1][1]=1.0;
    gas->ISPVM[1][1][1]=2;
    gas->ISPVM[2][1][1]=2;
    //species 2 is atomic hydrogen H
    gas->SP[1][2]=2.5e-10;      //estimate
    gas->SP[2][2]=273.e00;
    gas->SP[3][2]=0.8e00;
    gas->SP[4][2]=1.e00;
    gas->SP[5][2]=1.67e-27;
    gas->SP[6][2]=3.62e-19;
    gas->ISPR[1][2]=0;
    gas->ISPV[2]=0;
    //species 3 is oxygen O2
    gas->SP[1][3]=4.07e-10;
    gas->SP[2][3]=273.e00;
    gas->SP[3][3]=0.77e00;
    gas->SP[4][3]=1.e00;
    gas->SP[5][3]=5.312e-26;
    gas->SP[6][3]=0.e00;
    gas->ISPR[1][3]=2;
    gas->ISPR[2][3]=0;
    gas->SPR[1][3]=5.e00;
    gas->ISPV[3]=1;               // the number of vibrational modes
    gas->SPVM[1][1][3]=2256.e00;          // the characteristic vibrational temperature
    gas->SPVM[2][1][3]=18000.e00;  //90000.D00        // a constant Zv, or the reference Zv
    gas->SPVM[3][1][3]=2256.e00;       // -1 for a constant Zv, or the reference temperature
    gas->SPVM[5][1][3]=1.e00;
    gas->ISPVM[1][1][3]=4;
    gas->ISPVM[2][1][3]=4;
    //species 4 is atomic oxygen O
    gas->SP[1][4]=3.e-10;    //estimate
    gas->SP[2][4]=273.e00;
    gas->SP[3][4]=0.8e00;
    gas->SP[4][4]=1.e00;
    gas->SP[5][4]=2.656e-26;
    gas->SP[6][4]=4.099e-19;
    gas->ISPR[1][4]=0;
    gas->ISPV[4]=0;
    //species 5 is hydroxy OH
    gas->SP[1][5]=4.e-10;       //estimate
    gas->SP[2][5]=273.e00;
    gas->SP[3][5]=0.75e00;      //-estimate
    gas->SP[4][5]=1.0e00;
    gas->SP[5][5]=2.823e-26;
    gas->SP[6][5]=6.204e-20;
    gas->ISPR[1][5]=2;
    gas->ISPR[2][5]=0;
    gas->SPR[1][5]=5.e00;
    gas->ISPV[5]=1;
    gas->SPVM[1][1][5]=5360.e00;
    gas->SPVM[2][1][5]=20000.e00;   //estimate
    gas->SPVM[3][1][5]=2500.e00;    //estimate
    gas->SPVM[5][1][5]=1.0e00;
    gas->ISPVM[1][1][5]=2;
    gas->ISPVM[2][1][5]=4;
    //species 6 is water vapor H2O
    gas->SP[1][6]=4.5e-10;      //estimate
    gas->SP[2][6]=273.e00;
    gas->SP[3][6]=0.75e00 ;     //-estimate
    gas->SP[4][6]=1.0e00;
    gas->SP[5][6]=2.99e-26;
    gas->SP[6][6]=-4.015e-19;
    gas->ISPR[1][6]=3;
    gas->ISPR[2][6]=0;
    gas->SPR[1][6]=5.e00;
    gas->ISPV[6]=3;
    gas->SPVM[1][1][6]=5261.e00;  //symmetric stretch mode
    gas->SPVM[2][1][6]=20000.e00;   //estimate
    gas->SPVM[3][1][6]=2500.e00;    //estimate
    gas->SPVM[5][1][6]=1.e00;
    gas->SPVM[1][2][6]=2294.e00;  //bend mode
    gas->SPVM[2][2][6]=20000.e00;   //estimate
    gas->SPVM[3][2][6]=2500.e00;    //estimate
    gas->SPVM[5][2][6]=1.0e00;
    gas->SPVM[1][3][6]=5432.e00;  //asymmetric stretch mode
    gas->SPVM[2][3][6]=20000.e00;   //estimate
    gas->SPVM[3][3][6]=2500.e00 ;   //estimate
    gas->SPVM[5][3][6]=1.e00;
    gas->ISPVM[1][1][6]=2;
    gas->ISPVM[2][1][6]=5;
    gas->ISPVM[1][2][6]=2;
    gas->ISPVM[2][2][6]=5;
    gas->ISPVM[1][3][6]=2;
    gas->ISPVM[2][3][6]=5;
    //species 7 is hydroperoxy HO2
    gas->SP[1][7]=5.5e-10;       //estimate
    gas->SP[2][7]=273.e00;
    gas->SP[3][7]=0.75e00 ;     //-estimate
    gas->SP[4][7]=1.0e00;
    gas->SP[5][7]=5.479e-26;
    gas->SP[6][7]=2.04e-20;
    gas->ISPR[1][7]=2;    //assumes that HO2 is linear
    gas->ISPR[2][7]=0;
    gas->SPR[1][7]=5.e00;
    gas->ISPV[7]=3;
    gas->SPVM[1][1][7]=4950.e00;
    gas->SPVM[2][1][7]=20000.e00;   //estimate
    gas->SPVM[3][1][7]=2500.e00  ;  //estimate
    gas->SPVM[5][1][7]=1.e00;
    gas->SPVM[1][2][7]=2000.e00;
    gas->SPVM[2][2][7]=20000.e00;   //estimate
    gas->SPVM[3][2][7]=2500.e00;    //estimate
    gas->SPVM[5][2][7]=1.e00;
    gas->SPVM[1][3][7]=1580.e00;
    gas->SPVM[2][3][7]=20000.e00;   //estimate
    gas->SPVM[3][3][7]=2500.e00;    //estimate
    gas->SPVM[5][3][7]=1.e00;
    gas->ISPVM[1][1][7]=2;
    gas->ISPVM[2][1][7]=3;
    gas->ISPVM[1][2][7]=2;
    gas->ISPVM[2][2][7]=3;
    gas->ISPVM[1][3][7]=2;
    gas->ISPVM[2][3][7]=3;
    //Species 8 is argon
    gas->SP[1][8]=4.17e-10;
    gas->SP[2][8]=273.15;
    gas->SP[3][8]=0.81   ;
    gas->SP[4][8]=1.0;
    gas->SP[5][8]=6.63e-26;
    gas->SP[6][8]=0.e00;
    gas->ISPR[1][8]=0;
    gas->ISPV[8]=0;
    //
    for(int i=0;i<gas->MSP+1;i++){
        for(int j=0;j<gas->MSP+1;j++){
            gas->ISPRC[i][j]=0;
        }
    }
    //gas->ISPRC=0;    //data is zero unless explicitly set
    //
    gas->ISPRC[4][4]=3;    //O+O+M -> O2+M  recombined species code for an O+O recombination
    gas->ISPRK[4][4]=1;
    gas->SPRC[1][4][4][1]=0.26e00;
    gas->SPRC[2][4][4][1]=-1.3e00;
    gas->SPRC[1][4][4][2]=0.29e00;
    gas->SPRC[2][4][4][2]=-1.3e00;
    gas->SPRC[1][4][4][3]=0.04e00;
    gas->SPRC[2][4][4][3]=-1.5e00;
    gas->SPRC[1][4][4][4]=0.1e00;
    gas->SPRC[2][4][4][4]=-1.4e00;
    gas->SPRC[1][4][4][5]=0.1e00;
    gas->SPRC[2][4][4][5]=-1.4e00;
    gas->SPRC[1][4][4][6]=0.1e00;
    gas->SPRC[2][4][4][6]=-1.4e00;
    gas->SPRC[1][4][4][7]=0.07e00;
    gas->SPRC[2][4][4][7]=-1.5e00;
    gas->SPRC[1][4][4][8]=0.07e00;
    gas->SPRC[2][4][4][8]=-1.5e00;
    gas->SPRT[1][4][4]=1000.e00;
    gas->SPRT[2][4][4]=3000.e00;
    //
    gas->ISPRC[2][2]=1;   //H+H+M -> H2+M
    gas->ISPRK[2][2]=1;
    gas->SPRC[1][2][2][1]=0.07e00;
    gas->SPRC[2][2][2][1]=-2.e00;
    gas->SPRC[1][2][2][2]=0.11e00;
    gas->SPRC[2][2][2][2]=-2.2e00;
    gas->SPRC[1][2][2][3]=0.052e00;
    gas->SPRC[2][2][2][3]=-2.5e00;
    gas->SPRC[1][2][2][4]=0.052e00;
    gas->SPRC[2][2][2][4]=-2.5e00;
    gas->SPRC[1][2][2][5]=0.052e00;
    gas->SPRC[2][2][2][5]=-2.5e00;
    gas->SPRC[1][2][2][6]=0.052e00;
    gas->SPRC[2][2][2][6]=-2.5e00;
    gas->SPRC[1][2][2][7]=0.052e00;
    gas->SPRC[2][2][2][7]=-2.5e00;
    gas->SPRC[1][2][2][8]=0.04e00;
    gas->SPRC[2][2][2][7]=-2.5e00;
    gas->SPRT[1][2][2]=1000.e00;
    gas->SPRT[2][2][2]=3000.e00;
    //
    gas->ISPRC[2][4]=5;    //H+0+M -> OH+M
    gas->ISPRK[2][4]=1;
    gas->SPRC[1][2][4][1]=0.15e00;
    gas->SPRC[2][2][4][1]=-2.e00;
    gas->SPRC[1][2][4][2]=0.04e00;
    gas->SPRC[2][2][4][2]=-1.3e00;
    gas->SPRC[1][2][4][3]=0.04e00;
    gas->SPRC[2][2][4][3]=-1.3e00;
    gas->SPRC[1][2][4][4]=0.04e00;
    gas->SPRC[2][2][4][4]=-1.3e00;
    gas->SPRC[1][2][4][5]=0.04e00;
    gas->SPRC[2][2][4][5]=-1.3e00;
    gas->SPRC[1][2][4][6]=0.21e00;
    gas->SPRC[2][2][4][6]=-2.1e00;
    gas->SPRC[1][2][4][7]=0.18e00;
    gas->SPRC[2][2][4][7]=-2.3e00;
    gas->SPRC[1][2][4][8]=0.16e00;
    gas->SPRC[2][2][4][8]=-2.3e00;
    gas->SPRT[1][2][4]=1000.e00;
    gas->SPRT[2][2][4]=3000.e00;
    //
    gas->ISPRC[2][5]=6;    //H+OH+M -> H2O+M
    gas->ISPRK[2][5]=1;
    gas->SPRC[1][2][5][1]=0.1e00;
    gas->SPRC[2][2][5][1]=-2.0e00;
    gas->SPRC[1][2][5][2]=0.1e00;
    gas->SPRC[2][2][5][2]=-2.0e00;
    gas->SPRC[1][2][5][3]=0.0025e00;
    gas->SPRC[2][2][5][3]=-2.2e00;
    gas->SPRC[1][2][5][4]=0.0025e00;
    gas->SPRC[2][2][5][4]=-2.2e00;
    gas->SPRC[1][2][5][5]=0.0025e00;
    gas->SPRC[2][2][5][5]=-2.2e00;
    gas->SPRC[1][2][5][6]=0.0015e00;
    gas->SPRC[2][2][5][6]=-2.2e00;
    gas->SPRC[1][2][5][7]=0.0027e00;
    gas->SPRC[2][2][5][7]=-2.e00;
    gas->SPRC[1][2][5][8]=0.0025e00;
    gas->SPRC[2][2][5][8]=-2.e00;
    gas->SPRT[1][2][5]=1000.e00;
    gas->SPRT[2][2][5]=3000.e00;
    //
    gas->ISPRC[2][3]=7;   //H+O2+M -> H02+M
    gas->ISPRK[2][3]=1;
    gas->SPRC[1][2][3][1]=0.0001e00;
    gas->SPRC[2][2][3][1]=-1.7e00;
    gas->SPRC[1][2][3][2]=0.0001e00;
    gas->SPRC[2][2][3][2]=-1.7e00;
    gas->SPRC[1][2][3][3]=0.00003e00;
    gas->SPRC[2][2][3][3]=-1.5e00;
    gas->SPRC[1][2][3][4]=0.00003e00;
    gas->SPRC[2][2][3][4]=-1.7e00;
    gas->SPRC[1][2][3][5]=0.00003e00;
    gas->SPRC[2][2][3][5]=-1.7e00;
    gas->SPRC[1][2][3][6]=0.00003e00;
    gas->SPRC[2][2][3][6]=-1.7e00;
    gas->SPRC[1][2][3][7]=0.000012e00;
    gas->SPRC[2][2][3][7]=-1.7e00;
    gas->SPRC[1][2][3][8]=0.00002e00;
    gas->SPRC[2][2][3][8]=-1.7e00;
    gas->SPRT[1][2][3]=1000.e00;
    gas->SPRT[2][2][3]=3000.e00;
    //
    //set the exchange reaction data
    //  memset(gas->SPEX,0,sizeof(****gas->SPEX));//gas->SPEX=0.e00;    //all activation energies and heats of reaction are zero unless set otherwise
    for(int i=0;i<7;i++){
        for(int j=0;j<gas->MMEX+1;j++){
            for(int k=0;k<gas->MSP+1;k++){
                for(int l=0;l<gas->MSP+1;l++)
                    gas->SPEX[i][j][k][l]=0.e00;
            }
        }
    }
    //gas->ISPEX=0;       // ISPEX is also zero unless set otherwise
    for(int i=0;i<gas->MMEX+1;i++){
        for(int j=0;j<8;j++){
            for(int k=0;k<gas->MSP+1;k++){
                for(int l=0;l<gas->MSP+1;l++)
                    gas->ISPEX[i][j][k][l]=0.e00;
            }
        }
    }
    //gas->NSPEX=0;
    for(int i=0;i<gas->MSP+1;i++){
        for(int j=0;j<gas->MSP+1;j++){
            gas->NSPEX[i][j]=0;
        }
    }
    //set the number of exchange reactions for each species pair
    gas->NSPEX[1][3]=1;
    gas->NSPEX[2][7]=3;
    gas->NSPEX[2][3]=1;
    gas->NSPEX[4][5]=1;
    gas->NSPEX[1][4]=1;
    gas->NSPEX[2][5]=1;
    gas->NSPEX[1][5]=1;
    gas->NSPEX[2][6]=1;
    gas->NSPEX[4][6]=2;
    gas->NSPEX[5][5]=2;
    gas->NSPEX[4][7]=1;
    gas->NSPEX[3][5]=1;
    //set the information on the chain reactions
    //
    //H2+O2 -> HO2+H
    gas->ISPEX[1][1][1][3]=1;
    gas->ISPEX[1][2][1][3]=3;
    gas->ISPEX[1][3][1][3]=7;
    gas->ISPEX[1][4][1][3]=2;
    gas->ISPEX[1][5][1][3]=1;
    gas->ISPEX[1][6][1][3]=1;
    gas->SPEX[6][1][1][3]=0.e00;
    gas->NEX[1][1][3]=1;
    //
    //HO2+H -> H2+02
    gas->ISPEX[1][1][2][7]=7;
    gas->ISPEX[1][2][2][7]=2;
    gas->ISPEX[1][3][2][7]=1;
    gas->ISPEX[1][4][2][7]=3;
    gas->ISPEX[1][5][2][7]=1;
    gas->ISPEX[1][6][2][7]=1;
    gas->ISPEX[1][7][2][7]=1;
    //H02 is H-O-O so that not all vibrational modes contribute to this reaction, but the numbers here are guesses//
    gas->SPEX[1][1][2][7]=20.e00;
    gas->SPEX[2][1][2][7]=0.4e00;
    gas->SPEX[4][1][2][7]=2000.e00;
    gas->SPEX[5][1][2][7]=3000.e00;
    gas->SPEX[6][1][2][7]=0.e00;
    gas->NEX[1][2][7]=2;
    //
    //O2+H -> OH+O
    gas->ISPEX[1][1][2][3]=3;
    gas->ISPEX[1][2][2][3]=2;
    gas->ISPEX[1][3][2][3]=5;
    gas->ISPEX[1][4][2][3]=4;
    gas->ISPEX[1][5][2][3]=1;
    gas->ISPEX[1][6][2][3]=1;
    gas->SPEX[6][1][2][3]=0.e00;
    gas->NEX[1][2][3]=3;
    //
    //OH+O -> O2+H
    gas->ISPEX[1][1][4][5]=5;
    gas->ISPEX[1][2][4][5]=4;
    gas->ISPEX[1][3][4][5]=3;
    gas->ISPEX[1][4][4][5]=2;
    gas->ISPEX[1][5][4][5]=1;
    gas->ISPEX[1][6][4][5]=1;
    gas->ISPEX[1][7][4][5]=1;
    gas->SPEX[1][1][4][5]=0.65e00;
    gas->SPEX[2][1][4][5]=-0.26;
    gas->SPEX[4][1][4][5]=2000.e00;
    gas->SPEX[5][1][4][5]=3000.e00;
    gas->SPEX[6][1][4][5]=0.e00;
    gas->NEX[1][4][5]=4;
    //
    //H2+O -> OH+H
    gas->ISPEX[1][1][1][4]=1;
    gas->ISPEX[1][2][1][4]=4;
    gas->ISPEX[1][3][1][4]=5;
    gas->ISPEX[1][4][1][4]=2;
    gas->ISPEX[1][5][1][4]=1;
    gas->ISPEX[1][6][1][4]=1;
    gas->SPEX[6][1][1][4]=0.e00;
    gas->NEX[1][1][4]=5;
    //
    //OH+H -> H2+O
    gas->ISPEX[1][1][2][5]=5;
    gas->ISPEX[1][2][2][5]=2;
    gas->ISPEX[1][3][2][5]=1;
    gas->ISPEX[1][4][2][5]=4;
    gas->ISPEX[1][5][2][5]=1;
    gas->ISPEX[1][6][2][5]=1;
    gas->ISPEX[1][7][2][5]=1;
    gas->SPEX[1][1][2][5]=0.5e00;
    gas->SPEX[2][1][2][5]=-0.2e00;
    gas->SPEX[4][1][2][5]=2000.e00;
    gas->SPEX[5][1][2][5]=3000.e00;
    gas->SPEX[6][1][2][5]=0.e00;
    gas->NEX[1][2][5]=6;
    //
    //H20+H -> OH+H2
    gas->ISPEX[1][1][2][6]=6;
    gas->ISPEX[1][2][2][6]=2;
    gas->ISPEX[1][3][2][6]=5;
    gas->ISPEX[1][4][2][6]=1;
    gas->ISPEX[1][5][2][6]=1;
    gas->ISPEX[1][6][2][6]=1;
    gas->SPEX[6][1][2][6]=2.0e-19;
    gas->NEX[1][2][6]=7;
    
    //OH+H2 -> H2O+H
    gas->ISPEX[1][1][1][5]=5;
    gas->ISPEX[1][2][1][5]=1;
    gas->ISPEX[1][3][1][5]=6;
    gas->ISPEX[1][4][1][5]=2;
    gas->ISPEX[1][5][1][5]=1;
    gas->ISPEX[1][6][1][5]=1;
    gas->ISPEX[1][7][1][5]=1;
    gas->SPEX[1][1][1][5]=0.5;
    gas->SPEX[2][1][1][5]=-0.2;
    gas->SPEX[4][1][1][5]=2000.e00;
    gas->SPEX[5][1][1][5]=3000.e00;
    gas->SPEX[6][1][1][5]=0.e00;
    gas->NEX[1][1][5]=8;
    //
    //H2O+O -> OH+OH
    gas->ISPEX[1][1][4][6]=6;
    gas->ISPEX[1][2][4][6]=4;
    gas->ISPEX[1][3][4][6]=5;
    gas->ISPEX[1][4][4][6]=5;
    gas->ISPEX[1][5][4][6]=1;
    gas->ISPEX[1][6][4][6]=1;
    gas->SPEX[6][1][4][6]=0.e00;
    gas->NEX[1][4][6]=9;
    //
    //0H+OH -> H2O+O
    gas->ISPEX[1][1][5][5]=5;
    gas->ISPEX[1][2][5][5]=5;
    gas->ISPEX[1][3][5][5]=6;
    gas->ISPEX[1][4][5][5]=4;
    gas->ISPEX[1][5][5][5]=1;
    gas->ISPEX[1][6][5][5]=1;
    gas->ISPEX[1][7][5][5]=1;
    gas->SPEX[1][1][5][5]=0.35;
    gas->SPEX[2][1][5][5]=-0.2 ;
    gas->SPEX[4][1][5][5]=2000.e00;
    gas->SPEX[5][1][5][5]=3000.e00;
    gas->SPEX[6][1][5][5]=0.e00;
    gas->NEX[1][5][5]=10;
    //
    //OH+OH  -> HO2+H
    //
    gas->ISPEX[2][1][5][5]=5;
    gas->ISPEX[2][2][5][5]=5;
    gas->ISPEX[2][3][5][5]=7;
    gas->ISPEX[2][4][5][5]=2;
    gas->ISPEX[2][5][5][5]=1;
    gas->ISPEX[2][6][5][5]=1;
    gas->SPEX[6][2][5][5]=0.e00;
    gas->NEX[2][5][5]=11;
    //
    //H02+H -> 0H+OH
    gas->ISPEX[2][1][2][7]=7;
    gas->ISPEX[2][2][2][7]=2;
    gas->ISPEX[2][3][2][7]=5;
    gas->ISPEX[2][4][2][7]=5;
    gas->ISPEX[2][5][2][7]=1;
    gas->ISPEX[2][6][2][7]=1;
    gas->ISPEX[2][7][2][7]=1;
    gas->SPEX[1][2][2][7]=120.e00;
    gas->SPEX[2][2][2][7]=-0.05e00;
    gas->SPEX[4][2][2][7]=2000.e00;
    gas->SPEX[5][2][2][7]=3000.e00;
    gas->SPEX[6][2][2][7]=0.e00;
    gas->NEX[2][2][7]=12;
    //
    //H2O+O -> HO2+H
    //
    gas->ISPEX[2][1][4][6]=6;
    gas->ISPEX[2][2][4][6]=4;
    gas->ISPEX[2][3][4][6]=7;
    gas->ISPEX[2][4][4][6]=2;
    gas->ISPEX[2][5][4][6]=1;
    gas->ISPEX[2][6][4][6]=1;
    gas->SPEX[6][2][4][6]=0.e00;
    gas->NEX[2][4][6]=13;
    //
    //H02+H -> H2O+O
    //
    gas->ISPEX[3][1][2][7]=7;
    gas->ISPEX[3][2][2][7]=2;
    gas->ISPEX[3][3][2][7]=6;
    gas->ISPEX[3][4][2][7]=4;
    gas->ISPEX[3][5][2][7]=1;
    gas->ISPEX[3][6][2][7]=1;
    gas->ISPEX[3][7][2][7]=1;
    gas->SPEX[1][3][2][7]=40.e00;
    gas->SPEX[2][3][2][7]=-1.e00;
    gas->SPEX[4][3][2][7]=2000.e00;
    gas->SPEX[5][3][2][7]=3000.e00;
    gas->SPEX[6][3][2][7]=0.e00;
    gas->NEX[3][2][7]=14;
    //
    //OH+O2 -> HO2+O
    //
    gas->ISPEX[1][1][3][5]=5;
    gas->ISPEX[1][2][3][5]=3;
    gas->ISPEX[1][3][3][5]=7;
    gas->ISPEX[1][4][3][5]=4;
    gas->ISPEX[1][5][3][5]=1;
    gas->ISPEX[1][6][3][5]=1;
    gas->SPEX[6][1][3][5]=0.e00;
    gas->NEX[1][3][5]=15;
    //
    //H02+0 -> OH+O2
    //
    gas->ISPEX[1][1][4][7]=7;
    gas->ISPEX[1][2][4][7]=4;
    gas->ISPEX[1][3][4][7]=5;
    gas->ISPEX[1][4][4][7]=3;
    gas->ISPEX[1][5][4][7]=1;
    gas->ISPEX[1][6][4][7]=1;
    gas->ISPEX[1][7][4][7]=1;
    gas->SPEX[1][1][4][7]=100.e00;
    gas->SPEX[2][1][4][7]=0.15e00;
    gas->SPEX[4][1][4][7]=2000.e00;
    gas->SPEX[5][1][4][7]=3000.e00;
    gas->SPEX[6][1][4][7]=0.e00;
    gas->NEX[1][4][7]=16;
    
    //
    DERIVED_GAS_DATA();
    //
    cout<<"OXYGEN_HYDROGEN data done"<<endl;
    return;
}
//***************************************************************************
//*************************END OF GAS DATABASE*******************************
//***************************************************************************
//
void DERIVED_GAS_DATA()
{
    //
    //GAS gas;
    //CALC calc;
    int I,II,J,JJ,K,L,M,MM,N,JMAX,MOLSP,MOLOF,NSTEP,IMAX;
    double A,B,BB,C,X,T,CUR,EAD,TVD,ZVT,ERD,PETD,DETD,PINT,ETD,SUMD,VAL;
    double **BFRAC,**TOT;
    double ****VRRD;
    double *****VRREX;
    //
    //VRRD(1,L,M,K) dissociation rate coefficient to species L,M for vibrational level K at 5,000 K
    //VRRD(2,L,M,K) similar for 15,000 K
    //VRREX(1,J,L,M,K)  Jth exchange rate coefficient to species L,M for vibrational level K at 1,000 K
    //VRREX(2,J,L,M,K) similar for 3,000 K
    //BFRAC(2,J) Boltzmann fraction
    //JMAX imax-1
    //T temperature
    //CUR sum of level resolved rates
    //
    
    VRRD = new double ***[3];
    for (int i = 0; i < 3; ++i)
    {
        VRRD[i] = new double **[gas->MSP+1];
        for (int j = 0; j < gas->MSP+1; ++j)
        {
            VRRD[i][j] = new double *[gas->MSP+1];
            for(int k=0; k<gas->MSP+1; ++k)
                VRRD[i][j][k]=new double [gas->MVIBL+1];
        }
    }
    
    BFRAC = new double*[gas->MVIBL+1];
    for(int i =0; i< (gas->MVIBL+1); ++i)
        BFRAC[i] = new double[3];
    
    VRREX = new double ****[3];
    for (int i = 0; i < 3; ++i)
    {
        VRREX[i] = new double ***[gas->MMEX+1];
        for (int j = 0; j < gas->MMEX+1; ++j)
        {
            VRREX[i][j] = new double **[gas->MSP+1];
            for(int k=0; k<gas->MSP+1; ++k)
            {
                VRREX[i][j][k]=new double *[gas->MSP+1];
                for(int l=0; l<gas->MSP+1; ++l)
                    VRREX[i][j][k][l]= new double[gas->MVIBL+1];
            }
        }
    }
    
    TOT = new double*[gas->MVIBL+1];
    for(int i =0; i< (gas->MVIBL+1); ++i)
        TOT[i] = new double[3];
    
    // ALLOCATE (VRRD(2,MSP,MSP,0:MVIBL),BFRAC(0:MVIBL,2),VRREX(2,MMEX,MSP,MSP,0:MVIBL),TOT(0:MVIBL,2),STAT=ERROR)
    // IF (ERROR /= 0) THEN
    //   WRITE (*,*)'PROGRAM COULD NOT ALLOCATE VIB. RES. DISS. RATES',ERROR
    // END IF
    //
    cout<<"Setting derived gas data"<<endl;
    //copy the L,M data that has been specified for L < M so that it applies also for M>L
    for(L=1;L<=gas->MSP;L++){
        for(M=1;M<=gas->MSP;M++){
            if(L > M){
                gas->NSPEX[L][M]=gas->NSPEX[M][L];
                gas->ISPRC[L][M]=gas->ISPRC[M][L];
                gas->ISPRK[L][M]=gas->ISPRK[M][L];
                for(K=1;K<=gas->MSP;K++){
                    gas->SPRT[1][L][M]=gas->SPRT[1][M][L];
                    gas->SPRT[2][L][M]=gas->SPRT[2][M][L];
                    gas->SPRC[1][L][M][K]=gas->SPRC[1][M][L][K];
                    gas->SPRC[2][L][M][K]=gas->SPRC[2][M][L][K];
                }
                for(K=1;K<=gas->MMEX;K++){
                    gas->NEX[K][L][M]=gas->NEX[K][M][L];
                    for(J=1;J<=6;J++){
                        gas->SPEX[J][K][L][M]=gas->SPEX[J][K][M][L];
                    }
                    for(J=1;J<=7;J++){
                        gas->ISPEX[K][J][L][M]=gas->ISPEX[K][J][M][L];
                    }
                }
            }
        }
    }
    //
    if(gas->MMVM > 0){
        //set the characteristic dissociation temperatures
        for(L=1;L<=gas->MSP;L++){
            if(gas->ISPV[L] > 0){
                for(K=1;K<=gas->ISPV[L];K++)
                {
                    I=gas->ISPVM[1][K][L];
                    J=gas->ISPVM[2][K][L];
                    gas->SPVM[4][K][L]=(gas->SP[6][I]+gas->SP[6][J]-gas->SP[6][L])/BOLTZ;
                    //WRITE (9,*) 'Char. Diss temp of species',L,' is',SPVM(4,K,L)
                    file_9<<"Char. Diss temp of species "<<L<<" is "<<gas->SPVM[4][K][L]<<endl;
                }
            }
        }
    }
    //
    if(gas->MMEX > 0){
        //set the heats of reaction of the exchange and chain reactions
        for(L=1;L<=gas->MSP;L++){
            for(M=1;M<=gas->MSP;M++){
                for(J=1;J<=gas->MMEX;J++){
                    if((gas->ISPEX[J][3][L][M]> 0) && (gas->ISPEX[J][4][L][M]>0) && (gas->ISPEX[J][1][L][M]>0) && (gas->ISPEX[J][2][L][M]>0)){
                        gas->SPEX[3][J][L][M]=gas->SP[6][gas->ISPEX[J][1][L][M]]+gas->SP[6][gas->ISPEX[J][2][L][M]]-gas->SP[6][gas->ISPEX[J][3][L][M]]-gas->SP[6][gas->ISPEX[J][4][L][M]];
                        // WRITE (9,*) 'Reaction',NEX(J,L,M),' heat of reaction',SPEX(3,J,L,M)
                        file_9<<"Reaction "<<gas->NEX[J][L][M]<<" heat of reaction"<<gas->SPEX[3][J][L][M]<<endl;
                    }
                }
            }
        }
    }
    //
    if(gas->MELE > 1){
        //set the electronic cross-section ratios to a mean electronic relaxation collision number
        //(equipartition is not achieved unless there is a single number)
        for(L=1;L<=gas->MSP;L++){
            A=0.e00;
            for(K=1;K<=gas->NELL[L];K++){
                A=A+gas->QELC[3][K][L];
            }
            gas->QELC[3][1][L]=A/double(gas->NELL[L]);
        }
    }
    //
    //set the cumulative distributions of the post-recombination vibrational distributions for establishment of detailed balance
    for(L=1;L<=gas->MSP;L++){
        for(M=1;M<=gas->MSP;M++){
            if(gas->ISPRC[L][M] > 0){
                N=gas->ISPRC[L][M];   //recombined species
                K=gas->ISPRK[L][M];   //relevant vibrational mode
                //WRITE (9,*) 'SPECIES',L,M,' RECOMBINE TO',N
                file_9<<"SPECIES "<<L<<" "<<M<<" RECOMBINE TO"<<N<<endl;
                JMAX=gas->SPVM[4][K][N]/gas->SPVM[1][K][N];
                if(JMAX > gas->MVIBL){
                    cout<<" The variable MVIBL="<<gas->MVIBL<<" in the gas database must be increased to"<<JMAX<<endl;
                    cout<<"Enter 0 ENTER to stop";
                    cin>> A;
                    return ;
                }
                A=2.5e00-gas->SP[3][N];
                for(I=1;I<=2;I++){
                    if(I == 1) T=gas->SPRT[1][L][M];
                    if(I == 2) T=gas->SPRT[2][L][M];
                    //WRITE (9,*) 'TEMPERATURE',T
                    file_9<<"TEMPERATURE "<<T<<endl;
                    CUR=0.e00;
                    for(J=0;J<=JMAX;J++){
                        X=double(JMAX+1-J)*gas->SPVM[1][K][N]/T;
                        CQAX(A,X,B);
                        VRRD[I][L][M][J]=B*exp(-double(J)*gas->SPVM[1][K][N]/T);
                        CUR=CUR+VRRD[I][L][M][J];
                    }
                    B=0.e00;
                    for(J=0;J<=JMAX;J++){
                        B=B+VRRD[I][L][M][J]/CUR;
                        gas->SPRP[I][L][M][J]=B;
                        //WRITE (9,*) 'CDF level dissoc',J,SPRP(I,L,M,J)
                        file_9<< "CDF level dissoc "<<J<<" "<<gas->SPRP[I][L][M][J];
                    }
                }
            }
        }
    }
    //
    //READ (*,*)  //optionally pause program to check cumulative distributions for exchange and chain reactions
    //
    //set the cumulative distributions of the post-reverse vibrational distributions for establishment of detailed balance
    for(L=1;L<=gas->MSP;L++){
        for(M=1;M<=gas->MSP;M++){
            if(gas->NSPEX[L][M] > 0){
                for(K=1;K<=gas->NSPEX[L][M];K++){
                    if(gas->SPEX[3][K][L][M] > 0.e00){         //exothermic (reverse) exchange reaction
                        //L,M are the species in the reverse reaction, E_a of forward reaction is SPEX(3,K,L,M)
                        //WRITE (9,*) 'SPECIES',L,M,' REVERSE REACTION'
                        file_9<<"SPECIES "<<L<<" "<<M<<" REVERSE REACTION"<<endl;
                        MOLSP=gas->ISPEX[K][3][L][M];  //molecuke that splits in the forward reaction
                        MOLOF=gas->ISPEX[K][4][L][M];
                        JMAX=(gas->SPEX[3][K][L][M]+gas->SPEX[6][K][MOLSP][MOLOF])/(BOLTZ*gas->SPVM[1][gas->ISPEX[K][5][L][M]][MOLSP])+15;   //should always be less than the JMAX set by dissociation reactions
                        for(I=1;I<=2;I++){
                            if(I == 1) T=gas->SPEX[4][K][L][M];
                            if(I == 2) T=gas->SPEX[5][K][L][M];
                            for(J=0;J<=JMAX;J++){
                                EAD=(gas->SPEX[3][K][L][M]+gas->SPEX[6][K][MOLSP][MOLOF])/(BOLTZ*T);
                                TVD=gas->SPVM[1][gas->ISPEX[K][5][L][M]][MOLSP]/T;
                                ZVT=1.e00/(1.e00-exp(-TVD));
                                C=ZVT/(tgamma(2.5e00-gas->SP[3][MOLSP])*exp(-EAD));  //coefficient of integral
                                ERD=EAD-double(J)*TVD;
                                if(ERD < 0.e00) ERD=0.e00;
                                PETD=ERD;
                                DETD=0.01e00;
                                PINT=0.e00;  //progressive value of integral
                                NSTEP=0;
                                A=1.e00;
                                while(A > 1.e-10){
                                    NSTEP=NSTEP+1;
                                    ETD=PETD+0.5e00*DETD;
                                    SUMD=0.e00;  //normalizing sum in the denominator
                                    IMAX=ETD/TVD+J;
                                    for(II=0;II<=IMAX;II++){
                                        SUMD=SUMD+pow((1.e00-double(II)*TVD/(ETD+double(J)*TVD)),(1.5e00-gas->SP[3][MOLSP]));
                                    }
                                    VAL=(pow((ETD*(1.e00-EAD/(ETD+double(J)*TVD))),(1.5e00-gas->SP[3][MOLSP]))/SUMD)*exp(-ETD);
                                    PINT=PINT+VAL*DETD;
                                    A=VAL/PINT;
                                    PETD=ETD+0.5e00*DETD;
                                }
                                VRREX[I][K][L][M][J]=C*PINT;
                                //              WRITE (*,*) 'Level ratio exch',I,J,VRREX(I,K,L,M,J)
                            }
                        }
                        //
                        //memset(TOT,0.e00,sizeof(**TOT));//TOT=0.e00;
                        for(int i=0;i<gas->MVIBL+1;i++){
                            for(int j=0;j<gas->MVIBL+1;j++){
                                TOT[i][j]=0;
                            }
                        }
                        for(I=1;I<=2;I++){
                            if(I == 1) T=gas->SPEX[4][K][L][M];
                            if(I == 2) T=gas->SPEX[5][K][L][M];
                            for(J=0;J<=JMAX;J++){
                                TVD=gas->SPVM[1][gas->ISPEX[K][5][L][M]][MOLSP]/T;
                                ZVT=1.e00/(1.e00-exp(-TVD));
                                BFRAC[J][I]=exp(-J*gas->SPVM[1][gas->ISPEX[K][5][L][M]][MOLSP]/T)/ZVT;    //Boltzmann fraction
                                VRREX[I][K][L][M][J]=VRREX[I][K][L][M][J]*BFRAC[J][I];
                                //              WRITE (*,*) 'Contribution',I,J,VRREX(I,K,L,M,J)
                                for(MM=0;MM<=J;MM++)
                                    TOT[J][I]=TOT[J][I]+VRREX[I][K][L][M][MM];
                            }
                        }
                        //
                        for(I=1;I<=2;I++){
                            for(J=0;J<=JMAX;J++){
                                gas->SPREX[I][K][L][M][J]=TOT[J][I];
                                if(J == JMAX) gas->SPREX[I][K][L][M][J]=1.e00;
                                //WRITE (9,*) 'Cumulative',I,J,SPREX(I,K,L,M,J)
                                file_9<<"Cumulative "<<I<<" "<<J<<" "<<gas->SPREX[I][K][L][M][J];
                            }
                        }
                    }
                }
                gas->NSLEV=0;
                //memset(gas->SLER,0.e00,sizeof(*gas->SLER));//gas->SLER=0.e00;
                for(int i=0;i<gas->MSP+1;i++)
                    gas->SLER[i]=0.e00;
            }
        }
    }
    //
    //READ (*,*)  //optionally pause program to check cumulative distributions for exchange abd chain reactions
    return;
}

void READ_DATA()
{
    //CALC calc;
    //MOLECS molecs;
    //GAS gas;
    //OUTPUT output;
    //GEOM_1D geom;
    fstream file_3;
    fstream file_4;
    
    int NVERD,MVERD,N,K;
    if(calc->ICLASS==0)
    {
        cout<<"Reading the data file DS0D.DAT"<<endl;
        file_4.open("DS0D.DAT", ios::in);
        file_3.open("DS0D.TXT", ios::out);
        file_3<<"Data summary for program DSMC"<<endl;
        
        // OPEN (4,FILE='DS0D.DAT')
        // OPEN (3,FILE='DS0D.TXT')
        // WRITE (3,*) 'Data summary for program DSMC'
    }
    if(calc->ICLASS==1)
    {
        cout<<"Reading the data file DS1D.DAT"<<endl;
        file_4.open("DS1D.DAT", ios::in);
        file_3.open("DS1D.TXT", ios::out );
        file_3<<"Data summary for program DSMC"<<endl;
        // OPEN (4,FILE='DS1D.DAT')
        // OPEN (3,FILE='DS1D.TXT')
        // WRITE (3,*) 'Data summary for program DSMC'
    }
    //the following items are common to all classes of flow
    file_4>>NVERD;
    file_3<<"The n in version number n.m is "<<NVERD<<endl;
    file_4>>MVERD;
    file_3<<"The m in version number n.m is "<<MVERD<<endl;
    file_4>>calc->IMEG;
    file_3<<"The approximate number of megabytes for the calculation is "<<calc->IMEG<<endl;
    file_4>>gas->IGAS;
    file_3<<gas->IGAS<<endl;//gas->IGAS=1;
    // READ (4,*) NVERD
    // WRITE (3,*) 'The n in version number n.m is',NVERD
    // READ (4,*) MVERD
    // WRITE (3,*) 'The m in version number n.m is',MVERD
    // READ (4,*) IMEG //calc->IMEG
    // WRITE (3,*) 'The approximate number of megabytes for the calculation is',IMEG //calc->IMEG
    // READ (4,*) IGAS //gas->IGAS
    // WRITE (3,*) IGAS //gas->IGAS
    if(gas->IGAS==1)
    {
        file_3<<" Hard sphere gas "<<endl;
        // WRITE (3,*) 'Hard sphere gas'
        HARD_SPHERE();
    }
    if(gas->IGAS==2)
    {
        file_3<<"Argon "<<endl;
        // WRITE (3,*) 'Argon'
        ARGON();
    }
    if(gas->IGAS==3)
    {
        file_3<<"Ideal nitrogen"<<endl;
        // WRITE (3,*) 'Ideal nitrogen'
        IDEAL_NITROGEN();
    }
    if(gas->IGAS==4)
    {
        file_3<<"Real oxygen "<<endl;
        // WRITE (3,*) 'Real oxygen'
        REAL_OXYGEN();
    }
    if(gas->IGAS==5)
    {
        file_3<<"Ideal air "<<endl;
        // TE (3,*) 'Ideal air'
        IDEAL_AIR();
    }
    if(gas->IGAS==6)
    {
        file_3<<"Real air @ 7.5 km/s "<<endl;
        // RITE (3,*) 'Real air @ 7.5 km/s'
        REAL_AIR();
    }
    if(gas->IGAS==7)
    {
        file_3<<"Helium-argon-xenon mixture "<<endl;
        // WRITE (3,*) 'Helium-argon-xenon mixture'
        HELIUM_ARGON_XENON();
    }
    if(gas->IGAS==8)
    {
        file_3<<"Oxygen-hydrogen "<<endl;
        // WRRITE (3,*) 'Oxygen-hydrogen'
        OXYGEN_HYDROGEN();
    }
    file_3<<"The gas properties are:- "<<endl;
    file_4>>gas->FND[1];
    file_3<<"The stream number density is "<<gas->FND[1]<<endl;
    file_4>>gas->FTMP[1];
    file_3<<"The stream temperature is "<<gas->FTMP[1]<<endl;
    // WRITE (3,*) 'The gas properties are:-'
    // READ (4,*) FND(1) //gas->FND[1]
    // WRITE (3,*) '    The stream number density is',FND(1) ////gas->FND[1]
    // READ (4,*) FTMP(1) //gas->FTMP[1]
    // WRITE (3,*) '    The stream temperature is',FTMP(1) //gas->FTMP[1]
    if(gas->MMVM>0)
    {
        file_4>>gas->FVTMP[1];
        file_3<<"The stream vibrational and electronic temperature is "<<gas->FVTMP[1]<<endl;
        // READ (4,*) FVTMP(1) //gas->FVTMP;
        // WRITE (3,*) '    The stream vibrational and electronic temperature is',FVTMP(1) //gas->FVTMP[1]
    }
    if(calc->ICLASS==1)
    {
        file_4>>gas->VFX[1];
        file_3<<"The stream velocity in the x direction is "<<gas->VFX[1]<<endl;
        file_4>>gas->VFY[1];
        file_3<<"The stream velocity in the y direction is "<<gas->VFY[1]<<endl;
        // READ (4,*) VFX(1) //gas->VFX[1]
        // WRITE (3,*) '    The stream velocity in the x direction is',VFX(1) //gas->VFX[1]
        // READ (4,*) VFY(1) ////gas->VFY[1]
        // WRITE (3,*) '    The stream velocity in the y direction is',VFY(1) ////gas->VFY[1]
    }
    if(gas->MSP>1)
    {
        for(N=1;N<=gas->MSP;N++)
        {
            file_4>>gas->FSP[N][1];
            file_3<<" The fraction of species "<<N<<" is "<<gas->FSP[N][1]<<endl;
            // READ (4,*) FSP(N,1) //gas->FSP[N][1]
            // WRITE (3,*) '    The fraction of species',N,' is',FSP(N,1) //gas->FSP[N][1]
        }
    }
    else
    {
        gas->FSP[1][1]=1.0; //simple gas
    }
    if(calc->ICLASS==0){
        //       !--a homogeneous gas case is calculated as a one-dimensional flow with a single sampling cell
        // !--set the items that are required in the DS1D.DAT specification
        geom->IFX=0;
        geom->JFX=1;
        geom->XB[1]=0.e00;
        geom->XB[2]=0.0001e00*1.e25/gas->FND[1];
        geom->ITYPE[1]=1;
        geom->ITYPE[2]=1;
        gas->VFX[1]=0.e00;
        calc->IGS=1;
        calc->ISECS=0;
        calc->IREM=0;
        calc->MOLSC=10000*calc->IMEG; //a single sampling cell
    }
    if(calc->ICLASS==1)
    {
        file_4>>geom->IFX;
        // READ (4,*) IFX //geom->IFX
        if(geom->IFX==0)
            file_3<<"Plane Flow"<<endl;
        // WRITE (3,*) 'Plane flow'
        if(geom->IFX==0)
            file_3<<"Cylindrical flow"<<endl;
        // WRITE (3,*) 'Cylindrical flow'
        if(geom->IFX==0)
            file_3<<"Spherical flow"<<endl;
        // WRITE (3,*) 'Spherical flow'
        geom->JFX=geom->IFX+1;
        file_4>>geom->XB[1];
        // READ (4,*) XB(1) //geom->XB[1]
        file_3<<"The minimum x coordinate is "<<geom->XB[1]<<endl;
        // WRITE (3,*) 'The minimum x coordinate is',XB(1) //geom->XB[1]
        file_4>>geom->ITYPE[1];
        // READ (4,*) ITYPE(1) //geom->ITYPE[1]
        if(geom->ITYPE[1]==0)
            file_3<<"The minimum x coordinate is a stream boundary"<<endl;
        // WRITE (3,*) 'The minimum x coordinate is a stream boundary'
        if(geom->ITYPE[1]==1)
            file_3<<"The minimum x coordinate is a plane of symmetry"<<endl;
        // WRITE (3,*) 'The minimum x coordinate is a plane of symmetry'
        if(geom->ITYPE[1]==2)
            file_3<<"The minimum x coordinate is a solid surface"<<endl;
        // WRITE (3,*) 'The minimum x coordinate is a solid surface'
        if(geom->ITYPE[1]==3)
            file_3<<"The minimum x coordinate is a vacuum"<<endl;
        // WRITE (3,*) 'The minimum x coordinate is a vacuum'
        if(geom->ITYPE[1]==4)
            file_3<<"The minimum x coordinate is an axis or center"<<endl;
        // WRITE (3,*) 'The minimum x coordinate is an axis or center'
        if(geom->ITYPE[1]==2)
        {
            file_3<<"The minimum x boundary is a surface with the following properties"<<endl;
            file_4>>gas->TSURF[1];
            file_3<<"The temperature of the surface is "<<gas->TSURF[1]<<endl;
            file_4>>gas->FSPEC[1];
            file_3<<"The fraction of specular reflection is "<<gas->FSPEC[1]<<endl;
            file_4>>gas->VSURF[1];
            file_3<<"The velocity in the y direction of this surface is "<<gas->VSURF[1];
            // WRITE (3,*) 'The minimum x boundary is a surface with the following properties'
            // READ (4,*) TSURF(1) //gas->TSURF[1]
            // WRITE (3,*) '     The temperature of the surface is',TSURF(1) //gas->TSURF[1]
            // READ (4,*) FSPEC(1) //gas->FSPEC[1]
            // WRITE (3,*) '     The fraction of specular reflection is',FSPEC(1) //gas->FSPEC[1]
            // READ (4,*) VSURF(1) //gas->VSURF[1]
            // WRITE (3,*) '     The velocity in the y direction of this surface is',VSURF(1) //gas->VSURF[1]
        }
        file_4>>geom->XB[2];
        file_3<<"The maximum x coordinate is "<<geom->XB[2]<<endl;
        file_4>>geom->ITYPE[2];
        // READ (4,*) XB(2) //geom->XB[2]
        // WRITE (3,*) 'The maximum x coordinate is',XB(2)//geom->XB[2]
        // READ (4,*) ITYPE(2)//geom->ITYPE[2]
        if(geom->ITYPE[2]==0)
            file_3<<"The mmaximum  x coordinate is a stream boundary"<<endl;
        // WRITE (3,*) 'The mmaximum  x coordinate is a stream boundary'
        if(geom->ITYPE[2]==1)
            file_3<<"The maximum x coordinate is a plane of symmetry"<<endl;
        // WRITE (3,*) 'The maximum x coordinate is a plane of symmetry'
        if(geom->ITYPE[2]==2)
            file_3<<"The maximum  x coordinate is a solid surface"<<endl;
        // WRITE (3,*) 'The maximum  x coordinate is a solid surface'
        if(geom->ITYPE[2]==3)
            file_3<<"The maximum  x coordinate is a vacuum"<<endl;
        // WRITE (3,*) 'The maximum  x coordinate is a vacuum'
        calc->ICN=0;
        if(geom->ITYPE[2]==4)
        {
            file_3<<"The maximum x coordinate is a stream boundary with a fixed number of simulated molecules"<<endl;
            // WRITE (3,*) 'The maximum x coordinate is a stream boundary with a fixed number of simulated molecules'
            if(gas->MSP==1)
                calc->ICN=1;
        }
        if(geom->ITYPE[2]==2)
        {
            file_3<<"The maximum  x boundary is a surface with the following properties"<<endl;
            file_4>>gas->TSURF[1];
            file_3<<"The temperature of the surface is "<<gas->TSURF[1]<<endl;
            file_4>>gas->FSPEC[1];
            file_3<<"The fraction of specular reflection is "<<gas->FSPEC[1]<<endl;
            file_4>>gas->VSURF[1];
            file_3<<"The velocity in the y direction of this surface is "<<gas->VSURF[1]<<endl;
            // WRITE (3,*) 'The maximum  x boundary is a surface with the following properties'
            // READ (4,*) TSURF(1) //gas->TSURF[1]
            // WRITE (3,*) '     The temperature of the surface is',TSURF(1) //gas->TSURF[1]
            // READ (4,*) FSPEC(1) //gas->FSPEC[1]
            // WRITE (3,*) '     The fraction of specular reflection is',FSPEC(1) //gas->FSPEC[1]
            // READ (4,*) VSURF(1) //gas->VSURF[1]
            // WRITE (3,*) '     The velocity in the y direction of this surface is',VSURF(1) //gas->VSURF[1]
        }
        if(geom->IFX>0)
        {
            file_4>>geom->IWF;
            // READ (4,*) READ (4,*) IWF //geom->IWF
            if(geom->IWF==0)
                file_3<<"There are no radial weighting factors"<<endl;
            // WRITE (3,*) 'There are no radial weighting factors'
            if(geom->IWF==0)
                file_3<<"There are radial weighting factors"<<endl;
            // WRITE (3,*) 'There are radial weighting factors'
            if(geom->IWF==0)
            {
                file_4>>geom->WFM;
                file_3<<"The maximum value of the weighting factor is  "<<geom->WFM<<endl;
                // READ (4,*) WFM //geom->WFM
                // WRITE (3,*) 'The maximum value of the weighting factor is ',WFM //geom->WFM
                geom->WFM=(geom->WFM-1)/geom->XB[2];
            }
        }
        file_4>>calc->IGS;
        // READ (4,*) IGS //calc->IGS
        if(calc->IGS==0)
            file_3<<"The flowfield is initially a vacuum "<<endl;
        // WRITE (3,*) 'The flowfield is initially a vacuum'
        if(calc->IGS==1)
            file_3<<"The flowfield is initially the stream(s) or reference gas"<<endl;
        // WRITE (3,*) 'The flowfield is initially the stream(s) or reference gas'
        file_4>>calc->ISECS;
        // READ (4,*) ISECS //calc->ISECS
        if(calc->ISECS==0)
            file_3<<"There is no secondary stream initially at x > 0"<<endl;
        // WRITE (3,*) 'There is no secondary stream initially at x > 0'
        if(calc->ISECS==1 && geom->IFX==0)
            file_3<<"There is a secondary stream applied initially at x = 0 (XB(2) must be > 0)"<<endl;
        // WRITE (3,*) 'There is a secondary stream applied initially at x = 0 (XB(2) must be > 0)'
        if(calc->ISECS==1 && geom->IFX>0)
        {
            if(geom->IWF==1)
            {
                file_3<<"There cannot be a secondary stream when weighting factors are present"<<endl;
                // WRITE (3,*) 'There cannot be a secondary stream when weighting factors are present'
                return;//STOP//dout
            }
            file_3<<"There is a secondary stream"<<endl;
            // WRITE (3,*) 'There is a secondary stream'
            file_4>>geom->XS;
            // READ (4,*) XS //geom->XS
            file_3<<"The secondary stream boundary is at r= "<<geom->XS<<endl;
            // WRITE (3,*) 'The secondary stream boundary is at r=',XS //geom->XS
        }
        if(calc->ISECS==1)
        {
            file_3<<"The secondary stream (at x>0 or X>XS) properties are:-"<<endl;
            file_4>>gas->FND[2];
            file_3<<"The stream number density is "<<gas->FND[2]<<endl;
            file_4>>gas->FTMP[2];
            file_3<<"The stream temperature is "<<gas->FTMP[2]<<endl;
            // WRITE (3,*) 'The secondary stream (at x>0 or X>XS) properties are:-'
            // READ (4,*) FND(2) //gas->FND
            // WRITE (3,*) '    The stream number density is',FND(2) //gas->FND
            // READ (4,*) FTMP(2) //gas->FTMP
            // WRITE (3,*) '    The stream temperature is',FTMP(2) //gas->FTMP
            if(gas->MMVM>0)
            {
                file_4>>gas->FVTMP[2];
                file_3<<"The stream vibrational and electronic temperature is "<<gas->FVTMP[2]<<endl;
                // READ (4,*) FVTMP(2) //gas->FVTMP[2]
                // WRITE (3,*) '    The stream vibrational and electronic temperature is',FVTMP(2) //gas->FVTMP[2]
            }
            file_4>>gas->VFX[2];
            file_3<<"The stream velocity in the x direction is "<<gas->VFX[2]<<endl;
            file_4>>gas->VFY[2];
            file_3<<"The stream velocity in the y direction is "<<gas->VFY[2]<<endl;
            // READ (4,*) VFX(2) //gas->VFX
            // WRITE (3,*) '    The stream velocity in the x direction is',VFX(2) //gas->VFX
            // READ (4,*) VFY(2) //gas->VFY
            // WRITE (3,*) '    The stream velocity in the y direction is',VFY(2) //gas->VFY
            if(gas->MSP>1)
            {
                for(N=1;N<=gas->MSP;N++)
                {
                    file_4>>gas->FSP[N][2];
                    file_3<<"The fraction of species "<<N<<" is "<<gas->FSP[N][2]<<endl;
                    // READ (4,*) FSP(N,2) //gas->FSP
                    // WRITE (3,*) '    The fraction of species',N,' is',FSP(N,2) //gas->FSP
                }
            }
            else
            {
                gas->FSP[1][2]=1;
            }
        }
        if(geom->IFX==0 && geom->ITYPE[1]==0)
        {
            file_4>>calc->IREM;
            // READ (4,*) IREM //calc->IREM
            if(calc->IREM==0)
            {
                file_3<<"There is no molecule removal"<<endl;
                // WRITE (3,*) 'There is no molecule removal'
                geom->XREM=geom->XB[1]-1.e00;
                geom->FREM=0.e00;
            }
            else if(calc->IREM==1)
            {
                file_4>>geom->XREM;
                file_3<<"There is full removal of the entering (at XB(1)) molecules between "<<geom->XREM<<" and "<<geom->XB[2]<<endl;
                // READ (4,*) XREM //geom->XREM
                // WRITE (3,*) ' There is full removal of the entering (at XB(1)) molecules between',XREM,' and',XB(2) //geom->XREM ,geom->XB[2]
                geom->FREM=1.e00;
            }
            else if(calc->IREM==2)
            {
                file_3<<"Molecule removal is specified whenever the program is restarted"<<endl;
                // WRITE (3,*) ' Molecule removal is specified whenever the program is restarted'
                geom->XREM=geom->XB[1]-1.e00;
                geom->FREM=0.e00;
            }
            else
            {
                geom->XREM=geom->XB[1]-1.e00;
                geom->FREM=0.e00;
            }
        }
        geom->IVB=0;
        geom->VELOB=0.e00;
        if(geom->ITYPE[2]==1)
        {
            file_4>>geom->IVB;
            // READ (4,*) IVB
            if(geom->IVB==0)
                file_3<<"The outer boundary is stationary"<<endl;
            // WRITE (3,*) ' The outer boundary is stationary'
            if(geom->IVB==1)
            {
                file_3<<"The outer boundary moves with a constant speed"<<endl;
                file_4>>geom->VELOB;
                file_3<<" The speed of the outer boundary is "<<geom->VELOB<<endl;
                // WRITE (3,*) ' The outer boundary moves with a constant speed'
                // READ (4,*) VELOB //geom->VELOB
                // WRITE (3,*) ' The speed of the outer boundary is',VELOB //geom->VELOB
            }
        }
        file_4>>calc->MOLSC;
        file_3<<"The desired number of molecules in a sampling cell is "<<calc->MOLSC<<endl;
        // READ (4,*) MOLSC //calc->MOLSC
        // WRITE (3,*) 'The desired number of molecules in a sampling cell is',MOLSC ////calc->MOLSC
    }
    //set the speed of the outer boundary
    file_3.close();
    file_4.close();
    // CLOSE (3)
    // CLOSE (4)
    // set the stream at the maximum x boundary if there is no secondary stream
    if(calc->ISECS==0 && geom->ITYPE[2]==0)
    {
        gas->FND[2]=gas->FND[1];
        gas->FTMP[2]=gas->FTMP[1];
        if(gas->MMVM>0)
            gas->FVTMP[2]=gas->FVTMP[1];
        gas->VFX[2]=gas->VFX[1];
        if(gas->MSP>1)
        {
            for(N=1;N<=gas->MSP;N++)
            {
                gas->FSP[N][2]=gas->FSP[N][1];
            }
        }
        else
            gas->FSP[1][2]=1;
    }
    //dout
    //1234   CONTINUE;
    return;
}

void INITIALISE_SAMPLES()
{
    //start a new sample for all classes of flow
    //CALC calc;
    //GEOM_1D geom;
    //GAS gas;
    //OUTPUT output;
    //MOLECS molecs;
    
    int N;
    //
    output->NSAMP=0.0;
    output->TISAMP=calc->FTIME;
    output->NMISAMP=molecs->NM;
    //memset(output->COLLS,0.e00,sizeof(*output->COLLS));memset(output->WCOLLS,0.e00,sizeof(*output->WCOLLS));memset(output->CLSEP,0.e00,sizeof(*output->CLSEP));
   
    for(int i=0;i<geom->NCELLS+1;i++)
        output->COLLS[i]=0.e00;
    for(int i=0;i<geom->NCELLS+1;i++)
       output->WCOLLS[i]=0.e00;
    for(int i=0;i<geom->NCELLS+1;i++)
        output->CLSEP[i]=0.e00;
    //output->COLLS=0.e00 ; output->WCOLLS=0.e00 ; output->CLSEP=0.e00;
    //memset(calc->TCOL,0.0,sizeof(**calc->TCOL));//calc->TCOL=0.0;
    for(int i=0;i<gas->MSP+1;i++){
        for(int j=0;j<gas->MSP+1;j++){
            calc->TCOL[i][j]=0.0;
        }
    }
    //gas->TREACG=0;
    //gas->TREACL=0;
    for(int i=0;i<5;i++){
        for(int j=0;j<gas->MSP+1;j++){
            gas->TREACG[i][j]=0;
        }
    }
    for(int i=0;i<5;i++){
        for(int j=0;j<gas->MSP+1;j++){
            gas->TREACL[i][j]=0;
        }
    }
    //memset(output->CS,0.0,sizeof(***output->CS));memset(output->CSS,0.0,sizeof(****output->CSS));memset(output->CSSS,0.0,sizeof(**output->CSSS));
    for(int j=0;j<gas->MSP+10;j++){
        for(int k=0;k<geom->NCELLS+1;k++){
            for(int l=0;l<gas->MSP+1;l++)
                output->CS[j][k][l]=0.0;
        }
    }
    for(int i=0;i<9;i++){
        for(int j=0;j<3;j++){
            for(int k=0;k<gas->MSP+1;k++){
                for(int l=0;l<3;l++)
                    output->CSS[i][j][k][l]=0.0;
            }
        }
    }
    for(int k=0;k<7;k++){
        for(int l=0;l<3;l++)
            output->CSSS[k][l]=0.0;
    }
    //output->CS=0.0 ; output->CSS=0.0 ; output->CSSS=0.0;
    //memset(output->VIBFRAC,0.e00,sizeof(***output->VIBFRAC));//output->VIBFRAC=0.e00;
    //memset(output->SUMVIB,0.e00,sizeof(**output->SUMVIB));//output->SUMVIB=0.e00;
    for(int j=0;j<gas->MSP+1;j++){
        for(int k=0;k<gas->MMVM+1;k++){
            for(int l=0;l<151;l++)
                output->VIBFRAC[j][k][l]=0.0;
        }
    }
    for(int k=0;k<gas->MSP+1;k++){
        for(int l=0;l<gas->MMVM+1;l++)
            output->SUMVIB[k][l]=0.0;
    }
    
}
////
//
void SET_INITIAL_STATE_1D()
{
    //set the initial state of a homogeneous or one-dimensional flow
    //
    //MOLECS molecs;
    //GEOM_1D geom;
    //GAS gas;
    //CALC calc;
    //OUTPUT output;
    //
    //
    int J,L,K,KK,KN,II,III,INC,NSET,NSC;
    long long N,M;
    double A,B,AA,BB,BBB,SN,XMIN,XMAX,WFMIN,DENG,ELTI,EA,XPREV;
    double DMOM[4];
    double VB[4][3];
    double ROTE[3];
    //
    //NSET the alternative set numbers in the setting of exact initial state
    //DMOM(N) N=1,2,3 for x,y and z momentum sums of initial molecules
    //DENG the energy sum of the initial molecules
    //VB alternative sets of velocity components
    //ROTE alternative sets of rotational energy
    //EA entry area
    //INC counting increment
    //ELTI  initial electronic temperature
    //XPREV the pevious x coordinate
    //
    //memset(DMOM,0.e00,sizeof(DMOM));
    for(int i=0;i<4;i++)
        DMOM[i]=0.e00;
    DENG=0.e00;
    //set the number of molecules, divisions etc. based on stream 1
    //
    calc->NMI=10000*calc->IMEG+2;    //small changes in number for statistically independent runs
    geom->NDIV=calc->NMI/calc->MOLSC; //MOLSC molecules per division
    //WRITE (9,*) 'The number of divisions is',NDIV
    file_9<< "The number of divisions is "<<geom->NDIV<<endl;
    //
    geom->MDIV=geom->NDIV;
    geom->ILEVEL=0;
    //
    geom->i_allocate(geom->ILEVEL+1,geom->MDIV+1,geom->JDIV);
    // ALLOCATE (JDIV(0:ILEVEL,MDIV),STAT=ERROR)
    // IF (ERROR /= 0) THEN
    //   WRITE (*,*) 'PROGRAM COULD NOT ALLOCATE SPACE FOR JDIV ARRAY',ERROR
    // ENDIF
    //
    geom->DDIV=(geom->XB[2]-geom->XB[1])/double(geom->NDIV);
    geom->NCELLS=geom->NDIV;
    
    //WRITE (9,*) 'The number of sampling cells is',NCELLS
    file_9<<"The number of sampling cells is "<< geom->NCELLS<<endl;
    geom->NCIS=calc->MOLSC/calc->NMCC;
    geom->NCCELLS=geom->NCIS*geom->NDIV;
    //WRITE (9,*) 'The number of collision cells is',NCCELLS
    file_9<< "The number of collision cells is "<<geom->NCCELLS<<endl;
    //
    if(geom->IFX == 0) geom->XS=0.e00;
    //
    if(calc->ISECS == 0){
        if(geom->IFX == 0) calc->FNUM=((geom->XB[2]-geom->XB[1])*gas->FND[1])/double(calc->NMI);
        if(geom->IFX == 1) calc->FNUM=PI*(pow(geom->XB[2],2)-pow(geom->XB[1],2))*gas->FND[1]/double(calc->NMI);
        if(geom->IFX == 2) calc->FNUM=1.3333333333333333333333e00*PI*(pow(geom->XB[2],3)-pow(geom->XB[1],3))*gas->FND[1]/double(calc->NMI);
    }
    else{
        if(geom->IFX == 0) calc->FNUM=((geom->XS-geom->XB[1])*gas->FND[1]+(geom->XB[2]-geom->XS)*gas->FND[2])/double(calc->NMI);
        if(geom->IFX == 1) calc->FNUM=PI*((pow(geom->XS,2)-pow(geom->XB[1],2))*gas->FND[1]+(pow(geom->XB[2],2)-pow(geom->XS,2))*gas->FND[2])/double(calc->NMI);
        if(geom->IFX == 2) calc->FNUM=1.3333333333333333333333e00*PI*((pow(geom->XS,3)-pow(geom->XB[1],3))*gas->FND[1]+(pow(geom->XB[2],3)-pow(geom->XS,3))*gas->FND[2])/double(calc->NMI);
    }
    //
    calc->FNUM=calc->FNUM*calc->FNUMF;
    if(calc->FNUM < 1.e00) calc->FNUM=1.e00;
    //
    calc->FTIME=0.e00;
    //
    calc->TOTMOV=0.e00;
    calc->TOTCOL=0.e00;
    
    output->NDISSOC=0;
    //memset(calc->TCOL,0.e00,sizeof(**calc->TCOL));//calc->TCOL=0.e00;
    for(int i=0;i<gas->MSP+1;i++){
        for(int j=0;j<gas->MSP+1;j++){
            calc->TCOL[i][j]=0.e00;
        }
    }
    
    //memset(calc->TDISS,0.e00,sizeof(*calc->TDISS));//calc->TDISS=0.e00;
    //memset(calc->TRECOMB,0.e00,sizeof(*calc->TRECOMB));//calc->TRECOMB=0.e00;
    for(int i=0;i<gas->MSP+1;i++)
        calc->TDISS[i]=0.e00;
    for(int i=0;i<gas->MSP+1;i++)
        calc->TRECOMB[i]=0.e00;
    //gas->TREACG=0;
    //gas->TREACL=0;
    for(int i=0;i<5;i++){
        for(int j=0;j<gas->MSP+1;j++){
            gas->TREACG[i][j]=0;
        }
    }
    for(int i=0;i<5;i++){
        for(int j=0;j<gas->MSP+1;j++){
            gas->TREACL[i][j]=0;
        }
    }
    //memset(gas->TNEX,0.e00,sizeof(*gas->TNEX));//gas->TNEX=0.e00;
    for(int i=0;i<gas->MEX+1;i++)
        gas->TNEX[i]= 0.e00;
    for(N=1;N<=geom->NDIV;N++){
        geom->JDIV[0][N]=-N;
    }
    
    //
    geom->d_allocate(5,geom->NCELLS+1,geom->CELL);
    geom->i_allocate(geom->NCELLS+1,geom->ICELL);
    geom->d_allocate(6,geom->NCCELLS+1,geom->CCELL);
    geom->i_allocate(4,geom->NCCELLS+1,geom->ICCELL);
    calc->d_allocate(geom->NCCELLS+1,calc->COLL_TOTCOL);
    // ALLOCATE (CELL(4,NCELLS),ICELL(NCELLS),CCELL(5,NCCELLS),ICCELL(3,NCCELLS),STAT=ERROR)
    // IF (ERROR /= 0) THEN
    //   WRITE (*,*) 'PROGRAM COULD NOT ALLOCATE SPACE FOR CELL ARRAYS',ERROR
    // ENDIF
    //
    output->d_allocate(geom->NCELLS+1,output->COLLS);
    output->d_allocate(geom->NCELLS+1,output->WCOLLS);
    output->d_allocate(geom->NCELLS+1,output->CLSEP);
    output->d_allocate(gas->MNSR+1,output->SREAC);
    output->d_allocate(24,geom->NCELLS+1,output->VAR);
    output->d_allocate(13,geom->NCELLS+1,gas->MSP+1,output->VARSP);
    output->d_allocate(36+gas->MSP,3,output->VARS);
    output->d_allocate(10+gas->MSP,geom->NCELLS+1,gas->MSP+1,output->CS);
    output->d_allocate(9,3,gas->MSP+1,3,output->CSS);
    output->d_allocate(7,3,output->CSSS);
    
    // ALLOCATE (COLLS(NCELLS),WCOLLS(NCELLS),CLSEP(NCELLS),SREAC(MNSR),VAR(23,NCELLS),VARSP(0:12,NCELLS,MSP),    &
    //           VARS(0:35+MSP,2),CS(0:9+MSP,NCELLS,MSP),CSS(0:8,2,MSP,2),CSSS(6,2),STAT=ERROR)
    // IF (ERROR /= 0) THEN
    //   WRITE (*,*) 'PROGRAM COULD NOT ALLOCATE SPACE FOR SAMPLING ARRAYS',ERROR
    // ENDIF
    //
    if(gas->MMVM >= 0){
        
        output->d_allocate(gas->MSP+1,gas->MMVM+1,151,output->VIBFRAC);
        output->d_allocate(gas->MSP+1,gas->MMVM+1,output->SUMVIB);
        // ALLOCATE (VIBFRAC(MSP,MMVM,0:150),SUMVIB(MSP,MMVM),STAT=ERROR)
        // IF (ERROR /= 0) THEN
        //   WRITE (*,*) 'PROGRAM COULD NOT ALLOCATE SPACE FOR RECOMBINATION ARRAYS',ERROR
        // ENDIF
    }
    //
    INITIALISE_SAMPLES();
    //
    //Set the initial cells
    
    for(N=1;N<=geom->NCELLS;N++){
        geom->CELL[2][N]=geom->XB[1]+double(N-1)*geom->DDIV;
        geom->CELL[3][N]=geom->CELL[2][N]+geom->DDIV;
        geom->CELL[1][N]=geom->CELL[2][N]+0.5e00*geom->DDIV;
        if(geom->IFX == 0) geom->CELL[4][N]=geom->CELL[3][N]-geom->CELL[2][N];    //calculation assumes unit cross-section
        if(geom->IFX == 1) geom->CELL[4][N]=PI*(pow(geom->CELL[3][N],2)-pow(geom->CELL[2][N],2));  //assumes unit length of full cylinder
        if(geom->IFX == 2) geom->CELL[4][N]=1.33333333333333333333e00*PI*(pow(geom->CELL[3][N],3)-pow(geom->CELL[2][N],3));    //flow is in the full sphere
        geom->ICELL[N]=geom->NCIS*(N-1);
        for(M=1;M<=geom->NCIS;M++){
            L=geom->ICELL[N]+M;
            XMIN=geom->CELL[2][N]+double(M-1)*geom->DDIV/double(geom->NCIS);
            XMAX=XMIN+geom->DDIV/double(geom->NCIS);
            if(geom->IFX == 0) geom->CCELL[1][L]=XMAX-XMIN;
            if(geom->IFX == 1) geom->CCELL[1][L]=PI*(pow(XMAX,2)-pow(XMIN,2));  //assumes unit length of full cylinder
            if(geom->IFX == 2) geom->CCELL[1][L]=1.33333333333333333333e00*PI*(pow(XMAX,3)-pow(XMIN,3));    //flow is in the full sphere
            geom->CCELL[2][L]=0.e00;
            geom->ICCELL[3][L]=N;
        }
        output->VAR[11][N]=gas->FTMP[1];
        output->VAR[8][N]=gas->FTMP[1];
    }
    //
    if(geom->IWF == 0) geom->AWF=1.e00;
    if(geom->IWF == 1){
        //FNUM must be reduced to allow for the weighting factors
        A=0.e00;
        B=0.e00;
        for(N=1;N<=geom->NCELLS;N++){
            A=A+geom->CELL[4][N];
            B=B+geom->CELL[4][N]/(1.0+geom->WFM*pow(geom->CELL[1][N],geom->IFX));
        }
        geom->AWF=A/B;
        calc->FNUM=calc->FNUM*B/A;
    }
    //
    //WRITE (9,*) 'FNUM is',FNUM
    file_9<<"FNUM is "<<calc->FNUM<<endl;
    //
    //set the information on the molecular species
    //
    A=0.e00;
    B=0.e00;
    for(L=1;L<=gas->MSP;L++){
        A=A+gas->SP[5][L]*gas->FSP[L][1];
        B=B+(3.0+gas->ISPR[1][L])*gas->FSP[L][1];
        gas->VMP[L][1]=sqrt(2.e00*BOLTZ*gas->FTMP[1]/gas->SP[5][L]);
        if((geom->ITYPE[2]== 0) || (calc->ISECS == 1)) gas->VMP[L][2]=sqrt(2.e00*BOLTZ*gas->FTMP[2]/gas->SP[5][L]);
        calc->VNMAX[L]=3.0*gas->VMP[L][1];
        if(L == 1)
            gas->VMPM=gas->VMP[L][1];
        else
            if(gas->VMP[L][1] > gas->VMPM) gas->VMPM=gas->VMP[L][1];
    }
    //WRITE (9,*) 'VMPM =',VMPM
    file_9<< "VMPM = "<<gas->VMPM<<endl;
    gas->FDEN=A*gas->FND[1];
    gas->FPR=gas->FND[1]*BOLTZ*gas->FTMP[1];
    gas->FMA=gas->VFX[1]/sqrt((B/(B+2.e00))*BOLTZ*gas->FTMP[1]/A);
    //set the molecular properties for collisions between unlike molecles
    //to the average of the molecules
    for(L=1;L<=gas->MSP;L++){
        for(M=1;M<=gas->MSP;M++){
            gas->SPM[4][L][M]=0.5e00*(gas->SP[1][L]+gas->SP[1][M]);
            gas->SPM[3][L][M]=0.5e00*(gas->SP[3][L]+gas->SP[3][M]);
            gas->SPM[5][L][M]=0.5e00*(gas->SP[2][L]+gas->SP[2][M]);
            gas->SPM[1][L][M]=gas->SP[5][L]*(gas->SP[5][M]/(gas->SP[5][L]+gas->SP[5][M]));
            gas->SPM[2][L][M]=0.25e00*PI*pow((gas->SP[1][L]+gas->SP[1][M]),2);
            AA=2.5e00-gas->SPM[3][L][M];
            A=tgamma(AA);
            gas->SPM[6][L][M]=1.e00/A;
            gas->SPM[8][L][M]=0.5e00*(gas->SP[4][L]+gas->SP[4][M]);
            if((gas->ISPR[1][L] > 0) && (gas->ISPR[1][M] > 0))
                gas->SPM[7][L][M]=(gas->SPR[1][L]+gas->SPR[1][M])*0.5e00;
            if((gas->ISPR[1][L] > 0) && (gas->ISPR[1][M] == 0))
                gas->SPM[7][L][M]=gas->SPR[1][L];
            if((gas->ISPR[1][M] > 0) && (gas->ISPR[1][L] == 0))
                gas->SPM[7][L][M]=gas->SPR[1][M];
        }
    }
    if(gas->MSP == 1){   //set unscripted variables for the simple gas case
        gas->RMAS=gas->SPM[1][1][1];
        gas->CXSS=gas->SPM[2][1][1];
        gas->RGFS=gas->SPM[6][1][1];
    }
    //
    for(L=1;L<=gas->MSP;L++){
        gas->CR[L]=0.e00;
        for(M=1;M<=gas->MSP;M++){   //set the equilibrium collision rates
            gas->CR[L]=gas->CR[L]+2.e00*SPI*pow(gas->SPM[4][L][M],2)*gas->FND[1]*gas->FSP[M][1]*pow((gas->FTMP[1]/gas->SPM[5][L][M]),(1.0-gas->SPM[3][L][M]))*sqrt(2.0*BOLTZ*gas->SPM[5][L][M]/gas->SPM[1][L][M]);
        }
    }
    A=0.e00;
    for(L=1;L<=gas->MSP;L++)
        A=A+gas->FSP[L][1]*gas->CR[L];
    gas->CTM=1.e00/A;
    //WRITE (9,*) 'Collision time in the stream is',CTM
    file_9<< "Collision time in the stream is "<<gas->CTM;
    //
    for(L=1;L<=gas->MSP;L++){
        gas->FP[L]=0.e00;
        for(M=1;M<=gas->MSP;M++){
            gas->FP[L]=gas->FP[L]+PI*pow(gas->SPM[4][L][M],2)*gas->FND[1]*gas->FSP[M][1]*pow((gas->FTMP[1]/gas->SPM[5][L][M]),(1.0-gas->SPM[3][L][M]))*sqrt(1.e00+gas->SP[5][L]/gas->SP[5][M]);
        }
        gas->FP[L]=1.e00/gas->FP[L];
    }
    gas->FPM=0.e00;
    for(L=1;L<=gas->MSP;L++)
        gas->FPM=gas->FPM+gas->FSP[L][1]*gas->FP[L];
    //WRITE (9,*) 'Mean free path in the stream is',FPM
    file_9<<"Mean free path in the stream is "<<gas->FPM<<endl;
    //
    calc->TNORM=gas->CTM;
    if(calc->ICLASS == 1) calc->TNORM= (geom->XB[2]-geom->XB[1])/gas->VMPM;     //there may be alternative definitions
    //
    //set the initial time step
    calc->DTM=gas->CTM*calc->CPDTM;
    //
    if(fabs(gas->VFX[1]) > 1.e-6)
        A=(0.5e00*geom->DDIV/gas->VFX[1])*calc->TPDTM;
    else
        A=0.5e00*geom->DDIV/gas->VMPM;
    
    if(geom->IVB == 1){
        B=0.25e00*geom->DDIV/(fabs(geom->VELOB)+gas->VMPM);
        if(B < A) A=B;
    }
    if(calc->DTM > A) calc->DTM=A;
    //
    calc->DTM=0.1e00*calc->DTM;   //OPTIONAL MANUAL ADJUSTMENT that is generally used with a fixed time step (e.g for making x-t diagram)
    //
    calc->DTSAMP=calc->SAMPRAT*calc->DTM;
    calc->DTOUT=calc->OUTRAT*calc->DTSAMP;
    calc->TSAMP=calc->DTSAMP;
    calc->TOUT=calc->DTOUT;
    calc->ENTMASS=0.0;
    //
    //WRITE (9,*) 'The initial value of the overall time step is',DTM
    file_9<< "The initial value of the overall time step is "<<calc->DTM<<endl;
    //
    //initialise cell quantities associated with collisions
    //
    for(N=1;N<=geom->NCCELLS;N++){
        geom->CCELL[3][N]=calc->DTM/2.e00;
        geom->CCELL[4][N]=2.e00*gas->VMPM*gas->SPM[2][1][1];
        calc->RANF=((double)rand()/(double)RAND_MAX);
        // RANDOM_NUMBER(RANF)
        geom->CCELL[2][N]=calc->RANF;
        geom->CCELL[5][N]=0.e00;
    }
    //
    //set the entry quantities
    //
    for(K=1;K<=2;K++){
        if((geom->ITYPE[K] == 0) || ((K == 2) && (geom->ITYPE[K] == 4))){
            if(geom->IFX == 0) EA=1.e00;
            if(geom->IFX == 1) EA=2.e00*PI*geom->XB[K];
            if(geom->IFX == 2) EA=4.e00*PI*pow(geom->XB[K],2);
            for(L=1;L<=gas->MSP;L++){
                if(K == 1) SN=gas->VFX[1]/gas->VMP[L][1];
                if(K == 2) SN=-gas->VFX[2]/gas->VMP[L][2];
                AA=SN;
                A=1.e00+erf(AA);
                BB=exp(-pow(SN,2));
                gas->ENTR[3][L][K]=SN;
                gas->ENTR[4][L][K]=SN+sqrt(pow(SN,2)+2.e00);
                gas->ENTR[5][L][K]=0.5e00*(1.e00+SN*(2.e00*SN-gas->ENTR[4][L][K]));
                gas->ENTR[6][L][K]=3.e00*gas->VMP[L][K];
                B=BB+SPI*SN*A;
                gas->ENTR[1][L][K]=EA*gas->FND[K]*gas->FSP[L][K]*gas->VMP[L][K]*B/(calc->FNUM*2.e00*SPI);
                gas->ENTR[2][L][K]=0.e00;
            }
        }
    }
    //
    //Set the uniform stream
    //
    molecs->MNM=1.1e00*calc->NMI;
    //
    if(gas->MMVM > 0){
        molecs->d_allocate(calc->NCLASS+1,molecs->MNM+1,molecs->PX);
        molecs->d_allocate(molecs->MNM+1,molecs->PTIM);
        molecs->d_allocate(molecs->MNM+1,molecs->PROT);
        molecs->i_allocate(molecs->MNM+1,molecs->IPCELL);
        molecs->i_allocate(molecs->MNM+1,molecs->IPSP);
        molecs->i_allocate(molecs->MNM+1,molecs->ICREF);
        molecs->i_allocate(molecs->MNM+1,molecs->IPCP);
        molecs->d_allocate(4,molecs->MNM+1,molecs->PV);
        molecs->i_allocate(gas->MMVM+1,molecs->MNM+1,molecs->IPVIB);
        molecs->d_allocate(molecs->MNM+1,molecs->PELE);
        // ALLOCATE (PX(NCLASS,MNM),PTIM(MNM),PROT(MNM),IPCELL(MNM),IPSP(MNM),ICREF(MNM),IPCP(MNM),PV(3,MNM),     &
        //      IPVIB(MMVM,MNM),PELE(MNM),STAT=ERROR)
    }
    
    else{
        if(gas->MMRM > 0){
            molecs->d_allocate(calc->NCLASS+1,molecs->MNM+1,molecs->PX);
            molecs->d_allocate(molecs->MNM+1,molecs->PTIM);
            molecs->d_allocate(molecs->MNM+1,molecs->PROT);
            molecs->i_allocate(molecs->MNM+1,molecs->IPCELL);
            molecs->i_allocate(molecs->MNM+1,molecs->IPSP);
            molecs->i_allocate(molecs->MNM+1,molecs->ICREF);
            molecs->i_allocate(molecs->MNM+1,molecs->IPCP);
            molecs->d_allocate(4,molecs->MNM+1,molecs->PV);
            molecs->d_allocate(molecs->MNM+1,molecs->PELE);
            // ALLOCATE (PX(NCLASS,MNM),PTIM(MNM),PROT(MNM),IPCELL(MNM),IPSP(MNM),ICREF(MNM),IPCP(MNM),PV(3,MNM),PELE(MNM),STAT=ERROR)
        }
        else{
            molecs->d_allocate(calc->NCLASS+1,molecs->MNM+1,molecs->PX);
            molecs->d_allocate(molecs->MNM+1,molecs->PTIM);
            molecs->i_allocate(molecs->MNM+1,molecs->IPCELL);
            molecs->i_allocate(molecs->MNM+1,molecs->IPSP);
            molecs->i_allocate(molecs->MNM+1,molecs->ICREF);
            molecs->i_allocate(molecs->MNM+1,molecs->IPCP);
            molecs->d_allocate(4,molecs->MNM+1,molecs->PV);
            molecs->d_allocate(molecs->MNM+1,molecs->PELE);
            // ALLOCATE (PX(NCLASS,MNM),PTIM(MNM),IPCELL(MNM),IPSP(MNM),ICREF(MNM),IPCP(MNM),PV(3,MNM),PELE(MNM),STAT=ERROR)
        }
    }
    // IF (ERROR /= 0) THEN
    //   WRITE (*,*) 'PROGRAM COULD NOT ALLOCATE SPACE FOR MOLECULE ARRAYS',ERROR
    // ENDIF
    //
    molecs->NM=0;
    if(calc->IGS == 1){
        cout<<"Setting the initial gas"<<endl;
        for(L=1;L<=gas->MSP;L++){
            //memset(ROTE,0.0,sizeof(ROTE));
            for(int i=0;i<3;i++)
                ROTE[i]=0.0;
            for(K=1;K<=calc->ISECS+1;K++){
                if(calc->ISECS == 0){         //no secondary stream
                    M=(double(calc->NMI)*gas->FSP[L][1]*geom->AWF);
                    XMIN=geom->XB[1];
                    XMAX=geom->XB[2];
                }
                else{
                    A=(pow(geom->XS,geom->JFX)-pow(geom->XB[1],geom->JFX))*gas->FND[1]+(pow(geom->XB[2],geom->JFX)-pow(geom->XS,geom->JFX))*gas->FND[2];
                    if(K == 1){
                        M=int(double(calc->NMI)*((pow(geom->XS,geom->JFX)-pow(geom->XB[1],geom->JFX))*gas->FND[1]/A)*gas->FSP[L][1]);
                        XMIN=geom->XB[1];
                        XMAX=geom->XS;
                    }
                    else{
                        M=int(double(calc->NMI)*((pow(geom->XB[2],geom->JFX)-pow(geom->XS,geom->JFX))*gas->FND[2]/A)*gas->FSP[L][2]);
                        XMIN=geom->XS;
                        XMAX=geom->XB[2];
                    }
                }
                if((K == 1) || (calc->ISECS == 1)){
                    III=0;
                    WFMIN=1.e00+geom->WFM*pow(geom->XB[1],geom->IFX);
                    N=1;
                    INC=1;
                    if((K== 2) && (geom->JFX > 1)){
                        BBB=(pow(XMAX,geom->JFX)-pow(XMIN,geom->JFX))/double(M);
                        XPREV=XMIN;
                    }
                    while(N < M){
                        if((geom->JFX == 1) || (K == 1))
                            A=pow((pow(XMIN,geom->JFX)+((double(N)-0.5e00)/double(M))*pow((XMAX-XMIN),geom->JFX)),(1.e00/double(geom->JFX)));
                        else{
                            A=pow((pow(XPREV,geom->JFX)+BBB),(1.e00/double(geom->JFX)));
                            XPREV=A;
                        }
                        if(geom->IWF == 0)
                            B=1.e00;
                        else{
                            B=WFMIN/(1.e00+geom->WFM*pow(A,geom->IFX));
                            if((B < 0.1e00) && (INC == 1)) INC=10;
                            if((B < 0.01e00) && (INC == 10)) INC=100;
                            if((B < 0.001e00) && (INC == 100)) INC=1000;
                            if((B < 0.0001e00) && (INC == 1000)) INC=10000;
                        }
                        calc->RANF=((double)rand()/(double)RAND_MAX);
                        // CALL RANDOM_NUMBER(RANF)
                        if(B*double(INC) > calc->RANF){
                            molecs->NM=molecs->NM+1;
                            molecs->PX[1][molecs->NM]=A;
                            molecs->IPSP[molecs->NM]=L;
                            molecs->PTIM[molecs->NM]=0.0;
                            if(geom->IVB == 0) FIND_CELL_1D(molecs->PX[1][molecs->NM],molecs->IPCELL[molecs->NM],KK);
                            if(geom->IVB == 1) FIND_CELL_MB_1D(molecs->PX[1][molecs->NM],molecs->IPCELL[molecs->NM],KK,molecs->PTIM[molecs->NM]);
                            //
                            for(NSET=1;NSET<=2;NSET++){
                                for(KK=1;KK<=3;KK++){
                                    RVELC(A,B,gas->VMP[L][K]);
                                    if(A < B){
                                        if(DMOM[KK] < 0.e00)
                                            BB=B;
                                        else
                                            BB=A;
                                    }           
                                    else{
                                        if(DMOM[KK] < 0.e00)
                                            BB=A;
                                        else
                                            BB=B;
                                    }
                                    VB[KK][NSET]=BB;
                                }
                                if(gas->ISPR[1][L] > 0) SROT(L,gas->FTMP[K],ROTE[NSET]);
                            }
                            A=(0.5e00*gas->SP[5][L]*(pow(VB[1][1],2)+pow(VB[2][1],2)+pow(VB[3][1],2))+ROTE[1])/(0.5e00*BOLTZ*gas->FTMP[K])-3.e00-double(gas->ISPR[1][L]);
                            B=(0.5e00*gas->SP[5][L]*(pow(VB[1][2],2)+pow(VB[2][2],2)+pow(VB[3][2],2))+ROTE[2])/(0.5e00*BOLTZ*gas->FTMP[K])-3.e00-double(gas->ISPR[1][L]);
                            if(A < B){
                                if(DENG < 0.e00)
                                    KN=2;
                                else
                                    KN=1;
                            }
                            else{
                                if(DENG < 0.e00)
                                    KN=1;
                                else
                                    KN=2;
                            }
                            
                            for(KK=1;KK<=3;KK++){
                                molecs->PV[KK][molecs->NM]=VB[KK][KN];
                                DMOM[KK]=DMOM[KK]+VB[KK][KN];
                            }
                            molecs->PV[1][molecs->NM]=molecs->PV[1][molecs->NM]+gas->VFX[K];
                            molecs->PV[2][molecs->NM]=molecs->PV[2][molecs->NM]+gas->VFY[K];
                            if(gas->ISPR[1][L] > 0) molecs->PROT[molecs->NM]=ROTE[KN];
                            //           PROT(NM)=0.d00       //uncomment for zero initial rotational temperature (Figs. 6.1 and 6.2)
                            if(KN == 1) DENG=DENG+A;
                            if(KN == 2) DENG=DENG+B;
                            if(gas->MMVM > 0){
                                if(gas->ISPV[L] > 0){
                                    for(J=1;J<=gas->ISPV[L];J++)
                                        SVIB(L,gas->FVTMP[K],molecs->IPVIB[J][molecs->NM],J);
                                }
                                ELTI=gas->FVTMP[K];
                                if(gas->MELE > 1) SELE(L,ELTI,molecs->PELE[molecs->NM]);
                            }
                        }
                        N=N+INC;
                    }
                }
            }
        }
        //
        //WRITE (9,*) 'DMOM',DMOM
        //WRITE (9,*) 'DENG',DENG
        file_9<<"DMOM "<<DMOM<<endl;
        file_9<<"DENG "<<DENG<<endl;
    }
    //
    calc->NMI=molecs->NM;
    //
    
    //SPECIAL CODING FOR INITIATION OF COMBUSION IN H2-02 MIXTURE (FORCED IGNITION CASES in section 6.7)
    //set the vibrational levels of A% random molecules to 5
    //  A=0.05D00
    //  M=0.01D00*A*NM
    //  DO N=1,M
    //    CALL RANDOM_NUMBER(RANF)
    //    K=INT(RANF*DFLOAT(NM))+1
    //    IPVIB(1,K)=5
    //  END DO
    //
    SAMPLE_FLOW();
    OUTPUT_RESULTS();
    calc->TOUT=calc->TOUT-calc->DTOUT;
    return;
}

void MOLECULES_ENTER_1D()
{
    //molecules enter boundary at XB(1) and XB(2) and may be removed behind a wave
    //MOLECS molecs;
    //GAS gas;
    //CALC calc;
    //GEOM_1D geom;
    //OUTPUT output;
    //
    int K,L,M,N,NENT,II,J,JJ,KK,NTRY;
    double A,B,AA,BB,U,VN,XI,X,DX,DY,DZ;
    //
    //NENT number to enter in the time step
    //
    calc->ENTMASS=0.e00;
    //
    for(J=1;J<=2;J++){     //J is the end
        if((geom->ITYPE[J] == 0) || (geom->ITYPE[J] == 4)){
            KK=1;//the entry surface will normally use the reference gas (main stream) properties
            if((J == 2) && (calc->ISECS == 1) && (geom->XB[2] > 0.e00)) KK=2;    //KK is 1 for reference gas 2 for the secondary stream
            for(L=1;L<=gas->MSP;L++){
                A=gas->ENTR[1][L][J]*calc->DTM+gas->ENTR[2][L][J];
                if((geom->ITYPE[2] == 4) && (calc->ICN == 1)){
                    NENT=A;
                    if(J == 1) calc->EME[L]=NENT;
                    if(J == 2) {
                        A=calc->ALOSS[L]-calc->EME[L]-calc->AJM[L];
                        calc->AJM[L]=0.e00;
                        if(A < 0.e00){
                            calc->AJM[L]=-A;
                            A=0.e00;
                        }
                    }
                }
                NENT=A;
                gas->ENTR[2][L][J]=A-NENT;
                if((geom->ITYPE[2] == 4) && (J == 2) && (calc->ICN == 1)) gas->ENTR[2][L][J]=0.e00;
                if(NENT > 0){
                    for(M=1;M<=NENT;M++){
                        if(molecs->NM >= molecs->MNM){
                            cout<< "EXTEND_MNM from MOLECULES_ENTER "<<endl;
                            EXTEND_MNM(1.1);
                        }
                        molecs->NM=molecs->NM+1;
                        AA=max(0.e00,gas->ENTR[3][L][J]-3.e00);
                        BB=max(3.e00,gas->ENTR[3][L][J]+3.e00);
                        II=0;
                        while(II == 0){
                            calc->RANF=((double)rand()/(double)RAND_MAX);
                            // CALL RANDOM_NUMBER(RANF)
                            B=AA+(BB-AA)*calc->RANF;
                            U=B-gas->ENTR[3][L][J];
                            A=(2.e00*B/gas->ENTR[4][L][J])*exp(gas->ENTR[5][L][J]-U*U);
                            calc->RANF=((double)rand()/(double)RAND_MAX);
                            // CALL RANDOM_NUMBER(RANF)
                            if(A > calc->RANF) II=1;
                        }
                        molecs->PV[1][molecs->NM]=B*gas->VMP[L][KK];
                        if(J == 2) molecs->PV[1][molecs->NM]=-molecs->PV[1][molecs->NM];
                        //
                        RVELC(molecs->PV[2][molecs->NM],molecs->PV[3][molecs->NM],gas->VMP[L][KK]);
                        molecs->PV[2][molecs->NM]=molecs->PV[2][molecs->NM]+gas->VFY[J];
                        //
                        if(gas->ISPR[1][L] > 0) SROT(L,gas->FTMP[KK],molecs->PROT[molecs->NM]);
                        //
                        if(gas->MMVM > 0){
                            for(K=1;K<=gas->ISPV[L];K++)
                                SVIB(L,gas->FVTMP[KK],molecs->IPVIB[K][molecs->NM],K);
                        }
                        if(gas->MELE > 1) SELE(L,gas->FTMP[KK],molecs->PELE[molecs->NM]);
                        //
                        if(molecs->PELE[molecs->NM] > 0.e00)
                            continue;                     //DEBUG
                        //
                        molecs->IPSP[molecs->NM]=L;
                        //advance the molecule into the flow
                        calc->RANF=((double)rand()/(double)RAND_MAX);
                        // CALL RANDOM_NUMBER(RANF)
                        XI=geom->XB[J];
                        DX=calc->DTM*calc->RANF*molecs->PV[1][molecs->NM];
                        if((geom->IFX == 0) || (J == 2)) X=XI+DX;
                        if(J == 1){   //1-D move at outer boundary so molecule remains in flow
                            if(geom->IFX > 0) DY=calc->DTM*calc->RANF*molecs->PV[2][molecs->NM];
                            DZ=0.e00;
                            if(geom->IFX == 2) DZ=calc->DTM*calc->RANF*molecs->PV[3][molecs->NM];
                            if(geom->IFX > 0) AIFX(XI,DX,DY,DZ,X,molecs->PV[1][molecs->NM],molecs->PV[2][molecs->NM],molecs->PV[3][molecs->NM]);
                        }
                        molecs->PX[calc->NCLASS][molecs->NM]=X;
                        molecs->PTIM[molecs->NM]=calc->FTIME;
                        if(geom->IVB == 0) FIND_CELL_1D(molecs->PX[calc->NCLASS][molecs->NM],molecs->IPCELL[molecs->NM],JJ);
                        if(geom->IVB == 1) FIND_CELL_MB_1D(molecs->PX[calc->NCLASS][molecs->NM],molecs->IPCELL[molecs->NM],JJ,molecs->PTIM[molecs->NM]);
                        molecs->IPCP[molecs->NM]=0;
                        if(geom->XREM > geom->XB[1]) calc->ENTMASS=calc->ENTMASS+gas->SP[5][L];
                    }
                }
            }
            if((geom->ITYPE[2] == 4) && (J==2) && (molecs->NM != calc->NMP) && (calc->ICN == 1))
                continue;
        }
    }
    //
    //stagnation streamline molecule removal
    if(geom->XREM > geom->XB[1]){
        calc->ENTMASS=geom->FREM*calc->ENTMASS;
        NTRY=0;
        calc->ENTMASS=calc->ENTMASS+calc->ENTREM;
        while((calc->ENTMASS > 0.e00) && (NTRY < 10000)){
            NTRY=NTRY+1;
            if(NTRY == 10000){
                cout<<"Unable to find molecule for removal"<<endl;
                calc->ENTMASS=0.e00;
                //memset(calc->VNMAX,0.e00,sizeof(*calc->VNMAX));//calc->VNMAX=0.e00;
                for(int i=0;i<gas->MSP+1;i++)
                    calc->VNMAX[i]=0.e00;
            }
            calc->RANF=((double)rand()/(double)RAND_MAX);
            // CALL RANDOM_NUMBER(RANF)
            N=molecs->NM*calc->RANF+0.9999999e00;
            if(molecs->PX[calc->NCLASS][N] > geom->XREM){
                // CALL RANDOM_NUMBER(RANF)
                calc->RANF=((double)rand()/(double)RAND_MAX);
                //IF (RANF < ((PX(N)-XREM)/(XB(2)-XREM))**2) THEN
                if(fabs(gas->VFY[1]) < 1.e-3)
                    VN=sqrt(molecs->PV[2][N]*molecs->PV[2][N]+molecs->PV[3][N]*molecs->PV[3][N]);   //AXIALLY SYMMETRIC STREAMLINE
                else
                    VN=fabs(molecs->PV[3][N]);   //TWO-DIMENSIONAL STREAMLINE
                
                L=molecs->IPSP[N];
                if(VN > calc->VNMAX[L]) calc->VNMAX[L]=VN;
                // CALL RANDOM_NUMBER(RANF)
                calc->RANF=((double)rand()/(double)RAND_MAX);
                if(calc->RANF < VN/calc->VNMAX[L]){
                    REMOVE_MOL(N);
                    calc->ENTMASS=calc->ENTMASS-gas->SP[5][L];
                    NTRY=0;
                }
                //END IF
            }
        }
        calc->ENTREM=calc->ENTMASS;
    }
}

void FIND_CELL_1D(double &X,int &NCC,int &NSC)
{
    //find the collision and sampling cells at a givem location in a 0D or 1D case
    //MOLECS molecs;
    //GEOM_1D geom;
    //CALC calc;
    
    int N,L,M,ND;
    double FRAC,DSC;
    //
    //NCC collision cell number
    //NSC sampling cell number
    //X location
    //ND division number
    //DSC the ratio of the sub-division width to the division width
    //
    ND=(X-geom->XB[1])/geom->DDIV+0.99999999999999e00;
    //
    if(geom->JDIV[0][ND] < 0){    //the division is a level 0 (no sub-division) sampling cell
        NSC=-geom->JDIV[0][ND];
        //  IF (IFX == 0)
        NCC=geom->NCIS*(X-geom->CELL[2][NSC])/(geom->CELL[3][NSC]-geom->CELL[2][NSC])+0.9999999999999999e00;
        NCC=NCC+geom->ICELL[NSC];
        //  IF (NCC == 0) NCC=1
        return;
    }
    else{  //the molecule is in a subdivided division
        FRAC=(X-geom->XB[1])/geom->DDIV-double(ND-1);
        M=ND;
        for(N=1;N<=geom->ILEVEL;N++){
            DSC=1.e00/double(N+1);
            for(L=1;L<=2;L++){  //over the two level 1 subdivisions
                if(((L == 1) && (FRAC < DSC)) || ((L == 2) || (FRAC >= DSC))){
                    M=geom->JDIV[N-1][M]+L;  //the address in JDIV
                    if(geom->JDIV[N][M] < 0){
                        NSC=-geom->JDIV[N][M];
                        NCC=geom->NCIS*(X-geom->CELL[2][NSC])/(geom->CELL[3][NSC]-geom->CELL[2][NSC])+0.999999999999999e00;
                        if(NCC == 0) NCC=1;
                        NCC=NCC+geom->ICELL[NSC];
                        return;
                    }
                }
            }
            FRAC=FRAC-DSC;
        }
    }
    file_9<<"No cell for molecule at x= "<<X<<endl;
    return ;
}

void FIND_CELL_MB_1D(double &X,int &NCC,int &NSC,double &TIM)
{
    //find the collision and sampling cells at a givem location in a 0D or 1D case
    //when there is a moving boundary
    //MOLECS molecs;
    //GEOM_1D geom;
    //CALC calc;
    //
    // IMPLICIT NONE
    //
    int N,L,M,ND;
    double FRAC,DSC,A,B,C;
    //
    //NCC collision cell number
    //NSC sampling cell number
    //X location
    //ND division number
    //DSC the ratio of the sub-division width to the division width
    //TIM the time
    //
    A=(geom->XB[2]+geom->VELOB*TIM-geom->XB[1])/double(geom->NDIV);      //new DDIV
    ND=(X-geom->XB[1])/A+0.99999999999999e00;
    B=geom->XB[1]+double(ND-1)*A;
    //
    //the division is a level 0 sampling cell
    NSC=-geom->JDIV[0][ND];
    NCC=geom->NCIS*(X-B)/A+0.99999999999999e00;
    NCC=NCC+geom->ICELL[NSC];
    
    //WRITE (9,*) 'No cell for molecule at x=',X
    file_9<< "No cell for molecule at x= "<<X<<endl;
    return;
    //return ;
    //
}

void RVELC(double &U,double &V,double &VMP)
{
    //CALC calc;
    //generates two random velocity components U and V in an equilibrium
    //gas with most probable speed VMP
    //based on equations (4.4) and (4.5)
    double A,B;
    //
    // CALL RANDOM_NUMBER(RANF)
    calc->RANF=((double)rand()/(double)RAND_MAX);
    A=sqrt(-log(calc->RANF));
    // CALL RANDOM_NUMBER(RANF)
    calc->RANF=((double)rand()/(double)RAND_MAX);
    B=DPI*calc->RANF;
    U=A*sin(B)*VMP;
    V=A*cos(B)*VMP;
    return;
}

void SROT(int &L,double &TEMP,double &ROTE)
{
    //sets a typical rotational energy ROTE of species L
    //CALC calc;
    //GAS gas;
    //
    // IMPLICIT NONE
    //
    int I;
    double A,B,ERM;
    //
    if(gas->ISPR[1][L] == 2){
        // CALL RANDOM_NUMBER(RANF)
        calc->RANF=((double)rand()/(double)RAND_MAX);
        ROTE=-log(calc->RANF)*BOLTZ*TEMP;   //equation (4.8)
    }
    else{
        A=0.5e00*gas->ISPR[1][L]-1.e00;
        I=0;
        while(I == 0){
            // CALL RANDOM_NUMBER(RANF)
            calc->RANF=((double)rand()/(double)RAND_MAX);
            ERM=calc->RANF*10.e00;
            //there is an energy cut-off at 10 kT
            B=(pow((ERM/A),A))*exp(A-ERM);      //equation (4.9)
            // CALL RANDOM_NUMBER(RANF)
            calc->RANF=((double)rand()/(double)RAND_MAX);
            if(B > calc->RANF) I=1;
        }
        ROTE=ERM*BOLTZ*TEMP;
    }
    return;
}

void SVIB(int &L,double &TEMP,int &IVIB, int &K)
{
    //sets a typical vibrational state at temp. TEMP of mode K of species L
    //GAS gas;
    //CALC calc;
    //
    // IMPLICIT NONE
    //
    int N;
    //    double TEMP;
    //    int IVIB;
    //
    // CALL RANDOM_NUMBER(RANF)
    calc->RANF=((double)rand()/(double)RAND_MAX);
    N=-log(calc->RANF)*TEMP/gas->SPVM[1][K][L];                 //eqn(4.10)
    //the state is truncated to an integer
    IVIB=N;
}

void SELE(int &L,double &TEMP, double &ELE)
{
    //sets a typical electronic energy at temp. TEMP of species L
    //employs direct sampling from the Boltzmann distribution
    //GAS gas;
    //CALC calc;
    //
    // IMPLICIT NONE
    //
    int K,N;
    double EPF,A,B;
    double CTP[20];
    //
    //ELE electronic energy of a molecule
    //EPF electronic partition function
    //CTP(N) contribution of electronic level N to the electronic partition function
    //
    if(TEMP > 0.1){
        EPF=0.e00;
        for(N=1;N<=gas->NELL[L];N++)
            EPF=EPF+gas->QELC[1][N][L]*exp(-gas->QELC[2][N][L]/TEMP) ;
        //
        // CALL RANDOM_NUMBER(RANF)
        calc->RANF=((double)rand()/(double)RAND_MAX);
        //
        A=0.0;
        K=0; //becomes 1 when the energy is set
        N=0;  //level
        while(K == 0){
            N=N+1;
            A=A+gas->QELC[1][N][L]*exp(-gas->QELC[2][N][L]/TEMP);
            B=A/EPF;
            if(calc->RANF < B){
                K=1;
                ELE=BOLTZ*gas->QELC[2][N][L];
            }
        }
    }
    else
        ELE=0.e00;
    
    //
}

void CQAX(double &A,double &X,double &GAX)
{
    //calculates the function Q(a,x)=Gamma(a,x)/Gamma(a)
    //
    // IMPLICIT NONE
    double G,DT,T,PV,V;
    int NSTEP,N;
    //
    G=tgamma(A);
    //
    if(X < 10.e00){       //direct integration
        NSTEP=100000;
        DT=X/double(NSTEP);
        GAX=0.e00;
        PV=0.e00;
        for(N=1;N<=NSTEP;N++){
            T=double(N)*DT;
            V=exp(-T)*pow(T,(A-1));
            GAX=GAX+(PV+V)*DT/2.e00;
            PV=V;
        }
        GAX=1.e00-GAX/G;
    }
    else{      //asymptotic formula
        GAX=pow(X,(A-1.e00))*exp(-X)*(1.0+(A-1.e00)/X+(A-1.e00)*(A-2.e00)/pow(X,2)+(A-1.e00)*(A-2.e00)*(A-3.e00)/pow(X,3)+(A-1.e00)*(A-2.e00)*(A-3.e00)*(A-4.e00)/pow(X,4));
        GAX=GAX/G;
    }
    //
    return;
}
//*****************************************************************************
//
void LBS(double XMA,double XMB,double &ERM)
{
    //selects a Larsen-Borgnakke energy ratio using eqn (11.9)
    //
    double PROB,RANF;
    int I,N;
    //
    //I is an indicator
    //PROB is a probability
    //ERM ratio of rotational to collision energy
    //XMA degrees of freedom under selection-1
    //XMB remaining degrees of freedom-1
    //
    I=0;
    while(I == 0){
        // CALL RANDOM_NUMBER(RANF)
        RANF=((double)rand()/(double)RAND_MAX);
        ERM=RANF;
        if((XMA < 1.e-6) || (XMB < 1.e-6)){
            //    IF (XMA < 1.E-6.AND.XMB < 1.E-6) RETURN
            //above can never occur if one mode is translational
            if(XMA < 1.e-6) PROB=pow((1.e00-ERM),XMB);
            if(XMB < 1.e-6) PROB=pow((1.e00-ERM),XMA);
        }
        else
            PROB=pow(((XMA+XMB)*ERM/XMA),XMA)*pow(((XMA+XMB)*(1.e00-ERM)/XMB),XMB);
        
        // CALL RANDOM_NUMBER(RANF)
        RANF=((double)rand()/(double)RAND_MAX);
        if(PROB > RANF) I=1;
    }
    //
    return;
}

void REFLECT_1D(int &N,int J,double &X)
{
    //reflects molecule N and samples the surface J properties
    //MOLECS molecs;
    //GAS gas;
    //GEOM_1D geom;
    //CALC calc;
    //OUTPUT output;
    //
    // IMPLICIT NONE
    //
    int L,K,M;
    double A,B,VMPS,DTR,XI,DX,DY,DZ,WF;
    //
    //VMPS most probable velocity at the surface temperature
    //DTR time remaining after molecule hits a surface
    //
    L=molecs->IPSP[N];
    WF=1.e00;
    if(geom->IWF == 1) WF=1.e00+geom->WFM*pow(X,geom->IFX);
    output->CSS[0][J][L][1]=output->CSS[0][J][L][1]+1.e00;
    output->CSS[1][J][L][1]=output->CSS[1][J][L][1]+WF;
    output->CSS[2][J][L][1]=output->CSS[2][J][L][1]+WF*molecs->PV[1][N]*gas->SP[5][L];
    output->CSS[3][J][L][1]=output->CSS[3][J][L][1]+WF*(molecs->PV[2][N]-gas->VSURF[J])*gas->SP[5][L];
    output->CSS[4][J][L][1]=output->CSS[4][J][L][1]+WF*molecs->PV[3][N]*gas->SP[5][L];
    A=pow(molecs->PV[1][N],2)+pow((molecs->PV[2][N]-gas->VSURF[J]),2)+pow(molecs->PV[3][N],2);
    output->CSS[5][J][L][1]=output->CSS[5][J][L][1]+WF*0.5e00*gas->SP[5][L]*A;
    if(gas->ISPR[1][L] > 0) output->CSS[6][J][L][1]=output->CSS[6][J][L][1]+WF*molecs->PROT[N];
    if(gas->MELE > 1) output->CSS[8][J][L][1]=output->CSS[8][J][L][1]+WF*molecs->PELE[N];
    if(gas->MMVM > 0){
        if(gas->ISPV[L] > 0){
            for(K=1;K<=gas->ISPV[L];K++)
                output->CSS[7][J][L][1]=output->CSS[7][J][L][1]+WF*double(molecs->IPVIB[K][N])*BOLTZ*gas->SPVM[1][K][L];
        }
    }
    A=pow(molecs->PV[1][N],2)+pow(molecs->PV[2][N],2)+pow(molecs->PV[3][N],2);
    B=fabs(molecs->PV[1][N]);
    output->CSSS[1][J]=output->CSSS[1][J]+WF/B;
    output->CSSS[2][J]=output->CSSS[2][J]+WF*gas->SP[5][L]/B;
    output->CSSS[3][J]=output->CSSS[3][J]+WF*gas->SP[5][L]*molecs->PV[2][N]/B;
    //this assumes that any flow normal to the x direction is in the y direction
    output->CSSS[4][J]=output->CSSS[4][J]+WF*gas->SP[5][L]*A/B;
    if(gas->ISPR[1][L] > 0){
        output->CSSS[5][J]=output->CSSS[5][J]+WF*molecs->PROT[N]/B;
        output->CSSS[6][J]=output->CSSS[6][J]+WF*gas->ISPR[1][L]/B;
    }
    //
    // CALL RANDOM_NUMBER(RANF)
    calc->RANF=((double)rand()/(double)RAND_MAX);
    if(gas->FSPEC[J] > calc->RANF){      //specular reflection
        X=2.e00*geom->XB[J]-X;
        molecs->PV[1][N]=-molecs->PV[1][N];
        DTR=(X-geom->XB[J])/molecs->PV[1][N];
    }
    else{                         //diffuse reflection
        VMPS=sqrt(2.e00*BOLTZ*gas->TSURF[J]/gas->SP[5][L]);
        DTR=(geom->XB[J]-molecs->PX[1][N])/molecs->PV[1][N];
        // CALL RANDOM_NUMBER(RANF)
        calc->RANF=((double)rand()/(double)RAND_MAX);
        molecs->PV[1][N]=sqrt(-log(calc->RANF))*VMPS;
        if(J == 2) molecs->PV[1][N]=-molecs->PV[1][N];
        RVELC(molecs->PV[2][N],molecs->PV[3][N],VMPS);
        molecs->PV[2][N]=molecs->PV[2][N]+gas->VSURF[J];
        if(gas->ISPR[1][L] > 0) SROT(L,gas->TSURF[J],molecs->PROT[N]);
        if(gas->MMVM > 0){
            for(K=1;K<=gas->ISPV[L];K++)
                SVIB(L,gas->TSURF[J],molecs->IPVIB[K][N],K);
        }
        if(gas->MELE > 1) SELE(L,gas->TSURF[J],molecs->PELE[N]);
    }
    //
    output->CSS[2][J][L][2]=output->CSS[2][J][L][2]-WF*molecs->PV[1][N]*gas->SP[5][L];
    output->CSS[3][J][L][2]=output->CSS[3][J][L][2]-WF*(molecs->PV[2][N]-gas->VSURF[J])*gas->SP[5][L];
    output->CSS[4][J][L][2]=output->CSS[4][J][L][2]-WF*molecs->PV[3][N]*gas->SP[5][L];
    A=pow(molecs->PV[1][N],2)+pow((molecs->PV[2][N]-gas->VSURF[J]),2)+pow(molecs->PV[3][N],2);
    output->CSS[5][J][L][2]=output->CSS[5][J][L][2]-WF*0.5e00*gas->SP[5][L]*A;
    if(gas->ISPR[1][L] > 0) output->CSS[6][J][L][2]=output->CSS[6][J][L][2]-WF*molecs->PROT[N];
    if(gas->MELE > 1) output->CSS[8][J][L][2]=output->CSS[8][J][L][2]-WF*molecs->PELE[N];
    if(gas->MMVM > 0){
        if(gas->ISPV[L] > 0){
            for(K=1;K<=gas->ISPV[L];K++)
                output->CSS[7][J][L][2]=output->CSS[7][J][L][2]-WF*double(molecs->IPVIB[K][N])*BOLTZ*gas->SPVM[1][K][L];
        }
    }
    A=pow(molecs->PV[1][N],2)+pow(molecs->PV[2][N],2)+pow(molecs->PV[3][N],2);
    B=fabs(molecs->PV[1][N]);
    output->CSSS[1][J]=output->CSSS[1][J]+WF/B;
    output->CSSS[2][J]=output->CSSS[2][J]+WF*gas->SP[5][L]/B;
    output->CSSS[3][J]=output->CSSS[3][J]+WF*gas->SP[5][L]*molecs->PV[2][N]/B;
    //this assumes that any flow normal to the x direction is in the y direction
    output->CSSS[4][J]=output->CSSS[4][J]+WF*gas->SP[5][L]*A/B;
    if(gas->ISPR[1][L] > 0){
        output->CSSS[5][J]=WF*output->CSSS[5][J]+molecs->PROT[N]/B;
        output->CSSS[6][J]=output->CSSS[6][J]+WF*gas->ISPR[1][L]/B;
    }
    //
    XI=geom->XB[J];
    DX=DTR*molecs->PV[1][N];
    DZ=0.e00;
    if(geom->IFX > 0) DY=DTR*molecs->PV[2][N];
    if(geom->IFX == 2) DZ=DTR*molecs->PV[3][N];
    if(geom->IFX == 0) X=XI+DX;
    if(geom->IFX > 0) AIFX(XI,DX,DY,DZ,X,molecs->PV[1][N],molecs->PV[2][N],molecs->PV[3][N]);
    //
    return;
}

void RBC(double &XI, double &DX, double &DY,double &DZ, double &R,double &S)
{
    //calculates the trajectory fraction S from a point at radius XI with
    //note that the axis is in the y direction
    //--displacements DX, DY, and DZ to a possible intersection with a
    //--surface of radius R, IFX=1, 2 for cylindrical, spherical geometry
    //MOLECS molecs;
    //GAS gas;
    //GEOM_1D geom;
    //CALC calc;
    //OUTPUT output;
    //
    // IMPLICIT NONE
    //
    double A,B,C,DD,S1,S2;
    //
    DD=DX*DX+DZ*DZ;
    if(geom->IFX == 2) DD=DD+DY*DY;
    B=XI*DX/DD;
    C=(XI*XI-R*R)/DD;
    A=B*B-C;
    if(A >= 0.e00){
        //find the least positive solution to the quadratic
        A=sqrt(A);
        S1=-B+A;
        S2=-B-A;
        if(S2 < 0.e00){
            if(S1 > 0.e00)
                S=S1;
            else
                S=2.e00;
        }
        else if(S1 < S2)
            S=S1;
        else
            S=S2;
    }
    else
        S=2.e00;
    //setting S to 2 indicates that there is no intersection
    return;
    //
}

void AIFX(double &XI,double &DX, double &DY, double &DZ, double &X, double &U, double &V, double &W)
{
    //
    //calculates the new radius and realigns the velocity components in
    //--cylindrical and spherical flows
    //MOLECS molecs;
    //GAS gas;
    //GEOM_1D geom;
    //CALC calc;
    //OUTPUT output;
    //
    // IMPLICIT NONE
    //
    //INTEGER ::
    double A,B,C,DR,VR,S;
    //
    if(geom->IFX == 1){
        DR=DZ;
        VR=W;
    }
    else if(geom->IFX == 2){
        DR=sqrt(DY*DY+DZ*DZ);
        VR=sqrt(V*V+W*W);
    }
    A=XI+DX;
    X=sqrt(A*A+DR*DR);
    S=DR/X;
    C=A/X;
    B=U;
    U=B*C+VR*S;
    W=-B*S+VR*C;
    if(geom->IFX == 2){
        VR=W;
        // CALL RANDOM_NUMBER(RANF)
        calc->RANF=((double)rand()/(double)RAND_MAX);
        A=DPI*calc->RANF;
        V=VR*sin(A);
        W=VR*cos(A);
    }
    //
    return;
    //
}

void REMOVE_MOL(int &N)
{
    //remove molecule N and replaces it by NM
    //MOLECS molecs;
    //CALC calc;
    //GEOM_1D geom;
    //GAS gas;
    // IMPLICIT NONE
    //
    int NC,M,K;
    
    //N the molecule number
    //M,K working integer
    //
    if(N != molecs->NM){
        for(M=1;M<=calc->NCLASS;M++)
            molecs->PX[M][N]=molecs->PX[M][molecs->NM];
        for(M=1;M<=3;M++)
            molecs->PV[M][N]=molecs->PV[M][molecs->NM];
        
        if(gas->MMRM > 0) molecs->PROT[N]=molecs->PROT[molecs->NM];
        molecs->IPCELL[N]=fabs(molecs->IPCELL[molecs->NM]);
        molecs->IPSP[N]=molecs->IPSP[molecs->NM];
        molecs->IPCP[N]=molecs->IPCP[molecs->NM];
        if(gas->MMVM > 0){
            for(M=1;M<=gas->MMVM;M++)
                molecs->IPVIB[M][N]=molecs->IPVIB[M][molecs->NM];
        }
        if(gas->MELE > 1) molecs->PELE[N]=molecs->PELE[molecs->NM];
        molecs->PTIM[N]=molecs->PTIM[molecs->NM];
    }
    molecs->NM=molecs->NM-1;
    //
    return;
    //
}

void INDEX_MOLS()
{
    //index the molecules to the collision cells
    //MOLECS molecs;
    //CALC calc;
    //GEOM_1D geom;
    // IMPLICIT NONE
    //
    int N,M,K;
    //
    //N,M,K working integer
    //
    for(N=1;N<=geom->NCCELLS;N++)
        geom->ICCELL[2][N]=0;
    //
    if(molecs->NM != 0){
        for(N=1;N<=molecs->NM;N++){
            M=molecs->IPCELL[N];
            geom->ICCELL[2][M]=geom->ICCELL[2][M]+1;
        }
        //

        M=0;
        for(N=1;N<=geom->NCCELLS;N++){
            geom->ICCELL[1][N]=M;
            M=M+geom->ICCELL[2][N];
            geom->ICCELL[2][N]=0;
        }
        //

        for(N=1;N<=molecs->NM;N++){
            M=molecs->IPCELL[N];
            geom->ICCELL[2][M]=geom->ICCELL[2][M]+1;
            K=geom->ICCELL[1][M]+geom->ICCELL[2][M];
            molecs->ICREF[K]=N;
        }
        //cin.get();
        //
    }
    return;
}

void SAMPLE_FLOW()
{
    //sample the flow properties
    //MOLECS molecs;
    //CALC calc;
    //GEOM_1D geom;
    //GAS gas;
    //OUTPUT output;
    //
    // IMPLICIT NONE
    //
    int NC,NCC,LS,N,M,K,L,I,KV;
    double A,TE,TT,WF;
    //
    //NC the sampling cell number
    //NCC the collision cell number
    //LS the species code
    //N,M,K working integers
    //TE total translational energy
    //
    output->NSAMP=output->NSAMP+1;
    cout<<"Sample \t"<<output->NSAMP<<endl<<endl;
    //WRITE (9,*) NM,'Mols. at sample',NSAMP
    file_9<<molecs->NM<<"  Mols. at sample  "<<output->NSAMP<<endl;
    //
    for(N=1;N<=molecs->NM;N++){
        
        NCC=molecs->IPCELL[N];
        NC=geom->ICCELL[3][NCC];
        WF=1.e00;
        if(geom->IWF == 1) WF=1.e00+geom->WFM*pow(molecs->PX[1][N],geom->IFX);
        if((NC > 0) && (NC <= geom->NCELLS)){
            if(gas->MSP > 1)
                LS=fabs(molecs->IPSP[N]);
            else
                LS=1;
            
            output->CS[0][NC][LS]=output->CS[0][NC][LS]+1.e00;
            output->CS[1][NC][LS]=output->CS[1][NC][LS]+WF;
            for(M=1;M<=3;M++){
                output->CS[M+1][NC][LS]=output->CS[M+1][NC][LS]+WF*molecs->PV[M][N];
                output->CS[M+4][NC][LS]=output->CS[M+4][NC][LS]+WF*pow(molecs->PV[M][N],2);
            }
            if(gas->MMRM > 0) output->CS[8][NC][LS]=output->CS[8][NC][LS]+WF*molecs->PROT[N];
            if(gas->MELE > 1) output->CS[9][NC][LS]=output->CS[9][NC][LS]+WF*molecs->PELE[N];
            if(gas->MMVM > 0){
                if(gas->ISPV[LS] > 0){
                    for(K=1;K<=gas->ISPV[LS];K++)
                        output->CS[K+9][NC][LS]=output->CS[K+9][NC][LS]+WF*double(molecs->IPVIB[K][N]);
                }
            }
        }
        else{
            cout<<"Illegal sampling cell  "<<NC<<"  "<<NCC<<"  for MOL  "<<N<<"  at  "<<molecs->PX[1][N]<<endl;
            return;
        }
        
    }
    //
    if(calc->FTIME > 0.5e00*calc->DTM) calc->TSAMP=calc->TSAMP+calc->DTSAMP;
    //
    return;
}

void ADAPT_CELLS_1D()
{
    //adapt the sampling cells through the splitting of the divisions into successive levels
    //the collision cells are divisions of the sampling cells
    //MOLECS molecs;
    //GAS gas;
    //GEOM_1D geom;
    //CALC calc;
    //OUTPUT output;
    //
    // IMPLICIT NONE
    //
    int M,N,L,K,KK,I,J,JJ,MSEG,NSEG,NSEG1,NSEG2,MLEVEL;
    double A,B,DDE,DCRIT;
    int *KDIV,*NC;
    int **ISD;
    double *XMIN,*XMAX,*DRAT;
    // INTEGER, ALLOCATABLE, DIMENSION(:) :: KDIV,NC
    // INTEGER, ALLOCATABLE, DIMENSION(:,:) :: ISD
    // REAL(KIND=8), ALLOCATABLE, DIMENSION(:) :: XMIN,XMAX,DRAT
    //
    //DCRIT  the number density ratio that causes a cell to be subdivided
    //KDIV(N) the number of divisions/subdivisions (cells or further subdivisions) at level N
    //DRAT(N) the contriburion to the density ratio of element N
    //NC(I) the number of sampling cells at level I
    //DDE the width of an element
    //MSEG the maximum number of segments (a segment is the size of the smallest subdivision
    //NSEG1 the (first segment-1) in the subdivision
    //NSEG2 the final segment in the subdivision
    //ISD(N,M) 0,1 for cell,subdivided for level N subdivision
    //MLEVEL The maximum desired level ILEVEL of subdivision (cellS are proportional to 2**ILEVEL)
    //
    DCRIT=1.5e00;    //may be altered
    MLEVEL=2;    //may be altered
    //
    //determine the level to which the divisions are to be subdivided
    //
    A=1.e00;
    for(N=1;N<=geom->NCELLS;N++)
        if(output->VAR[3][N]/gas->FND[1] > A) A=output->VAR[3][N]/gas->FND[1];
    
    geom->ILEVEL=0;
    while(A > DCRIT){
        geom->ILEVEL=geom->ILEVEL+1;
        A=A/2.e00;
    }
    if(geom->ILEVEL > MLEVEL) geom->ILEVEL=MLEVEL;
    //WRITE (9,*) 'ILEVEL =',ILEVEL
    file_9<<"ILEVEL = "<<geom->ILEVEL<<endl;
    NSEG=pow(2,geom->ILEVEL);
    MSEG=geom->NDIV*NSEG;
    //
    
    KDIV = new int[geom->ILEVEL+1];
    DRAT = new double[MSEG+1];
    NC = new int[geom->ILEVEL+1];
    
    ISD = new int*[geom->ILEVEL+1];
    for(int i =0; i< (geom->ILEVEL+1); ++i)
        ISD[i] = new int[MSEG+1];
    
    
    // ALLOCATE (KDIV(0:ILEVEL),DRAT(MSEG),NC(0:ILEVEL),ISD(0:ILEVEL,MSEG),STAT=ERROR)
    // IF (ERROR /= 0) THEN
    //   WRITE (*,*) 'PROGRAM COULD NOT ALLOCATE SPACE FOR KDIV ARRAY',ERROR
    // ENDIF
    //
    DDE=(geom->XB[2]-geom->XB[1])/double(MSEG);
    for(N=1;N<=MSEG;N++){
        A=geom->XB[1]+(double(N)-0.5e00)*DDE;
        FIND_CELL_1D(A,M,L);
        DRAT[N]=output->VAR[3][L]/(gas->FND[1]*double(NSEG));
    }
    //
    //calculate the number of subdivisions at the various levels of subdivision
    KDIV=0;
    //also the number of sampling cells at each level
    NC=0;
    //
    for(N=1;N<=geom->NDIV;N++){  //divisions
        ISD=0;
        ISD[0][1]=1;
        KDIV[0]=KDIV[0]+1;
        //  WRITE (9,*) 'DIVISION',N
        for(I=0;I<=geom->ILEVEL;I++){  //level of subdivision
            //    WRITE (9,*) 'LEVEL',I
            J=pow(2,I);  //number of possible subdivisions at this level
            JJ=NSEG/J;  //number of segments in a subdivision
            for(M=1;M<=J;M++){
                //      WRITE (9,*) 'SUBDIVISION',M
                if(ISD[I][M] == 1){
                    NSEG1=(N-1)*NSEG+(M-1)*JJ+1;
                    NSEG2=NSEG1+JJ-1;
                    A=0.e00;
                    //        WRITE (9,*) 'NSEG RANGE',NSEG1,NSEG2
                    for(L=NSEG1;L<=NSEG2;L++)
                        A=A+DRAT[L];
                    
                    //        WRITE (9,*) 'DENS CONTRIB',A
                    if(A < DCRIT){
                        NC[I]=NC[I]+1;
                        //          WRITE (9,*) 'LEVEL',I,' CELLS TO', NC(I)
                    }
                    else{
                        KDIV[I+1]=KDIV[I+1]+2;
                        //          WRITE (9,*) 'LEVEL',I+1,' SUBDIVISIONS TO',KDIV(I+1)
                        for(L=NSEG1-(N-1)*NSEG;L<=NSEG2-(N-1)*NSEG;L++)
                            ISD[I+1][L]=1;
                        
                    }
                }
            }
        }
    }
    //
    //WRITE (9,*) 'KDIV',KDIV
    file_9<<"KDIV "<<KDIV<<endl;
    //
    //WRITE (9,*) 'NC',NC
    file_9<< "NC "<<NC<<endl;
    cin.get();
    //WRITE (9,*) 'Number of divisions',NDIV
    file_9<<"Number of divisions "<<geom->NDIV<<endl;
    A=0;
    geom->NCELLS=0;
    for(N=0;N<=geom->ILEVEL;N++){
        A=A+double(NC[N])/(pow(2.e00,N));
        geom->NCELLS=geom->NCELLS+NC[N];
    }
    //WRITE (9,*) 'Total divisions from sampling cells',A
    //WRITE (9,*) 'Adapted sampling cells',NCELLS
    file_9<< "Total divisions from sampling cells "<<A<<endl;
    file_9<< "Adapted sampling cells "<<geom->NCELLS<<endl;
    geom->NCCELLS=geom->NCELLS*geom->NCIS;
    //WRITE (9,*) 'Adapted collision cells',NCCELLS
    file_9<< "Adapted collision cells "<<geom->NCCELLS<<endl;
    //
    
    for (int i = 0; i < geom->ILEVEL+1; i++) {
        cudaFree(geom->JDIV[i]); //delete [] geom->JDIV[i];
    }
    cudaFree(geom->JDIV); //delete [] geom->JDIV;  // <- because they won't exist anymore after this
    
    for (int i = 0; i < 5; i++) {
        cudaFree(geom->CELL[i]); //delete [] geom->CELL[i];
    }
    cudaFree(geom->CELL); //delete [] geom->CELL;  // <- because they won't exist anymore after this
    
    
    cudaFree(geom->ICELL); //delete[] geom->ICELL;
    
    for (int i = 0; i < 6; i++) {
        cudaFree(geom->CCELL[i]); //delete [] geom->CCELL[i];
    }
    cudaFree(geom->CCELL); //delete [] geom->CCELL;  // <- because they won't exist anymore after this
    
    for (int i = 0; i < 4; i++) {
        cudaFree(geom->ICCELL[i]); //delete [] geom->ICCELL[i];
    }
    cudaFree(geom->ICCELL); //delete [] geom->ICCELL;  // <- because they won't exist anymore after this
    
    cudaFree(output->COLLS);  //delete[] output->COLLS;
    
    cudaFree(output->WCOLLS); //delete[] output->WCOLLS;
    
    cudaFree(output->CLSEP); //delete[] output->CLSEP;
    
    for (int i = 0; i < 24; i++) {
        cudaFree(output->VAR[i]); //delete [] output->VAR[i];
    }
    cudaFree(output->VAR); //delete [] output->VAR;  // <- because they won't exist anymore after this
    
    
    for(int i = 0; i < 13; i++)
    {
        for(int j = 0; j < geom->NCELLS+1; j++)
        {
            cudaFree(output->VARSP[i][j]); //delete [] output->VARSP[i][j];
        }
        cudaFree(output->VARSP[i]); //delete [] output->VARSP[i];
    }
    cudaFree(output->VARSP); //delete [] output->VARSP;
    
    for(int i = 0; i < (10+gas->MSP); i++)
    {
        for(int j = 0; j < geom->NCELLS+1; j++)
        {
            cudaFree(output->CS[i][j]); //delete [] output->CS[i][j];
        }
        cudaFree(output->CS[i]); //delete [] output->CS[i];
    }
    cudaFree(output->CS); //delete [] output->CS;
    /*DEALLOCATE (JDIV,CELL,ICELL,CCELL,ICCELL,COLLS,WCOLLS,CLSEP,VAR,VARSP,CS,STAT=ERROR)
     IF (ERROR /= 0) THEN
     WRITE (*,*)'PROGRAM COULD NOT DEALLOCATE ARRAYS IN ADAPT',ERROR
     END IF*/
    //
    for(N=0;N<=geom->ILEVEL;N++)
        if(KDIV[N] > geom->MDIV) geom->MDIV=KDIV[N];
    //
    
    geom->i_allocate(geom->ILEVEL+1,geom->MDIV, geom->JDIV);
    //    ALLOCATE (JDIV(0:ILEVEL,MDIV),STAT=ERROR)
    //    IF (ERROR /= 0) THEN
    //    WRITE (*,*) 'PROGRAM COULD NOT ALLOCATE SPACE FOR JDIV ARRAY IN ADAPT',ERROR
    //    ENDIF
    //
    
    geom->d_allocate(5,geom->NCELLS+1, geom->CELL);
    geom->i_allocate(geom->NCELLS+1, geom->ICELL);
    geom->d_allocate(6, geom->NCCELLS+1, geom->CCELL);
    geom->i_allocate(4, geom->NCCELLS+1,geom->ICCELL);
    
    XMIN= new double[geom->NCCELLS+1];
    XMAX = new double[geom->NCCELLS+1];
    //
    //    ALLOCATE (CELL(4,NCELLS),ICELL(NCELLS),CCELL(5,NCCELLS),ICCELL(3,NCCELLS),XMIN(NCCELLS),XMAX(NCCELLS),STAT=ERROR)
    //    IF (ERROR /= 0) THEN
    //    WRITE (*,*) 'PROGRAM COULD NOT ALLOCATE SPACE FOR CELL ARRAYS IN ADAPT',ERROR
    //    ENDIF
    //
    
    output->d_allocate(geom->NCELLS+1,output->COLLS);
    output->d_allocate(geom->NCELLS+1, output->WCOLLS);
    output->d_allocate(geom->NCELLS+1, output->CLSEP);
    output->d_allocate(24, geom->NCELLS+1, output->VAR);
    output->d_allocate(13,geom->NCELLS+1,gas->MSP+1, output->VARSP);
    output->d_allocate(10+gas->MSP+1,geom->NCELLS+1,gas->MSP+1,output->CS);
    
    
    //    ALLOCATE (COLLS(NCELLS),WCOLLS(NCELLS),CLSEP(NCELLS),VAR(23,NCELLS),VARSP(0:12,NCELLS,MSP),CS(0:9+MSP,NCELLS,MSP),STAT=ERROR)
    //    IF (ERROR /= 0) THEN
    //    WRITE (*,*) 'PROGRAM COULD NOT ALLOCATE SPACE FOR SAMPLING ARRAYS IN ADAPT',ERROR
    //    ENDIF
    //
    geom->NCCELLS=0;
    geom->NCELLS=0;
    //
    //set the JDIV arrays and the sampling cells at the various levels of subdivision
    KDIV=0;
    geom->JDIV=0;
    //
    for(N=1;N<=geom->NDIV;N++){    //divisions
        ISD=0;
        ISD[0][1]=1;
        KDIV[0]=KDIV[0]+1;
        for(I=0;I<=geom->ILEVEL;I++){  //level of subdivision
            J=pow(2,I);  //number of possible subdivisions at this level
            JJ=NSEG/J;  //number of segments in a subdivision
            for(M=1;M<=J;M++){
                if(ISD[I][M] == 1){
                    NSEG1=(N-1)*NSEG+(M-1)*JJ+1;
                    NSEG2=NSEG1+JJ-1;
                    A=0.e00;
                    for(L=NSEG1;L<=NSEG2;L++)
                        A=A+DRAT[L];
                    
                    if(A < DCRIT){
                        geom->NCELLS=geom->NCELLS+1;
                        output->VAR[11][geom->NCELLS]=gas->FTMP[1];
                        XMIN[geom->NCELLS]=geom->XB[1]+double(NSEG1-1)*DDE;
                        XMAX[geom->NCELLS]=XMIN[geom->NCELLS]+double(NSEG2-NSEG1+1)*DDE;
                        //WRITE (9,*) NCELLS,I,' XMIN,XMAX',XMIN(NCELLS),XMAX(NCELLS)
                        file_9<< geom->NCELLS<<" "<<I<<" XMIN,XMAX "<<XMIN[geom->NCELLS]<<" , "<<XMAX[geom->NCELLS]<<endl;
                        geom->JDIV[I][KDIV[I]-(J-M)]=-geom->NCELLS;
                        //          WRITE (9,*) 'JDIV(',I,',',KDIV(I)-(J-M),')=',-NCELLS
                    }
                    else{
                        geom->JDIV[I][KDIV[I]-(J-M)]=KDIV[I+1];
                        //          WRITE (9,*) 'JDIV(',I,',',KDIV(I)-(J-M),')=',KDIV(I+1)
                        KDIV[I+1]=KDIV[I+1]+2;
                        for(L=NSEG1-(N-1)*NSEG;L<=NSEG2-(N-1)*NSEG;L++)
                            ISD[I+1][L]=1;
                    }
                }
            }
        }
    }
    //
    //set the other quantities associated with the sampling cells and the collision cells
    //
    geom->NCCELLS=0;
    for(N=1;N<=geom->NCELLS;N++){
        geom->CELL[1][N]=(XMIN[N]+XMAX[N])/2.e00;
        geom->CELL[2][N]=XMIN[N];
        geom->CELL[3][N]=XMAX[N];
        if(geom->IFX == 0) geom->CELL[4][N]=XMAX[N]-XMIN[N];    //calculation assumes unit cross-section
        if(geom->IFX == 1) geom->CELL[4][N]=PI*(pow(XMAX[N],2)-pow(XMIN[N],2));
        if(geom->IFX == 2) geom->CELL[4][N]=1.33333333333333333333e00*PI*(pow(XMAX[N],3)-pow(XMIN[N],3));
        geom->ICELL[N]=geom->NCCELLS;
        for(M=1;M<=geom->NCIS;M++){
            geom->NCCELLS=geom->NCCELLS+1;
            geom->ICCELL[3][geom->NCCELLS]=N;
            geom->CCELL[1][geom->NCCELLS]=geom->CELL[4][N]/double(geom->NCIS);
            geom->CCELL[3][geom->NCCELLS]=calc->DTM/2.e00;
            geom->CCELL[4][geom->NCCELLS]=2.e00*gas->VMPM*gas->SPM[2][1][1];
            // CALL RANDOM_NUMBER(RANF)
            calc->RANF=((double)rand()/(double)RAND_MAX);
            geom->CCELL[2][geom->NCCELLS]=calc->RANF;
            geom->CCELL[5][geom->NCCELLS]=calc->FTIME;
        }
    }
    //
    //assign the molecules to the cells
    //
    for(N=1;N<=molecs->NM;N++){
        FIND_CELL_1D(molecs->PX[1][N],molecs->IPCELL[N],JJ);
        M=molecs->IPCELL[N];
    }
    //
    //deallocate the local variables
    for (int i = 0; i < geom->ILEVEL+1; i++) {
        delete [] ISD[i];
    }
    delete [] ISD;
    delete [] NC;
    delete[] KDIV;
    delete [] XMAX;
    delete [] XMIN;
    delete [] DRAT;
    /*DEALLOCATE (KDIV,NC,ISD,XMIN,XMAX,DRAT,STAT=ERROR)
     IF (ERROR /= 0) THEN
     WRITE (*,*)'PROGRAM COULD NOT DEALLOCATE LOCAL ARRAYS IN ADAPT',ERROR
     END IF*/
    //
    return;
}

void EXTEND_MNM(double FAC)
{  //
    //the maximum number of molecules is increased by a specified factor
    //the existing molecules are copied TO disk storage
    //MOLECS molecs;
    //CALC calc;
    //GAS gas;
    //
    // IMPLICIT NONE
    //
    int M,N,MNMN;
    fstream file_7;
    // REAL :: FAC
    //
    //M,N working integers
    //MNMN extended value of MNM
    //FAC the factor for the extension
    MNMN=FAC*molecs->MNM;
    cout<< "Maximum number of molecules is to be extended from "<<molecs->MNM<<" to "<<MNMN<<endl;
    cout<< "( if the additional memory is available //// )"<<endl;
    
    file_7.open("EXTMOLS.SCR", ios::binary | ios::out);
    if(file_7.is_open()){
        cout<<"EXTMOLS.SCR is opened"<<endl;
    }
    else{
        cout<<"EXTMOLS.SCR not opened"<<endl;
    }
    cout<<"Start write to disk storage"<<endl;
    //OPEN (7,FILE='EXTMOLS.SCR',FORM='BINARY')
    //WRITE (*,*) 'Start write to disk storage'
    
    for(N=1;N<=molecs->MNM;N++){
        if(gas->MMVM > 0){
            file_7<<molecs->PX[calc->NCLASS][N]<<endl<<molecs->PTIM[N]<<endl<<molecs->PROT[N]<<endl;
            for(M=1;M<=3;M++)
                file_7<<molecs->PV[M][N]<<endl;
            file_7<<molecs->IPSP[N]<<endl<<molecs->IPCELL[N]<<endl<<molecs->ICREF[N]<<endl<<molecs->IPCP[N]<<endl;
            for(M=1;M<=gas->MMVM;M++)
                file_7<<molecs->IPVIB[M][N]<<endl;
            file_7<<molecs->PELE[N]<<endl;//WRITE (7) PX(NCLASS,N),PTIM(N),PROT(N),(PV(M,N),M=1,3),IPSP(N),IPCELL(N),ICREF(N),IPCP(N),(IPVIB(M,N),M=1,MMVM),PELE(N)
        }
        else{
            if(gas->MMRM > 0){
                file_7<<molecs->PX[calc->NCLASS][N]<<endl<<molecs->PTIM[N]<<endl<<molecs->PROT[N]<<endl;
                for(M=1;M<=3;M++)
                    file_7<<molecs->PV[M][N]<<endl;
                file_7<<molecs->IPSP[N]<<endl<<molecs->IPCELL[N]<<endl<<molecs->ICREF[N]<<endl<<molecs->IPCP[N]<<endl<<molecs->PELE[N]<<endl;//WRITE (7) PX(NCLASS,N),PTIM(N),PROT(N),(PV(M,N),M=1,3),IPSP(N),IPCELL(N),ICREF(N),IPCP(N),PELE(N)
            }
            else{
                file_7<<molecs->PX[calc->NCLASS][N]<<endl<<molecs->PTIM[N]<<endl;
                for(M=1;M<=3;M++)
                    file_7<<molecs->PV[M][N]<<endl;
                file_7<<molecs->IPSP[N]<<endl<<molecs->IPCELL[N]<<endl<<molecs->ICREF[N]<<endl<<molecs->IPCP[N]<<endl<<molecs->PELE[N]<<endl;//WRITE (7) PX(NCLASS,N),PTIM(N),(PV(M,N),M=1,3),IPSP(N),IPCELL(N),ICREF(N),IPCP(N),PELE(N)
            }
            
        }
    }
    cout<<"Disk write completed"<<endl;
    // WRITE (*,*) 'Disk write completed'
    // CLOSE (7)
    file_7.close();
    if(gas->MMVM > 0){
        for(int i=0;i<calc->NCLASS+1;i++){
                cudaFree(molecs->PX[i]); //delete [] molecs->PX[i];
        }
        cudaFree(molecs->PX); //delete [] molecs->PX;

        cudaFree(molecs->PTIM); //delete [] molecs->PTIM;

        cudaFree(molecs->PROT);

        for(int i=0;i<4;i++){
            cudaFree(molecs->PV[i]); //delete [] molecs->PV[i];
        }
        cudaFree(molecs->PV); //delete [] molecs->PV;

        cudaFree(molecs->IPSP);
        cudaFree(molecs->IPCELL);
        cudaFree(molecs->ICREF);
        cudaFree(molecs->IPCP);
        cudaFree(molecs->PELE);
        for(int i=0;i<gas->MMVM;i++){
            cudaFree(molecs->IPVIB[i]); //delete [] molecs->IPVIB[i];
        }
        cudaFree(molecs->IPVIB); //delete molecs->IPVIB;
        // for(int i=0;i<calc->NCLASS+1;i++){
        //     delete [] molecs->PX[i];
        // }
        // delete [] molecs->PX;
        // delete [] molecs->PTIM;
        // delete [] molecs->PROT;
        // for(int i=0;i<4;i++){
        //     delete [] molecs->PV[i];
        // }
        // delete [] molecs->PV;
        // delete [] molecs->IPSP;
        // delete [] molecs->IPCELL;
        // delete [] molecs->ICREF;
        // delete [] molecs->IPCP;
        // delete [] molecs->PELE;
        // for(int i=0;i<gas->MMVM;i++){
        //     delete [] molecs->IPVIB[i];
        // }
        // delete molecs->IPVIB;
        //DEALLOCATE (PX,PTIM,PROT,PV,IPSP,IPCELL,ICREF,IPCP,IPVIB,PELE,STAT=ERROR)
    }
    else{
        if(gas->MMRM > 0){
            for(int i=0;i<calc->NCLASS+1;i++){
                cudaFree(molecs->PX[i]); //delete [] molecs->PX[i];
            }
            cudaFree(molecs->PX); //delete [] molecs->PX;

            cudaFree(molecs->PTIM); //delete [] molecs->PTIM;

            cudaFree(molecs->PROT);

            for(int i=0;i<4;i++){
                cudaFree(molecs->PV[i]); //delete [] molecs->PV[i];
            }
            cudaFree(molecs->PV); //delete [] molecs->PV;

            cudaFree(molecs->IPSP);
            cudaFree(molecs->IPCELL);
            cudaFree(molecs->ICREF);
            cudaFree(molecs->IPCP);
            cudaFree(molecs->PELE);
            // delete [] molecs->IPSP;
            // delete [] molecs->IPCELL;
            // delete [] molecs->ICREF;
            // delete [] molecs->IPCP;
            // delete [] molecs->PELE;//DEALLOCATE (PX,PTIM,PV,IPSP,IPCELL,ICREF,IPCP,PELE,STAT=ERROR)
            // for(int i=0;i<calc->NCLASS+1;i++){
            //     delete [] molecs->PX[i];
            // }
            // delete [] molecs->PX;
            // delete [] molecs->PTIM;
            // delete [] molecs->PROT;
            // for(int i=0;i<4;i++){
            //     delete [] molecs->PV[i];
            // }
            // delete [] molecs->PV;
            // delete [] molecs->IPSP;
            // delete [] molecs->IPCELL;
            // delete [] molecs->ICREF;
            // delete [] molecs->IPCP;
            // delete [] molecs->PELE;
            //DEALLOCATE (PX,PTIM,PROT,PV,IPSP,IPCELL,ICREF,IPCP,PELE,STAT=ERROR)
        }
        else{
            for(int i=0;i<calc->NCLASS+1;i++){
                cudaFree(molecs->PX[i]); //delete [] molecs->PX[i];
            }
            cudaFree(molecs->PX); //delete [] molecs->PX;

            cudaFree(molecs->PTIM); //delete [] molecs->PTIM;

            for(int i=0;i<4;i++){
                cudaFree(molecs->PV[i]); //delete [] molecs->PV[i];
            }
            cudaFree(molecs->PV); //delete [] molecs->PV;

            cudaFree(molecs->IPSP);
            cudaFree(molecs->IPCELL);
            cudaFree(molecs->ICREF);
            cudaFree(molecs->IPCP);
            cudaFree(molecs->PELE);
            // delete [] molecs->IPSP;
            // delete [] molecs->IPCELL;
            // delete [] molecs->ICREF;
            // delete [] molecs->IPCP;
            // delete [] molecs->PELE;//DEALLOCATE (PX,PTIM,PV,IPSP,IPCELL,ICREF,IPCP,PELE,STAT=ERROR)
        }
    }
    // IF (ERROR /= 0) THEN
    //   WRITE (*,*)'PROGRAM COULD NOT DEALLOCATE MOLECULES',ERROR
    // !  STOP
    // END IF
    // !
    
    if(gas->MMVM > 0){
        molecs->d_allocate(calc->NCLASS+1,MNMN+1,molecs->PX);
        molecs->d_allocate(MNMN+1,molecs->PTIM);
        molecs->d_allocate(MNMN+1,molecs->PROT);
        molecs->d_allocate(4,MNMN+1,molecs->PV);
        molecs->i_allocate(MNMN+1,molecs->IPSP);
        molecs->i_allocate(MNMN+1,molecs->IPCELL);
        molecs->i_allocate(MNMN+1,molecs->ICREF);
        molecs->i_allocate(MNMN+1,molecs->IPCP);
        molecs->i_allocate(gas->MMVM+1,MNMN+1,molecs->IPVIB);
        molecs->d_allocate(MNMN+1,molecs->PELE);
        // ALLOCATE (PX(NCLASS,MNMN),PTIM(MNMN),PROT(MNMN),PV(3,MNMN),IPSP(MNMN),IPCELL(MNMN),ICREF(MNMN),IPCP(MNMN),IPVIB(MMVM,MNMN),PELE(MNMN),STAT=ERROR)
    }
    else{
        if(gas->MMRM > 0){
            molecs->d_allocate(calc->NCLASS+1,MNMN+1,molecs->PX);
            molecs->d_allocate(MNMN+1,molecs->PTIM);
            molecs->d_allocate(MNMN+1,molecs->PROT);
            molecs->d_allocate(4,MNMN+1,molecs->PV);
            molecs->i_allocate(MNMN+1,molecs->IPSP);
            molecs->i_allocate(MNMN+1,molecs->IPCELL);
            molecs->i_allocate(MNMN+1,molecs->ICREF);
            molecs->i_allocate(MNMN+1,molecs->IPCP);
            molecs->d_allocate(MNMN+1,molecs->PELE);
            // ALLOCATE (PX(NCLASS,MNMN),PTIM(MNMN),PROT(MNMN),PV(3,MNMN),IPSP(MNMN),IPCELL(MNMN),ICREF(MNMN),IPCP(MNMN),PELE(MNMN),STAT=ERROR)
        }
        else{
            molecs->d_allocate(calc->NCLASS+1,MNMN+1,molecs->PX);
            molecs->d_allocate(MNMN+1,molecs->PTIM);
            molecs->d_allocate(4,MNMN+1,molecs->PV);
            molecs->i_allocate(MNMN+1,molecs->IPSP);
            molecs->i_allocate(MNMN+1,molecs->IPCELL);
            molecs->i_allocate(MNMN+1,molecs->ICREF);
            molecs->i_allocate(MNMN+1,molecs->IPCP);
            molecs->d_allocate(MNMN+1,molecs->PELE);
            // ALLOCATE (PX(NCLASS,MNMN),PTIM(MNMN),PV(3,MNMN),IPSP(MNMN),IPCELL(MNMN),ICREF(MNMN),IPCP(MNMN),PELE(MNMN),STAT=ERROR)
        }
    }
    // IF (ERROR /= 0) THEN
    //   WRITE (*,*)'PROGRAM COULD NOT ALLOCATE SPACE FOR EXTEND_MNM',ERROR
    // !  STOP
    // END IF
    // !
    //memset(molecs->PX,0.0,sizeof(**molecs->PX)); memset(molecs->PTIM,0.0,sizeof(*molecs->PTIM)); memset(molecs->PV,0.0,sizeof(**molecs->PV)); memset(molecs->IPSP,0,sizeof(*molecs->IPSP)); memset(molecs->IPCELL,0,sizeof(*molecs->IPCELL)); memset(molecs->ICREF,0,sizeof(*molecs->ICREF)); memset(molecs->IPCP,0,sizeof(*molecs->IPCP)); memset(molecs->PELE,0,sizeof(*molecs->PELE));
    
    for(int i=0;i<calc->NCLASS+1;i++){
        for(int j=0;j<MNMN+1;j++)
            molecs->PX[i][j]=0.0;
    }
    
    for(int i=0;i<4;i++){
        for(int j=0;j<MNMN+1;j++)
            molecs->PV[i][j]=0.0;
    }
    for(int i=0;i<MNMN+1;i++){
        molecs->PTIM[i]=0.0;
        molecs->IPSP[i]=0;
        molecs->IPCELL[i]=0;
        molecs->ICREF[i]=0;
        molecs->IPCP[i]=0;
        molecs->PELE[i]=0;
    }
        
    
    if(gas->MMRM > 0) {
        for(int i=0;i<MNMN+1;i++)
            molecs->PROT[i]=0.0;
        //memset(molecs->PROT,0.0,sizeof(*molecs->PROT));
    }
    if(gas->MMVM > 0) {
        for(int i=0;i<gas->MMVM+1;i++){
            for(int j=0;j<MNMN+1;j++)
                molecs->IPVIB[i][j]=0;
        }
        //memset(molecs->IPVIB,0,sizeof(**molecs->IPVIB));
    }
    //restore the original molecules
    // OPEN (7,FILE='EXTMOLS.SCR',FORM='BINARY')
    // WRITE (*,*) 'Start read back from disk storage'
    file_7.open("EXTMOLS.SCR", ios::binary | ios::in);
    if(file_7.is_open()){
        cout<<"EXTMOLS.SCR is opened"<<endl;
    }
    else{
        cout<<"EXTMOLS.SCR not opened"<<endl;
    }
    for(N=1;N<=molecs->MNM;N++){
        if(gas->MMVM > 0){
            file_7>>molecs->PX[calc->NCLASS][N]>>molecs->PTIM[N]>>molecs->PROT[N];
            for(M=1;M<=3;M++)
                file_7>>molecs->PV[M][N];
            file_7>>molecs->IPSP[N]>>molecs->IPCELL[N]>>molecs->ICREF[N]>>molecs->IPCP[N];
            for(M=1;M<=gas->MMVM;M++)
                file_7>>molecs->IPVIB[M][N];
            file_7>>molecs->PELE[N];//READ (7) PX(NCLASS,N),PTIM(N),PROT(N),(PV(M,N),M=1,3),IPSP(N),IPCELL(N),ICREF(N),IPCP(N),(IPVIB(M,N),M=1,MMVM),PELE(N)
        }
        else{
            if(gas->MMRM > 0){
                file_7>>molecs->PX[calc->NCLASS][N]>>molecs->PTIM[N]>>molecs->PROT[N];
                for(M=1;M<=3;M++)
                    file_7>>molecs->PV[M][N];
                file_7>>molecs->IPSP[N]>>molecs->IPCELL[N]>>molecs->ICREF[N]>>molecs->IPCP[N]>>molecs->PELE[N];//READ (7) PX(NCLASS,N),PTIM(N),PROT(N),(PV(M,N),M=1,3),IPSP(N),IPCELL(N),ICREF(N),IPCP(N),PELE(N)
            }
            else{
                file_7>>molecs->PX[calc->NCLASS][N]>>molecs->PTIM[N];
                for(M=1;M<=3;M++)
                    file_7>>molecs->PV[M][N];
                file_7>>molecs->IPSP[N]>>molecs->IPCELL[N]>>molecs->ICREF[N]>>molecs->IPCP[N]>>molecs->PELE[N];//READ (7) PX(NCLASS,N),PTIM(N),(PV(M,N),M=1,3),IPSP(N),IPCELL(N),ICREF(N),IPCP(N),PELE(N)
            }
        }
    }
    cout<<"Disk read completed"<<endl;
    // WRITE (*,*) 'Disk read completed'
    // CLOSE (7,STATUS='DELETE')
    file_7.close();
    //
    molecs->MNM=MNMN;
    //
    return;
}

void DISSOCIATION()
{
    //dissociate diatomic molecules that have been marked for dissociation by -ve level or -99999 for ground state
    //MOLECS molecs;
    //GAS gas;
    //CALC calc;
    //
    // IMPLICIT NONE
    //
    int K,KK,L,N,M,LS,MS,KV,IDISS;
    double A,B,C,EA,VRR,VR,RMM,RML;
    double VRC[4],VCM[4],VRCP[4];
    //
    N=0;
    while(N < molecs->NM){
        N=N+1;
        IDISS=0;
        L=molecs->IPSP[N];
        if(gas->ISPV[L] > 0){
            for(K=1;K<=gas->ISPV[L];K++){
                M=molecs->IPVIB[K][N];
                if(M < 0){
                    //dissociation
                    calc->TDISS[L]=calc->TDISS[L]+1.e00;
                    IDISS=1;
                }
            }
            if(IDISS == 1){
                EA=molecs->PROT[N];    //EA is energy available for relative translational motion of atoms
                if(gas->MELE > 1) EA=EA+molecs->PELE[N];
                if(molecs->NM >= molecs->MNM) EXTEND_MNM(1.1);
                molecs->NM=molecs->NM+1;
                //set center of mass velocity as that of molecule
                VCM[1]=molecs->PV[1][N];
                VCM[2]=molecs->PV[2][N];
                VCM[3]=molecs->PV[3][N];
                molecs->PX[calc->NCLASS][molecs->NM]=molecs->PX[calc->NCLASS][N];
                molecs->IPCELL[molecs->NM]=molecs->IPCELL[N];
                LS=molecs->IPSP[N];
                gas->TREACL[1][LS]=gas->TREACL[1][LS]-1;
                molecs->IPSP[molecs->NM]=gas->ISPVM[1][1][L];
                MS=molecs->IPSP[molecs->NM];
                molecs->IPSP[N]=gas->ISPVM[2][1][L];
                LS=molecs->IPSP[N];
                gas->TREACG[1][LS]=gas->TREACG[1][LS]+1;
                gas->TREACG[1][MS]=gas->TREACG[1][MS]+1;
                molecs->PTIM[molecs->NM]=molecs->PTIM[N];
                VRR=2.e00*EA/gas->SPM[1][LS][MS];
                VR=sqrt(VRR);
                RML=gas->SPM[1][LS][MS]/gas->SP[5][MS];
                RMM=gas->SPM[1][LS][MS]/gas->SP[5][LS];
                // CALL RANDOM_NUMBER(RANF)
                calc->RANF=((double)rand()/(double)RAND_MAX);
                B=2.e00*calc->RANF-1.e00;
                A=sqrt(1.e00-B*B);
                VRCP[1]=B*VR;
                // CALL RANDOM_NUMBER(RANF)
                calc->RANF=((double)rand()/(double)RAND_MAX);
                C=2.e00*PI*calc->RANF;
                VRCP[2]=A*cos(C)*VR;
                VRCP[3]=A*sin(C)*VR;
                for(KK=1;KK<=3;KK++){
                    molecs->PV[KK][N]=VCM[KK]+RMM*VRCP[KK];
                    molecs->PV[KK][molecs->NM]=VCM[KK]-RML*VRCP[KK];
                }
                
                if((fabs(molecs->PV[1][N]) > 100000.e00) || (fabs(molecs->PV[1][molecs->NM]) > 100000.e00)) {
                    cout<< "EXCESSIVE SPEED, DISS "<< N<< " "<<molecs->PV[1][N]<<" "<<molecs->NM<<" "<<molecs->PV[1][molecs->NM]<<endl;
                   
                }
                
                
                
                //set any internal modes to the ground state
                if(gas->ISPV[LS] > 0){
                    for(KV=1;KV<=gas->ISPV[LS];KV++)
                        molecs->IPVIB[KV][N]=0;
                }
                if(gas->ISPR[1][LS] > 0) molecs->PROT[N]=0.e00;
                if(gas->MELE > 1) molecs->PELE[N]=0.e00;
                if(gas->ISPV[MS] > 0){
                    for(KV=1;KV<=gas->ISPV[MS];KV++)
                        molecs->IPVIB[KV][molecs->NM]=0;
                }
                if(gas->ISPR[1][MS] > 0) molecs->PROT[molecs->NM]=0.0;
                if(gas->MELE > 1) molecs->PELE[molecs->NM]=0.e00;
            }
        }
    }
    return;
}
//************************************************************************************
//

void ENERGY(int I,double &TOTEN)
{
    //calculate the total energy (all molecules if I=0, otherwise molecule I)
    //I>0 used for dianostic purposes only
    //MOLECS molecs;
    //GAS gas;
    //CALC calc;
    //
    // IMPLICIT NONE
    //
    int K,L,N,II,M,IV,KV,J;
    double TOTENI,TOTELE;
    //
    TOTEN=0.0;
    TOTELE=0;
    //
    
    if(I == 0){
        for(N=1;N<=molecs->NM;N++){
            if(molecs->IPCELL[N] > 0){
                L=molecs->IPSP[N];
                TOTENI=TOTEN;
                TOTEN=TOTEN+gas->SP[6][L];
                TOTEN=TOTEN+0.5e00*gas->SP[5][L]*(pow(molecs->PV[1][N],2)+pow(molecs->PV[2][N],2)+pow(molecs->PV[3][N],2));
                if(gas->ISPR[1][L] > 0) TOTEN=TOTEN+molecs->PROT[N];
                if(gas->ISPV[L] > 0){
                    for(KV=1;KV<=gas->ISPV[L];KV++){
                        J=molecs->IPVIB[KV][N];
                        //         IF (J <0) THEN
                        //           J=-J
                        //           IF (J == 99999) J=0
                        //         END IF
                        TOTEN=TOTEN+double(J)*BOLTZ*gas->SPVM[1][KV][L];
                    }
                }
            }
            if(gas->MELE > 1){
                TOTEN=TOTEN+molecs->PELE[N];
                TOTELE=TOTELE+molecs->PELE[N];
            }
            if((TOTEN-TOTENI) > 1.e-16) cout<<"MOL "<<N<<" ENERGY "<<TOTEN-TOTENI<<endl;
        }
        //
        //WRITE (9,*) 'Total Energy =',TOTEN,NM
        //WRITE (*,*) 'Total Energy =',TOTEN,NM
        file_9<<"Total Energy =  "<<setprecision(25)<<TOTEN<<"\t"<<molecs->NM<<endl;
        cout<<"Total Energy =  "<<setprecision(20)<<TOTEN<<"\t"<<molecs->NM<<endl;
        //  WRITE (*,*) 'Electronic Energy =',TOTELE
    }
    else{
        N=I;
        if(molecs->IPCELL[N] > 0){
            L=molecs->IPSP[N];
            TOTEN=TOTEN+gas->SP[6][L];
            TOTEN=TOTEN+0.5e00*gas->SP[5][L]*(pow(molecs->PV[1][N],2)+pow(molecs->PV[2][N],2)+pow(molecs->PV[3][N],2));
            if(gas->ISPR[1][L] > 0) TOTEN=TOTEN+molecs->PROT[N];
            if(gas->ISPV[L] > 0){
                for(KV=1;KV<=gas->ISPV[L];KV++){
                    J=molecs->IPVIB[KV][N];
                    //         IF (J <0) THEN
                    //           J=-J
                    //           IF (J == 99999) J=0
                    //         END IF
                    TOTEN=TOTEN+double(J)*BOLTZ*gas->SPVM[1][KV][L];
                }
            }
        }
    }
    
    //
    return;   //
}



void SETXT()
{
    //generate TECPLOT files for displaying an x-t diagram of an unsteady flow
    //this employs ordered data, therefore the cells MUST NOT BE ADAPTED
    //N.B. some custom coding for particular problems
    //
    //
    //MOLECS molecs;
    //CALC calc;
    //GEOM_1D geom;
    //GAS gas;
    //OUTPUT output;
    //
    
    // IMPLICIT NONE
    //
    int N,M,IOUT;
    double A,C;
    double **VALINT;
    // REAL(KIND=8), ALLOCATABLE, DIMENSION(:,:) :: VALINT
    //
    //VALINT(N,M) the interpolated values at sampling cell M boundaries and extrapolated values at boundaries
    //    N=1 distance
    //    N=2 time
    //    N=3 number density
    //    N=4 radial velocity
    //    N=5 pressure (nkT)
    //    N=6 temperature
    //    N=7 h2o fraction (Sec. 7.9 only)
    //
    //the variables in VALINT may be altered for particular problems
    //
    VALINT = new double*[7];
    for(int i =0; i< 7; ++i)
        VALINT[i] = new double[geom->NCELLS+2];
    
    // ALLOCATE (VALINT(6,NCELLS+1),STAT=ERROR)
    //
    //777 FORMAT(12G14.6)
    //24[]
    
    //Internal options
    IOUT=0;    //0 for dimensioned output, 1 for non-dimensional output
    //
    A=1.e00;   //dt/dt for selection of v velocity component in TECPLOT to draw particle paths as "streamlines"
    //
    if(calc->FTIME < 0.5e00*calc->DTM){
        //Headings and zero time record
        //        IF (ERROR /= 0) THEN
        //        WRITE (*,*) 'PROGRAM COULD NOT ALLOCATE SPACE FOR VALINT ARRAY',ERROR
        //        ENDIF
        calc->NLINE=1;
        file_9<< "J in tecplot file = "<<calc->NLINE*(geom->NCELLS+1)<<endl;
        //  WRITE (18,*) 'VARIABLES = "Distance","Time","n","u","p","T","H2O","A"'   //for combustion wave output(Sec. 7.9)
        
        file_18<<"VARIABLES = 'Distance','Time','n','u','p','T','A' "<<endl;
        file_18<<"ZONE I= "<<geom->NCELLS+1<<", J=  (set to number of output intervals+1), F=POINT"<<endl;
        //
        for(N=1;N<=geom->NCELLS+1;N++){
            VALINT[1][N]=geom->XB[1]+(N-1)*geom->DDIV;    //distance
            VALINT[1][N]=VALINT[1][N];         //time
            VALINT[2][N]=0.0;
            VALINT[3][N]=gas->FND[1];
            VALINT[4][N]=0;
            VALINT[5][N]=gas->FND[1]*BOLTZ*gas->FTMP[1];
            VALINT[6][N]=gas->FTMP[1];
            //   VALINT(7,N)=FSP(6,1)   //FSP(6 for combustion wave
            if((VALINT[1][N] > geom->XS) && (calc->ISECS == 1)){
                VALINT[3][N]=gas->FND[2];
                VALINT[5][N]=gas->FND[2]*BOLTZ*gas->FTMP[2];
                VALINT[6][N]=gas->FTMP[2];
                //      VALINT(7,N)=FSP(6,2)
            }
            if(IOUT == 1){
                VALINT[3][N]=1.e00;
                VALINT[5][N]=1.e00;
                VALINT[6][N]=1.e00;
            }
            for(M=1;M<=6;M++)
                file_18<<VALINT[M][N]<<"\t";//WRITE (18,777) (VALINT(M,N),M=1,6),A
            file_18<<A<<endl;
        }
    }
    else{
        calc->NLINE=calc->NLINE+1;
        cout<<"J in tecplot file = "<<calc->NLINE<<endl;
        if(geom->IVB == 0) C=geom->DDIV;
        if(geom->IVB == 1) C=(geom->XB[2]+geom->VELOB*calc->FTIME-geom->XB[1])/double(geom->NDIV);
        for(N=1;N<=geom->NCELLS+1;N++){
            VALINT[1][N]=geom->XB[1]+(N-1)*C;
            VALINT[2][N]=calc->FTIME;
            if((N > 1) && (N < geom->NCELLS+1)){
                VALINT[3][N]=0.5e00*(output->VAR[3][N]+output->VAR[3][N-1]);
                VALINT[4][N]=0.5e00*(output->VAR[5][N]+output->VAR[5][N-1]);
                VALINT[5][N]=0.5e00*(output->VAR[18][N]+output->VAR[18][N-1]);
                VALINT[6][N]=0.5e00*(output->VAR[11][N]+output->VAR[11][N-1]);
                //     VALINT(7,N)=0.5D00*(VARSP(1,N,6)+VARSP(1,N-1,6))   //H2O fraction for Sec 7.9
            }
        }
        for(N=3;N<=6;N++)
            VALINT[N][1]=0.5e00*(3.e00*VALINT[N][2]-VALINT[N][3]);
        
        //
        for(N=3;N<=6;N++)
            VALINT[N][geom->NCELLS+1]=0.5e00*(3.e00*VALINT[N][geom->NCELLS]-VALINT[N][geom->NCELLS-1]);
        
        //
        for(N=1;N<=geom->NCELLS+1;N++){
            if(IOUT == 1){
                VALINT[1][N]=(VALINT[1][N]-geom->XB[1])/(geom->XB[2]-geom->XB[1]);
                VALINT[2][N]=VALINT[2][N]/calc->TNORM;
                VALINT[3][N]=VALINT[3][N]/gas->FND[1];
                VALINT[4][N]=VALINT[4][N]/gas->VMPM;
                VALINT[5][N]=VALINT[5][N]/(gas->FND[1]*BOLTZ*gas->FTMP[1]);
                VALINT[6][N]=VALINT[6][N]/gas->FTMP[1];
            }
            for(M=1;M<=6;M++)
                file_18<<VALINT[M][N]<<"\t";//WRITE (18,777) (VALINT[M][N],M=1,6),A       //
            file_18<<A<<endl;
        }
    }
    //
    return;
}


void MOLECULES_MOVE_1D()
{//
    //molecule moves appropriate to the time step
    //for homogeneous and one-dimensional flows
    //(homogeneous flows are calculated as one-dimensional)
    //MOLECS molecs;
    //GAS gas;
    //GEOM_1D geom;
    //CALC calc;
    //OUTPUT output;
    //
    // IMPLICIT NONE
    //
    int N,L,M,K,NCI,J,II,JJ;
    double A,B,X,XI,XC,DX,DY,DZ,DTIM,S1,XM,R,TI,DTC,POB,UR,WFI,WFR,WFRI;
    //
    //N working integer
    //NCI initial cell time
    //DTIM time interval for the move
    //POB position of the outer boundary
    //TI initial time
    //DTC time interval to collision with surface
    //UR radial velocity component
    //WFI initial weighting factor
    //WFR weighting factor radius
    //WFRI initial weighting factor radius
    //
    if((geom->ITYPE[2] == 4) && (calc->ICN == 1)){
        //memset(calc->ALOSS,0.e00,sizeof(*calc->ALOSS));//calc->ALOSS=0.e00;
        for(int i=0;i<gas->MSP+1;i++)
            calc->ALOSS[i]=0.e00;
        
        calc->NMP=molecs->NM;
    }
    //
    N=1;

    while(N <= molecs->NM){
        //
        NCI=molecs->IPCELL[N];
        if((calc->IMTS == 0) || (calc->IMTS == 2)) DTIM=calc->DTM;
        if(calc->IMTS == 1) DTIM=2.e00*geom->CCELL[3][NCI];
        if(calc->FTIME-molecs->PTIM[N] > 0.5*DTIM){
            WFI=1.e00;
            if(geom->IWF == 1) WFI=1.e00+geom->WFM*pow(molecs->PX[1][N],geom->IFX);
            II=0; //becomes 1 if a molecule is removed
            TI=molecs->PTIM[N];
            molecs->PTIM[N]=TI+DTIM;
            calc->TOTMOV=calc->TOTMOV+1;
            //
            XI=molecs->PX[1][N];
            DX=DTIM*molecs->PV[1][N];
            X=XI+DX;
            //
            if(geom->IFX > 0){
                DY=0.e00;
                DZ=DTIM*molecs->PV[3][N];
                if(geom->IFX == 2) DY=DTIM*molecs->PV[2][N];
                R=sqrt(X*X+DY*DY+DZ*DZ);
            }
            //
            if(geom->IFX == 0){
                for(J=1;J<=2;J++){    // 1 for minimum x boundary, 2 for maximum x boundary
                    if(II == 0){
                        if(((J == 1) && (X < geom->XB[1])) || ((J == 2) && (X > (geom->XB[2]+geom->VELOB*molecs->PTIM[N])))){  //molecule crosses a boundary
                            if((geom->ITYPE[J] == 0) || (geom->ITYPE[J] == 3) || (geom->ITYPE[J] == 4)){
                                if(geom->XREM > geom->XB[1]){
                                    L=molecs->IPSP[N];
                                    calc->ENTMASS=calc->ENTMASS-gas->SP[5][L];
                                }
                                if((geom->ITYPE[2] == 4) && (calc->ICN == 1)){
                                    L=molecs->IPSP[N];
                                    calc->ALOSS[L]=calc->ALOSS[L]+1.e00;
                                }
                                REMOVE_MOL(N);
                                N=N-1;
                                II=1;
                            }
                            //
                            if(geom->ITYPE[J] == 1){
                                if((geom->IVB == 0) || (J == 1)){
                                    X=2.e00*geom->XB[J]-X;
                                    molecs->PV[1][N]=-molecs->PV[1][N];
                                }
                                else if((J == 2) && (geom->IVB == 1)){
                                    DTC=(geom->XB[2]+TI*geom->VELOB-XI)/(molecs->PV[1][N]-geom->VELOB);
                                    XC=XI+molecs->PV[1][N]*DTC;
                                    molecs->PV[1][N]=-molecs->PV[1][N]+2.*geom->VELOB;
                                    X=XC+molecs->PV[1][N]*(DTIM-DTC);
                                }
                            }
                            //
                            if(geom->ITYPE[J] == 2)
                                REFLECT_1D(N,J,X);
                            // END IF
                        }
                    }
                }
            }
            else{         //cylindrical or spherical flow
                //check boundaries
                if((X <geom-> XB[1]) && (geom->XB[1] > 0.e00)){
                    RBC(XI,DX,DY,DZ,geom->XB[1],S1);
                    if(S1 < 1.e00){     //intersection with inner boundary
                        if(geom->ITYPE[1] == 2){//solid surface
                            DX=S1*DX;
                            DY=S1*DY;
                            DZ=S1*DZ;
                            AIFX(XI,DX,DY,DZ,X,molecs->PV[1][N],molecs->PV[2][N],molecs->PV[3][N]);
                            REFLECT_1D(N,1,X);
                        }
                        else{
                            REMOVE_MOL(N);
                            N=N-1;
                            II=1;
                        }
                    }
                }
                else if((geom->IVB == 0) && (R > geom->XB[2])){
                    RBC(XI,DX,DY,DZ,geom->XB[2],S1);
                    if(S1 < 1.e00){     //intersection with outer boundary
                        if(geom->ITYPE[2] == 2){ //solid surface
                            DX=S1*DX;
                            DY=S1*DY;
                            DZ=S1*DZ;
                            AIFX(XI,DX,DY,DZ,X,molecs->PV[1][N],molecs->PV[2][N],molecs->PV[3][N]);
                            X=1.001e00*geom->XB[2];
                            while(X > geom->XB[2])
                                REFLECT_1D(N,2,X);
                            // END DO
                        }
                        else{
                            REMOVE_MOL(N);
                            N=N-1;
                            II=1;
                        }
                    }
                }
                else if((geom->IVB == 1) && (R > (geom->XB[2]+molecs->PTIM[N]*geom->VELOB))){
                    if(geom->IFX == 1) UR=sqrt(pow(molecs->PV[1][N],2)+pow(molecs->PV[2][N],2));
                    if(geom->IFX == 2) UR=sqrt(pow(molecs->PV[1][N],2)+pow(molecs->PV[2][N],2)+pow(molecs->PV[3][N],2));
                    DTC=(geom->XB[2]+TI*geom->VELOB-XI)/(UR-geom->VELOB);
                    S1=DTC/DTIM;
                    DX=S1*DX;
                    DY=S1*DY;
                    DZ=S1*DZ;
                    AIFX(XI,DX,DY,DZ,X,molecs->PV[1][N],molecs->PV[2][N],molecs->PV[3][N]);
                    molecs->PV[1][N]=-molecs->PV[1][N]+2.0*geom->VELOB;
                    X=X+molecs->PV[1][N]*(DTIM-DTC);
                }
                else
                    AIFX(XI,DX,DY,DZ,X,molecs->PV[1][N],molecs->PV[2][N],molecs->PV[3][N]);
                
                
                //DIAGNOSTIC
                if(II == 0){
                    if(X > geom->XB[2]+molecs->PTIM[N]*geom->VELOB){
                        //WRITE (*,*) N,calc->FTIME,X,geom->XB[2]+molecs->PTIM[N]*geom->VELOB;
                        cout<<N<<" "<<calc->FTIME<<" "<<X<<" "<<(geom->XB[2]+molecs->PTIM[N]*geom->VELOB)<<endl;
                    }
                }
                
                //Take action on weighting factors
                if((geom->IWF == 1) && (II == 0)){
                    WFR=WFI/(1.e00+geom->WFM*pow(X,geom->IFX));
                    L=0;
                    WFRI=WFR;
                    if(WFR >= 1.e00){
                        while(WFR >= 1.e00){
                            L=L+1;
                            WFR=WFR-1.e00;
                        }
                    }
                    // CALL RANDOM_NUMBER(RANF)
                    calc->RANF=((double)rand()/(double)RAND_MAX);
                    if(calc->RANF <= WFR) L=L+1;
                    if(L == 0){
                        REMOVE_MOL(N);
                        N=N-1;
                        II=1;
                    }
                    L=L-1;
                    if(L > 0){
                        for(K=1;K<=L;K++){
                            if(molecs->NM >= molecs->MNM) EXTEND_MNM(1.1);
                            molecs->NM=molecs->NM+1;
                            molecs->PX[1][molecs->NM]=X;
                            for(M=1;M<=3;M++)
                                molecs->PV[M][molecs->NM]=molecs->PV[M][N];
                            
                            if(gas->MMRM > 0) molecs->PROT[molecs->NM]=molecs->PROT[N];
                            molecs->IPCELL[molecs->NM]=fabs(molecs->IPCELL[N]);
                            molecs->IPSP[molecs->NM]=molecs->IPSP[N];
                            molecs->IPCP[molecs->NM]=molecs->IPCP[N];
                            if(gas->MMVM > 0){
                                for(M=1;M<=gas->MMVM;M++)
                                    molecs->IPVIB[M][molecs->NM]=molecs->IPVIB[M][N];
                                
                            }
                            molecs->PTIM[molecs->NM]=molecs->PTIM[N];    //+5.D00*DFLOAT(K)*DTM
                            //note the possibility of a variable time advance that may take the place of the duplication buffer in earlier programs
                            
                            if(molecs->PX[1][molecs->NM] > geom->XB[2]+molecs->PTIM[molecs->NM]*geom->VELOB)
                                //WRITE (*,*) 'DUP',NM,FTIME,PX(1,NM),XB(2)+PTIM(NM)*VELOB
                                cout<<"DUP "<<molecs->NM<<" "<<calc->FTIME<<" "<<molecs->PX[1][molecs->NM]<<" "<<(geom->XB[2]+molecs->PTIM[molecs->NM]*geom->VELOB)<<endl;
                            
                        }
                    }
                }
            }
            //
            if(II == 0){
                molecs->PX[1][N]=X;
                
                if(molecs->PX[1][N] > geom->XB[1] && (molecs->PX[1][N] < geom->XB[2]))
                    continue;
                else{
                    cout<< N<<" OUTSIDE FLOWFIELD AT "<<molecs->PX[1][N]<<" VEL "<<molecs->PV[1][N]<<endl;
                    REMOVE_MOL(N);
                    N=N-1;
                    II=1;
                }
            }
            //
            if(II == 0){
                if(geom->IVB == 0) FIND_CELL_1D(molecs->PX[1][N],molecs->IPCELL[N],JJ);
                if(geom->IVB == 1) FIND_CELL_MB_1D(molecs->PX[1][N],molecs->IPCELL[N],JJ,molecs->PTIM[N]);
            }
            //
        }
        //
        N=N+1;
    }
    //
    return;
}



void READ_RESTART()
{
    //MOLECS molecs;
    //GEOM_1D geom;
    //GAS gas;
    //CALC calc;
    //OUTPUT output;
    // IMPLICIT NONE
    //
    fstream file_7;
    int ZCHECK;
    //
    //    101 CONTINUE
    _101:
    file_7.open("PARAMETERS.DAT", ios::in | ios::binary);
    if(file_7.is_open()){
        cout<<"PARAMETERS.DAT opened successfully"<<endl;
        file_7>>geom->NCCELLS>>geom->NCELLS>>gas->MMRM>>gas->MMVM>>molecs->MNM>>gas->MNSR>>gas->MSP>>geom->ILEVEL>>geom->MDIV>>gas->MMEX>>gas->MEX>>gas->MELE>>gas->MVIBL>>calc->NCLASS;
        file_7.close();
    }
    else{
        cout<<"PARAMETERS.DAT not opening"<<endl;
        goto _101;
    }
    //cout<<geom->NCCELLS<<endl<<geom->NCELLS<<endl<<gas->MMRM<<endl<<gas->MMVM<<endl<<molecs->MNM<<endl;
    // OPEN (7,FILE='PARAMETERS.DAT',FORM='BINARY',ERR=101)
    // READ (7) NCCELLS,NCELLS,MMRM,MMVM,MNM,MNSR,MSP,ILEVEL,MDIV,MMEX,MEX,MELE,MVIBL,NCLASS
    // CLOSE(7)
    //
    if(gas->MMVM > 0){
        
        molecs->d_allocate(calc->NCLASS+1,molecs->MNM+1,molecs->PX);
        molecs->d_allocate(molecs->MNM+1,molecs->PTIM);
        molecs->d_allocate(molecs->MNM+1,molecs->PROT);
        molecs->i_allocate(molecs->MNM+1,molecs->IPCELL);
        molecs->i_allocate(molecs->MNM+1,molecs->IPSP);
        molecs->i_allocate(molecs->MNM+1,molecs->ICREF);
        molecs->i_allocate(molecs->MNM+1,molecs->IPCP);
        molecs->d_allocate(4,molecs->MNM+1,molecs->PV);
        molecs->i_allocate(gas->MMVM+1,molecs->MNM+1,molecs->IPVIB);
        molecs->d_allocate(molecs->MNM+1,molecs->PELE);
        // ALLOCATE (PX(NCLASS,MNM),PTIM(MNM),PROT(MNM),IPCELL(MNM),IPSP(MNM),ICREF(MNM),IPCP(MNM),PV(3,MNM),      &
        //      IPVIB(MMVM,MNM),PELE(MNM),STAT=ERROR)
    }
    else{
        if(gas->MMRM > 0){
            molecs->d_allocate(calc->NCLASS+1,molecs->MNM+1,molecs->PX);
            molecs->d_allocate(molecs->MNM+1,molecs->PTIM);
            molecs->d_allocate(molecs->MNM+1,molecs->PROT);
            molecs->i_allocate(molecs->MNM+1,molecs->IPCELL);
            molecs->i_allocate(molecs->MNM+1,molecs->IPSP);
            molecs->i_allocate(molecs->MNM+1,molecs->ICREF);
            molecs->i_allocate(molecs->MNM+1,molecs->IPCP);
            molecs->d_allocate(4,molecs->MNM+1,molecs->PV);
            molecs->d_allocate(molecs->MNM+1,molecs->PELE);
            // ALLOCATE (PX(NCLASS,MNM),PTIM(MNM),PROT(MNM),IPCELL(MNM),IPSP(MNM),ICREF(MNM),IPCP(MNM),PV(3,MNM),PELE(MNM),STAT=ERROR)
        }
        else{
            molecs->d_allocate(calc->NCLASS+1,molecs->MNM+1,molecs->PX);
            molecs->d_allocate(molecs->MNM+1,molecs->PTIM);
            molecs->i_allocate(molecs->MNM+1,molecs->IPCELL);
            molecs->i_allocate(molecs->MNM+1,molecs->IPSP);
            molecs->i_allocate(molecs->MNM+1,molecs->ICREF);
            molecs->i_allocate(molecs->MNM+1,molecs->IPCP);
            molecs->d_allocate(4,molecs->MNM+1,molecs->PV);
            molecs->d_allocate(molecs->MNM+1,molecs->PELE);
            // ALLOCATE (PX(NCLASS,MNM),PTIM(MNM),IPCELL(MNM),IPSP(MNM),ICREF(MNM),IPCP(MNM),PV(3,MNM),PELE(MNM),STAT=ERROR)
        }
        
    }
     
    // IF (ERROR /= 0) THEN
    //   WRITE (*,*) 'PROGRAM COULD NOT ALLOCATE SPACE FOR MOLECULE ARRAYS',ERROR
    // ENDIF
    //
    geom->i_allocate(geom->ILEVEL+1,geom->MDIV+1,geom->JDIV);
    // ALLOCATE (JDIV(0:ILEVEL,MDIV),STAT=ERROR)
    // IF (ERROR /= 0) THEN
    //   WRITE (*,*) 'PROGRAM COULD NOT ALLOCATE SPACE FOR JDIV ARRAY',ERROR
    // ENDIF
    geom->d_allocate(5,geom->NCELLS+1,geom->CELL);
    geom->i_allocate(geom->NCELLS+1,geom->ICELL);
    geom->d_allocate(6,geom->NCCELLS+1,geom->CCELL);
    geom->i_allocate(4,geom->NCCELLS+1,geom->ICCELL);
    // ALLOCATE (CELL(4,NCELLS),ICELL(NCELLS),CCELL(5,NCCELLS),ICCELL(3,NCCELLS),STAT=ERROR)
    // IF (ERROR /= 0) THEN
    //   WRITE (*,*) 'PROGRAM COULD NOT ALLOCATE SPACE FOR CELL ARRAYS',ERROR
    // ENDIF
    
    output->d_allocate(geom->NCELLS+1,output->COLLS);
    output->d_allocate(geom->NCELLS+1,output->WCOLLS);
    output->d_allocate(geom->NCELLS+1,output->CLSEP);
    output->d_allocate(gas->MNSR+1,output->SREAC);
    output->d_allocate(24,geom->NCELLS+1,output->VAR);
    output->d_allocate(13,geom->NCELLS+1,gas->MSP+1,output->VARSP);
    output->d_allocate(36+gas->MSP,3,output->VARS);
    output->d_allocate(10+gas->MSP,geom->NCELLS+1,gas->MSP+1,output->CS);
    output->d_allocate(9,3,gas->MSP+1,3,output->CSS);
    output->d_allocate(7,3,output->CSSS);
    // ALLOCATE (COLLS(NCELLS),WCOLLS(NCELLS),CLSEP(NCELLS),SREAC(MNSR),VAR(23,NCELLS),    &
    //           VARSP(0:12,NCELLS,MSP),VARS(0:35+MSP,2),CS(0:9+MSP,NCELLS,MSP),CSS(0:8,2,MSP,2),CSSS(6,2),STAT=ERROR)
    // IF (ERROR /= 0) THEN
    //   WRITE (*,*) 'PROGRAM COULD NOT ALLOCATE SPACE FOR SAMPLING ARRAYS',ERROR
    // ENDIF
    //

    if(gas->MMVM >= 0){
        output->d_allocate(gas->MSP+1,gas->MMVM+1,151,output->VIBFRAC);
        output->d_allocate(gas->MSP+1,gas->MMVM+1,output->SUMVIB);
        // ALLOCATE (VIBFRAC(MSP,MMVM,0:150),SUMVIB(MSP,MMVM),STAT=ERROR)
        // IF (ERROR /= 0) THEN
        //   WRITE (*,*) 'PROGRAM COULD NOT ALLOCATE SPACE FOR RECOMBINATION ARRAYS',ERROR
        // ENDIF
    }
    //
    ALLOCATE_GAS();
    //
    //102 CONTINU
_102:
    file_7.open("RESTART.DAT", ios::in | ios::binary);
    if(file_7.is_open()){
        cout<<"RESTART.DAT opened successfully"<<endl;
        /*file_7>>calc->AJM>>calc->ALOSS>>output->AVDTM>>BOLTZ>>geom->CCELL>>geom->CELL>>output->CLSEP>>output->COLLS>>calc->CPDTM>>gas->CR>>output->CS>>output->CSS>>output->CSSS>>gas->CTM>>gas->CXSS>>geom->DDIV>>DPI>>calc->DTM>>calc->DTSAMP>>calc->DTOUT>>calc->EME>>calc->ENTMASS>>gas->ENTR>>calc->ENTREM>>calc->ERROR>>gas->ERS>>gas->FDEN>>gas->FMA>>gas->FND>>calc->FNUM>>calc->FRACSAM>>gas->FSP>>gas->FP>>gas->FPM>>gas->FPR>>geom->FREM>>gas->FSPEC>>gas->FTMP>>calc->FTIME>>gas->FVTMP>>geom->ICCELL>>geom->ICELL>>calc->ICLASS>>calc->ICN>>molecs->ICREF>>geom->IFX>>gas->IGAS>>calc->IMTS>>molecs->IPCELL>>molecs->IPCP>>molecs->IPSP>>molecs->IPVIB>>calc->IREM>>calc->ISAD>>calc->ISECS>>calc->ISF>>gas->ISPEX>>gas->ISPR>>gas->ISPRC>>gas->ISPRK>>gas->ISPV>>gas->ISPVM>>gas->ISRCD>>geom->ITYPE>>geom->IVB>>geom->IWF>>geom->JDIV>>gas->LIS>>gas->LRS>>calc->MOLSC>>calc->MVER>>geom->NCCELLS>>geom->NCELLS>>geom->NCIS>>geom->NDIV>>gas->NELL>>gas->NEX>>calc->NLINE>>molecs->NM>>output->NMISAMP>>calc->NNC>>output->NOUT>>output->NSAMP>>gas->NSLEV>>gas->NSPEX>>calc->NREL>>calc->NVER>>molecs->PELE>>PI>>molecs->PROT>>molecs->PTIM>>molecs->PV>>molecs->PX>>gas->QELC>>gas->RGFS>>gas->RMAS>>gas->SLER>>gas->SP>>gas->SPEX>>SPI>>gas->SPM>>gas->SPR>>gas->SPRC>>gas->SPREX>>gas->SPRP>>gas->SPRT>>gas->SPV>>gas->SPVM>>output->SREAC>>output->SUMVIB>>calc->TCOL>>calc->TDISS>>calc->TRECOMB>>output->TISAMP>>calc->TPOUT>>calc->TREF>>calc->TLIM>>calc->TOTCOL>>calc->TOTMOV>>gas->TREACG>>gas->TREACL>>calc->TOUT>>calc->TPDTM>>calc->TREF>>calc->TSAMP>>gas->TSURF>>output->VAR>>output->VARS>>output->VARSP>>geom->VELOB>>gas->VFX>>gas->VFY>>output->VIBFRAC>>gas->VMP>>gas->VMPM>>calc->VNMAX>>gas->VSURF>>output->WCOLLS>>geom->WFM>>geom->XB>>geom->XREM>>output->XVELS>>output->YVELS>>gas->TNEX>>ZCHECK>>endl;*/
        file_7.read((char*)&calc,sizeof(calc));
        file_7.read((char*)&molecs,sizeof(molecs));
        file_7.read((char*)&gas,sizeof(gas));
        file_7.read((char*)&geom,sizeof(geom));
        file_7.read((char*)&output,sizeof(output));
        file_7.close();
    }
    else{
        cout<<"Restart.DAT not opening"<<endl;
        goto _102;
    }
    // OPEN (7,FILE='RESTART.DAT',FORM='BINARY',ERR=102)
    // READ (7) AJM,ALOSS,AVDTM,BOLTZ,CCELL,CELL,CLSEP,COLLS,    &
    //          CPDTM,CR,CS,CSS,CSSS,CTM,CXSS,DDIV,DPI,DTM,DTSAMP,DTOUT,EME,      &
    //          ENTMASS,ENTR,ENTREM,ERROR,ERS,FDEN,FMA,FND,FNUM,FRACSAM,FSP,FP,FPM,FPR,FREM,FSPEC,     &
    //          FTMP,FTIME,FVTMP,ICCELL,ICELL,ICLASS,ICN,ICREF,IFX,IGAS,IMTS,IPCELL,IPCP,     &
    //          IPSP,IPVIB,IREM,ISAD,ISECS,ISF,ISPEX,ISPR,ISPRC,ISPRK,ISPV,ISPVM,ISRCD,ITYPE,IVB,IWF,     &
    //          JDIV,LIS,LRS,MOLSC,MVER,NCCELLS,NCELLS,    &
    //          NCIS,NDIV,NELL,NEX,NLINE,NM,NMISAMP,NNC,NOUT,NSAMP,NSLEV,NSPEX,NREL,NVER,PELE,PI,PROT,PTIM,PV,PX,     &
    //          QELC,RGFS,RMAS,SLER,SP,SPEX,SPI,SPM,SPR,SPRC,SPREX,SPRP,SPRT,SPV,SPVM,SREAC,SUMVIB,    &
    //          TCOL,TDISS,TRECOMB,TISAMP,TPOUT,TREF,TLIM,TOTCOL,TOTMOV,     &
    //          TREACG,TREACL,TOUT,TPDTM,TREF,TSAMP,TSURF,VAR,VARS,VARSP,VELOB,VFX,VFY,VIBFRAC,VMP,     &
    //          VMPM,VNMAX,VSURF,WCOLLS,WFM,XB,XREM,XVELS,YVELS,TNEX,ZCHECK
    // //
    // CLOSE(7)
    //
    if(ZCHECK != 1234567){
        file_9<<molecs->NM<<" Molecules, Check integer = "<<ZCHECK<<endl;
        //WRITE (9,*) NM,' Molecules, Check integer =',ZCHECK
        return ;
    }
    else
        file_9<<"Restart file read, Check integer= "<<ZCHECK<<endl;
    //WRITE (9,*) 'Restart file read, Check integer=',ZCHECK
    
    //
    return;
    
    //
}
//*****************************************************************************

void WRITE_RESTART()
{
    //MOLECS molecs;
    //GEOM_1D geom;
    //GAS gas;
    //CALC calc;
    //OUTPUT output;
    // IMPLICIT NONE
    //
    int ZCHECK;
    //
    fstream file_7;
    ZCHECK=1234567;
    //
    //101 CONTINUE
_101:
    file_7.open("PARAMETERS.DAT", ios::out | ios::binary);
    if(file_7.is_open()){
        file_7<<geom->NCCELLS<<endl<<geom->NCELLS<<endl<<gas->MMRM<<endl<<gas->MMVM<<endl<<molecs->MNM<<endl<<gas->MNSR<<endl<<gas->MSP<<endl<<geom->ILEVEL<<endl<<geom->MDIV<<endl<<gas->MMEX<<endl<<gas->MEX<<endl<<gas->MELE<<endl<<gas->MVIBL<<endl<<calc->NCLASS<<endl;
        file_7.close();
    }
    else{
        cout<<"Parameters.DAT file not opening(write)"<<endl;
        goto _101;
    }
    // OPEN (7,FILE='PARAMETERS.DAT',FORM='BINARY',ERR=101)
    // WRITE (7) NCCELLS,NCELLS,MMRM,MMVM,MNM,MNSR,MSP,ILEVEL,MDIV,MMEX,MEX,MELE,MVIBL,NCLASS
    // CLOSE(7)
    //
    //    102 CONTINUE
_102:
    file_7.open("RESTART.DAT", ios::out | ios::binary);
    if(file_7.is_open()){
        /*file_7<<calc->AJM<<calc->ALOSS<<output->AVDTM<<BOLTZ<<geom->CCELL<<geom->CELL<<output->CLSEP<<output->COLLS<<calc->CPDTM<<gas->CR<<output->CS<<output->CSS<<output->CSSS<<gas->CTM<<gas->CXSS<<geom->DDIV<<DPI<<calc->DTM<<calc->DTSAMP<<calc->DTOUT<<calc->EME<<calc->ENTMASS<<gas->ENTR<<calc->ENTREM<<calc->ERROR<<gas->ERS<<gas->FDEN<<gas->FMA<<gas->FND<<calc->FNUM<<calc->FRACSAM<<gas->FSP<<gas->FP<<gas->FPM<<gas->FPR<<geom->FREM<<gas->FSPEC<<gas->FTMP<<calc->FTIME<<gas->FVTMP<<geom->ICCELL<<geom->ICELL<<calc->ICLASS<<calc->ICN<<molecs->ICREF<<geom->IFX<<gas->IGAS<<calc->IMTS<<molecs->IPCELL<<molecs->IPCP<<molecs->IPSP<<molecs->IPVIB<<calc->IREM<<calc->ISAD<<calc->ISECS<<calc->ISF<<gas->ISPEX<<gas->ISPR<<gas->ISPRC<<gas->ISPRK<<gas->ISPV<<gas->ISPVM<<gas->ISRCD<<geom->ITYPE<<geom->IVB<<geom->IWF<<geom->JDIV<<gas->LIS<<gas->LRS<<calc->MOLSC<<calc->MVER<<geom->NCCELLS<<geom->NCELLS<<geom->NCIS<<geom->NDIV<<gas->NELL<<gas->NEX<<calc->NLINE<<molecs->NM<<output->NMISAMP<<calc->NNC<<output->NOUT<<output->NSAMP<<gas->NSLEV<<gas->NSPEX<<calc->NREL<<calc->NVER<<molecs->PELE<<PI<<molecs->PROT<<molecs->PTIM<<molecs->PV<<molecs->PX<<gas->QELC<<gas->RGFS<<gas->RMAS<<gas->SLER<<gas->SP<<gas->SPEX<<SPI<<gas->SPM<<gas->SPR<<gas->SPRC<<gas->SPREX<<gas->SPRP<<gas->SPRT<<gas->SPV<<gas->SPVM<<output->SREAC<<output->SUMVIB<<calc->TCOL<<calc->TDISS<<calc->TRECOMB<<output->TISAMP<<calc->TPOUT<<calc->TREF<<calc->TLIM<<calc->TOTCOL<<calc->TOTMOV<<gas->TREACG<<gas->TREACL<<calc->TOUT<<calc->TPDTM<<calc->TREF<<calc->TSAMP<<gas->TSURF<<output->VAR<<output->VARS<<output->VARSP<<geom->VELOB<<gas->VFX<<gas->VFY<<output->VIBFRAC<<gas->VMP<<gas->VMPM<<calc->VNMAX<<gas->VSURF<<output->WCOLLS<<geom->WFM<<geom->XB<<geom->XREM<<output->XVELS<<output->YVELS<<gas->TNEX<<ZCHECK<<endl;*/
        file_7.write((char*)&calc,sizeof(calc));
        file_7.write((char*)&molecs,sizeof(molecs));
        file_7.write((char*)&gas,sizeof(gas));
        file_7.write((char*)&geom,sizeof(geom));
        file_7.write((char*)&output,sizeof(output));
        file_7.close();
    }
    else{
        cout<<"Restart.DAT file not opening(write)"<<endl;
        goto _101;
    }
    // OPEN (7,FILE='RESTART.DAT',FORM='BINARY',ERR=102)
    // WRITE (7)AJM,ALOSS,AVDTM,BOLTZ,CCELL,CELL,CLSEP,COLLS,    &
    //          CPDTM,CR,CS,CSS,CSSS,CTM,CXSS,DDIV,DPI,DTM,DTSAMP,DTOUT,EME,      &
    //          ENTMASS,ENTR,ENTREM,ERROR,ERS,FDEN,FMA,FND,FNUM,FRACSAM,FSP,FP,FPM,FPR,FREM,FSPEC,     &
    //          FTMP,FTIME,FVTMP,ICCELL,ICELL,ICLASS,ICN,ICREF,IFX,IGAS,IMTS,IPCELL,IPCP,     &
    //          IPSP,IPVIB,IREM,ISAD,ISECS,ISF,ISPEX,ISPR,ISPRC,ISPRK,ISPV,ISPVM,ISRCD,ITYPE,IVB,IWF,     &
    //          JDIV,LIS,LRS,MOLSC,MVER,NCCELLS,NCELLS,    &
    //          NCIS,NDIV,NELL,NEX,NLINE,NM,NMISAMP,NNC,NOUT,NSAMP,NSLEV,NSPEX,NREL,NVER,PELE,PI,PROT,PTIM,PV,PX,     &
    //          QELC,RGFS,RMAS,SLER,SP,SPEX,SPI,SPM,SPR,SPRC,SPREX,SPRP,SPRT,SPV,SPVM,SREAC,SUMVIB,    &
    //          TCOL,TDISS,TRECOMB,TISAMP,TPOUT,TREF,TLIM,TOTCOL,TOTMOV,     &
    //          TREACG,TREACL,TOUT,TPDTM,TREF,TSAMP,TSURF,VAR,VARS,VARSP,VELOB,VFX,VFY,VIBFRAC,VMP,     &
    //          VMPM,VNMAX,VSURF,WCOLLS,WFM,XB,XREM,XVELS,YVELS,TNEX,ZCHECK
    // //
    // CLOSE(7)
    //
    file_9<<"Restart files written"<<endl;
    //WRITE (9,*) 'Restart files written'
    //
    return;
}

void OUTPUT_RESULTS()
{
    //--calculate the surface and flowfield properties
    //--generate TECPLOT files for displaying these properties
    //--calculate collisiion rates and flow transit times and reset time intervals
    //--add molecules to any flow plane molecule output files
    //CALC calc;
    //MOLECS molecs;
    //GAS gas;
    //OUTPUT output;
    //GEOM_1D geom;
    
    fstream file_3;
    fstream file_10;
    fstream file_7;
    
    int IJ,J,JJ,K,L,LL,M,N,NN,NMCR,CTIME,II;
    long long NNN;
    double AS,AT,C1,C2,C3,C4,C5,C6,C7,C8,C9;
    double A,B,C,SDTM,SMCR,DOF,AVW,UU,VDOFM,TVIBM,VEL,DTMI,TT;
    //dout
    double SUM[14];
    double SUMS[10][3];
    double *TVIB,*VDOF,*PPA,*TEL,*ELDOF,*SDOF,*CDTM;
    double **TV,**THCOL;
    double ***DF;
    int *NMS;
    //    REAL(KIND=8), ALLOCATABLE, DIMENSION(:) :: TVIB,VDOF,PPA,TEL,ELDOF,SDOF,CDTM
    //    REAL(KIND=8), ALLOCATABLE, DIMENSION(:,:) :: TV,THCOL
    //    REAL(KIND=8), ALLOCATABLE, DIMENSION(:,:,:) :: DF
    //    INTEGER, ALLOCATABLE, DIMENSION(:) :: NMS
    //INTEGER, ALLOCATABLE, DIMENSION(:,:) ::
    string F,E;
    //--CTIME  computer time (microseconds)
    //--SUMS(N,L) sum over species of CSS(N,J,L,M) for surface properties
    //
    //--For flowfield properties,where <> indicates sampled sum
    //--SUM(0) the molecular number sum over all species
    //--SUM(1) the weighted number sum over all species
    //--SUM(2) the weighted sum of molecular masses
    //--SUM(3),(4),(5) the weighted sum over species of m*<u>,<v>,<w>
    //--SUM(6) the weighted sum over species of m*(<u**2>+<v**2>+<w**2>)
    //--SUM(7) the weighted sum over species of <u**2>+<v**2>+<w**2>
    //--SUM(8) the weighted sum of rotational energy
    //--SUM(9) the weighted sum of rotational degrees of freedom
    //--SUM(10) the weighted sum over species of m*<u**2>
    //--SUM(11) the weighted sum over species of m*<v**2>
    //--SUM(12) sum over species of m*<w**2>
    //--SUM(13) the weighted sum of electronic energy
    //--UU velocity squared
    //--DOF degrees of freedom
    //--AVW the average value of the viscosity-temperature exponent
    //--DVEL velocity difference
    //--TVEL thermal speed
    //--SMCR sum of mcs/mfp over cells
    //--NMCR number in the sum
    //--VDOFM effective vibrational degrees of freedom of mixture
    //--TVIB(L)
    //--VDOF(L)
    //--TV(K,L) the temperature of vibrational mode K of species L
    //--PPA particles per atom
    //--NMS number per species
    //--SDOF(L) total degrees of freedom for species L
    //
    //
    //--calculate the flowfield properties in the cells
    //dout
    
    
    TV = new double*[gas->MMVM+1];
    for(int i =0; i< gas->MMVM+1; ++i)
        TV[i] = new double[gas->MSP+1];
    
    TVIB = new double[gas->MSP+1];
    
    DF = new double **[geom->NCELLS+1];
    for (int i = 0; i < geom->NCELLS+1; ++i)
    {
        DF[i] = new double *[gas->MMVM+1];
        for (int j = 0; j < gas->MMVM+1; ++j)
            DF[i][j] = new double [gas->MSP+1];
    }
    
    VDOF= new double[gas->MSP+1];
    
    TEL = new double[gas->MSP+1];
    
    ELDOF = new double[gas->MSP+1];
    
    PPA = new double[gas->MSP+1];
    
    NMS = new int[gas->MSP+1];
    
    THCOL = new double*[gas->MSP+1];
    for(int i =0; i< gas->MSP+1; ++i)
        THCOL[i] = new double[gas->MSP+1];
    
    SDOF = new double[gas->MSP+1];
    
    CDTM = new double[geom->NCELLS+1];
    
    
    //    ALLOCATE (TV(MMVM,MSP),TVIB(MSP),DF(NCELLS,MMVM,MSP),VDOF(MSP),TEL(MSP),ELDOF(MSP),PPA(MSP),NMS(MSP),THCOL(MSP,MSP)    &
    //              ,SDOF(MSP),CDTM(NCELLS),STAT=ERROR)
    //    if(calc->ERROR!=0)
    //    {
    //        cout<<"ROGRAM COULD NOT ALLOCATE OUTPUT VARIABLES"<<calc->ERROR<<endl;
    //    }
    if(calc->FTIME>0.5e00*calc->DTM)
    {
        output->NOUT+=1;
        if(output->NOUT>9999)
            output->NOUT=output->NOUT-9999;
        cout<<"Generating files for output interval"<<output->NOUT<<endl;
        if(calc->ISF==0)
        {
            //dout
            //OPEN (3,FILE='DS1OUT.DAT')
            file_3.open("DS1OUT.DAT" , ios::out);
            if(file_3.is_open()){
                cout<<"DS1OUT.DAT is opened"<<endl;
            }
            else{
                cout<<"DS1OUT.DAT not opened"<<endl;
            }
            //F='DS';//E//'.OUT'
        }
        else
        {
            //--the files are DS1n.DAT, where n is a four digit integer equal to NOUT
            //dout
            //500 FORMAT(I5)
            //ENCODE(5,500,E) 10000+NOUT
            int a=output->NOUT+10000;
            E=to_string(a);
            F="DS" + E + "OUT.DAT";
            //dout
            file_3.open(F.c_str(), ios::out);
            if(file_3.is_open()){
                cout<<F<<" is opened"<<endl;
            }
            else{
                cout<<F<<" not opened"<<endl;
            }
            //OPEN (3,FILE=F)
        }
    }
    //dout
    //memset(output->VAR,0.e00,sizeof(**output->VAR));
    for(int i=0;i<24;i++){
        for(int j=0;j<geom->NCELLS+1;j++)
            output->VAR[i][j]=0.e00;
    }
    if(geom->IFX==0)
        A=calc->FNUM/(calc->FTIME-output->TISAMP);
    for(JJ=1;JJ<=2;JJ++)
    {
        if(geom->IFX==1)
            A=calc->FNUM/(2.e00*PI*geom->XB[JJ])*(calc->FTIME-output->TISAMP);
        if(geom->IFX==2)
            A=calc->FNUM/(4.e00*PI*geom->XB[JJ])*geom->XB[JJ]*(calc->FTIME-output->TISAMP);
        //--JJ=1 for surface at XB(1), JJ=2 for surface at XB(2)
        if(geom->ITYPE[JJ]==2)
        {
            //dout
            //memset(SUMS,0.e00,sizeof(SUMS));
            for(int i=0;i<10;i++){
                for(int j=0;j<3;j++)
                    SUMS[i][j]=0.e00;
            }
            for( L=1;L<=gas->MSP;L++)
            {
                for(J=0;J<=8;J++)
                {
                    for(IJ=1;IJ<=2;IJ++)
                    {
                        SUMS[J][IJ]=SUMS[J][IJ]+output->CSS[J][JJ][L][IJ];
                    }
                }
            }
            output->VARS[0][JJ]=SUMS[0][1];
            output->VARS[1][JJ]=SUMS[1][1];
            output->VARS[2][JJ]=SUMS[1][2];
            output->VARS[3][JJ]=SUMS[1][1]*A;
            output->VARS[4][JJ]=SUMS[1][2]*A;
            output->VARS[5][JJ]=SUMS[2][1]*A;
            output->VARS[6][JJ]=SUMS[2][2]*A;
            output->VARS[7][JJ]=SUMS[3][1]*A;
            output->VARS[8][JJ]=SUMS[3][2]*A;
            output->VARS[9][JJ]=SUMS[4][1]*A;
            output->VARS[10][JJ]=SUMS[4][2]*A;
            output->VARS[11][JJ]=SUMS[5][1]*A;
            output->VARS[12][JJ]=SUMS[5][2]*A;
            output->VARS[13][JJ]=SUMS[6][1]*A;
            output->VARS[14][JJ]=SUMS[6][2]*A;
            output->VARS[15][JJ]=SUMS[7][1]*A;
            output->VARS[16][JJ]=SUMS[7][2]*A;
            output->VARS[33][JJ]=SUMS[8][1]*A;
            output->VARS[34][JJ]=SUMS[8][2]*A;
            //   VARS(17,JJ)=SUMS(9,1)*A        //--SURFACE REACTIONS NOT YET IMPLEMENTED
            //   VARS(18,JJ)=SUMS(9,2)*A
            if(output->CSSS[1][JJ]>1.e-6)
            {
                output->VARS[19][JJ]=output->CSSS[3][JJ]/output->CSSS[2][JJ]; ////--n.b. must be modified to include second component in 3D
                output->VARS[20][JJ]=(output->CSSS[4][JJ]-output->CSSS[2][JJ]*output->VARS[19][JJ]*output->VARS[19][JJ])/(output->CSSS[1][JJ]*3.e00*BOLTZ)-gas->TSURF[JJ];
                output->VARS[19][JJ]=output->VARS[19][JJ]-gas->VSURF[JJ];
                if(output->CSSS[6][JJ]>1.e-6)
                {
                    output->VARS[21][JJ]=(2.e000/BOLTZ)*(output->CSSS[5][JJ]/output->CSSS[6][JJ])-gas->TSURF[JJ];
                }
                else
                {
                    output->VARS[21][JJ]=0.e00;
                }
            }
            else
            {
                output->VARS[19][JJ]=0.e00;
                output->VARS[20][JJ]=0.e00;
                output->VARS[21][JJ]=0.e00;
            }
            output->VARS[22][JJ]=(SUMS[2][1]+SUMS[2][2])*A;
            output->VARS[23][JJ]=(SUMS[3][1]+SUMS[3][2])*A;
            output->VARS[24][JJ]=(SUMS[4][1]+SUMS[4][2])*A;
            output->VARS[25][JJ]=(SUMS[5][1]+SUMS[5][2])*A;
            output->VARS[26][JJ]=(SUMS[6][1]+SUMS[6][2])*A;
            output->VARS[27][JJ]=(SUMS[7][1]+SUMS[7][2])*A;
            output->VARS[28][JJ]=(SUMS[9][1]+SUMS[9][2])*A;
            output->VARS[29][JJ]=output->VARS[11][JJ]+output->VARS[13][JJ]+output->VARS[15][JJ]+output->VARS[33][JJ];
            output->VARS[30][JJ]=output->VARS[12][JJ]+output->VARS[14][JJ]+output->VARS[16][JJ]+output->VARS[34][JJ];
            output->VARS[31][JJ]=output->VARS[29][JJ]+output->VARS[30][JJ];
            output->VARS[35][JJ]=output->VARS[33][JJ]+output->VARS[34][JJ];
            for(L=1;gas->MSP;L++)
            {
                if(SUMS[1][1]>0)
                {
                    output->VARS[35+L][JJ]=100*output->CSS[1][JJ][L][1]/SUMS[1][1];
                }
                else
                {
                    output->VARS[35+L][JJ]=0.0;
                }
            }
        }
    }
    //output->VARSP=0;
    for(int i=0;i<13;i++){
        for(int j=0;j<geom->NCELLS+1;j++){
            for(int k=0;k<gas->MSP+1;k++)
                output->VARSP[i][j][k]=0;
        }
    }
    SMCR=0;
    NMCR=0;
    for(N=1;N<=geom->NCELLS;N++)
    {
        if(N==120)
        {
            continue;
        }
        A=calc->FNUM/(geom->CELL[4][N])*output->NSAMP;
        if(geom->IVB==1)
            A=A*pow((geom->XB[2]-geom->XB[1])/(geom->XB[2]+geom->VELOB*0.5e00*(calc->FTIME-output->TISAMP)-geom->XB[1]),geom->IFX+1);
        //--check the above for non-zero XB(1)
        //dout
        //memset(SUM,0,sizeof(SUM));
        for(int i=0;i<14;i++)
            SUM[i]=0;
        
        NMCR+=1;
        for(L=1;L<=gas->MSP;L++)
        {
            SUM[0]=SUM[0]+output->CS[0][N][L];
            SUM[1]=SUM[1]+output->CS[1][N][L];
            SUM[2]=SUM[2]+gas->SP[5][L]*output->CS[0][N][L];
            for(K=1;K<=3;K++)
            {
                SUM[K+2]=SUM[K+2]+gas->SP[5][L]*output->CS[K+1][N][L];
                if(output->CS[1][N][L]>1.1e00)
                {
                    output->VARSP[K+1][N][L]=output->CS[K+4][N][L]/output->CS[1][N][L];
                    //--VARSP(2,3,4 are temporarily the mean of the squares of the velocities
                    output->VARSP[K+8][N][L]=output->CS[K+1][N][L]/output->CS[1][N][L];
                }
            }
            SUM[6]=SUM[6]+gas->SP[5][L]*(output->CS[5][N][L]+output->CS[6][N][L]+output->CS[7][N][L]);
            SUM[10]=SUM[10]+gas->SP[5][L]*output->CS[5][N][L];
            SUM[12]=SUM[11]+gas->SP[5][L]*output->CS[6][N][L];
            SUM[12]=SUM[12]+gas->SP[5][L]*output->CS[7][N][L];
            SUM[13]=SUM[13]+output->CS[9][N][L];
            if(output->CS[1][N][L]>0.5e00)
                SUM[7]=SUM[7]+output->CS[5][N][L]+output->CS[6][N][L]+output->CS[7][N][L];
            if(gas->ISPR[1][L]>0)
            {
                SUM[8]=SUM[8]+output->CS[8][N][L];
                SUM[9]=SUM[9]+output->CS[1][N][L]*gas->ISPR[1][L];
            }
        }
        AVW=0;
        for(L=1;L<=gas->MSP;L++)
        {
            output->VARSP[0][N][L]=output->CS[1][N][L];
            output->VARSP[1][N][L]=0.e00;
            output->VARSP[6][N][L]=0.0;
            output->VARSP[7][N][L]=0.0;
            output->VARSP[8][N][L]=0.0;
            if(SUM[1]>0.1)
            {
                output->VARSP[1][N][L]=output->CS[1][N][L]/SUM[1];
                AVW=AVW+gas->SP[3][L]*output->CS[1][N][L]/SUM[1];
                if(gas->ISPR[1][L]>0 && output->CS[1][N][L]>0.5)
                    output->VARSP[6][N][L]=(2.e00/BOLTZ)*output->CS[8][N][L]/((double)(gas->ISPR[1][L])*output->CS[1][N][L]);
            }
            output->VARSP[5][N][L]=0;
            for(K=1;K<=3;K++)
            {
                output->VARSP[K+1][N][L]=(gas->SP[5][L]/BOLTZ)*(output->VARSP[K+1][N][L]-pow(output->VARSP[K+8][N][L],2));
                output->VARSP[5][N][L]=output->VARSP[5][N][L]+output->VARSP[K+1][N][L];
            }
            output->VARSP[5][N][L]=output->VARSP[5][N][L]/3.e00;
            output->VARSP[8][N][L]=(3.e00*output->VARSP[5][N][L]+(double)gas->ISPR[1][L]*output->VARSP[6][N][L])/(3.e00+(double)(gas->ISPR[1][L]));
        }
        if(geom->IVB==0)
            output->VAR[1][N]=geom->CELL[1][N];
        if(geom->IVB==1)
        {
            C=(geom->XB[2]+geom->VELOB*calc->FTIME-geom->XB[1])/(double)(geom->NDIV); //new DDIV
            output->VAR[1][N]=geom->XB[1]+((double)(N-1)+0.5)*C;
        }
        output->VAR[2][N]=SUM[0];
        if(SUM[1]>0.5)
        {
            output->VAR[3][N]=SUM[1]*A;//--number density Eqn. (4.28)
            output->VAR[4][N]=output->VAR[3][N]*SUM[2]/SUM[1]; //--density  Eqn. (4.29)
            output->VAR[5][N]=SUM[3]/SUM[2];//--u velocity component  Eqn. (4.30)
            output->VAR[6][N]=SUM[4]/SUM[2]; //--v velocity component  Eqn. (4.30)
            output->VAR[7][N]=SUM[5]/SUM[2]; //--w velocity component  Eqn. (4.30)
            UU= pow(output->VAR[5][N],2)+pow(output->VAR[6][N],2)+pow(output->VAR[7][N],2);
            if(SUM[1]>1)
            {   
                output->VAR[8][N]=(fabs(SUM[6]-SUM[2]*UU))/(3.e00*BOLTZ*SUM[1]); //Eqn. (4.39)
                //--translational temperature
                output->VAR[19][N]=fabs(SUM[10]-SUM[2]*pow(output->VAR[5][N],2))/(BOLTZ*SUM[1]);
                output->VAR[20][N]=fabs(SUM[11]-SUM[2]*pow(output->VAR[6][N],2))/(BOLTZ*SUM[1]);
                output->VAR[21][N]=fabs(SUM[12]-SUM[2]*pow(output->VAR[7][N],2))/(BOLTZ*SUM[1]);
            }
            else
            {
                output->VAR[8][N]=1.0;
                output->VAR[19][N]=1.0;
                output->VAR[20][N]=1.0;
                output->VAR[21][N]=1.0;
            }
            if(SUM[9]>0.1e00)
            {
                output->VAR[9][N]=(2.e00/BOLTZ)*SUM[8]/SUM[9]; ////--rotational temperature Eqn. (4.36)
            }
            else
                output->VAR[9][N]=0.0;
            
            output->VAR[10][N]=gas->FTMP[1]; ////vibration default
            DOF=(3.e00+SUM[9])/SUM[1];
            output->VAR[11][N]=(3.0*output->VAR[8][N]+(SUM[9]/SUM[1]))*output->VAR[9][N]/DOF;
            //--overall temperature based on translation and rotation
            output->VAR[18][N]=output->VAR[3][N]*BOLTZ*output->VAR[8][N];
            //--scalar pressure (now (from V3) based on the translational temperature)
            if(gas->MMVM>0)
            {
                for(L=1;L<=gas->MSP;L++)
                {
                    VDOF[L]=0.0;
                    //dout
                    if(gas->ISPV[L] > 0)
                    {
                        for(K=1;K<=gas->ISPV[L];K++)
                        {
                            if(output->CS[K+9][N][L]<BOLTZ)
                            {
                                TV[K][L]=0.0;
                                DF[N][K][L]=0.0;
                            }
                            else
                            {
                                TV[K][L]=gas->SPVM[1][K][L]/log(1.0+output->CS[1][N][L]/output->CS[K+9][N][L]) ;//--Eqn.(4.45)
                                DF[N][K][L]=2.0*(output->CS[K+9][N][L]/output->CS[1][N][L])*log(1.0+output->CS[1][N][L]/output->CS[K+9][N][L]); //--Eqn. (4.46)
                            }
                            VDOF[L]=VDOF[L]+DF[N][K][L];
                        }
                        //memset(TVIB,0.0,sizeof(*TVIB));
                        for(int i=0;i<gas->MSP+1;i++)
                            TVIB[i]=0.0;
                        
                        for(K=1;K<=gas->ISPV[L];K++)
                        {
                            if(VDOF[L]>1.e-6)
                            {
                                TVIB[L]=TVIB[L]+TV[K][L]*DF[N][K][L]/VDOF[L];
                            }
                            else
                                TVIB[L]=gas->FVTMP[1];
                        }
                    }
                    else
                    {
                        TVIB[L]=calc->TREF;
                        VDOF[L]=0.0;
                    }
                    output->VARSP[7][N][L]=TVIB[L];
                }
                VDOFM=0.0;
                TVIBM=0.0;
                A=0.e00;
                for(L=1;L<=gas->MSP;L++)
                {
                    //dout
                    if(gas->ISPV[L] > 0)
                    {
                        A=A+output->CS[1][N][L];
                    }
                }
                for(L=1;L<=gas->MSP;L++)
                {
                    //dout
                    if(gas->ISPV[L] > 0)
                    {
                        VDOFM=VDOFM+VDOF[L]-output->CS[1][N][L]/A;
                        TVIBM=TVIBM+TVIB[L]-output->CS[1][N][L]/A;
                    }
                }
                output->VAR[10][N]=TVIBM;
            }
            for(L=1;L<=gas->MSP;L++)
            {
                if(output->VARSP[0][N][L]>0.5)
                {
                    //--convert the species velocity components to diffusion velocities
                    for(K=1;K<=3;K++)
                    {
                        output->VARSP[K+8][N][L]=output->VARSP[K+8][N][L]-output->VAR[K+4][N];
                    }
                    if(gas->MELE>1)
                    {
                        //--calculate the electronic temperatures for the species
                        //memset(ELDOF,0.e00,sizeof(*ELDOF));
                        for(int i=0;i<gas->MSP+1;i++)
                            ELDOF[i] = 0.e00;
                        //dout
                        //memset(TEL,0.e00,sizeof(*TEL));
                        for(int i=0;i<gas->MSP+1;i++)
                            TEL[i] = 0.e00;
                        if(gas->MELE>1)
                        {
                            A=0.e00;
                            B=0.e00;
                            for(M=1;M<=gas->NELL[L];M++)
                            {
                                if(output->VARSP[5][N][L]>1.e00)
                                {
                                    C=gas->QELC[2][M][L]/output->VARSP[5][N][L];
                                    A=A+gas->QELC[1][M][L]*exp(-C);
                                    B=B+gas->QELC[1][M][L]*C*exp(-C);
                                }
                            }
                            if(B>1.e-10)
                            {
                                TEL[L]=output->CS[9][N][L]/output->CS[1][N][L]/(BOLTZ*B/A);
                            }
                            else
                                TEL[L]=output->VAR[11][N];
                            output->VARSP[12][N][L]=TEL[L];
                            ELDOF[L]=0.e00;
                            if(output->VARSP[5][N][L]>1.e00)
                                ELDOF[L]=2.e00*output->CS[9][N][L]/output->CS[1][N][L]/(BOLTZ*output->VARSP[5][N][L]);
                            if(ELDOF[L]<0.01)
                            {
                                output->VARSP[12][N][L]=output->VAR[11][N];
                            }
                        }
                        else
                        {
                            ELDOF[L]=0.0;
                        }
                    }
                }
                else
                {
                    for(K=8;K<=12;K++)
                    {
                        output->VARSP[K][N][L]=0.e00;
                    }
                }
            }
            //--set the overall electronic temperature
            if(gas->MELE>1)
            {
                C=0.e00;
                for(L=1;L<=gas->MSP;L++)
                {
                    if(ELDOF[L]>1.e-5)
                        C=C+output->CS[1][N][L];
                }
                if(C>0.e00)
                {
                    A=0.e00;
                    B=0.e00;
                    for(L=1;L<=gas->MSP;L++)
                    {
                        if(ELDOF[L]>1.e-5)
                        {
                            A=A+output->VARSP[12][N][L]*output->CS[1][N][L];
                            B=B+output->CS[1][N][L];
                        }
                    }
                    output->VAR[22][N]=A/B;
                }
                else{
                    output->VAR[22][N]=output->VAR[11][N];
                }
            }
            else{
                output->VAR[22][N]=gas->FTMP[1];
            }
            if(gas->MMVM>0)
            {
                //--set the overall temperature and degrees of freedom for the individual species
                for(L=1;L<=gas->MSP;L++)
                {
                    if(gas->MELE>1){
                        SDOF[L]=3.e00+gas->ISPR[1][L]+VDOF[L]+ELDOF[L];
                        output->VARSP[8][N][L]=(3.0*output->VARSP[5][N][L]+gas->ISPR[1][L]*output->VARSP[6][N][L]+VDOF[L]*output->VARSP[7][N][L]+ELDOF[L]*output->VARSP[12][N][L])/SDOF[L];
                    }
                    else{
                        SDOF[L]=3.e00+gas->ISPR[1][L]+VDOF[L]+ELDOF[L];
                        output->VARSP[8][N][L]=(3.0*output->VARSP[5][N][L]+gas->ISPR[1][L]*output->VARSP[6][N][L]+VDOF[L]*output->VARSP[7][N][L])/SDOF[L];
                    }
                }
                //--the overall species temperature now includes vibrational and electronic excitation
                //--the overall gas temperature can now be set
                A=0.e00;
                B=0.e00;
                for(L=1;L<=gas->MSP;L++)
                {
                    A=A+SDOF[L]+output->VARSP[8][N][L]*output->CS[1][N][L];
                    B=B+SDOF[L]*output->CS[1][N][L];
                }
                output->VAR[11][N]=A/B;
            }
            VEL=sqrt(pow(output->VAR[5][N],2)+pow(output->VAR[6][N],2)+pow(output->VAR[7][N],2));
            output->VAR[12][N]=VEL/sqrt((DOF+2.e00)*output->VAR[11][N]*(SUM[1]*BOLTZ/SUM[2]))/DOF;
            //--Mach number
            output->VAR[13][N]=SUM[0]/output->NSAMP; ////--average number of molecules in cell
            //dout
            if(output->COLLS[N] > 2.0)
            {
                output->VAR[14][N]=0.5e00*(calc->FTIME-output->TISAMP)*(SUM[1]/output->NSAMP)/output->WCOLLS[N];
                //--mean collision time
                output->VAR[15][N]=0.92132e00*sqrt(fabs(SUM[7]/SUM[1]-UU))*output->VAR[14][N];
                //--mean free path (based on r.m.s speed with correction factor based on equilibrium)
                output->VAR[16][N]=output->CLSEP[N]/(output->COLLS[N]*output->VAR[15][N]);
            }
            else{
                output->VAR[14][N]=1.e10;
                output->VAR[15][N]=1.e10/output->VAR[3][N];
                //--m.f.p set by nominal values
            }
        }
        else
        {
            for(L=3;L<=22;L++)
            {
                output->VAR[L][N]=0.0;
            }
        }
        output->VAR[17][N]=VEL;
    }
    if(calc->FTIME>0.e00*calc->DTM)
    {
        if(calc->ICLASS==1){
            if(geom->IFX==0)
                file_3<<"DSMC program for a one-dimensional plane flow"<<endl;//WRITE (3,*) 'DSMC program for a one-dimensional plane flow';
            if(geom->IFX==1)
                file_3<<"DSMC program for a cylindrical flow"<<endl;//WRITE (3,*) 'DSMC program for a one-dimensional plane flow';
            if(geom->IFX==2)
                file_3<<"DSMC program for a spherical flow"<<endl;//WRITE (3,*) 'DSMC program for a one-dimensional plane flow';
        }
        file_3<<endl;//WRITE (3,*)
        file_3<<"Interval "<<output->NOUT<<" Time "<<calc->FTIME<< " with "<<output->NSAMP<<" samples from "<<output->TISAMP<<endl;
        //WRITE (3,*) 'Interval',output->NOUT,'Time ',calc->FTIME, ' with',output->NSAMP,' samples from',output->TISAMP
        //990 FORMAT(I7,G13.5,I7,G13.5)
        //Dout
        NNN=calc->TOTMOV;
        cout<<"TOTAL MOLECULES = "<< molecs->NM<<endl;
        //dout
        //NMS=0;
        for(int i=0;i<gas->MSP+1;i++)
            NMS[i]=0;

        for(N=1;N<=molecs->NM;N++)
        {
            M=molecs->IPSP[N];
            NMS[M]+=1;
        }
        file_3<<"Total simulated molecules = "<<molecs->NM<<endl;
        for(N=1;N<=gas->MSP;N++)
        {
            cout<< " SPECIES "<<N<<" TOTAL = "<<NMS[N]<<endl;
            file_3<<"Species "<<N<<" total = "<<NMS[N]<<endl;
        }
        if(gas->MEX>0)
        {
            ENERGY(0,A);
            for(N=1;N<=gas->MSP;N++)
            {
                if(gas->ISPV[N]>0){
                    file_9<< "SP "<<N<<" DISSOCS "<<calc->TDISS[N]<<" RECOMBS "<<calc->TRECOMB[N]<<endl;
                    cout<<"SP"<<N<<"DISSOCS"<<calc->TDISS[N]<<" RECOMBS "<<calc->TRECOMB[N]<<endl;
                    file_3<<"SP "<<N<<" DISSOCS "<<calc->TDISS[N]<<" RECOMBS "<<calc->TRECOMB[N]<<endl;
                }
            }
            for(N=1;N<=gas->MSP;N++)
            {
                cout<<"EX,C reaction"<<N<<" number"<<gas->TNEX[N]<<endl;
                file_9<<"EX,C reaction "<<N<<" number "<<gas->TNEX[N]<<endl;
                file_3<<"EX,C reaction "<<N<<" number "<<gas->TNEX[N]<<endl;
                
            }
        }
        
        file_3<<"Total molecule moves   = "<<NNN<<endl;
        //dout
        NNN=calc->TOTCOL;
        file_3<<"Total collision events = "<<NNN<<endl;
        //
        file_3<<"Species dependent collision numbers in current sample"<<endl;
        for(N=1;N<=gas->MSP;N++)
        {
            if(gas->IGAS!=8){
                for(M=1;M<=gas->MSP;M++)
                    file_3<<calc->TCOL[N][M]<<"\t";
                file_3<<endl;
                //WRITE(3,901) (calc->TCOL[N][M],M=1,gas->MSP);
            }
            if(gas->IGAS==8){
                for(M=1;M<=gas->MSP;M++)
                    file_3<<calc->TCOL[N][M]<<"\t";
                file_3<<endl;
                // WRITE(3,902) (calc->TCOL[N][M],M=1,gas->MSP);
            }
        }
        //Dout
        //901 FORMAT(5G13.5)
        //902 FORMAT(8G13.5)
        //dout
        CTIME=clock();
        file_3<<"Computation time "<<(double)CTIME/1000.0<< "seconds"<<endl;
        file_3<<"Collision events per second "<<(calc->TOTCOL-calc->TOTCOLI)*1000.e00/(double)CTIME<<endl;
        file_3<<"Molecule moves per secon "<<(calc->TOTMOV-calc->TOTMOVI)*1000.e00/(double)CTIME<<endl;
        if(calc->ICLASS==0&& gas->MMVM==0&&calc->ISF==0){
            //--a homogeneous gas with no vibratioal modes - assume that it is a collision test run
            //******PRODUCES DATA FOR TABLES 6.1 AND 6.2 IN SECTION 6.2*******
            //
            A=0.e00;
            B=0.e00;
            C=0.e00;
            for(N=1;N<=geom->NCCELLS;N++)
            {
                A+=geom->CCELL[5][N];
                B+=geom->CCELL[4][N];
                C+=geom->CCELL[3][N];
            }
            file_3<<"Overall time step "<<calc->DTM<<endl;
            file_3<<"Molecules per collision cell "<<(double)(molecs->NM)/(double)(geom->NCCELLS)<<endl;
            file_3<<"Mean cell time ratio "<< A/((double)(geom->NCCELLS)*calc->FTIME)<<endl;
            file_3<<"Mean value of cross-section and relative speed "<<B/(double)(geom->NCCELLS)<<endl;
            file_3<<"Mean half collision cell time step "<<C/(double)(geom->NCCELLS)<<endl;
            if(gas->MSP==1){
                A=2.e00*SPI*output->VAR[3][1]  *(pow(gas->SP[1][1],2))*sqrt(4.e00*BOLTZ*gas->SP[2][1]/gas->SP[5][1])*pow((output->VAR[11][1])/gas->SP[2][1],(1.e00-gas->SP[3][1]));
                //--Eqn. (2.33) for equilibhrium collision rate
                file_3<<"Coll. rate ratio to equilib "<<calc->TCOL[1][1]/((double)(molecs->NM)*(calc->FTIME-output->TISAMP))/A<<endl;
            }
            else{
                file_3<<"Species collision rate ratios to equilibrium"<<endl;
                for(N=1;N<=gas->MSP;N++){
                    file_3<<"Collision rate for species "<<N<<endl;
                    for(M=1;M<=gas->MSP;M++)
                    {
                        THCOL[N][M]=2.e00*(1.e00/SPI)*output->VAR[3][1]*output->VARSP[1][1][M]*gas->SPM[2][N][M]*sqrt(2.e00*BOLTZ*gas->SPM[5][N][M]/gas->SPM[1][N][M])*pow(output->VAR[11][1]/gas->SPM[5][N][M],1.e00-gas->SPM[3][N][M]);
                        //--Eqn. (2.36) for equilibhrium collision rate of species N with species M
                        file_3<<"with species "<<M<<" "<<calc->TCOL[N][M]/((double)(molecs->NM)*gas->FSP[N][1]*(calc->FTIME-output->TISAMP))/THCOL[N][M]<<endl;
                    }
                }
                file_3<<endl;
                for(N=1;N<=gas->MSP;N++){
                    file_3<<"Collision numbers for species "<<N<<endl;
                    for(M=1;M<=gas->MSP;M++){
                        file_3<<"with species "<<M<<" "<<calc->TCOL[N][M]<<endl;
                    }
                }
            }
        }
        file_3<<endl;
        if(geom->ITYPE[1]==2|| geom->ITYPE[2]==1)
            file_3<<"Surface quantities"<<endl;
        for(JJ=1;JJ<=2;JJ++)
        {
            if(geom->ITYPE[JJ]==2){
                file_3<<endl;
                file_3<<"Surface at "<<geom->XB[JJ]<<endl;
                file_3<<"Incident sample "<<output->VARS[0][JJ]<<endl;
                file_3<<"Number flux "<<output->VARS[3][JJ]<<" /sq m/s"<<endl;
                file_3<<"Inc pressure "<<output->VARS[5][JJ]<<" Refl pressure "<<output->VARS[6][JJ]<<endl;
                file_3<<"Pressure "<< output->VARS[5][JJ]+output->VARS[6][JJ]<<" N/sq m"<<endl;
                file_3<<"Inc y shear "<<output->VARS[7][JJ]<<" Refl y shear "<<output->VARS[8][JJ]<<endl;
                file_3<<"Net y shear "<<output->VARS[7][JJ]-output->VARS[8][JJ]<<" N/sq m"<<endl;
                file_3<<"Net z shear "<<output->VARS[9][JJ]-output->VARS[10][JJ]<<" N/sq m"<<endl;
                file_3<<"Incident translational heat flux "<<output->VARS[11][JJ]<<" W/sq m"<<endl;
                if(gas->MMRM>0)
                    file_3<<"Incident rotational heat flux "<<output->VARS[13][JJ]<<" W/sq m"<<endl;
                if(gas->MMVM>0)
                    file_3<<"Incident vibrational heat flux "<<output->VARS[15][JJ]<<" W/sq m"<<endl;
                if(gas->MELE>1)
                    file_3<<"Incident electronic heat flux "<<output->VARS[33][JJ]<<" W/sq m"<<endl;
                file_3<<"Total incident heat flux "<<output->VARS[29][JJ]<<" W/sq m"<<endl;
                file_3<<"Reflected translational heat flux "<<output->VARS[12][JJ]<<" W/sq m"<<endl;
                if(gas->MMRM>0)
                    file_3<<"Reflected rotational heat flux "<<output->VARS[14][JJ]<<" W/sq m"<<endl;
                if(gas->MMVM>0)
                    file_3<<"Reflected vibrational heat flux "<<output->VARS[16][JJ]<<" W/sq m"<<endl;
                if(gas->MELE>1)
                    file_3<<"Reflected electronic heat flux "<<output->VARS[34][JJ]<<" W/sq m"<<endl;
                file_3<<"Total reflected heat flux "<<output->VARS[30][JJ]<<" W/sq m"<<endl;
                file_3<<"Net heat flux "<<output->VARS[31][JJ]<<" W/sq m"<<endl;
                file_3<<"Slip velocity (y direction) "<<output->VARS[19][JJ]<<" m/s"<<endl;
                file_3<<"Translational temperature slip"<<output->VARS[20][JJ]<<" K"<<endl;
                if(gas->MMRM>0)
                    file_3<<"Rotational temperature slip "<<output->VARS[21][JJ]<<" K"<<endl;
                if(gas->MSP>1)
                {
                    for(L=1;L<=gas->MSP;L++)
                    {
                        file_3<<"Species "<<L<<" percentage "<<output->VARS[L+35][JJ]<<endl;
                    }
                }
            }
        }

        file_3<<endl;
        //PPA=0;
        for(int i=0;i<gas->MSP+1;i++)
            PPA[i]=0;

        for(N=1;N<=geom->NCELLS;N++)
        {
            for(M=1;M<=gas->MSP;M++){
                PPA[M]=PPA[M]+output->VARSP[0][N][M];
            }
        }
        // WRITE (*,*)
        //cin.get();
        if(gas->MSP>1)
        {
            file_3<<"GAINS FROM REACTIONS"<<endl;
            file_3<<"                          Dissoc.     Recomb. Endo. Exch.  Exo. Exch."<<endl;
            for(M=1;M<=gas->MSP;M++){
                file_3<<"                          SPECIES "<<M<<" "<<gas->TREACG[1][M]<<" "<<gas->TREACG[2][M]<<" "<<gas->TREACG[3][M]<<" "<<gas->TREACG[4][M]<<endl;
            }
            file_3<<endl;
            file_3<<"LOSSES FROM REACTIONS"<<endl;
            file_3<<"                          Dissoc.     Recomb. Endo. Exch.  Exo. Exch."<<endl;
            for(M=1;M<=gas->MSP;M++){
                file_3<<"                          SPECIES "<<M<<" "<<gas->TREACL[1][M]<<" "<<gas->TREACL[2][M]<<" "<<gas->TREACL[3][M]<<" "<<gas->TREACL[4][M]<<endl;
            }
            file_3<<endl;
            file_3<<"TOTALS"<<endl;
            for(M=1;M<=gas->MSP;M++){
                file_3<<"                        SPECIES "<<M<<" GAINS "<<gas->TREACG[1][M]+gas->TREACG[2][M]+gas->TREACG[3][M]+gas->TREACG[4][M]<<" LOSSES "<<gas->TREACL[1][M]+gas->TREACL[2][M]+gas->TREACL[3][M]+gas->TREACL[4][M]<<endl;
            }
        }
        file_3<<endl;
        file_3<<"Flowfield properties "<<endl;
        file_3<< output->NSAMP<<" Samples"<<endl;
        file_3<<"Overall gas"<<endl;
        file_3<<"Cell x coord.      Sample       Number Dens. Density      u velocity   v velocity   w velocity   Trans. Temp. Rot. Temp.   Vib. Temp.    El. Temp.  Temperature  Mach no.     Mols/cell    m.c.t        m.f.p        mcs/mfp        speed      Pressure      TTX         TTY         TTZ   Species Fractions "<<endl;
        for(N=1;N<=geom->NCELLS;N++)
        {
            file_3<< N<<" ";
            for(M=1;M<=10;M++){
                file_3<<output->VAR[M][N]<<" ";
            }
            file_3<<output->VAR[22][N]<<" ";
            for(M=11;M<=21;M++){
                file_3<<output->VAR[M][N]<<" ";
            }
            for(L=1;M<=gas->MSP;M++){
                file_3<<output->VARSP[1][N][L]<<" ";
            }
            file_3<<endl;
        }
        file_3<<"Individual molecular species"<<endl;
        for(L=1;L<=gas->MSP;L++){
            file_3<<"Species "<<L<<endl;
            file_3<<"Cell x coord.      Sample       Percentage   Species TTx   Species TTy  Species TTz  Trans. Temp.  Rot. Temp.  Vib. Temp.   Spec. Temp  u Diff. Vel. v Diff. Vel. w. Diff. Vel. Elec. Temp."<<endl;
            for(N=1;N<=geom->NCELLS;N++){
                file_3<< N<<" "<<output->VAR[1][N]<<" ";
                for(M=0;M<=12;M++)
                    file_3<<output->VARSP[M][N][L]<<" ";
                file_3<<endl;
            }
        }
        //dout
        //999 FORMAT (I5,30G13.5)
        //998 FORMAT (G280.0)
        // 997 FORMAT (G188.0)
        // CLOSE (3)
        file_3.close();
    }
    if(calc->ICLASS==0 && calc->ISF==1){
        //--a homogeneous gas and the "unsteady sampling" option has been chosen-ASSUME THAT IT IS A RELAXATION TEST CASE FOR SECTION 6.2
        INITIALISE_SAMPLES();
        //write a special output file for internal temperatures and temperature versus collision number
        //dout
        file_10.open("RELAX.DAT", ios::app | ios::out);
        if(file_10.is_open()){
            cout<<"RELAX.DAT is opened"<<endl;
        }
        else{
            cout<<"RELAX.DAT not opened"<<endl;
        }
        // OPEN (10,FILE='RELAX.DAT',ACCESS='APPEND')
        A=2.0*calc->TOTCOL/molecs->NM; //--mean collisions
        //--VAR(11,N)   //--overall
        //--VAR(8,N)    //--translational
        //--VAR(9,N)    //--rotational
        //--VAR(10,N)   //--vibrational
        //--VAR(22,N)   //--electronic
        //file_10<<std::right<<setw(15)<<A<<setw(15)<<output->VAR[8][1]<<setw(15)<<output->VAR[9][1]<<setw(15)<<output->VAR[8][1]-output->VAR[9][1]<<endl;
        file_10<<std::right<<setw(15)<<A<<setw(15)<<output->VAR[11][1]<<setw(15)<<output->VAR[8][1]<<setw(15)<<output->VAR[9][1]<<setw(15)<<output->VAR[10][1]<<setw(15)<<output->VAR[22][1]<<endl;
        //file_10<<std::right<<setw(15)<<A<<setw(15)<<output->VAR[8][1]<<setw(15)<<output->VAR[9][1]<<setw(15)<<output->VAR[8][1]-output->VAR[9][1]<<endl;
        //  WRITE (10,950) A,VAR(8,1),VAR(9,1),VAR(8,1)-VAR(9,1)   //--Generates output for Figs. 6.1 and 6.2
        //  WRITE (10,950) A,VAR(11,1),VAR(8,1),VAR(9,1),VAR(10,1),VAR(22,1)   //--Generates output for modal temperatures in Figs. 6.3, 6.5 +
        //  WRITE (10,950) A,0.5D00*(VAR(8,1)+VAR(9,1)),VAR(10,1),0.5D00*(VAR(8,1)+VAR(9,1))-VAR(10,1)  //--Generates output for Figs. 6.4
        //
        //--VARSP(8,N,L) //--overall temperature of species L
        //  WRITE (10,950) A,VARSP(8,1,3),VARSP(8,1,2),VARSP(8,1,5),VARSP(8,1,4),A  //--output for Fig 6.17
        // CLOSE (10)
        file_10.close();
    }
    //dout
    // 950 FORMAT (6G13.5)
    if(gas->IGAS==8||gas->IGAS==6||gas->IGAS==4)
    {
        //--Write a special output file for the composition of a reacting gas as a function of time
        //dout
        //OPEN (10,FILE='COMPOSITION.DAT',ACCESS='APPEND')
        file_10.open("COMPOSITION.DAT", ios::app | ios::out);
        if(file_10.is_open()){
            cout<<"COMPOSITION.DAT is opened"<<endl;
        }
        else{
            cout<<"COMPOSITION.DAT not opened"<<endl;
        }
        AS=molecs->NM;
        //dout
        AT=calc->FTIME*1.e6;
        if (gas->IGAS == 4)
            file_10<< AT <<" "<<(double)(NMS[1])/1000000<<" "<<A<<" "<<output->VAR[11][1]<<endl;    //--Data for fig
        if (gas->IGAS == 8)
            file_10<<AT<<" "<<NMS[1]/AS<<" "<<NMS[2]/AS<<" "<<NMS[3]/AS<<" "<<NMS[4]/AS<<" "<<NMS[5]/AS<<" "<<NMS[6]/AS<<" "<<NMS[7]/AS<<" "<<NMS[8]/AS<<" "<<output->VAR[11][1]<<endl;
        if (gas->IGAS == 6)
            file_10<<AT<<" "<<NMS[1]/AS<<" "<<NMS[2]/AS<<" "<<NMS[3]/AS<<" "<<NMS[4]/AS<<" "<<NMS[5]/AS<<" "<<output->VAR[11][1]<<endl;
        //dout
        // 888 FORMAT(10G13.5)
        file_10.close();
    }
    if(calc->FTIME>0.5e00*calc->DTM){
        //
        //--reset collision and transit times etc.
        //
        cout<<"Output files written "<<endl;
        DTMI=calc->DTM;
        if(calc->IMTS<2){
            if(calc->ICLASS>0)
                calc->DTM*=2;
            //--this makes it possible for DTM to increase, it will be reduced as necessary
            for(NN=1;NN<=geom->NCELLS;NN++)
            {
                CDTM[NN]=calc->DTM;
                B=geom->CELL[3][NN]-geom->CELL[2][NN] ;//--sampling cell width
                if(output->VAR[13][NN]>20.e00){
                    //consider the local collision rate
                    CDTM[NN]=output->VAR[14][NN]*calc->CPDTM;
                    //look also at sampling cell transit time based on the local flow speed
                    A=(B/(fabs(output->VAR[5][NN])))*calc->TPDTM;
                    if(A<CDTM[NN])
                        CDTM[NN]=A;
                }
                else{
                    //-- base the time step on a sampling cell transit time at the refence vmp
                    A=calc->TPDTM*B/gas->VMPM;
                    if(A<CDTM[NN])
                        CDTM[NN]=A;
                }
                if(CDTM[NN]<calc->DTM)
                    calc->DTM=CDTM[NN];
            }
        }
        else
        {
            //dout
            //memset(CDTM, calc->DTM, sizeof(*CDTM));
            for(int i=0;i<geom->NCELLS+1;i++)
                CDTM[i]= calc->DTM;
            //CDTM=calc->DTM;
        }
        for(N=1;N<=geom->NCELLS;N++){
            NN=geom->ICCELL[3][N];
            geom->CCELL[3][N]=0.5*CDTM[NN];
        }
        file_9<<"DTM changes  from "<<DTMI<<" to "<<calc->DTM<<endl;
        calc->DTSAMP=calc->DTSAMP*calc->DTM/DTMI;
        calc->DTOUT=calc->DTOUT*calc->DTM/DTMI;
    }
    else
    {
        INITIALISE_SAMPLES();
    }
    if(calc->ICLASS==1&& calc->ISF==1)
    {
        //*************************************************************************
        //--write TECPLOT data files for x-t diagram (unsteady calculation only)
        //--comment out if not needed
        //dout
        file_18.open("DS1xt.DAT", ios::app | ios::out);
        if(file_18.is_open()){
            cout<<"DS1xt.DAT is opened"<<endl;
        }
        else
            cout<<"DS1xt.DAT not opened"<<endl;
        // OPEN (18,FILE='DS1xt.DAT',ACCESS='APPEND')
        //--make sure that it is empty at the stary of the run
        SETXT();
        // CLOSE (18)
        file_18.close();
        //**************************************************************************
    }
    //WRITE (19,*) calc->FTIME,-output->VARS[5][1],-output->VARS[5][1]-output->VARS[6][1]
    
    file_7.open("PROFILE.DAT" , ios::out);
    if(file_7.is_open()){
        cout<<"PROFILE.DAT is opened"<<endl;
    }
    else
        cout<<"PROFILE.DAT not opened"<<endl;
    // OPEN (7,FILE='PROFILE.DAT',FORM='FORMATTED')
    //
    //OPEN (8,FILE='ENERGYPROF.DAT',FORM='FORMATTED')
    //
    // 995 FORMAT (22G13.5)
    // 996 FORMAT (12G14.6)
    for(N=1;N<=geom->NCELLS;N++)
    {
        //
        //--the following line is the default output
        //  WRITE (7,995) VAR(1,N),VAR(4,N),VAR(3,N),VAR(11,N),VAR(18,N),VAR(5,N),VAR(12,N),VAR(8,N),VAR(9,N),VAR(10,N),VAR(22,N),     &
        //        (VARSP(8,N,M),M=1,MSP),(VARSP(1,N,M),M=1,MSP)
        //
        //--calculate energies per unit mass (employed for re-entry shock wave in Section 7.5)
        C1=0.5e00*pow(output->VAR[5][N],2);    //--Kinetic
        C2=0.e00;                 //--Thermal
        C3=0.e00;                //--Rotational
        C4=0.e00;               //--Vibrational
        C5=0.e00;              //--Electronic
        C6=0.e00;             //--Formation
        for(L=1;L<=gas->MSP;L++)
        {
            //    C2=C2+3.D00*BOLTZ*VARSP(5,N,L)*VARSP(1,N,L)/SP(5,L)
            A=(output->CS[1][N][L]/output->VARSP[1][N][L])*gas->SP[5][L];
            if(output->CS[1][N][L]>0.5e00){
                C2=C2+0.5e00*(output->CS[5][N][L]+output->CS[6][N][L]+output->CS[7][N][L])*gas->SP[5][L]/A;
                if(gas->ISPR[1][L]>0)
                    C3=C3+output->CS[8][N][L];
                if(gas->ISPV[L]>0)
                    C4=C4+output->CS[10][N][L]*BOLTZ*gas->SPVM[1][1][L]/A;
                if(gas->NELL[L]>1)
                    C5=C5+output->CS[9][N][L]/A;
                C6=C6+gas->SP[6][L]*output->CS[1][N][L]/A;
            }
        }
        C2=C2-C1;
        //  A=0.5D00*VFX(1)**2+2.5D00*BOLTZ*FTMP(1)/(0.75*SP(5,2)+0.25*SP(5,1))
        C7=C1+C2+C3+C4+C5+C6;
        //
        //  WRITE (8,995) VAR(1,N),C1/A,C2/A,C3/A,C4/A,C5/A,C6/A,C7/A
        //
        //--the following lines are for normalised shock wave output in a simple gas (Sec 7.3)
        C1=gas->FND[2]-gas->FND[1];
        C2=gas->FTMP[2]-gas->FTMP[1];
        
        file_7<<output->VAR[1][N]<<" "<<output->VAR[2][N]<<" "<<(0.5*(output->VAR[20][N]+output->VAR[21][N])-gas->FTMP[1])/C2<<" "<<(output->VAR[19][N]-gas->FTMP[1])/C2<<" "<<(output->VAR[11][N]-gas->FTMP[1])/C2<<" "<<(output->VAR[3][N]-gas->FND[1])/C1<<endl;
        //--the following replaces sample size with density
        //C3=0.D00
        //DO L=1,MSP
        //  C3=C3+FND(1)*FSP(L,1)*SP(5,L)  //--upstream density
        //END DO
        //C4=0.D00
        //DO L=1,MSP
        //  C4=C4+FND(2)*FSP(L,2)*SP(5,L)  //--upstream density
        //END DO
        //
        //  WRITE (7,996) VAR(1,N),(VAR(4,N)-C3)/(C4-C3),(0.5*(VAR(20,N)+VAR(21,N))-FTMP(1))/C2,(VAR(19,N)-FTMP(1))/C2,(VAR(11,N)-FTMP(1))/C2,    &
        //        (VAR(3,N)-FND(1))/C1
        //--the following lines is for a single species in a gas mixture
        //  C1=C1*FSP(3,1)
        //  WRITE (7,996) VAR(1,N),VARSP(1,N,3),(0.5*(VARSP(3,N,3)+VARSP(4,N,3))-FTMP(1))/C2,(VARSP(2,N,3)-FTMP(1))/C2,(VARSP(5,N,3)-FTMP(1))/C2,(VAR(3,N)*VARSP(1,N,3)-FND(1)*FSP(3,1))/C1
        //
        //--the following line is for Couette flow (Sec 7.4)
        //  WRITE (7,996) VAR(1,N),VAR(2,N),VAR(5,N),VAR(6,N),VAR(7,N),VAR(11,N)
        //--the following line is for the breakdown of equilibrium in expansions (Sec 7.10)
        //  WRITE (7,996) VAR(1,N),VAR(2,N),VAR(12,N),VAR(4,N),VAR(5,N),VAR(8,N),VAR(9,N),VAR(10,N),VAR(11,N),VAR(19,N),VAR(20,N),VAR(21,N)
        //
    }
    if(calc->ISF==1)
        INITIALISE_SAMPLES();
    // CLOSE(7)
    file_7.close();
    //
    //--deallocate local variables
    //
    //dout
    for(int i=0;i<gas->MMVM+1;i++){
        delete [] TV[i];
    }
    delete [] TV;
    delete [] TVIB;
    delete [] VDOF;
    for(int i=0;i<gas->MSP+1;i++){
        delete [] THCOL[i];
    }
    delete [] THCOL;
    // DEALLOCATE (TV,TVIB,VDOF,THCOL,STAT=ERROR)
    // if(calc->ERROR)
    //     cout<<"PROGRAM COULD NOT DEALLOCATE OUTPUT VARIABLES"<<calc->ERROR;
    calc->TOUT=calc->TOUT+calc->DTOUT;
    return;
}



// __global__ void kernel(curandState* globalState, test &testy)
// {
//     // generate random numbers
//     //for(int i=0;i<40;i++)
//   //  {
//         float k = generate(globalState, 1);
//         //N[i] = k;
//         printf("yo %.6f\n", k);
//     //}
//         testy.a = generate(globalState, 0);
// }


__global__ void cuda_collisons(curandState* globalState, MOLECS *molecs, OUTPUT *output, GEOM_1D *geom, GAS *gas, CALC *calc)
{
    //CALC calc;
    //MOLECS molecs;
    //GAS gas;
    //OUTPUT output;
    //GEOM_1D geom;
    int NN,M,MM,L,LL,K,KK,KT,J,I,II,III,NSP,MAXLEV,IV,NSEL,KV,LS,MS,KS,JS,IIII,LZ,KL,IS,IREC,NLOOP,IA,IDISS,IEX,NEL,NAS,NPS,
    JJ,LIMLEV,KVV,KW,INIL,INIM,JI,LV,IVM,NMC,NVM,LSI,JX,MOLA,KR,JKV,NSC,KKV,IAX,NSTEP,NTRY,NLEVEL,NSTATE,IK,NK,MSI ;
    double A,AA,AAA,AB,B,BB,BBB,ABA,ASEL,DTC,SEP,VR,VRR,ECT,EVIB,ECC,ZV,ERM,C,OC,SD,D,CVR,PROB,RML,RMM,ECTOT,ETI,EREC,ET2,
    XMIN,XMAX,WFC,CENI,CENF,VRRT,EA,DEN,E1,E2,VRI,VRA ;
    double VRC[4],VCM[4],VRCP[4],VRCT[4];
    //   //N,M,K working integer
    // //LS,MS,KS,JS molecular species
    // //VRC components of the relative velocity
    // //RML,RMM molecule mass parameters
    // //VCM components of the center of mass velocity
    // //VRCP post-collision components of the relative velocity
    // //SEP the collision partner separation
    // //VRR the square of the relative speed
    // //VR the relative speed
    // //ECT relative translational energy
    // //EVIB vibrational energy
    // //ECC collision energy (rel trans +vib)
    // //MAXLEV maximum vibrational level
    // //ZV vibration collision number
    // //SDF the number of degrees of freedom associated with the collision
    // //ERM rotational energy
    // //NSEL integer number of selections
    // //NTRY number of attempts to find a second molecule
    // //CVR product of collision cross-section and relative speed
    // //PROB a probability
    // //KT third body molecule code
    // //ECTOT energy added at recmbination
    // //IREC initially 0, becomes 1 of a recombination occurs
    // //WFC weighting factor in the cell
    // //IEX is the reaction that occurs (1 if only one is possible)
    // //EA activation energy
    // //NPS the number of possible electronic states
    // //NAS the number of available electronic states
    //cout<<"START COLLISIONS"<<endl;
    
       int N = threadIdx.x + blockIdx.x * blockDim.x+1;
        if((calc->FTIME-geom->CCELL[5][N]) > (geom->CCELL[3][N]))
        {
            DTC=2.e00*geom->CCELL[3][N];
            //calculate collisions appropriate to  time DTC
            if(geom->ICCELL[2][N]>1)
            {
                //no collisions calculated if there are less than two molecules in collision cell
                NN=geom->ICCELL[3][N];
                WFC=1.e00;
                if(geom->IWF==1 && geom->IVB==0)
                {
                    //dout
                    WFC=1.e00+geom->WFM*powf(geom->CELL[1][NN],geom->IFX);
                }
                geom->CCELL[5][N]=geom->CCELL[5][N]+DTC;
                if(geom->IVB==0)
                {
                    AAA=geom->CCELL[1][N];
                }
                if(geom->IVB==1)
                {
                    C=(geom->XB[2]+geom->VELOB*calc->FTIME-geom->XB[1])/(double)(geom->NDIV*geom->NCIS);
                    //dout
                    XMIN=geom->XB[1]+(double)(N-1)*C;
                    XMAX=XMIN+C;
                    //dout
                    WFC=1.e00+geom->WFM*powf((0.5e00*(XMIN+XMAX)),geom->IFX);
                    if(geom->IFX==0)
                    {
                        AAA=XMAX-XMIN;
                    }
                    if(geom->IFX==1)
                    {
                        AAA=PI*(powf(XMAX,2)-powf(XMIN,2)); //assumes unit length of full cylinder
                    }
                    if(geom->IFX==2)
                    {
                        AAA=1.33333333333333333333e00*PI*(powf(XMAX,3)-powf(XMIN,3));    //flow is in the full sphere
                    }
                }
                //these statements implement the N(N-1) scheme
                ASEL=0.5e00*geom->ICCELL[2][N]*(geom->ICCELL[2][N]-1)*WFC*calc->FNUM*geom->CCELL[4][N]*DTC/AAA+geom->CCELL[2][N];
                NSEL=ASEL;
                //dout
                geom->CCELL[2][N]=ASEL-(double)(NSEL);
                if(NSEL>0)
                {
                    I=0; //counts the number of selections
                    KL=0; //becomes 1 if it is the last selection
                    IIII=0; //becomes 1 if there is a recombination
                    for(KL=1;KL<=NSEL;KL++)
                    {
                        I=I+1;
                        III=0; //becomes 1 if there is no valid collision partner
                        if(geom->ICCELL[2][N]==2)
                        {
                            K=1+geom->ICCELL[1][N];
                            //dout
                            L=molecs->ICREF[K];
                            K=2+geom->ICCELL[1][N];
                            //dout
                            M=molecs->ICREF[K];
                            if(M==molecs->IPCP[L])
                            {
                                III=1;
                                geom->CCELL[5][N]=geom->CCELL[5][N]-DTC;
                            }
                        }
                        else
                        {
                            //dout
                            //                            RANDOM_NUMBER(RANF);
                            calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                            K=(int)(calc->RANF*(double)(geom->ICCELL[2][N]))+geom->ICCELL[1][N]+1;
                            //dout
                            L=molecs->ICREF[K];
                            //one molecule has been selected at random
                            if(calc->NNC==0)
                            {
                                //select the collision partner at random
                                M=L;
                                NTRY=0;
                                while(M==L)
                                {
                                    //dout
                                    //                                    RANDOM_NUMBER(RANF);
                                    calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                    K=(int)(calc->RANF*(double)(geom->ICCELL[2][N]))+geom->ICCELL[1][N]+1;
                                    M=molecs->ICREF[K];
                                    if(M==molecs->IPCP[L])
                                    {
                                        if(NTRY<5*geom->ICCELL[2][N])
                                        {
                                            M=L;
                                        }
                                        else
                                        {
                                            III = 1;
                                            geom->CCELL[5][N]=geom->CCELL[5][N]-DTC/ASEL;
                                            M=L+1;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                //elect the nearest from the total number (< 30) or a random 30
                                if(geom->ICCELL[2][N]<30)
                                {
                                    LL=geom->ICCELL[2][N];
                                }
                                else
                                {
                                    LL=30;
                                }
                                SEP=1.0e10;
                                M=0;
                                for(J=1;J<=LL;J++)
                                {
                                    if(LL<30)
                                    {
                                        K=J+geom->ICCELL[1][N];
                                    }
                                    else
                                    {
                                        //                                        RANDOM_NUMBER(RANF);
                                        calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                        K=(int)(calc->RANF*(double)(geom->ICCELL[2][N]))+geom->ICCELL[1][N]+1;
                                    }
                                    MM=molecs->ICREF[K];
                                    if(MM != L)
                                    {
                                        //exclude the already selected molecule
                                        if(MM != molecs->IPCP[L])
                                        {
                                            //exclude the previous collision partner
                                            //dout
                                            A=fabsf(molecs->PX[1][L]-molecs->PX[1][MM]);
                                            if(A<SEP&& A>1.e-8*geom->DDIV)
                                            {
                                                M=MM;
                                                SEP=A;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if(III==0)
                        {
                            for(KK=1;KK<=3;KK++)
                            {
                                VRC[KK]=molecs->PV[KK][L]-molecs->PV[KK][M];
                            }
                            VRR=VRC[1]*VRC[1]+VRC[2]*VRC[2]+VRC[3]*VRC[3];
                            VR=sqrtf(VRR);
                            VRI=VR;
                            //Simple GAs
                            if(gas->MSP==1)
                            {
                                //dout
                                CVR=VR*gas->CXSS*powf(2.e00*BOLTZ*gas->SP[2][1]/(gas->RMAS*VRR),(gas->SP[3][1]-0.5e00))*gas->RGFS;
                                if(CVR>geom->CCELL[4][N])
                                {
                                    geom->CCELL[4][N]=CVR;
                                }
                                //dout
                                //                                RANDOM_NUMBER(RANF);
                                calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                if(calc->RANF<CVR/geom->CCELL[4][N])
                                {
                                    // the collision occurs
                                    if(M==molecs->IPCP[L]&& L==molecs->IPCP[M])
                                    {
                                        //file_9<<"Duplicate collision"<<endl;
                                    }
                                    //atomicAdd(&calc->TOTCOL,1.e00);
                                    //calc->TOTCOL=calc->TOTCOL+1.e00;
                                    calc->COLL_TOTCOL[N]=calc->COLL_TOTCOL[N]+1.e00;
                                    calc->TCOL[1][1]=calc->TCOL[1][1]+2.e00;
                                    output->COLLS[NN]=output->COLLS[NN]+1.e000;
                                    output->WCOLLS[NN]=output->WCOLLS[NN]+WFC;
                                    //dout
                                    SEP=fabsf(molecs->PX[1][L]-molecs->PX[1][M]);
                                    output->CLSEP[NN]=output->CLSEP[NN]+SEP;
                                    if(gas->ISPR[1][1]>0)
                                    {
                                        //Larsen-Borgnakke serial redistribution
                                        ECT=0.5e00*gas->RMAS*VRR;
                                        for(NSP=1;NSP<=2;NSP++)
                                        {
                                            //consider the molecules in turn
                                            if(NSP==1)
                                            {
                                                K=L;
                                            }
                                            else
                                            {
                                                K=M;
                                            }
                                            if(gas->MMVM>0)
                                            {
                                                if(gas->ISPV[1]>0)
                                                {
                                                    for(KV=1;KV<=gas->ISPV[1];KV++)
                                                    {
                                                        EVIB=(double)(molecs->IPVIB[KV][K]*BOLTZ*gas->SPVM[1][KV][1]);
                                                        ECC=ECT+EVIB;
                                                        if(gas->SPVM[3][KV][1]>0.0)
                                                        {
                                                            MAXLEV=ECC/(BOLTZ*gas->SPVM[1][KV][1]);
                                                            B=gas->SPVM[4][KV][1]/gas->SPVM[3][KV][1]; //Tdiss/Tref
                                                            A= gas->SPVM[4][KV][1]/output->VAR[8][NN] ;//Tdiss/Ttrans
                                                            //ZV=(A**SPM(3,1,1))*(SPVM(3,KV,1)*(B**(-SPM(3,1,1))))**(((A**0.3333333D00)-1.D00)/((B**0.33333D00)-1.D00))
                                                            ZV=powf(A,gas->SPM[3][1][1])*powf(gas->SPVM[3][KV][1]*powf(B,-gas->SPM[3][1][1]),((powf(A,0.3333333e00)-1e00)/(powf(B,33333e00)-1.e00)));
                                                        }
                                                        else
                                                        {
                                                            ZV=gas->SPVM[2][KV][1];
                                                            MAXLEV=ECC/(BOLTZ*gas->SPVM[1][KV][1])+1;
                                                        }
                                                        //dout
                                                        //                                                        RANDOM_NUMBER(RANF);
                                                        calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                                        if(1.e00/ZV>calc->RANF)
                                                        {
                                                            II=0;
                                                            while(II==0)
                                                            {
                                                                //dout
                                                                //                                                                RANDOM_NUMBER(RANF);
                                                                calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                                                IV=calc->RANF*(MAXLEV+0.99999999e00);
                                                                molecs->IPVIB[KV][K]=IV;
                                                                EVIB=(double)(IV)*BOLTZ;
                                                                if(EVIB<ECC)
                                                                {
                                                                    PROB=powf((1.e00-EVIB/ECC),(1.5e00-gas->SPM[3][KV][1]));
                                                                    //PROB is the probability ratio of eqn (3.28)
                                                                    //dout
                                                                    //                                                                    RANDOM_NUMBER(RANF);
                                                                    calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                                                    if(PROB>calc->RANF)
                                                                        II=1;
                                                                }
                                                            }
                                                            ECT=ECC-EVIB;
                                                        }
                                                    }
                                                }
                                            }
                                            //now rotation of this molecule
                                            //dout
                                            if(gas->ISPR[1][1] > 0)
                                            {
                                                if(gas->ISPR[2][1]==0)
                                                {
                                                    B=1.e00/gas->SPR[1][1];
                                                }
                                                else //use molecule rather than mean value
                                                {
                                                    B=1.e00/(gas->SPR[1][1]+gas->SPR[2][1]*output->VAR[8][NN]+gas->SPR[3][1]*powf(output->VAR[8][NN],2));
                                                }
                                                //dout
                                                //                                                RANDOM_NUMBER(RANF);
                                                calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                                if(B>calc->RANF)
                                                {
                                                    ECC=ECT +molecs->PROT[K];
                                                    if(gas->ISPR[1][1]==2)
                                                    {
                                                        //dout
                                                        //                                                        RANDOM_NUMBER(RANF);
                                                        calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                                        ERM=1.e00-powf(calc->RANF,1.e00/(2.5e00-gas->SP[3][1])); //eqn(5.46)
                                                    }
                                                    else
                                                    {
                                                        //dout
                                                        LBS(globalState, 0.5e00*gas->ISPR[1][1]-1.e00,1.5e00-gas->SP[3][1],ERM);
                                                    }
                                                    molecs->PROT[K]=ERM*ECC;
                                                    ECT=ECC-molecs->PROT[K];
                                                }
                                            }
                                        }
                                        //adjust VR for the change in energy;
                                        VR=sqrtf(2.e00*ECT/gas->SPM[1][1][1]);
                                    }
                                    //end of L-B redistribution
                                    for(KK=1;KK<=3;KK++)
                                    {
                                        VCM[KK]=0.5e00*(molecs->PV[KK][L]+molecs->PV[KK][M]);
                                    }
                                    //dout
                                    if(fabsf(gas->SP[4][1]-1.0) < 0.001)
                                    {
                                        //use the VHS logic //dout
                                        //                                        RANDOM_NUMBER(RANF);
                                        calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                        B=2.e00*calc->RANF-1.e00;
                                        //B is the cosine of a random elevation angle
                                        A=sqrtf(1.e00-B*B);
                                        VRCP[1]=B*VR;
                                        //dout
                                        //                                        RANDOM_NUMBER(RANF);
                                        calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                        C=2.e00*PI*calc->RANF;
                                        //C is a random azimuth angle
                                        //dout
                                        VRCP[2]=A*cos(C)*VR;
                                        VRCP[3]=A*sin(C)*VR;
                                    }
                                    else
                                    {
                                        //use the VSS logic //dout
                                        //                                        RANDOM_NUMBER(RANF);
                                        calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                        B=2.e00*(powf(calc->RANF,gas->SP[4][1]))-1.e00;
                                        //B is the cosine of the deflection angle for the VSS model (Eqn. 11.8) of Bird(1994))
                                        A=sqrtf(1.e00-B*B);
                                        //dout
                                        //                                                 RANDOM_NUMBER(RANF);
                                        calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                        C=2.e00*PI*calc->RANF;
                                        //dout
                                        OC=(double)cos(C);
                                        SD=(double)sin(C);
                                        D=sqrtf(powf(VRC[2],2)+powf(VRC[3],2));
                                        VRA=VR/VRI;
                                        VRCP[1]=(B*VRC[1]+A*SD*D)*VRA;
                                        VRCP[2]=(B*VRC[2]+A*(VRI*VRC[3]*OC-VRC[1]*VRC[2]*SD)/D)*VRA;
                                        VRCP[3]=(B*VRC[2]+A*(VRI*VRC[2]*OC-VRC[1]*VRC[3]*SD)/D)*VRA;
                                        //the post-collision rel. velocity components are based on eqn (3.18)
                                    }
                                    for(KK=1;KK<=3;KK++)
                                    {
                                        molecs->PV[KK][L]=VCM[KK]+0.5e00*VRCP[KK];
                                        molecs->PV[KK][M]=VCM[KK]-0.5e00*VRCP[KK];
                                    }
                                    molecs->IPCP[L]=M;
                                    molecs->IPCP[M]=L;
                                }
                            } //collision occurrence
                            else
                            {
                                //Gas Mixture
                                LS=fabsf(molecs->IPSP[L]);
                                MS=fabsf(molecs->IPSP[M]);
                                CVR=VR*gas->SPM[2][LS][MS]*powf(((2.e00*BOLTZ*gas->SPM[5][LS][MS])/((gas->SPM[1][LS][MS])*VRR)),(gas->SPM[3][LS][MS]-0.5e00))*gas->SPM[6][LS][MS];
                                if(CVR>geom->CCELL[4][N])
                                {
                                    geom->CCELL[4][N]=CVR;
                                }
                                //dout
                                //                                    RANDOM_NUMBER(RANF);
                                calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                if(calc->RANF<CVR/geom->CCELL[4][N] && molecs->IPCELL[L]>0 && molecs->IPCELL[M]>0)
                                {
                                    //the collision occurs (-ve IPCELL indicates recombined molecule marled for removal)
                                    if(M==molecs->IPCP[L] && L==molecs->IPCP[M])
                                    {
                                        //file_9<<"Duplicate collision";
                                    }
                                    //atomicAdd(&calc->TOTCOL,1.e00);
                                    //calc->TOTCOL=calc->TOTCOL+1.e00;
                                    calc->COLL_TOTCOL[N]=calc->COLL_TOTCOL[N]+1.e00;
                                    calc->TCOL[LS][MS]=calc->TCOL[LS][MS]+1.e00;
                                    calc->TCOL[MS][LS]=calc->TCOL[MS][LS]+1.e00;
                                    output->COLLS[NN]=output->COLLS[NN]+1.e00;
                                    output->WCOLLS[NN]=output->WCOLLS[NN]+WFC;
                                    SEP=fabsf(molecs->PX[1][L]-molecs->PX[1][M]);
                                    output->CLSEP[NN]=output->CLSEP[NN]+SEP;
                                    RML=gas->SPM[1][LS][MS]/gas->SP[5][MS];
                                    RMM=gas->SPM[1][LS][MS]/gas->SP[5][LS];
                                    for(KK=1;KK<=3;KK++)
                                    {
                                        VCM[KK]=RML*molecs->PV[KK][L]+RMM*molecs->PV[KK][M];
                                    }
                                    IDISS=0;
                                    IREC=0;
                                    IEX=0;
                                    //check for dissociation
                                    if(gas->ISPR[1][LS]>0 || gas->ISPR[1][MS]>0)
                                    {
                                        ECT=0.5e00*gas->SPM[1][LS][MS]*VRR;
                                        for(NSP=1;NSP<=2;NSP++)
                                        {
                                            if(NSP==1)
                                            {
                                                K=L; KS=LS; JS=MS;
                                            }
                                            else
                                            {
                                                K=M ; KS=MS ; JS=LS;
                                            }
                                            if(gas->MMVM>0)
                                            {
                                                if(gas->ISPV[KS]>0)
                                                {
                                                    for(KV=1;KV<=gas->ISPV[KS];KV++)
                                                    {
                                                        if(molecs->IPVIB[KV][K]>=0 && IDISS==0)
                                                        {
                                                            //do not redistribute to a dissociating molecule marked for removal
                                                            EVIB=(double)(molecs->IPVIB[KV][K]*BOLTZ*gas->SPVM[1][KV][KS]);
                                                            ECC=ECT+EVIB;
                                                            MAXLEV=ECC/(BOLTZ*gas->SPVM[1][KV][KS]);
                                                            LIMLEV=gas->SPVM[4][KV][KS]/gas->SPVM[1][KV][KS];
                                                            if(MAXLEV > LIMLEV)
                                                            {
                                                                //dissociation occurs subject to reduction factor  -  reflects the infinity of levels past the dissociation limit
                                                                //dout
                                                                //                                                                    RANDOM_NUMBER(RANF)
                                                                calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                                                if(calc->RANF<gas->SPVM[5][KV][KS])
                                                                {
                                                                    IDISS=1;
                                                                    LZ=molecs->IPVIB[KV][K];
                                                                    output->NDISSL[LZ]=output->NDISSL[LZ]+1;
                                                                    ECT=ECT-BOLTZ*gas->SPVM[4][KV][KS]+EVIB;
                                                                    //adjust VR for the change in energy
                                                                    VRR=2.e00*ECT/gas->SPM[1][LS][MS];
                                                                    VR=sqrtf(VRR);
                                                                    molecs->IPVIB[KV][K]=-1;
                                                                    //a negative IPVIB marks a molecule for dissociation
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    IEX=0;    //becomes the reaction number if a reaction occurs
                                    IREC=0;   //becomes 1 if a recombination occurs
                                    if(IDISS==0)
                                    {
                                        //dissociation has not occurred
                                        //consider possible recombinations
                                        if(gas->ISPRC[LS][MS]>0 && geom->ICCELL[2][N]>2)
                                        {
                                            //possible recombination using model based on collision volume for equilibrium
                                            KT=L;
                                            //NTRY=0
                                            while(KT==L||KT==M)
                                            {
                                                NTRY+=1;
                                                // if(NTRY>100)
                                                // {
                                                //  cout>>"NTRY 3rd body"<<NTRY;
                                                // }
                                                //RANDOM_NUMBER(RANF);
                                                calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);\
                                                K=(int)(calc->RANF*(double)(geom->ICCELL[2][N]))+geom->ICCELL[1][N]+1;
                                                KT=molecs->ICREF[K];
                                            }
                                            KS=molecs->IPSP[KT];
                                            //the potential third body is KT OF species KS
                                            AA=(PI/6.e00)*powf((gas->SP[1][LS]+gas->SP[1][MS]+gas->SP[1][KS]),3); //reference volume
                                            BB=AA*gas->SPRC[1][LS][MS][KS]*powf(output->VAR[8][NN]/gas->SPVM[1][gas->ISPRK[LS][MS]][gas->ISPRC[LS][MS]],gas->SPRC[2][LS][MS][KS]);//collision volume
                                            B=BB*geom->ICCELL[2][N]*calc->FNUM/AAA;
                                            if(B>1.e00)
                                            {
                                                printf("THREE BODY PROBABILITY %f\n", B);
                                                //cout<<"THREE BODY PROBABILITY"<<B;
                                                //for low density flows in which three-body collisions are very rare, it is advisable to consider recombinations in only a small
                                                //fraction of collisions and to increase the pribability by the inverse of this fraction.  This message provides a warning if this
                                                //factor has been set to an excessively large value
                                            }
                                            //dout
                                            //                                                RANDOM_NUMBER(RANF);
                                            calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                            if(calc->RANF<B)
                                            {
                                                IREC=1;
                                                calc->TRECOMB[gas->ISPRC[LS][MS]]=calc->TRECOMB[gas->ISPRC[LS][MS]]+1.e00;
                                                //the collision now becomes a collision between these with L having the center of mass velocity
                                                A=0.5e00*gas->SPM[1][LS][MS]*VRR ;//the relative energy of the recombining molecules
                                                if(gas->ISPR[1][LS]>0)
                                                    A=A+molecs->PROT[L];
                                                if(gas->MELE>1)
                                                    A=A+molecs->PELE[L];
                                                if(gas->ISPV[LS]>0)
                                                {
                                                    for(KVV=1;KVV<=gas->ISPV[LS];KVV++)
                                                    {
                                                        JI=molecs->IPVIB[KVV][L];
                                                        if(JI<0)
                                                            JI=-JI;
                                                        if(JI==99999)
                                                            JI=0;
                                                        A=A+(double)(JI)*BOLTZ*gas->SPVM[1][KVV][LS];
                                                    }
                                                }
                                                if(gas->ISPR[1][MS]>0)
                                                    A+=molecs->PROT[M];
                                                if(gas->MELE>1)
                                                    A=A+molecs->PELE[M];
                                                if(gas->ISPV[MS]>0)
                                                {
                                                    for(KVV=1;KVV<=gas->ISPV[MS];KVV++)
                                                    {
                                                        JI=molecs->IPVIB[KVV][M];
                                                        if(JI<0)
                                                            JI=-JI;
                                                        if(JI==99999)
                                                            JI=0;
                                                        A=A+(double)(JI)*BOLTZ*gas->SPVM[1][KVV][MS];
                                                    }
                                                }
                                                gas->TREACL[2][LS]=gas->TREACL[2][LS]-1;
                                                gas->TREACL[2][MS]=gas->TREACL[2][MS]-1;
                                                LSI=LS;
                                                MSI=MS;
                                                LS=gas->ISPRC[LS][MS];
                                                molecs->IPSP[L]=LS;
                                                //any additional vibrational modes must be set to zero
                                                IVM=gas->ISPV[LSI];
                                                NMC=molecs->IPSP[L];
                                                NVM=gas->ISPV[NMC];
                                                if(NVM>IVM)
                                                {
                                                    for(KV=IVM+1;KV<=NVM;KV++)
                                                    {
                                                        molecs->IPVIB[KV][L]=0;
                                                    }
                                                }
                                                if(gas->MELE>1)
                                                    molecs->PELE[KV]=0.e00;

                                                molecs->IPCELL[M]=-molecs->IPCELL[M]; //recombining molecule M marked for removal
                                                M=KT; //third body molecule is set as molecule M
                                                MS=KS;
                                                gas->TREACG[2][LS]=gas->TREACG[2][LS]+1;
                                                if(gas->ISPR[1][LS]>0)
                                                {
                                                    molecs->PROT[L]=0.e00;
                                                }
                                                if(gas->MELE>1)
                                                    molecs->PELE[L]=0.e00;
                                                if(gas->ISPV[LS]>0)
                                                {
                                                    for(KVV=1;KVV<=gas->ISPV[LS];KVV++)
                                                    {
                                                        if(molecs->IPVIB[KVV][L]<0)
                                                        {
                                                            molecs->IPVIB[KVV][L]=-99999;
                                                        }
                                                        else
                                                        {
                                                            molecs->IPVIB[KVV][L]=0;
                                                        }
                                                    }
                                                }
                                                if(gas->ISPR[1][MS]>0)
                                                {
                                                    molecs->PROT[M]=molecs->PROT[KT];
                                                }
                                                if(gas->MELE>1)
                                                    molecs->PELE[M]=molecs->PELE[KT];
                                                if(gas->ISPV[MS]>0)
                                                {
                                                    for(KVV=1;KVV<=gas->ISPV[MS];KVV++)
                                                    {
                                                        molecs->IPVIB[KVV][M]=molecs->IPVIB[KVV][KT];
                                                    }
                                                }
                                                ECTOT=A+gas->SPVM[4][1][LS]*BOLTZ ; //the energy added to this collision
                                                for(KK=1;KK<=3;KK++)
                                                {
                                                    molecs->PV[KK][L]=VCM[KK];
                                                }
                                                for(KK=1;KK<=3;KK++)
                                                {
                                                    VRC[KK]=molecs->PV[KK][L]-molecs->PV[KK][M];
                                                }
                                                VRR=VRC[1]*VRC[1]+VRC[2]*VRC[2]+VRC[3]*VRC[3];
                                                ECT=0.5e00*gas->SPM[1][LS][MS]*VRR*ECTOT;
                                                //set the vibrational energy of the recombined molecule L to enforce detailed balance
                                                IK=-1;
                                                NK=-1;
                                                //dout
                                                //                                                    RANDOM_NUMBER(RANF);
                                                calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                                //NTRY=0;
                                                while(IK<0)
                                                {
                                                    // NTRY+=1;
                                                    // if(NTRY>100)
                                                    //   cout<<"NTRY VibEn"<<NTRY;
                                                    NK=NK+1;
                                                    BB=(output->VAR[8][NN]-gas->SPRT[1][LSI][MSI])*(gas->SPRP[2][LSI][MSI][NK]-gas->SPRP[1][LSI][MSI][NK])/(gas->SPRT[2][LSI][MSI]-gas->SPRT[1][LSI][MSI])-gas->SPRP[1][LSI][MSI][NK];
                                                    if(calc->RANF<BB)
                                                        IK=NK;
                                                }
                                                molecs->IPVIB[1][L]=IK;
                                                ECT=ECT-(double)(IK)*BOLTZ*gas->SPVM[1][gas->ISPRK[LSI][MSI]][LS];
                                                VRR=2.e00*ECT/gas->SPM[1][LS][MS];
                                                VR=sqrtf(VRR);
                                                RML=gas->SPM[1][LS][MS]/gas->SP[5][MS];
                                                RMM=gas->SPM[1][LS][MS]/gas->SP[5][LS];
                                                for(KK=1;KK<=3;KK++)
                                                {
                                                    VCM[KK]=RML*molecs->PV[KK][L]+RMM*molecs->PV[KK][M];
                                                }
                                            }
                                        }
                                        //consider exchange and chain reactions
                                        if(gas->NSPEX[LS][MS]>0 && IREC==0 && IDISS==0)
                                        {
                                            //possible exchange reaction
                                            //memset(gas->PSF,0.e00,sizeof(*gas->PSF));//gas->PSF=0.e00; //PSF(MMEX) PSF is the probability that this reaction will occur in this collision
                                            for(int i=0;i<gas->MMEX+1;i++)
                                                gas->PSF[i]=0.e00;
                                            
                                            for(JJ=1;JJ<=gas->NSPEX[LS][MS];JJ++)
                                            {
                                                if(LS==gas->ISPEX[JJ][1][LS][MS])
                                                {
                                                    K=L; KS=LS;JS=MS;
                                                }
                                                else
                                                {
                                                    K=M; KS=MS; JS=LS;
                                                }
                                                //the pre-collision molecule that splits is K of species KS
                                                if(gas->SPEX[3][JJ][LS][MS]<0.e00)
                                                    KV=gas->ISPEX[JJ][5][LS][MS];
                                                if(gas->SPEX[3][JJ][LS][MS]>0.e00)
                                                {
                                                    KV=gas->ISPEX[JJ][7][LS][MS];
                                                }
                                                JI=molecs->IPVIB[KV][K];
                                                if(JI<0)
                                                    JI=-JI;
                                                if(JI==99999)
                                                    JI=0;
                                                ECC=0.5e00*gas->SPM[1][LS][MS]*VRR+(double)(JI)*BOLTZ*gas->SPVM[1][KV][KS];
                                                if(gas->SPEX[3][JJ][KS][JS]>0.e00)
                                                {
                                                    //reverse exothermic reaction
                                                    gas->PSF[JJ]=(gas->SPEX[1][JJ][KS][JS]*powf(output->VAR[8][NN]/273.e00,gas->SPEX[2][JJ][KS][JS]))*expf(-gas->SPEX[6][JJ][KS][JS]/(BOLTZ*output->VAR[8][NN]));
                                                }
                                                else
                                                {
                                                    //forward endothermic reaction
                                                    MAXLEV=ECC/(BOLTZ*gas->SPVM[1][KV][KS]);
                                                    EA=fabsf(gas->SPEX[3][JJ][KS][JS]); //temporarily just the heat of reaction;
                                                    if(ECC>EA)
                                                    {
                                                        //the collision energy must exceed the heat of reaction
                                                        EA=EA+gas->SPEX[6][JJ][KS][JS]; //the activation energy now includes the energy barrier
                                                        DEN=0.e00;
                                                        for(IAX=0;IAX<=MAXLEV;IAX++)
                                                        {
                                                            DEN=DEN+powf((1.e00-(double)(IAX)*BOLTZ*gas->SPVM[1][KV][KS]/ECC),(1.5e00-gas->SPM[3][KS][JS]));
                                                        }
                                                        gas->PSF[JJ]=(double)(gas->ISPEX[JJ][6][LS][MS])*powf((1.e00-EA/ECC),(1.5e00-gas->SPM[3][KS][JS]))/DEN;
                                                    }
                                                }
                                            }
                                            if(gas->NSPEX[LS][MS]>1)
                                            {
                                                BB=0.e00;
                                                for(JJ=1;JJ<=gas->NSPEX[LS][MS];JJ++)
                                                {
                                                    BB=BB+gas->PSF[JJ];
                                                }
                                                //BB is the sum of the probabilities
                                                //dout
                                                //                                                    RANDOM_NUMBER(RANF);
                                                calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                                if(BB>calc->RANF)
                                                {
                                                    BB=0.e00;
                                                    IEX=0;
                                                    JJ=0;
                                                    //NTRY=0;
                                                    while(JJ<gas->NSPEX[LS][MS]&& IEX==0)
                                                    {
                                                        // NTRY=NTRY+1;
                                                        // if(NTRY>100)
                                                        // {
                                                        //   cout<<"NTRY find IEX"<<NTRY;
                                                        // }
                                                        JJ+=1;
                                                        BB+=gas->PSF[JJ];
                                                        if(BB>calc->RANF)
                                                            IEX=JJ;
                                                    }
                                                }
                                            }
                                            else
                                            {
                                                //dout
                                                //                                                    RANDOM_NUMBER(RANF);
                                                calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                                IEX=0;
                                                if(gas->PSF[1]>calc->RANF)
                                                    IEX=1;
                                            }
                                            if(IEX>0)
                                            {
                                                //exchange or chain reaction occurs
                                                JX=gas->NEX[IEX][LS][MS];
                                                //cout<<"Reaction"<<JX;
                                                gas->TNEX[JX]=gas->TNEX[JX]+1.e00;
                                                //cout<<IEX<<L<<M<<LS<<MS;
                                                molecs->IPSP[L]=gas->ISPEX[IEX][3][LS][MS]; //L is now the new molecule that splits
                                                molecs->IPSP[M]=gas->ISPEX[IEX][4][LS][MS];
                                                LSI=LS;
                                                MSI=MS;
                                                //any additional vibrational modes must be set to zero
                                                IVM=gas->ISPV[LS];
                                                NMC=molecs->IPCP[L];
                                                NVM=gas->ISPV[NMC];
                                                if(NVM>IVM)
                                                {
                                                    for(KV=IVM+1;KV<=NVM;KV++)
                                                    {
                                                        molecs->IPVIB[KV][L]=0;
                                                    }
                                                }
                                                IVM=gas->ISPV[MS];
                                                NMC=molecs->IPCP[M];
                                                NVM=gas->ISPV[NMC];
                                                if(NVM>IVM)
                                                {
                                                    for(KV=IVM+1;KV<=NVM;KV++)
                                                    {
                                                        molecs->IPVIB[KV][M]=0;
                                                    }
                                                }
                                                //put all pre-collision energies into the relative translational energy and adjust for the reaction energy
                                                ECT=0.5e00*gas->SPM[1][LS][MS]*VRR;
                                                if(gas->ISPR[1][LS]>0)
                                                    ECT=ECT+molecs->PROT[L];
                                                if(gas->MELE>1)
                                                    ECT=ECT+molecs->PELE[L];
                                                if(gas->ISPV[LS]>0)
                                                {
                                                    for(KV=1;KV<=gas->ISPV[LS];KV++)
                                                    {
                                                        JI=molecs->IPVIB[KV][L];
                                                        if(JI<0)
                                                            JI=-JI;
                                                        if(JI==99999)
                                                            JI=0;
                                                        ECT=ECT+(double)(JI)*BOLTZ*gas->SPVM[1][KV][LS];
                                                    }
                                                }
                                                if(gas->ISPR[1][MS]>0)
                                                    ECT=ECT+molecs->PROT[M];
                                                if(gas->MELE>1)
                                                    ECT=ECT+molecs->PELE[M];
                                                if(gas->ISPV[MS]>0)
                                                {
                                                    for(KV=1;KV<=gas->ISPV[MS];KV++)
                                                    {
                                                        JI=molecs->IPVIB[KV][M];
                                                        if(JI<0)
                                                            JI=-JI;
                                                        if(JI==99999)
                                                            JI=0;
                                                        ECT=ECT+(double)(JI)*BOLTZ*gas->SPVM[1][KV][MS];
                                                    }
                                                }
                                                ECT=ECT+gas->SPEX[3][IEX][LS][MS];
                                                if(ECT<0.0)
                                                {
                                                    printf ("-VE ECT %f\n",ECT);
                                                    printf ("REACTION %d",JJ," BETWEEN %d",LS," & %d\n",MS);
                                                    // cout<<"-VE ECT "<<ECT<<endl;
                                                    // cout<<"REACTION "<<JJ<<" BETWEEN "<<LS<<" "<<MS<<endl;
                                                    //dout
                                                    //cin.get();
                                                    return ;
                                                }
                                                if(gas->SPEX[3][IEX][LS][MS]<0.e00)
                                                {
                                                    gas->TREACL[3][LS]=gas->TREACL[3][LS]-1;
                                                    gas->TREACL[3][MS]=gas->TREACL[3][MS]-1;
                                                    LS=molecs->IPSP[L] ;
                                                    MS=molecs->IPSP[M] ;
                                                    gas->TREACG[3][LS]=gas->TREACG[3][LS]+1;
                                                    gas->TREACG[3][MS]=gas->TREACG[3][MS]+1;
                                                }
                                                else
                                                {
                                                    gas->TREACL[4][LS]=gas->TREACL[4][LS]-1;
                                                    gas->TREACL[4][MS]=gas->TREACL[4][MS]-1;
                                                    LS=molecs->IPSP[L] ;
                                                    MS=molecs->IPSP[M] ;
                                                    gas->TREACG[4][LS]=gas->TREACG[4][LS]+1;
                                                    gas->TREACG[4][MS]=gas->TREACG[4][MS]+1;
                                                }
                                                RML=gas->SPM[1][LS][MS]/gas->SP[5][MS];
                                                RMM=gas->SPM[1][LS][MS]/gas->SP[5][LS];
                                                //calculate the new VRR to match ECT using the new molecular masses
                                                VRR=2.e00*ECT/gas->SPM[1][LS][MS];
                                                if(gas->ISPV[LS]>0)
                                                {
                                                    for(KV=1;gas->ISPV[LS];KV++)
                                                    {
                                                        if(molecs->IPVIB[KV][L]<0)
                                                        {
                                                            molecs->IPVIB[KV][L]=-99999;
                                                        }
                                                        else
                                                        {
                                                            molecs->IPVIB[KV][L]=0;
                                                        }
                                                    }
                                                }
                                                if(gas->ISPR[1][LS]>0)
                                                    molecs->PROT[L]=0;
                                                if(gas->MELE>1)
                                                    molecs->PELE[L]=0.e00;
                                                if(gas->ISPV[MS]>0)
                                                {
                                                    for(KV=1;gas->ISPV[MS];KV++)
                                                    {
                                                        if(molecs->IPVIB[KV][M]<0)
                                                        {
                                                            molecs->IPVIB[KV][M]=-99999;
                                                        }
                                                        else
                                                        {
                                                            molecs->IPVIB[KV][M]=0;
                                                        }
                                                    }
                                                }
                                                if(gas->ISPR[1][MS]>0)
                                                    molecs->PROT[M]=0;
                                                if(gas->MELE>1)
                                                    molecs->PELE[M]=0.e00;
                                                //set vibrational level of product molecule in exothermic reaction to enforce detailed balance
                                                if(gas->SPEX[3][IEX][LSI][MSI]>0.e00)
                                                {
                                                    //exothermic exchange or chain reaction
                                                    IK=-1; //becomes 0 when the level is chosen
                                                    NK=-1;
                                                    //dout
                                                    //                                                        RANDOM_NUMBER(RANF);
                                                    calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                                    //NTRY=0;
                                                    while(IK<0)
                                                    {
                                                        // NTRY=NTRY+1;
                                                        // if(NTRY>100)
                                                        // {
                                                        //   cout>>"NTRY VibProd"<<NTRY<<endl;
                                                        // }
                                                        NK=NK+1;
                                                        BB=(output->VAR[8][NN]-gas->SPEX[4][IEX][LSI][MSI])*  (gas->SPREX[2][IEX][LSI][MSI][NK]-gas->SPREX[1][IEX][LSI][MSI][NK])/(gas->SPEX[5][IEX][LSI][MSI]-gas->SPEX[4][IEX][LSI][MSI])+gas->SPREX[1][IEX][LSI][MSI][NK];
                                                        if(calc->RANF<BB)
                                                            IK=NK;
                                                    }
                                                    if(gas->NSLEV[1][LS]>0)
                                                    {
                                                        IK+=gas->NSLEV[1][LS];
                                                        gas->NSLEV[1][LS]=0;
                                                    }
                                                    KV=gas->ISPEX[IEX][7][LSI][MSI];
                                                    molecs->IPVIB[KV][L]=IK;
                                                    EVIB=(double)(IK)*BOLTZ*gas->SPVM[1][KV][LS];
                                                    ECT=ECT-EVIB;
                                                    if(ECT<0.e00)
                                                    {
                                                        //NTRY=0;
                                                        while(ECT<0.e00)
                                                        {
                                                            //NTRY+=1;
                                                            // if(NTRY>100)
                                                            //     cout<<"NTRY ECT<0"<<NTRY<<endl;
                                                            molecs->IPVIB[KV][L]=molecs->IPVIB[KV][L]-1;
                                                            gas->NSLEV[1][LS]+=1;
                                                            ECT=ECT+BOLTZ*gas->SPVM[1][KV][LS];
                                                        }
                                                    }
                                                }
                                                else
                                                {
                                                    //for endothermic reaction, select vibration from vib. dist. at macroscopic temperature
                                                    //normal L-B selection would be from the excessively low energy after the endo. reaction
                                                    KV=gas->ISPEX[IEX][5][LS][MS];
                                                    //dout
                                                    SVIB(globalState, LS,output->VAR[8][NN],IK,KV,gas,calc);
                                                    if(gas->NSLEV[2][LS]>0)
                                                    {
                                                        IK=IK+gas->NSLEV[2][LS];
                                                        gas->NSLEV[2][LS]=0;
                                                    }
                                                    molecs->IPVIB[KV][L]=IK;
                                                    EVIB=(double)(IK)*BOLTZ*gas->SPVM[1][KV][LS];
                                                    ECT=ECT-EVIB;
                                                    if(ECT<0.e00)
                                                    {
                                                        //NTRY=0;
                                                        while(ECT<0.e00)
                                                        {
                                                            //NTRY+=1;
                                                            molecs->IPVIB[KV][L]-=1;
                                                            gas->NSLEV[2][LS]+=1;
                                                            ECT=ECT+BOLTZ*gas->SPVM[1][KV][LS];
                                                            // if(NTRY>100)
                                                            // {
                                                            //cout<<"NTRY ECT<0#2"<<NTRY<<endl;
                                                            // molecs->IPVIB[KV][L]=0;
                                                            //   ECT+=EVIB;
                                                            //   gas->NSLEV[2][LS]=0;
                                                            // }
                                                        }
                                                    }
                                                }
                                                //set rotational energy of molecule L to equilibrium at the macroscopic temperature
                                                SROT(globalState, LS,output->VAR[8][NN],molecs->PROT[L],gas,calc);
                                                if(gas->SLER[LS]>1.e-21)
                                                {
                                                    molecs->PROT[L]+=gas->SLER[LS];
                                                    gas->SLER[LS]=1.e-21;
                                                }
                                                ECT-=molecs->PROT[L];
                                                ABA=molecs->PROT[L];
                                                if(ECT<0.e00)
                                                {
                                                    //NTRY=0;
                                                    while(ECT<0.e00)
                                                    {
                                                        //NTRY+=1;
                                                        BB=0.5e00*molecs->PROT[L];
                                                        gas->SLER[LS]+=BB;
                                                        molecs->PROT[L]=BB;
                                                        ECT+=BB;
                                                        // if(NTRY>100)
                                                        // {
                                                        //   cout<<"NTRY ECT<0#3"<<NTRY<<L<<endl;
                                                        //   ECT+=ABA;
                                                        //   molecs->PROT[L]=0;
                                                        //   gas->SLER[LS]=1.e-21;
                                                        // }
                                                    }
                                                }
                                                //calculate the new VRR to match ECT using the new molecular masses
                                                VRR=2.e00*ECT/gas->SPM[1][LS][MS];
                                            }
                                        }
                                    }
                            
                                        //end of reactions other than the deferred dissociation action in the DISSOCIATION subroutine
                                    if(IREC==0 && IDISS==0)
                                    {
                                        //recombined redistribution already made and there is a separate subroutine for dissociation
                                        //Larsen-Borgnakke serial redistribution
                                        ECT=0.5e00*gas->SPM[1][LS][MS]*VRR;
                                        for(NSP=1;NSP<=2;NSP++)
                                        {
                                            if(NSP==1)
                                            {
                                                K=L;KS=LS;JS=MS;
                                            }
                                            else
                                            {
                                                K=M; KS=MS; JS=LS;
                                            }
                                            //now electronic energy for this molecule
                                            if(gas->MELE>1)
                                            {
                                                B=1.e00/gas->QELC[3][1][KS];
                                                //dout
                                                //                                                        RANDOM_NUMBER(RANF);
                                                calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                                if(B>calc->RANF)
                                                {
                                                    NPS=0;
                                                    ECC=ECT+molecs->PELE[K];
                                                    if(gas->NELL[KS]==1){
                                                        NPS=gas->QELC[1][1][KS]; //number of possible states is at least the degeneracy of the ground state
                                                    }
                                                    if(gas->NELL[KS]>1)
                                                    {
                                                        for(NEL=1;NEL<=gas->NELL[KS];NEL++)
                                                        {
                                                            if(ECC>BOLTZ*gas->QELC[2][NEL][KS])
                                                                NPS=NPS+gas->QELC[1][NEL][KS];
                                                        }
                                                        II=0;
                                                        //NTRY=0;
                                                        while(II==0)
                                                        {
                                                            //NTRY+=1;
                                                            // if(NTRY>100)
                                                            //           cout<<"NTRY ElecEn"<<NTRY<<endl;
                                                            //dout
                                                            //                                                                    RANDOM_NUMBER(RANF);
                                                            calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                                            NSTATE=ceil(calc->RANF*NPS);//random state, now determine the energy level
                                                            NAS=0;
                                                            NLEVEL=-1;
                                                            for(NEL=1;NEL<=gas->NELL[KS];NEL++)
                                                            {
                                                                NAS= NAS+gas->QELC[1][NEL][KS];
                                                                if(NSTATE<=NAS && NLEVEL<0)
                                                                    NLEVEL=NEL;
                                                            }
                                                            //dout
                                                            //                                                                    RANDOM_NUMBER(RANF);
                                                            calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                                            if((1.e00/(B*gas->QELC[3][NLEVEL][KS]))<calc->RANF)
                                                            {
                                                                II=1;
                                                            }
                                                            else
                                                            {
                                                                if(ECC>BOLTZ*gas->QELC[2][NLEVEL][KS])
                                                                {
                                                                    PROB=powf(1.e00-BOLTZ*gas->QELC[2][NLEVEL][KS]/ECC,(1.5e00-gas->SPM[3][KS][JS]));
                                                                    //dout
                                                                    //                                                                            RANDOM_NUMBER(RANF);
                                                                    calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                                                    if(PROB>calc->RANF)
                                                                    {
                                                                        II=1;
                                                                        molecs->PELE[K]=BOLTZ*gas->QELC[2][NLEVEL][KS];
                                                                    }
                                                                }
                                                            }
                                                        }
                                                        ECT=ECC-molecs->PELE[K];
                                                    }
                                                }
                                            }
                                            //now the vibrational energy for this molecule
                                            if(gas->MMVM>0 && IEX==0)
                                            {
                                                if(gas->ISPV[KS]>0)
                                                {
                                                    for(KV=1;KV<=gas->ISPV[KS];KV++)
                                                    {
                                                        if(molecs->IPVIB[KV][K]>=0 && IDISS==0) //do not redistribute to a dissociating molecule marked for removal
                                                        {
                                                            EVIB=(double)(molecs->IPVIB[KV][K])*BOLTZ*gas->SPVM[1][KV][KS];
                                                            ECC=ECT+EVIB;
                                                            MAXLEV=ECC/(BOLTZ*gas->SPVM[1][KV][KS]);
                                                            if(gas->SPVM[3][KV][KS]>0.0)
                                                            {   
                                                                B=gas->SPVM[4][KV][KS]/gas->SPVM[3][KV][KS];
                                                                A=gas->SPVM[4][KV][KS]/output->VAR[8][NN];
                                                               ZV = powf(A,gas->SPM[3][KS][JS])*powf((gas->SPVM[2][KV][KS]*powf(B,-gas->SPM[3][KS][JS])),((powf(A,0.3333333e00)-1.e00)/(powf(B,0.33333e00)-1.e00)));
                                                               
                                                            }
                                                            else
                                                                ZV=gas->SPVM[2][KV][KS];
                                                            //                                                                    RANDOM_NUMBER(RANF) //dout
                                                            calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                                            if(1.e00/ZV>calc->RANF ||IREC==1)
                                                            {
                                                                II=0;
                                                                NSTEP=0;
                                                                while(II==0 && NSTEP<100000)
                                                                {
                                                                    NSTEP+=1;
                                                                    if(NSTEP>99000)
                                                                    {
                                                                        printf("%d %f %d\n",NSTEP,ECC,MAXLEV);
                                                                        //cout<<NSTEP<<" "<<ECC<<" "<<MAXLEV<<endl;
                                                                        //dout
                                                                        return ;
                                                                    }
                                                                    //                                                                            RANDOM_NUMBER(RANF);
                                                                    calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                                                    IV=calc->RANF*(MAXLEV+0.99999999e00);
                                                                    molecs->IPVIB[KV][K]=IV;
                                                                    EVIB=(double)(IV)*BOLTZ*gas->SPVM[1][KV][KS];
                                                                    if(EVIB<ECC)
                                                                    {
                                                                        PROB=powf(1.e00-EVIB/ECC,1.5e00-gas->SPVM[3][KS][JS]);
                                                                        //PROB is the probability ratio of eqn (3.28)
                                                                        //                                                                                RANDOM_NUMBER(RANF);
                                                                        calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                                                        if(PROB>calc->RANF)
                                                                            II=1;
                                                                    }
                                                                }
                                                                ECT=ECC-EVIB;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            //now rotation of this molecule
                                            //dout
                                            if(gas->ISPR[1][KS] > 0)
                                            {
                                                if(gas->ISPR[2][KS]==0 && gas->ISPR[2][JS]==0)
                                                {
                                                    B=1.e00/gas->SPM[7][KS][JS];
                                                }
                                                else
                                                    B=1.e00/(gas->SPR[1][KS])+gas->SPR[2][KS]*output->VAR[8][NN]+gas->SPR[3][KS]*powf(output->VAR[8][NN],2);
                                                //                                                        RANDOM_NUMBER(RANF);
                                                calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                                if(B>calc->RANF|| IREC==1)
                                                {
                                                    ECC=ECT+molecs->PROT[K];
                                                    if(gas->ISPR[1][KS]==2)
                                                    {
                                                        //                                                                RANDOM_NUMBER(RANF);
                                                        calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                                        ERM=1.e00-powf(calc->RANF,(1.e00/(2.5e00-gas->SPM[3][KS][JS])));//eqn(5.46)
                                                    }
                                                    else
                                                        LBS(globalState, 0.5e00*gas->ISPR[1][KS]-1.e00,1.5e00-gas->SPM[3][KS][JS],ERM);
                                                    molecs->PROT[K]=ERM*ECC;
                                                    ECT=ECC-molecs->PROT[K];
                                                }
                                            }
                                        }
                                        //adjust VR for the change in energy
                                        VR=sqrtf(2.e00*ECT/gas->SPM[1][LS][MS]);
                                    }//end of L-B redistribution
                                    if(fabsf(gas->SPM[8][LS][MS]-1.0)<0.001)
                                    {
                                        //use the VHS logic
                                        //                                                RANDOM_NUMBER(RANF);
                                        calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                        B=2.e00*calc->RANF-1.e00;
                                        //B is the cosine of a random elevation angle
                                        A=sqrtf(1.e00-B*B);
                                        VRCP[1]=B*VR;
                                        //                                                RANDOM_NUMBER(RANF);
                                        calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                        C=2.e00*PI*calc->RANF;
                                        //C is a random azimuth angle;
                                        VRCP[2]=A*(double)cos(C)*VR;
                                        VRCP[3]=A*(double)sin(C)*VR;
                                    }
                                    else
                                    {
                                        //use the VSS logic
                                        //the VRCP terms do not allow properly for the change in VR - see new book  !STILL TO BE FIXED
                                        VRA=VR/VRI;
                                        //                                                RANDOM_NUMBER(RANF);
                                        calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                        B=2.e00*powf(calc->RANF,gas->SP[4][1])-1.e00;
                                        // B is the cosine of the deflection angle for the VSS model
                                        A=sqrtf(1.e00-B*B);
                                        //                                                RANDOM_NUMBER(RANF);
                                        calc->RANF=generate(globalState, 0);//((double)rand()/(double)RAND_MAX);
                                        C=2.e00*PI*calc->RANF;
                                        OC=(double)cos(C);
                                        SD=(double)sin(C);
                                        D=sqrtf(powf(VRC[2],2)+powf(VRC[3],2));
                                        VRCP[1]=(B*VRC[1]+A*SD*D)*VRA;
                                        VRCP[2]=(B*VRC[2]+A*(VRI*VRC[3]*OC-VRC[1]*VRC[2]*SD)/D)*VRA;
                                        VRCP[3]=(B*VRC[3]+A*(VRI*VRC[2]*OC+VRC[1]*VRC[3]*SD)/D)*VRA;
                                        //the post-collision rel. velocity components are based on eqn (3.18)
                                    }
                                    for(KK=1;KK<=3;KK++)
                                    {
                                        molecs->PV[KK][L]=VCM[KK]+RMM*VRCP[KK];
                                        molecs->PV[KK][M]=VCM[KK]-RMM*VRCP[KK];
                                    }
                                    molecs->IPCP[L]=M;
                                    molecs->IPCP[M]=L;
                                    //call energy(0,E2)
                                    // !              IF (Dfabs(E2-E1) > 1.D-14) read(*,*)
                                }////collision occurrence
                            }
                        }//separate simplegas / mixture coding
                    }
                }
            }
        }
    //remove any recombined atoms
    
}

void COLLISIONS()
{   
    start =clock();
    double duration;
    int N=geom->NCCELLS;
    int gridSize;
    int blockSize=128;
    curandState* devStates;
    memset(calc->COLL_TOTCOL,0.e00,(N+1)*sizeof(double));
    
    gridSize = (N + blockSize - 1) / blockSize;
    cudaMalloc ( &devStates, sizeof( curandState ) );
    // setup seeds
    setup_kernel <<< 1, 1 >>> ( devStates,unsigned(time(NULL)) );

    cuda_collisons<<<gridSize,blockSize >>>(devStates, molecs, output, geom, gas, calc);
    cudaDeviceSynchronize();

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

    //std::cout<<"printf: "<< duration <<'\n';
    colltime=duration;
    for(N=1;N<=geom->NCCELLS;N++){
        calc->TOTCOL=calc->TOTCOL+calc->COLL_TOTCOL[N];
    }
    for(int N=1;N<=molecs->NM;N++)
    {
        if(molecs->IPCELL[N]<0)
            REMOVE_MOL(N); 
    }
    return;
} 


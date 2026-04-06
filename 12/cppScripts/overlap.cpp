// ═══════════════════════════════════════════════════════════════════════════
//  main.cpp  –  Optimised fused pipeline
//
//  Key changes vs. previous version
//  ──────────────────────────────────
//  1. FUSED STAGE 1+2: a single pass over grid points.
//       WignerH runs ONCE per point (not once per (ell,m,spin) triple).
//       Both s=+2 and s=-2 share that single H computation.
//       No Yp/Ym intermediate storage – values are accumulated immediately.
//
//  2. PRECOMPUTED HALF-ANGLE TRIG TABLES: cos/sin(theta/2) and cos/sin(phi/2)
//       computed once per 1-D grid (N_theta + N_phi ops) instead of
//       4 × N_theta × N_phi inside the hot loop.
//
//  3. NO FLAT GRID ARRAYS: theta_full / phi_full removed.
//       The (k,j) index pair maps directly to the 1-D tables.
//
//  4. PER-THREAD ACCUMULATORS: 4-double tuple per (thread, work-item)
//       avoids atomic/reduction overhead across all modes.
//       Merged in O(nthreads × n_work) after the parallel section.
//
//  5. COMPACT Y-INDEX TABLE: Yindex(ell,m) precomputed once outside the
//       point loop so no arithmetic inside the innermost wi-loop.
//
//  6. ONE SphericalHarmonics OBJECT PER THREAD: reused for the entire
//       column of theta values, avoiding repeated allocation/init.
// ═══════════════════════════════════════════════════════════════════════════

#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <string>
#include <limits>
#include <iomanip>
#include <map>
#include <array>

#include "spin_weighted_sh.hpp"

#ifdef _OPENMP
#  include <omp.h>
#endif

using namespace std;
using cd = complex<double>;
const double PI = 3.14159265358979323846;

// ─────────────────────── parameter parsing ────────────────────────────────
map<string,string> read_params_file(const string& path) {
    map<string,string> out;
    ifstream f(path);
    if (!f) { cerr<<"Warning: cannot open '"<<path<<"'. Using defaults.\n"; return out; }
    string line;
    while (getline(f, line)) {
        auto pos_c = line.find('#');
        if (pos_c != string::npos) line = line.substr(0, pos_c);
        auto trim = [](string& s) {
            while (!s.empty() && isspace((unsigned char)s.front())) s.erase(s.begin());
            while (!s.empty() && isspace((unsigned char)s.back()))  s.pop_back();
        };
        trim(line); if (line.empty()) continue;
        auto pos = line.find('='); if (pos == string::npos) continue;
        string key=line.substr(0,pos), val=line.substr(pos+1);
        trim(key); trim(val); out[key]=val;
    }
    return out;
}
double to_double_or(const map<string,string>&m,const string&k,double def){auto it=m.find(k);if(it==m.end())return def;try{return stod(it->second);}catch(...){return def;}}
int    to_int_or   (const map<string,string>&m,const string&k,int    def){auto it=m.find(k);if(it==m.end())return def;try{return stoi(it->second);}catch(...){return def;}}
string to_string_or(const map<string,string>&m,const string&k,const string&def){auto it=m.find(k);if(it==m.end())return def;return it->second;}
bool   to_bool_or  (const map<string,string>&m,const string&k,bool   def){
    auto it=m.find(k);if(it==m.end())return def;
    string v=it->second;for(char&c:v)c=(char)tolower((unsigned char)c);
    if(v=="true"||v=="1"||v=="yes"||v=="y"||v=="on") return true;
    if(v=="false"||v=="0"||v=="no"||v=="n"||v=="off") return false;
    return def;
}

// ─────────────────────── Gauss-Legendre ───────────────────────────────────
void gauss_legendre(int N, vector<double>& nodes, vector<double>& weights) {
    nodes.clear(); weights.clear(); if (N==0) return;
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(N,N);
    for (int n=1;n<N;n++){double b=(double)n/sqrt(4.0*(double)n*(double)n-1.0);J(n-1,n)=b;J(n,n-1)=b;}
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(J);
    if (es.info()!=Eigen::Success) throw runtime_error("Eigen decomp failed");
    nodes.resize(N); weights.resize(N);
    for (int i=0;i<N;i++){nodes[i]=es.eigenvalues()[i];double v0=es.eigenvectors()(0,i);weights[i]=2.0*v0*v0;}
}

// ─────────────────────── tensor helpers ───────────────────────────────────
Eigen::Matrix3cd S_of  (const Eigen::Matrix3cd& T) { return 0.5*(T+T.transpose()); }
Eigen::Matrix3cd TT_of (const Eigen::Matrix3cd& P, const Eigen::Matrix3cd& T) {
    return P*T*P.transpose() - 0.5*P*(P*T).trace();
}
Eigen::Matrix3cd STT_of(const Eigen::Matrix3cd& P, const Eigen::Matrix3cd& T) {
    return S_of(TT_of(P,T));
}

// ─────────────────────── work item (metadata only – no Y storage) ─────────
struct WorkItem {
    int ell, m, mindex;
    int yidx;   // Yindex(ell, m) – precomputed once
};

// ══════════════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    string params_path = "params.txt";
    if (argc >= 2) params_path = argv[1];
    auto params = read_params_file(params_path);
    int lMax = (argc > 2) ? std::stoi(argv[2]) : 10;

    //int    lMax       = to_int_or   (params,"lMax",10);
    int    lMin       = to_int_or   (params,"lMin",2);
    double threshold  = to_double_or(params,"threshold",1e-12);
    double lam        = to_double_or(params,"lam",1.0);
    double dtheta     = to_double_or(params,"dtheta",0.01);
    double deltaTheta = to_double_or(params,"deltaTheta",PI);
    double deltaPhi   = to_double_or(params,"deltaPhi",PI/2.0);
    int    N_theta    = to_int_or   (params,"N_theta",100);
    int    N_phi      = to_int_or   (params,"N_phi",100);
    string mode       = to_string_or(params,"mode","pm2");
    string out_prefix = to_string_or(params,"out_prefix","data_gauss");
    bool   use_phi    = to_bool_or  (params,"use_phi",false);
    int    precision  = to_int_or   (params,"precision",8);

    cout<<"Parameters:\n"
        <<" lMax="<<lMax<<" threshold="<<threshold<<" lam="<<lam<<" dtheta="<<dtheta<<"\n"
        <<" N_theta="<<N_theta<<" N_phi="<<N_phi<<" mode="<<mode<<" use_phi="<<use_phi<<"\n"
        <<" precision="<<precision<<" deltaTheta="<<deltaTheta<<" deltaPhi="<<deltaPhi<<"\n"
        <<" out_prefix="<<out_prefix<<"\n";
    #ifdef _OPENMP
        cout<<"OpenMP enabled. Max threads = "<<omp_get_max_threads()<<"\n"<<flush;
    #else
        cout<<"OpenMP disabled.\n";
    #endif

    // ── Gauss-Legendre ────────────────────────────────────────────────────
    vector<double> u_nodes, u_weights;
    gauss_legendre(N_theta, u_nodes, u_weights);
    vector<double> theta_nodes(N_theta);
    for (int k=0;k<N_theta;k++) theta_nodes[k] = acos(u_nodes[k]);

    vector<double> phi_nodes(N_phi);
    for (int j=0;j<N_phi;j++) phi_nodes[j] = 2.0*PI*j/N_phi;
    double phi_weight = 2.0*PI/N_phi;

    // ── Trig tables (full resolution, 1-D only) ───────────────────────────
    vector<double> cos_theta(N_theta), cos2phi(N_phi), sin2phi(N_phi);
    for (int k=0;k<N_theta;k++) cos_theta[k] = cos(theta_nodes[k]);
    for (int j=0;j<N_phi;j++) { cos2phi[j]=cos(2.0*phi_nodes[j]); sin2phi[j]=sin(2.0*phi_nodes[j]); }

    // ── Half-angle trig tables (optimisation: replaces 4×N transcendentals) ──
    // cos/sin(theta/2) and cos/sin(phi/2) are used to build unit quaternions.
    // Computing them here costs N_theta + N_phi ops rather than 4 × N_theta × N_phi.
    vector<double> cos_t2(N_theta), sin_t2(N_theta);
    vector<double> cos_p2(N_phi),   sin_p2(N_phi);
    for (int k=0;k<N_theta;k++) { cos_t2[k]=cos(theta_nodes[k]*0.5); sin_t2[k]=sin(theta_nodes[k]*0.5); }
    for (int j=0;j<N_phi;j++)   { cos_p2[j]=cos(phi_nodes[j]*0.5);   sin_p2[j]=sin(phi_nodes[j]*0.5); }

    const size_t N = (size_t)N_theta*(size_t)N_phi;

    // ── Fused weight array (unchanged) ───────────────────────────────────
    vector<double> wlam(N);
    {
        double inv2sig2 = 1.0/(2.0*dtheta*dtheta);
        for (int k=0;k<N_theta;k++) {
            double tk=theta_nodes[k];
            double ef_th=exp(-(tk*tk)*inv2sig2);
            double tterm=(tk-deltaTheta)*(tk-deltaTheta);
            double wk=u_weights[k]*phi_weight*lam;
            size_t base=(size_t)k*(size_t)N_phi;
            for (int j=0;j<N_phi;j++) {
                double ef = use_phi
                    ? exp(-(tterm+(phi_nodes[j]-deltaPhi)*(phi_nodes[j]-deltaPhi))*inv2sig2)
                    : ef_th;
                wlam[base+(size_t)j] = wk*ef;
            }
        }
    }

    // ── Coefficient matrices C1/C2 (unchanged) ───────────────────────────
    Eigen::Matrix3cd P = Eigen::Matrix3cd::Identity();
    {Eigen::Vector3cd e1; e1<<cd(1,0),cd(0,0),cd(0,0); P-=e1*e1.transpose();}

    Eigen::Vector3cd M_vec, MB_vec;
    M_vec <<cd(0,0),cd(1,0),cd(0, 1);
    MB_vec<<cd(0,0),cd(1,0),cd(0,-1);
    Eigen::Matrix3cd MM=M_vec*M_vec.transpose(), MBB=MB_vec*MB_vec.transpose();

    cd conjMM[9], conjMBB[9];
    for (int r=0;r<3;r++) for (int c=0;c<3;c++) { conjMM[r*3+c]=std::conj(MM(r,c)); conjMBB[r*3+c]=std::conj(MBB(r,c)); }

    Eigen::Matrix3cd C1=Eigen::Matrix3cd::Zero(), C2=Eigen::Matrix3cd::Zero();
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) {
        Eigen::Matrix3cd Tb=Eigen::Matrix3cd::Zero(); Tb(a,b)=cd(1,0);
        Eigen::Matrix3cd E0=STT_of(P,Tb);
        cd c1(0,0), c2(0,0);
        for (int r=0;r<3;r++) for (int c_=0;c_<3;c_++) { cd e=E0(r,c_); int i=r*3+c_; c1+=e*conjMM[i]; c2+=e*conjMBB[i]; }
        C1(a,b)=c1; C2(a,b)=c2;
    }
    const cd C1_11=C1(1,1),C1_12=C1(1,2),C1_21=C1(2,1),C1_22=C1(2,2);
    const cd C2_11=C2(1,1),C2_12=C2(1,2),C2_21=C2(2,1),C2_22=C2(2,2);

    // ── F1/F2 arrays (unchanged) ──────────────────────────────────────────
    vector<cd> F1(N), F2(N);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int k=0;k<N_theta;k++) {
        double ct=cos_theta[k];
        size_t base=(size_t)k*(size_t)N_phi;
        for (int j=0;j<N_phi;j++) {
            size_t idx=base+(size_t)j;
            double c2p=cos2phi[j], s2p=sin2phi[j];
            cd T11(c2p*ct*ct,0), T12(-s2p*ct,0), T21=T12, T22(c2p,0);
            cd S1b=T11*C1_11+T12*C1_12+T21*C1_21+T22*C1_22;
            cd S2b=T11*C2_11+T12*C2_12+T21*C2_21+T22*C2_22;
            double w=wlam[idx];
            F1[idx]=cd(w,0)*S1b;
            F2[idx]=cd(w,0)*S2b;
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    //  Build work-item list (metadata only – no SH computation here)
    //
    //  Previously Stage 1 called sYlm_batch per (ell,m,spin) triple and
    //  stored N complex values per call.  Now we only need the Yindex.
    // ════════════════════════════════════════════════════════════════════════
    vector<WorkItem> work_items;
    {
        int M_modes = 2*lMax+1;
        for (int ell=lMin; ell<=lMax; ++ell) {
            for (int mindex=0; mindex<2*ell+1; ++mindex) {
                int m = mindex - ell;
                if (mode!="all" && abs(m)!=2) continue;
                WorkItem wi;
                wi.ell    = ell;
                wi.m      = m;
                wi.mindex = mindex;
                wi.yidx   = sylm::Yindex(ell, m);  // precomputed once
                work_items.push_back(wi);
            }
        }
    }
    const int n_work = (int)work_items.size();
    cout<<"Work items: "<<n_work<<"\n"<<flush;

    // Convenience: compact array of Y-indices for the hot inner loop
    vector<int> yindices(n_work);
    for (int wi=0; wi<n_work; ++wi) yindices[wi] = work_items[wi].yidx;

    // ════════════════════════════════════════════════════════════════════════
    //  FUSED STAGE 1 + STAGE 2 – single pass over grid points
    //
    //  Threading model:
    //    - outer loop is over theta rows (k), parallelised with OpenMP
    //    - each thread owns:
    //        * one SphericalHarmonics object (mp_max=2, covers s=±2)
    //        * two Y-value buffers (Yp for s=+2, Ym for s=-2)
    //        * one block of local accumulators [n_work][4]
    //    - after the parallel section, per-thread accumulators are merged
    //
    //  Per grid point cost:
    //    OLD: n_work calls to sYlm_batch, each constructing WignerH
    //         ≈ n_work × (WignerH cost + N alloc)
    //    NEW: 1 × compute_H + 2 × fill_Y
    //         ≈ (WignerH cost) + 2 × fill_Y(lMax)
    // ════════════════════════════════════════════════════════════════════════
    cout<<"[Fused Stage 1+2] Single pass over "<<N_theta<<"×"<<N_phi<<" points...\n"<<flush;

    int nthreads = 1;
    #ifdef _OPENMP
    nthreads = omp_get_max_threads();
    #endif

    // Per-thread accumulator block: [thread][wi][{Ar,Ai,Br,Bi}]
    // Using a flat vector with stride n_work to avoid false sharing concerns.
    // Each block is 4 doubles × n_work, allocated once outside the loop.
    const int ACC_STRIDE = 4;
    vector<double> thread_accs((size_t)nthreads * (size_t)n_work * ACC_STRIDE, 0.0);

    #ifdef _OPENMP
    #pragma omp parallel default(none) \
        shared(theta_nodes,phi_nodes,cos_t2,sin_t2,cos_p2,sin_p2, \
               F1,F2,work_items,yindices,thread_accs, \
               N_theta,N_phi,n_work,lMax)
    #endif
    {
        int tid = 0;
    #ifdef _OPENMP
        tid = omp_get_thread_num();
    #endif
        // This thread's accumulator slice
        double* local_acc = thread_accs.data() + (size_t)tid * (size_t)n_work * ACC_STRIDE;

        // One SH object per thread, reused across all points.
        // mp_max=2 covers both s=+2 and s=-2.
        sylm::SphericalHarmonics sh(lMax, /*mp_max=*/2);
        vector<cd> Yp, Ym;
        Yp.reserve(sylm::Ysize(lMax));
        Ym.reserve(sylm::Ysize(lMax));

        #ifdef _OPENMP
        #pragma omp for schedule(static)
        #endif
        for (int k = 0; k < N_theta; ++k) {
            const double ct2 = cos_t2[k], st2 = sin_t2[k];
            const size_t base = (size_t)k * (size_t)N_phi;

            for (int j = 0; j < N_phi; ++j) {
                const double cp2 = cos_p2[j], sp2 = sin_p2[j];

                // Unit quaternion R = Rz(phi) * Ry(theta)
                // w=ct2*cp2, x=+st2*sp2, y=st2*cp2, z=ct2*sp2
                const array<double,4> R = { ct2*cp2, +st2*sp2, st2*cp2, ct2*sp2 };

                // ── One WignerH recurrence for this point ──────────────────
                sh.compute_H(R);

                // ── Two fill_Y calls sharing that H ────────────────────────
                sh.fill_Y(+2, Yp);
                sh.fill_Y(-2, Ym);

                const size_t pt = base + (size_t)j;
                const double f1r = F1[pt].real(), f1i = F1[pt].imag();
                const double f2r = F2[pt].real(), f2i = F2[pt].imag();

                // ── Inner accumulation over work items ─────────────────────
                // yindices[] was precomputed; no Yindex arithmetic here.
                for (int wi = 0; wi < n_work; ++wi) {
                    const int idx = yindices[wi];

                    const double ymr  =  Ym[idx].real();
                    const double ymni = -Ym[idx].imag();   // conj(Ym).imag negated
                    const double ypr  =  Yp[idx].real();
                    const double ypni = -Yp[idx].imag();

                    // conj(Ym)*F1
                    const double p1r = ymr*f1r - ymni*f1i;
                    const double p1i = ymr*f1i + ymni*f1r;
                    // conj(Yp)*F2
                    const double p2r = ypr*f2r - ypni*f2i;
                    const double p2i = ypr*f2i + ypni*f2r;

                    double* a = local_acc + wi*ACC_STRIDE;
                    a[0] += p1r + p2r;           // A real
                    a[1] += p1i + p2i;           // A imag
                    const double dr=p1r-p2r, di=p1i-p2i;
                    a[2] += -di;                 // B real
                    a[3] +=  dr;                 // B imag
                }
            }
        }
    } // end parallel

    cout<<"[Fused Stage 1+2] Done.\n\n"<<flush;

    // ── Merge per-thread accumulators ─────────────────────────────────────
    vector<double> Acc_A_re(n_work,0.0), Acc_A_im(n_work,0.0);
    vector<double> Acc_B_re(n_work,0.0), Acc_B_im(n_work,0.0);
    for (int t=0; t<nthreads; ++t) {
        const double* p = thread_accs.data() + (size_t)t*(size_t)n_work*ACC_STRIDE;
        for (int wi=0; wi<n_work; ++wi) {
            Acc_A_re[wi] += p[wi*ACC_STRIDE+0];
            Acc_A_im[wi] += p[wi*ACC_STRIDE+1];
            Acc_B_re[wi] += p[wi*ACC_STRIDE+2];
            Acc_B_im[wi] += p[wi*ACC_STRIDE+3];
        }
    }

    // ── Scatter into flat result matrix (unchanged) ───────────────────────
    int M_len = 2*lMax+1;
    auto NAN_ = numeric_limits<double>::quiet_NaN();
    vector<double> A_re_flat((size_t)(lMax)*(size_t)M_len, NAN_);
    vector<double> A_im_flat((size_t)(lMax)*(size_t)M_len, NAN_);
    vector<double> B_re_flat((size_t)(lMax)*(size_t)M_len, NAN_);
    vector<double> B_im_flat((size_t)(lMax)*(size_t)M_len, NAN_);

    for (int wi=0; wi<n_work; ++wi) {
        int ell=work_items[wi].ell, mindex=work_items[wi].mindex;
        int col=(lMax-ell)+mindex, row=ell-1;
        if (col<0||col>=M_len) continue;
        size_t fi=(size_t)row*(size_t)M_len+(size_t)col;
        A_re_flat[fi]=Acc_A_re[wi]; A_im_flat[fi]=Acc_A_im[wi];
        B_re_flat[fi]=Acc_B_re[wi]; B_im_flat[fi]=Acc_B_im[wi];
    }

    // ── Output (unchanged) ────────────────────────────────────────────────
    auto to_str=[](double x,int p){ostringstream ss;ss<<setprecision(p)<<scientific<<x;return ss.str();};
    vector<string> A_re_list{"A_re_V"},A_im_list{"A_im_V"};
    vector<string> B_re_list{"B_re_V"},B_im_list{"B_im_V"};
    vector<string> m_list{"M"},l_list{"L"};

    for (int li=lMin;li<=lMax;li++) for (int mi=0;mi<M_len;mi++) {
        bool got=false; double Ar=0,Ai=0,Br=0,Bi=0;
        size_t idx=(size_t)(li-1)*(size_t)M_len+(size_t)mi;
        auto chk=[&](double v,double&d){if(!std::isnan(v)&&fabs(v)>threshold){got=true;d=v;}};
        chk(A_re_flat[idx],Ar);chk(A_im_flat[idx],Ai);
        chk(B_re_flat[idx],Br);chk(B_im_flat[idx],Bi);
        if (got) {
            A_re_list.push_back(to_str(Ar,precision)); A_im_list.push_back(to_str(Ai,precision));
            B_re_list.push_back(to_str(Br,precision)); B_im_list.push_back(to_str(Bi,precision));
            m_list.push_back(to_str(mi-(M_len/2),precision));
            l_list.push_back(to_str(li,precision));
        }
    }

    auto safe=[&](double x)->string{
        ostringstream ss;ss<<fixed<<setprecision(precision)<<x;string s=ss.str();
        for(char&c:s)if(c=='.')c='p';
        while(!s.empty()&&s.back()=='0')s.pop_back();
        if(!s.empty()&&s.back()=='p')s.pop_back();
        return s;
    };
    string fname=out_prefix+"_"+safe(dtheta)+"_lMax_"+to_string(lMax)+".csv";
    ofstream csvf(fname);
    if (!csvf) { cerr<<"Error: cannot open '"<<fname<<"'\n"; return 1; }
    auto write_row=[&](const vector<string>& row){
        for(size_t i=0;i<row.size();i++){csvf<<row[i];if(i+1<row.size())csvf<<",";}csvf<<"\n";
    };
    write_row(A_re_list); write_row(A_im_list);
    write_row(B_re_list); write_row(B_im_list);
    write_row(m_list);    write_row(l_list);
    csvf.close();
    cout<<"Wrote CSV to '"<<fname<<"'\nDone.\n";
    return 0;
}
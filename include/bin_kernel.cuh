#ifndef CHIP_BIN_KERNEL_H
#define CHIP_BIN_KERNEL_H

#include "bin.cuh"
#include "cuda_string.cuh"
#include "incgammabeta.h"
#include "reduction_kernel.cuh"

__device__ void gpu_try_assign_kernel(uint64_t bin_start, uint64_t bin_end,
                                      uint32_t id, uint64_t *d_starts,
                                      int32_t numOfEntry, Assist *d_assist) {
    uint64_t start;
    int32_t left = 0, right = numOfEntry, mid_l, mid_r;
    // search for minimum boundary
    while (left < right) {
        mid_l = (left + right) / 2;
        start = d_starts[mid_l];
        if (start < bin_start) left = mid_l + 1;
        else right = mid_l;
    }
    //while(right>0 and d_starts[right]==d_starts[right-1]){
    //	right = right-1;
    //	}
    if (left != numOfEntry) d_assist[id].start_ = right;
    else {
        d_assist[id].start_ = d_assist[id].end_ = 0;
        return;
    }
    // search for maximum boundary
    left = 0;
    right = numOfEntry;
    while (left < right) {
        mid_r = (left + right) / 2;
        start = d_starts[mid_r];
        if (start < bin_end) left = mid_r + 1;
        else right = mid_r;
    }
    //while(left<numOfEntry and d_starts[left]==d_starts[left+1]){
    //	left = left+1;
    //	}

    if (left) d_assist[id].end_ = left;
    else {
        d_assist[id].start_ = d_assist[id].end_ = 0;
    }
}

__global__ void gpu_assign_nj_read_kernel(d_Bins d_bins, int32_t numOfBin,
                                       d_nj_Reads d_reads, int32_t numOfRead,
                                       Assist *d_assist,int32_t *d_read2bin_start,int32_t *d_read2bin_end) {
    int32_t binId = blockDim.x * blockIdx.x + threadIdx.x;
    int temp = 0;

    if (binId < numOfBin) {
        // try assign
        gpu_try_assign_kernel(
            d_bins.start_[binId], d_bins.end_[binId],
            binId, d_reads.start_, numOfRead, d_assist
        );
        __threadfence();
        d_read2bin_start[binId] = d_assist[binId].start_;
        d_read2bin_end[binId] = d_assist[binId].end_;
        for (int readId = d_assist[binId].start_; readId < d_assist[binId].end_; readId++) {
            if ((d_reads.strand[readId] != d_bins.strand[binId]) ||
                    (d_reads.end_[readId] > d_bins.end_[binId])) temp++;
        }
        d_bins.core[binId].readCount  = d_assist[binId].end_ - d_assist[binId].start_ - temp;
// #define DEBUG
#ifdef DEBUG
        printf("read count: %d\n", d_bins.core[binId].readCount);
#endif
    }
}

__global__ void gpu_assign_read_kernel(d_Bins d_bins, int32_t numOfBin,
                                       d_Reads d_reads, int32_t numOfRead,
                                       Assist *d_assist,int32_t *d_read2bin_start,int32_t *d_read2bin_end) {
    int32_t binId = blockDim.x * blockIdx.x + threadIdx.x;
    int temp = 0;

    if (binId < numOfBin) {
        // try assign
        gpu_try_assign_kernel(
            d_bins.start_[binId], d_bins.end_[binId],
            binId, d_reads.start_, numOfRead, d_assist
        );
        __threadfence();
        d_read2bin_start[binId] = d_assist[binId].start_;
        d_read2bin_end[binId] = d_assist[binId].end_;
        for (int readId = d_assist[binId].start_; readId < d_assist[binId].end_; readId++) {
            if ((d_reads.strand[readId] != d_bins.strand[binId]) ||
                    (d_reads.end_[readId] > d_bins.end_[binId])) temp++;
        }
        d_bins.core[binId].readCount += d_assist[binId].end_ - d_assist[binId].start_ - temp;
// #define DEBUG
#ifdef DEBUG
        printf("read count: %d\n", d_bins.core[binId].readCount);
#endif
    }
}

__global__ void gpu_count_tempTPM(d_Bins d_bins, int32_t numOfBin, float *d_tempTPM) {
    int32_t binId = blockDim.x * blockIdx.x + threadIdx.x;

    if (binId < numOfBin) {
        d_tempTPM[binId] = float(d_bins.core[binId].readCount) / \
                  float(d_bins.end_[binId] -  d_bins.start_[binId]);
//#define DEBUG
#ifdef DEBUG
        printf("d_tempTPM: %f\n", d_tempTPM[binId]);
#endif
    }
}

__global__ void gpu_count_TPM(d_Bins d_bins, int32_t numOfBin,
                              float *d_tempTPM, float *d_tpmCounter, float *d_tpmStore) {
    int32_t binId = blockDim.x * blockIdx.x + threadIdx.x;

    if (binId < numOfBin) {
        // make sure d_tpmCounter is not zero
        if (*d_tpmCounter == 0) return;
        // compute tpmCount for each bin
        float tmp;
        tmp = 1000000 * d_tempTPM[binId] / (*d_tpmCounter);
        d_bins.core[binId].tpmCount = tmp;
       	d_tpmStore[binId] = tmp;
//#define DEBUG
#ifdef DEBUG
        printf("d_tpmCounter: %f\n", *d_tpmCounter);
#endif
    }
}

__global__ void gpu_assign_ASE_kernel(d_Bins d_bins, int32_t numOfBin,
                                      d_ASEs d_ases, int32_t numOfASE,
                                      Assist *d_assist, int32_t *d_bin2ase) {
    int32_t binId = blockDim.x * blockIdx.x + threadIdx.x;

    if (binId < numOfBin) {
        // try assign
        gpu_try_assign_kernel(
                d_bins.start_[binId], d_bins.end_[binId],
                binId, d_ases.start_, numOfASE, d_assist
        );
        __threadfence();

        // assign
        for (int aseId = d_assist[binId].start_; aseId < d_assist[binId].end_; aseId++) {
            if ((d_ases.strand[aseId] == d_bins.strand[binId]) &&
                    (d_ases.end_[aseId] <= d_bins.end_[binId])) {
                d_ases.core[aseId].bin_h = d_bins.core[binId].name_h;
                d_bin2ase[aseId] = binId;
            } else {
                d_ases.core[aseId].bin_h = 0;
                d_bin2ase[aseId] = binId;
//                d_bin2ase[aseId] = 0;
            }
        }

//        if (d_bins.core[binId].name_h == 3115368877389788556L) {
//            printf("%lu %lu %d %d\n", d_bins.start_[binId], d_bins.end_[binId],
//                   d_assist[binId].start_, d_assist[binId].end_);
//            for (int i = 0; i < numOfASE; i++) {
//                if (d_ases.start_[i] == 247002400L) printf("%lu %lu\n", d_ases.end_[i], d_ases.core[i].bin_h);
//            }
//        }
    }
}

__global__ void gpu_assign_read_ASE_kernel(d_ASEs d_ases, int32_t numOfASE,
                                           d_Reads d_reads, int32_t numOfRead,
                                           Assist *d_assist, ASECounter *ACT) {
    int32_t aseId = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t read_strand, ase_strand, junctionCount;
    uint32_t read_s, read_e, junction_s, junction_e;
    int32_t *coord;

    if (aseId < numOfASE) {
        // try assign
        gpu_try_assign_kernel(
                d_ases.start_[aseId], d_ases.end_[aseId],
                aseId, d_reads.start_, numOfRead, d_assist
        );
        __threadfence();

        // for calc psi
        coord = d_ases.core[aseId].coordinates;
        ACT[aseId].artRange.start_ = coord[2];
        ACT[aseId].artRange.end_ = coord[3];

        // assign
        for (int readId = d_assist[aseId].start_; readId < d_assist[aseId].end_; readId++) {
            read_strand = d_reads.strand[readId];
            ase_strand = d_ases.strand[aseId];
            if (read_strand == ase_strand) {
                read_s = uint32_t(d_reads.start_[readId] & (refLength - 1));
                read_e = uint32_t(d_reads.end_[readId] & (refLength - 1));
#ifdef SE_ANCHOR
                // JTAT
                junctionCount = d_reads.core[readId].junctionCount;
                if (junctionCount) {
                    #pragma unroll
                    for (int jId = 0; jId < junctionCount; jId++) {
                        junction_s = d_reads.core[readId].junctions[jId].start_ + read_s - 1;
                        junction_e = d_reads.core[readId].junctions[jId].end_ + read_s;
                        if (ase_strand) {
                            if (junction_s == coord[1] && junction_e == coord[2]) ACT[aseId].anchor[0]++;
                            if (junction_s == coord[3] && junction_e == coord[4]) ACT[aseId].anchor[1]++;
                            if (junction_s == coord[1] && junction_e == coord[4]) ACT[aseId].anchor[2]++;
                        } else {
                            if (junction_s == coord[5] && junction_e == coord[2]) ACT[aseId].anchor[0]++;
                            if (junction_s == coord[3] && junction_e == coord[0]) ACT[aseId].anchor[1]++;
                            if (junction_s == coord[5] && junction_e == coord[0]) ACT[aseId].anchor[2]++;
                        }

                    }
                } else {
                    // ART
                    if ((read_s >= coord[2] && read_s <= coord[3]) ||
                        (read_e >= coord[2] && read_e <= coord[3])) {
                        ACT[aseId].anchor[3]++;
                    }
                }
#elif defined(RI_ANCHOR)
                // JTAT
                junctionCount = d_reads.core[readId].junctionCount;
                if (junctionCount) {
                    #pragma unroll
                    for (int jId = 0; jId < junctionCount; jId++) {
                        junction_s = d_reads.core[readId].junctions[jId].start_ + read_s - 1;
                        junction_e = d_reads.core[readId].junctions[jId].end_ + read_s;
                        if (ase_strand) {
                            if (junction_s == coord[1] && junction_e == coord[2])
                                ACT[aseId].anchor[0]++;
                        } else {
                            if (junction_s == coord[2] && junction_e == coord[1])
                                ACT[aseId].anchor[0]++;
                        }

                    }
                } else {
                    // ART
                    if (ase_strand) {
                        if ((read_s >= coord[1] && read_s <= coord[2]) ||
                            (read_e >= coord[1] && read_e <= coord[2])) {
                            ACT[aseId].anchor[1]++;
                        }
                    } else {
                        if ((read_s >= coord[2] && read_s <= coord[1]) ||
                            (read_e >= coord[2] && read_e <= coord[1])) {
                            ACT[aseId].anchor[1]++;
                        }
                    }
                }
#endif
            }
        }
//        if (d_ases.start_[aseId] == 764383L && d_ases.end_[aseId] == 787490L) {
//            printf("%d %d\n", d_assist[aseId].end_, d_assist[aseId].start_);
//            for (int i = 0; i < anchorCount; i++) printf("%d\n", ACT[aseId].anchor[i]);
//        }
// #define DEBUG
#ifdef DEBUG
    if (aseId == 0)
        for (int i = 0; i < anchorCount; i++) printf("%d %d\n", aseId, ACT[aseId].anchor[i]);
#endif
    }
}
__global__ void gpu_assign_read_ASE_kernel2(d_ASEs d_ases, int32_t numOfASE,
                                           d_Reads d_reads, int32_t numOfRead,
                                            Assist *d_assist, ASECounter *ACT,int32_t *d_bin2ase,int32_t * d_read2bin_start,int32_t *d_read2bin_end) {
    int32_t aseId = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t read_strand, ase_strand, junctionCount;
    uint32_t read_s, read_e;
    uint64_t junction_s, junction_e;
    int32_t *coord;

    if (aseId < numOfASE) {
        // try assign
        /*
        gpu_try_assign_kernel(
                d_ases.start_[aseId], d_ases.end_[aseId],
                aseId, d_reads.start_, numOfRead, d_assist
        );
        */
        //__threadfence();

        // for calc psi
        coord = d_ases.core[aseId].coordinates;
        ACT[aseId].artRange.start_ = coord[2];
        ACT[aseId].artRange.end_ = coord[3];
        uint32_t binId;
        binId = d_bin2ase[aseId];
        // assign
        //for (int readId = d_assist[binId].start_; readId < d_assist[binId].end_; readId++) {
        for (int readId = d_read2bin_start[binId]; readId < d_read2bin_end[binId]; readId++) {
            read_strand = d_reads.strand[readId];
            ase_strand = d_ases.strand[aseId];
            //if (read_strand == ase_strand) {
            if ( true ) {
                read_s = uint32_t(d_reads.start_[readId] & (refLength - 1));
                read_e = uint32_t(d_reads.end_[readId] & (refLength - 1));
#ifdef SE_ANCHOR
                // JTAT
                junctionCount = d_reads.core[readId].junctionCount;
                if (junctionCount) {
                    //#pragma unroll
                    for (int jId = 0; jId < junctionCount; jId++) {
                        //junction_s = d_reads.core[readId].junctions[jId].start_ + read_s - 1;
                        junction_s = d_reads.core[readId].junctions[jId].start_ + read_s - 1;
                        //junction_e = d_reads.core[readId].junctions[jId].end_ + read_s;
                        junction_e = d_reads.core[readId].junctions[jId].end_ + read_s;
                        
                        if (ase_strand) {
                            if (junction_s == coord[1] && junction_e == coord[2]) ACT[aseId].anchor[0]++;
                            if (junction_s == coord[3] && junction_e == coord[4]) ACT[aseId].anchor[1]++;
                            if (junction_s == coord[1] && junction_e == coord[4]) ACT[aseId].anchor[2]++;
                        } else {
                            if (junction_s == coord[5] && junction_e == coord[2]) ACT[aseId].anchor[0]++;
                            if (junction_s == coord[3] && junction_e == coord[0]) ACT[aseId].anchor[1]++;
                            if (junction_s == coord[5] && junction_e == coord[0]) ACT[aseId].anchor[2]++;
                        }
                        /*
                            if (junction_s == coord[1] && junction_e == coord[2]) ACT[aseId].anchor[0]++;
                            if (junction_s == coord[3] && junction_e == coord[4]) ACT[aseId].anchor[1]++;
                            if (junction_s == coord[1] && junction_e == coord[4]) ACT[aseId].anchor[2]++;
			*/

                    }
                } else {
                    // ART
                    if ((read_s >= coord[2] && read_s <= coord[3]) ||
                        (read_e >= coord[2] && read_e <= coord[3])) {
                        ACT[aseId].anchor[3]++;
                    }
                }
#elif defined(RI_ANCHOR)
                // JTAT
                junctionCount = d_reads.core[readId].junctionCount;
                if (junctionCount) {
                    #pragma unroll
                    for (int jId = 0; jId < junctionCount; jId++) {
                        junction_s = d_reads.core[readId].junctions[jId].start_ + read_s - 1;
                        junction_e = d_reads.core[readId].junctions[jId].end_ + read_s;
                        if (ase_strand) {
                            if (junction_s == coord[1] && junction_e == coord[2])
                                ACT[aseId].anchor[0]++;
                        } else {
                            if (junction_s == coord[2] && junction_e == coord[1])
                                ACT[aseId].anchor[0]++;
                        }

                    }
                } else {
                    // ART
                    if (ase_strand) {
                        if ((read_s >= coord[1] && read_s <= coord[2]) ||
                            (read_e >= coord[1] && read_e <= coord[2])) {
                            ACT[aseId].anchor[1]++;
                        }
                    } else {
                        if ((read_s >= coord[2] && read_s <= coord[1]) ||
                            (read_e >= coord[2] && read_e <= coord[1])) {
                            ACT[aseId].anchor[1]++;
                        }
                    }
                }
#endif
            }
        }
//        if (d_ases.start_[aseId] == 764383L && d_ases.end_[aseId] == 787490L) {
//            printf("%d %d\n", d_assist[aseId].end_, d_assist[aseId].start_);
//            for (int i = 0; i < anchorCount; i++) printf("%d\n", ACT[aseId].anchor[i]);
//        }
// #define DEBUG
#ifdef DEBUG
    if (aseId == 0)
        for (int i = 0; i < anchorCount; i++) printf("%d %d\n", aseId, ACT[aseId].anchor[i]);
#endif
    }
}

 __global__ void gpu_count_PSI(d_ASEs d_ases, int32_t numOfASE,
                               ASEPsi *d_ase_psi, ASECounter *ACT) {
     int32_t aseId = blockDim.x * blockIdx.x + threadIdx.x;
     float countOut;
     float countIn, psi;
     ASECounter act;
     if (aseId == 17){
	printf("aseId 17");
	}
     if (aseId < numOfASE) {
        act = ACT[aseId];
#ifdef SE_ANCHOR
        //countIn = act.anchor[0] + act.anchor[1] + \
        //       act.anchor[3] / float(act.artRange.end_ - act.artRange.start_);
	countIn = act.anchor[0] + act.anchor[1] + \
               act.anchor[3] ;

        countOut = act.anchor[2];
        if (act.anchor[3]) {
            psi = (countIn / 3) / (countIn / 3 + countOut);
        } else {
            psi = (countIn / 2) / (countIn / 2 + countOut);
        }
#elif defined(RI_ANCHOR)
        countIn = float(act.anchor[0]);
        countOut = act.anchor[1];
        psi = countIn / (countIn + countOut);
#endif

        // store into d_ase_psi
        d_ase_psi[aseId] = ASEPsi {
             d_ases.core[aseId].gid_h,
             d_ases.core[aseId].bin_h,
             countIn, countOut, psi, 0, 0
        };

        __threadfence();

 //#define DEBUG
 #ifdef DEBUG
         if (aseId == 0) {
             for (int i = 0; i < numOfASE; i++) {
                 printf("gid: %d, countIn: %.2f,  countOut: %.2f\n",
                            d_ase_psi[i].gid_h, d_ase_psi[i].countIn, d_ase_psi[i].countOut);
             }
         }
 #endif
     }
 }
#define BETAINV
#ifdef BETAINV
__global__ void gpu_post_PSI(ASEPsi *d_ase_psi,ASECounter *ACT, float *PSI_UB,
                             float *PSI_LB, int32_t numOfASE)
{
    int32_t aseId = blockDim.x * blockIdx.x + threadIdx.x;
    double countIn;
    double countOut;
    float psi_ub;
    float psi_lb;
    float eps;
    eps = 1.0e-5;
    ASECounter act;
    if (aseId < numOfASE) {
        act = ACT[aseId];
        if (act.anchor[3]){
        countIn = d_ase_psi[aseId].countIn/3;
	}
	else{
	countIn = d_ase_psi[aseId].countIn/2;
	}
        countOut = d_ase_psi[aseId].countOut;
        psi_ub = 1 - invbetai(0.025, countOut, countIn + 1);
        psi_lb = 1 - invbetai(0.975, countOut + 1, countIn);

        if (fabs(countIn) <eps || fabs(countOut) < eps) {
            if (countIn+countOut>=5){
            psi_ub = 1;
            psi_lb = 1;
            } 
            else {
            psi_ub = 1;
            psi_lb = 0;
                }
        }
        PSI_UB[aseId]=psi_ub;
        PSI_LB[aseId]=psi_lb;
    }
}
#endif
template <class T>
__global__ void gather(int* indices,T *source,T *out,uint32_t numOfEntry){
	int32_t threadId = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadId < numOfEntry){
		int32_t targetId= indices[threadId];
		out[threadId] = source[targetId];
	}		
}
#endif //CHIP_BIN_KERNEL_H

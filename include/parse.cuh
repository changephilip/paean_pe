#ifndef _CHIP_PARSE_CUH_
#define _CHIP_PARSE_CUH_

#include "bin.cuh"
#include "gff.h"
#include "htslib/sam.h"
#include "sys/sysinfo.h"

#include <string>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <stack>
#include <set>
#include <map>
// global variables
typedef std::unordered_map<size_t, std::string> UMAP;
typedef std::unordered_map<size_t, size_t> NMAP;
UMAP g_gid_map, g_name_map;
NMAP bin_id_map;
std::hash<std::string> HASH;

uint64_t _offset(char *chr) {
    char *chr_t = chr + 3;
    if (c_strlen(chr_t) > 2) return invalidLength;
    if (is_digitstr(chr_t)) return ((atoi(chr_t) - 1) * refLength);
    else if (*chr_t == 'M') return (22 * refLength);
    else if (*chr_t == 'X') return (23 * refLength);
    else if (*chr_t == 'Y') return (24 * refLength);
    else return invalidLength;
}

void LoadBinFromGff(h_Bins& h_bins, char* gff_file) {
    GffReader reader(gff_file);
    reader.readAll(true);

    std::string gene_name;
    size_t hash_t;
    size_t nfeat = reader.gflst.Count();
    for (size_t i=0; i < nfeat; ++i) {
        GffObj* f = reader.gflst[i];
        if (f->isGene()) {
            // gene name
            gene_name = std::string(f->getGeneName());
            hash_t = HASH(gene_name);
            g_name_map.insert({hash_t, gene_name});
            // chromosome id
            uint64_t offset = _offset(const_cast<char *>(f->getGSeqName()));

            h_bins.start_.push_back(f->start + offset);
            h_bins.end_.push_back(f->end + offset);
            h_bins.strand.push_back((f->strand == '+'));
            h_bins.core.push_back(bin_core_t(hash_t));
//#define DEBUG
#ifdef DEBUG
            std::cout << "gene name: " << hash_t << std::endl;
            std::cout << "gene start: " << f->start << std::endl;
            std::cout << "gene end: " << f->end << std::endl;
            std::cout << "gene strand: " << (f->strand == '+') << std::endl;
#endif
        }
    }
}

void LoadAseFromGFF(h_ASEs& h_ases, char *gff_file) {
    GffReader reader(gff_file);
    reader.readAll(true);

    std::string gid;
    size_t hash_t;
    size_t nfeat = reader.gflst.Count();
    for (size_t i=0; i < nfeat; ++i) {
        GffObj *f = reader.gflst[i];
        if (f->isGene()) {
            // gid
            gid = std::string(f->getAttr("gid"));
            hash_t = HASH(gid);
            g_gid_map.insert({hash_t, gid});

            if (f->attrs) {
                // parse gid to coordinates
                char tGid[gidSize];
                std::strncpy(tGid, gid.data(), sizeof(char) * gidSize);

                int coordinateId = 0;
                char *save_ptr = NULL;
                char *split_str = c_strtok_r(tGid, ":-", &save_ptr);

                ase_core_t ase_core(hash_t);
                while (split_str) {
                    if (is_digitstr(split_str)) {
                        ase_core.coordinates[coordinateId++] = (uint32_t)c_atoi(split_str);
                    }
                    split_str = c_strtok_r(NULL, ":-", &save_ptr);
                }
                h_ases.core.push_back(ase_core);

                uint64_t offset = _offset(const_cast<char *>(f->getGSeqName()));

                h_ases.start_.push_back(f->start + offset);
                h_ases.end_.push_back(f->end + offset);
                h_ases.strand.push_back((f->strand == '+'));
                // set to zero
                c_memset(tGid, '\0', sizeof(char) * gidSize);
// #define DEBUG
#ifdef DEBUG
                std::cout << "ase start: " << f->start << std::endl;
                std::cout << "ase end: " << f->end << std::endl;
                std::cout << "ase strand: " << (f->strand == '+') << std::endl;
                std::cout << "ase gid: ";
                for (int i = 0; i < coordinateId; i++) {
                    std::cout << ase_core.coordinates[i] << " ";
                }
                std::cout << std::endl;
#endif
            }
        }
    }
}

void LoadReadFromBam_1(h_Reads& h_reads, char* bam_file) {
    samFile *fp;
    if ((fp = sam_open(bam_file, "r")) == NULL) {
        std::cerr << "Could not open this file: " << bam_file << std::endl;
        return;
    }

    // threads
    hts_set_threads(fp, (get_nprocs() << 1));
    // header
    bam_hdr_t *header;
    if ((header = sam_hdr_read(fp)) == NULL) {
        std::cerr << "Could not read header for this file: " << bam_file << std::endl;
        return;
    }
    // parse
    uint32_t prev;
    uint64_t offset;
    char **qname;
    bam1_t *b = bam_init1();
#ifdef PAIR_END
    int32_t start_pos, end_pos;
    bam1_t *b_mate = bam_init1();
#endif
    while (sam_read1(fp, header, b) >= 0) {
        if ((b->core).tid < 0 || (b->core).tid >= header->n_targets) continue;

        if (!((b->core).flag & BAM_FUNMAP) && (b->core).qual) {
            // offset
            qname = header->target_name;
            offset = _offset(qname[(b->core).tid]);
            if (offset == invalidLength) continue;

            // junctions
            uint32_t *cigars = bam_get_cigar(b);
            prev = (cigars[0] >> 4);

            read_core_t read_core;
#ifdef SINGLE_END
            h_reads.start_.push_back((b->core).pos + offset + 1);
            h_reads.end_.push_back(bam_endpos(b) + offset + 1);
            h_reads.strand.push_back((uint32_t)(!bam_is_rev(b)));

            for (int i = 1; i < (b->core).n_cigar; i++) {
                // cigar operation, low 4 bit
                if ((cigars[i] & 15) == 3) {
                    read_core.junctions[read_core.junctionCount].start_ = prev;
                    read_core.junctions[read_core.junctionCount].end_ = prev + (cigars[i] >> 4);
                    read_core.junctionCount++;
                }
                prev = prev + (cigars[i] >> 4);
            }
            h_reads.core.push_back(read_core);
#elif defined(PAIR_END)
            if ((b->core).flag & BAM_FPAIRED) {
                if ((b->core).mpos == 0 || !((b->core).flag & BAM_FREAD1)) continue;

                // get read2
                while (sam_read1(fp, header, b_mate) >= 0) {
                    if (((b_mate->core).flag & BAM_FREAD2) &&
                        (b_mate->core).pos == (b->core).mpos) break;
                }

                // make sure the order of read1 and read2 correct.
                assert((b->core).flag & BAM_FREAD1);
                assert((b_mate->core).flag & BAM_FREAD2);

//                if (!((b_mate->core).flag & BAM_FREAD2))
//                    std::cout << b->core.flag << b_mate->core.flag << std::endl;

                // push strand
                h_reads.strand.push_back((uint32_t) (!bam_is_rev(b)));
#define bam_startpos(b) \
            ((b->core).pos - bam_cigar2rlen((b->core).n_cigar, bam_get_cigar(b)))

                if (!bam_is_rev(b) && bam_is_rev(b_mate)) {
                    start_pos = std::min((b_mate->core).mpos, bam_startpos(b_mate));
                    end_pos = std::max((b->core).mpos, bam_endpos(b));

                    if ((b_mate->core).mpos > bam_startpos(b_mate))
                        prev += (b_mate->core).mpos - bam_startpos(b_mate);

                    // cigars of read1
                    for (int i = 1; i < (b->core).n_cigar; i++) {
                        if ((cigars[i] & 15) == 3) {
                            read_core.junctions[read_core.junctionCount].start_ = prev;
                            read_core.junctions[read_core.junctionCount].end_ = prev + (cigars[i] >> 4);
                            read_core.junctionCount++;
                        }
                        prev = prev + (cigars[i] >> 4);
                    }
                    // cigars of read2
                    cigars = bam_get_cigar(b_mate);
                    prev = (cigars[0] >> 4);
                    for (int i = 1; i < (b_mate->core).n_cigar; i++) {
                        if ((cigars[i] & 15) == 3) {
                            read_core.junctions[read_core.junctionCount].end_ = (b_mate->core).pos - start_pos - prev;
                            read_core.junctions[read_core.junctionCount].start_ =
                                    (b_mate->core).pos - start_pos - (prev + (cigars[i] >> 4));
                            read_core.junctionCount++;
                        }
                        prev = prev + (cigars[i] >> 4);
                    }
                } else if (bam_is_rev(b) && !bam_is_rev(b_mate)) {
                    start_pos = std::min((b->core).mpos, bam_startpos(b));
                    end_pos = std::max((b_mate->core).mpos, bam_endpos(b_mate));

                    // cigars of read1
                    for (int i = 1; i < (b->core).n_cigar; i++) {
                        if ((cigars[i] & 15) == 3) {
                            read_core.junctions[read_core.junctionCount].end_ = (b->core).pos - start_pos - prev;
                            read_core.junctions[read_core.junctionCount].start_ =
                                    (b->core).pos - start_pos - (prev + (cigars[i] >> 4));
                            read_core.junctionCount++;
                        }
                        prev = prev + (cigars[i] >> 4);
                    }
                    // cigars of read2
                    cigars = bam_get_cigar(b_mate);
                    prev = (cigars[0] >> 4);

                    if ((b->core).mpos > bam_startpos(b))
                        prev += (b->core).mpos - bam_startpos(b);

                    for (int i = 1; i < (b->core).n_cigar; i++) {
                        if ((cigars[i] & 15) == 3) {
                            read_core.junctions[read_core.junctionCount].start_ = prev;
                            read_core.junctions[read_core.junctionCount].end_ = prev + (cigars[i] >> 4);
                            read_core.junctionCount++;
                        }
                        prev = prev + (cigars[i] >> 4);
                    }

                } else {
                    std::cout << "error parse in bam_is_rev!!!" << std::endl;
                    exit(1);
                }
                h_reads.start_.push_back(start_pos + offset + 1);
                h_reads.end_.push_back(end_pos + offset + 1);
                h_reads.core.push_back(read_core);
            }
#endif
        }
    }
    bam_destroy1(b);

//#define DEBUG
#ifdef DEBUG
    // print reads
    for (int itr = 0; itr < h_reads.start_.size(); itr++) {
        std::cout << "read start coordinate: " << h_reads.start_[itr] << std::endl;
        std::cout << "read end coordinate: " << h_reads.end_[itr] << std::endl;
        std::cout << "read strand: " << h_reads.strand[itr] << std::endl;
        std::cout << "read junction count: " << h_reads.core[itr].junctionCount << std::endl;
        if (h_reads.core[itr].junctionCount != 0) {
            std::cout << "read junctions: ";
            for (int itj = 0; itj < h_reads.core[itr].junctionCount; itj++) {
                std::cout << "N1: " << h_reads.core[itr].junctions[itj].start_ << " " <<
                             "N2: " << h_reads.core[itr].junctions[itj].end_ << " ";
            }
            std::cout << std::endl;
        }
    }
#endif
}

void LoadDataFromSerialization(h_Bins &h_bins, h_ASEs &h_ases) {
    std::ifstream ifs(serFilename);
    boost::archive::text_iarchive ia(ifs);
    ia & h_bins;
    ia & g_name_map;
    ia & h_ases;
    ia & g_gid_map;
}

void SaveDataToSerialization(h_Bins &h_bins, h_ASEs &h_ases) {
    std::ofstream ofs(serFilename);
    boost::archive::text_oarchive oa(ofs);
    oa & h_bins;
    oa & g_name_map;
    oa & h_ases;
    oa & g_gid_map;
}

void LoadReadFromBam(h_Reads &h_reads, char *bam_file)
{
    samFile *fp;
    if ((fp = sam_open(bam_file, "r")) == NULL) {
	std::cerr << "Could not open this file: " << bam_file << std::endl;
	return;
    }

    // threads
    hts_set_threads(fp, (get_nprocs() << 1));
    // header
    bam_hdr_t *header;
    if ((header = sam_hdr_read(fp)) == NULL) {
	std::cerr << "Could not read header for this file: " << bam_file
		  << std::endl;
	return;
    }
    // parse
    uint32_t prev;
    uint64_t offset;
    char **qname;
    // std::vector<std::string> chrList;
    std::set<std::string> chrList;
    bam1_t *b = bam_init1();
    typedef std::pair<int32_t, int32_t> junctionInline;
    std::vector<junctionInline> JunctionList;
    std::vector<uint64_t> start_posList;
    std::vector<uint64_t> end_posList;
    int32_t bam_core_stack_count = 0;
    //#ifdef PAIR_END
    uint32_t start_pos, end_pos;
    //#endif
    std::string qname_reg;
    while (sam_read1(fp, header, b) >= 0) {
	if ((b->core).tid < 0 || (b->core).tid >= header->n_targets) continue;
	// if mapping
	if (!((b->core).flag & BAM_FUNMAP) && (b->core).qual) {
	    qname = header->target_name;
	    offset = _offset(qname[(b->core).tid]);
	    if (offset == invalidLength) continue;
	    uint32_t *cigars = bam_get_cigar(b);
	    prev = (cigars[0] >> 4);
	    if (!((b->core).flag & BAM_FPAIRED)) {
		// if not paired, SE
		read_core_t read_core;
		h_reads.start_.push_back((b->core).pos + offset + 1);
		h_reads.end_.push_back(bam_endpos(b) + offset + 1);
		h_reads.strand.push_back((uint32_t)(!bam_is_rev(b)));
		for (int i = 1; i < (b->core).n_cigar; i++) {
		    if ((cigars[i] & 15) == 3) {
			read_core.junctions[read_core.junctionCount].start_ =
			    prev;
			read_core.junctions[read_core.junctionCount].end_ =
			    prev + (cigars[i] >> 4);
			read_core.junctionCount++;
		    }
		    prev = prev + (cigars[i] >> 4);
		}
		h_reads.core.push_back(read_core);
	    } else {
		// if paried, PE
		if (qname_reg.empty() ) {
		    // if the first qname
		    qname_reg = (char *)(b->data);
		    for (int i = 1; i < (b->core).n_cigar; i++) {
			if ((cigars[i] & 15) == 3) {
			    junctionInline tempJ;
			    tempJ.first = prev;
			    tempJ.second = prev + (cigars[i] >> 4);
			    JunctionList.push_back(tempJ);
			    start_posList.push_back((b->core).pos + offset + 1);
			    end_posList.push_back(bam_endpos(b) + offset + 1);
			}
			prev = prev + (cigars[i] >> 4);
		    }
		    std::string tmp_qname = (char *)qname[(b->core).tid];
		    chrList.insert(tmp_qname);
		} else {
		    // if the next qname
		    std::string qname_string;
		    qname_string = (char *)(b->data);
		    if (!qname_string.compare(qname_reg)) {
			for (int i = 1; i < (b->core).n_cigar; i++) {
			    if ((cigars[i] & 15) == 3) {
				junctionInline tempJ;
				tempJ.first = prev;
				tempJ.second = prev + (cigars[i] >> 4);
				JunctionList.push_back(tempJ);
				start_posList.push_back((b->core).pos + offset +
							1);
				end_posList.push_back(bam_endpos(b) + offset +
						      1);
			    }
			    prev = prev + (cigars[i] >> 4);
			}
			std::string tmp_qname = (char *)qname[(b->core).tid];
			chrList.insert(tmp_qname);
		    } else {
			// To strike the JunctionList
			// unique chrList to determine whether only mapping to
			// one chromosome
			if (chrList.size() == 1 and JunctionList.size()!=0) {
			    read_core_t read_core;
			    int32_t delta[start_posList.size()];
			    std::vector<junctionInline>::iterator it;
			    uint64_t left_startpos;
			    left_startpos = *(std::min_element(
				start_posList.begin(), start_posList.end()));
			    for (int i = 0; i < start_posList.size(); ++i) {
				delta[i] = start_posList[i] - left_startpos;
			    }
			    for (int i = 0; i < start_posList.size(); i++) {
				JunctionList[i].first += delta[i];
				JunctionList[i].second += delta[i];
			    }
			    std::set<junctionInline> junctionSet;
			    for (int i = 0; i < JunctionList.size(); i++) {
				junctionSet.insert(JunctionList[i]);
			    }
		            if (junctionSet.size()>junctionSize){
				printf("catch a larger junctionCount in %s!\n",qname_reg.c_str());
			    }
			    std::set<junctionInline>::iterator its;
			    for (its = junctionSet.begin();
				 its != junctionSet.end(); its++) {
				read_core.junctions[read_core.junctionCount]
				    .start_ = its->first;
				read_core.junctions[read_core.junctionCount]
				    .end_ = its->second;
				read_core.junctionCount++;
			    }
			    h_reads.start_.push_back(*std::min_element(
				start_posList.begin(), start_posList.end()));
			    h_reads.end_.push_back(*std::max_element(
				end_posList.begin(), end_posList.end()));
			    h_reads.strand.push_back((uint32_t)(!bam_is_rev(b)));

			    h_reads.core.push_back(read_core);
			    //junctionSet.clear();
			}
			chrList.clear();
			qname_reg = qname_string;
			JunctionList.clear();
			start_posList.clear();
			end_posList.clear();
			for (int i = 1; i < (b->core).n_cigar; i++) {
			    if ((cigars[i] & 15) == 3) {
				junctionInline tempJ;
				tempJ.first = prev;
				tempJ.second = prev + (cigars[i] >> 4);
				JunctionList.push_back(tempJ);
				start_posList.push_back((b->core).pos + offset +
							1);
				end_posList.push_back(bam_endpos(b) + offset +
						      1);
			    }
			    prev = prev + (cigars[i] >> 4);
			}
		        std::string tmp_qname = (char *)qname[(b->core).tid];
			chrList.insert(tmp_qname);
		    }
		}
	    }
	}
    }
    // after read all reads, process remains junctions in JunctionList
    printf("hello world!\n");
    if (JunctionList.size() != 0) {
	if (chrList.size() == 1) {
	    read_core_t read_core;
	    int32_t delta[start_posList.size()];
	    std::vector<junctionInline>::iterator it;
	    uint64_t left_startpos;
	    left_startpos =
		*(std::min_element(start_posList.begin(), start_posList.end()));
	    for (int i = 0; i < start_posList.size(); ++i) {
		delta[i] = start_posList[i] - left_startpos;
	    }
	    for (int i = 0; i < start_posList.size(); i++) {
		JunctionList[i].first += delta[i];
		JunctionList[i].second += delta[i];
	    }
	    std::set<junctionInline> junctionSet;
	    for (int i = 0; i < JunctionList.size(); i++) {
		junctionSet.insert(JunctionList[i]);
	    }
	    std::set<junctionInline>::iterator its;
	    for (its = junctionSet.begin(); its != junctionSet.end(); its++) {
		read_core.junctions[read_core.junctionCount].start_ =
		    its->first;
		read_core.junctions[read_core.junctionCount].end_ = its->second;
		read_core.junctionCount++;
	    }
	    
	    h_reads.start_.push_back(
		*std::min_element(start_posList.begin(), start_posList.end()));
	    h_reads.end_.push_back(
		*std::max_element(end_posList.begin(), end_posList.end()));
	    h_reads.strand.push_back((uint32_t)(true));

	    h_reads.core.push_back(read_core);
	    chrList.clear();
	    JunctionList.clear();
	    start_posList.clear();
	    end_posList.clear();
	    //junctionSet.clear();
	}
    }
bam_destroy1(b);
/*std::vector<uint64_t> s_reads_s(h_reads.start_.size());
std::copy(h_reads.start_.beign(),h_reads.start_.end(),s_reads_s.begin());
std::sort(s_reads_s.begin(),s_reads_s.end());

std::vector<uint64_t> s_reads_e(h_reads.end_.size());
std::copy(h_reads.end_.beign(),h_reads.end_.end(),s_reads_e.begin());
std::sort(s_reads_e.begin(),s_reads_e.end());

std::multimap<uint64_t,std::pair<uint64_t,read_core_t>> s_map;
for (int32_t i=0;i<h_reads.start_.size();i++){
	s_map.insert(std::pair<uint64_t,std::pair<uint64_t,read_core_t>>(h_reads.start_[i],std::pair<uint64_t,read_core_t>(h_reads.end_[i],h_reads.core[i])));
}
*/
FILE *f;
f=fopen("/home/qianjiaqiang/h.txt","w");
/*
if (f!=NULL){
	for (std::multimap<uint64_t,std::pair<uint64_t,read_core_t>>::iterator iter = s_map.begin();iter!=s_map.end();++iter){
		fprintf(f,"%lu\t%lu\t%lu\t%lu\n",iter->first,(iter->second).first,(iter->second).second.junctions[0].start_,(iter->second).second.junctions[0].end_);	
	}
}
*/
if (f!=NULL){
	for (uint32_t i=0;i < h_reads.start_.size();i++){
		fprintf(f,"%lu\t%lu\t%lu\t%lu\t%lu\t%d\n",h_reads.start_[i],h_reads.end_[i],h_reads.strand[i],h_reads.core[i].junctions[0].start_,h_reads.core[i].junctions[0].end_,i);
	}
}
fclose(f);
}


#endif //_CHIP_PARSE_CUH_

//===- SigModeSection.h - SigMode section merging for RISC-V ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file handles the merging of SigMode sections across multiple object
// files. When linking multiple .o files, each file has its own ID space.
// This module relocates the IDs so they form a continuous global ID space.
//
// Section Layout:
// - .sig_got: GOT table entries { addr(64), PointToID(32) }
// - .sig_ptr_header_counter: All header counts (32-bit each)
// - .sig_ptr_header_single: Single pointer headers { addr(64), DataID(32), PointToID(32) }
// - .sig_ptr_header_contig_same: ContiguousSame headers { addr(64), count(64), DataID(32), PointToID(32) }
// - .sig_ptr_header_contig_diff: ContiguousDiff headers { addr(64), count(64) }
// - .sig_ptr_header_sparse_same: SparseSame headers { addr(64), count(64), DataID(32), PointToID(32) }
// - .sig_ptr_header_sparse_diff: SparseDiff headers { addr(64), count(64) }
// - .sig_id_contig_diff: ContiguousDiff IDs [DataID(32), PointToID[count](32)]...
// - .sig_id_sparse_diff: SparseDiff IDs [DataID(32), PointToID[count](32)]...
// - .sig_offset_sparse_same: SparseSame offsets (64-bit each)
// - .sig_offset_sparse_diff: SparseDiff offsets (64-bit each)
//
// ID Relocation Algorithm:
// 1. For the first file, keep IDs unchanged, record total ID count
// 2. For subsequent files:
//    a. Add accumulated ID count to all ID fields (except special IDs 0 and 0xFFFFFF)
//    b. Accumulate this file's ID count
// 3. Concatenate all sections
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_SIGMODESECTION_H
#define LLD_ELF_SIGMODESECTION_H

#include "lld/Common/LLVM.h"

namespace lld {
namespace elf {

struct Ctx;

// Section names for SigMode
constexpr const char *sigSectionGOT = ".sig_got";
constexpr const char *sigSectionCounter = ".sig_ptr_header_counter";
constexpr const char *sigSectionHeaderSingle = ".sig_ptr_header_single";
constexpr const char *sigSectionHeaderContigSame = ".sig_ptr_header_contig_same";
constexpr const char *sigSectionHeaderContigDiff = ".sig_ptr_header_contig_diff";
constexpr const char *sigSectionHeaderSparseSame = ".sig_ptr_header_sparse_same";
constexpr const char *sigSectionHeaderSparseDiff = ".sig_ptr_header_sparse_diff";
constexpr const char *sigSectionIDContigDiff = ".sig_id_contig_diff";
constexpr const char *sigSectionIDSparseDiff = ".sig_id_sparse_diff";
constexpr const char *sigSectionOffsetSparseSame = ".sig_offset_sparse_same";
constexpr const char *sigSectionOffsetSparseDiff = ".sig_offset_sparse_diff";
constexpr const char *sigSectionSymtab = ".sig_symtab";
constexpr const char *sigSectionExtGOT = ".sig_ext_got";
constexpr const char *sigSectionExtFixupData = ".sig_ext_fixup_data";

// Special ID values that should not be relocated
constexpr uint32_t sigExternalID = 0xFFFFFF;
constexpr uint32_t sigNullID = 0;

// Check if a section name is a SigMode section
bool isSigModeSection(StringRef name);

// Process SigMode sections: relocate IDs across multiple object files
// This should be called after all input sections are collected but before
// they are written to the output.
template <class ELFT>
void processSigModeSections(Ctx &ctx);

// Convert .sig_got entries in-place from {addr, id} to {gotIndex, id} format
// This should be called after GOT addresses are finalized (after finalizeAddressDependentContent)
// 
// Original format: { addr(64), id(64) } = 16 bytes per entry
// Converted format: { gotIndex(64), id(64) } = 16 bytes per entry
//
// The section size is preserved to avoid layout changes.
// Only entries that match symbols in GOT are kept; unmatched entries are zeroed.
template <class ELFT>
void convertSigGotInPlace(Ctx &ctx);

// Resolve external GOT references across object files
// This should be called after all sections are allocated
//
// This function:
// 1. Builds a global symbol name -> ID mapping from all .sig_got sections
// 2. Resolves external references in .sig_ext_got using this mapping
// 3. Updates ID values at locations specified in .sig_ext_fixup_data
// 4. Removes temporary sections (.sig_ext_got, .sig_symtab, .sig_ext_fixup_data)
template <class ELFT>
void resolveExternalGotReferences(Ctx &ctx);

} // namespace elf
} // namespace lld

#endif // LLD_ELF_SIGMODESECTION_H

//===- SigModeSection.cpp - SigMode section merging for RISC-V ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SigModeSection.h"
#include "Config.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "OutputSections.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include <algorithm>
#include <cstddef>

#define DEBUG_TYPE "lld-sigmode"

using namespace llvm;
using namespace llvm::support::endian;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

bool lld::elf::isSigModeSection(StringRef name) {
  return name.starts_with(".sig_");
}

namespace {

// Helper class to track ID counts per input file and perform ID relocation
class SigModeRelocator {
public:
  SigModeRelocator(Ctx &ctx) : ctx(ctx) {}

  // Process all SigMode sections across all input files
  template <class ELFT>
  void process();

private:
  Ctx &ctx;
  uint32_t accumulatedIDCount = 0;
  InputSectionBase *mergedCounterSec = nullptr;  // The first counter section (will be updated)
  uint8_t *mergedCounterData = nullptr;          // Mutable copy of counter data

  // Get the ID count from an input file by reading its .sig_ptr_header_counter section
  uint32_t getFileIDCount(InputSectionBase *counterSec);

  // Merge counter from a subsequent file into the merged counter
  void mergeCounter(InputSectionBase *counterSec);

  // Relocate IDs in a section's data
  // Returns true if the section was modified
  template <class ELFT>
  void relocateIDs(InputSectionBase *sec, uint32_t idOffset);

  // Relocate IDs in different section types
  void relocateGOT(MutableArrayRef<uint8_t> data, uint32_t idOffset, bool is64Bit);
  void relocateHeaderSingle(MutableArrayRef<uint8_t> data, uint32_t idOffset, bool is64Bit);
  void relocateHeaderContigSame(MutableArrayRef<uint8_t> data, uint32_t idOffset, bool is64Bit);
  void relocateHeaderSparseSame(MutableArrayRef<uint8_t> data, uint32_t idOffset, bool is64Bit);
  void relocateIDArray(MutableArrayRef<uint8_t> data, uint32_t idOffset);

  // Helper to relocate a single 32-bit ID value
  void relocateID32(uint8_t *ptr, uint32_t idOffset);
};

uint32_t SigModeRelocator::getFileIDCount(InputSectionBase *counterSec) {
  if (!counterSec)
    return 0;
  
  ArrayRef<uint8_t> data = counterSec->content();
  uint32_t count = read32le(data.data());
  
  return count;
}

void SigModeRelocator::relocateID32(uint8_t *ptr, uint32_t idOffset) {
  uint32_t id = read32le(ptr);
  
  // Don't relocate special IDs
  if (id == sigNullID || id == sigExternalID)
    return;
  
  write32le(ptr, id + idOffset);
}

void SigModeRelocator::relocateGOT(MutableArrayRef<uint8_t> data, 
                                    uint32_t idOffset, bool is64Bit) {
  // GOT entry: { addr(64), PointToID(64) }
  size_t entrySize = is64Bit ? 16 : 8;  // 64-bit addr + 32-bit ID, or 32-bit addr + 32-bit ID
  size_t idFieldOffset = is64Bit ? 8 : 4;   // ID starts after addr
  
  for (size_t i = 0; i + entrySize <= data.size(); i += entrySize) {
    relocateID32(data.data() + i + idFieldOffset, idOffset);
  }
}

void SigModeRelocator::relocateHeaderSingle(MutableArrayRef<uint8_t> data,
                                             uint32_t idOffset, bool is64Bit) {
  // Single header: { addr(64), DataID(32), PointToID(32) }
  size_t entrySize = is64Bit ? 16 : 12;
  size_t dataIDOffset = is64Bit ? 8 : 4;
  size_t pointToIDOffset = dataIDOffset + 4;
  
  for (size_t i = 0; i + entrySize <= data.size(); i += entrySize) {
    relocateID32(data.data() + i + dataIDOffset, idOffset);
    relocateID32(data.data() + i + pointToIDOffset, idOffset);
  }
}

void SigModeRelocator::relocateHeaderContigSame(MutableArrayRef<uint8_t> data,
                                                 uint32_t idOffset, bool is64Bit) {
  // ContiguousSame/SparseSame header: { addr(64), count(64), DataID(32), PointToID(32) }
  size_t entrySize = is64Bit ? 24 : 16;
  size_t dataIDOffset = is64Bit ? 16 : 8;
  size_t pointToIDOffset = dataIDOffset + 4;
  
  for (size_t i = 0; i + entrySize <= data.size(); i += entrySize) {
    relocateID32(data.data() + i + dataIDOffset, idOffset);
    relocateID32(data.data() + i + pointToIDOffset, idOffset);
  }
}

void SigModeRelocator::relocateHeaderSparseSame(MutableArrayRef<uint8_t> data,
                                                 uint32_t idOffset, bool is64Bit) {
  // Same as ContiguousSame
  relocateHeaderContigSame(data, idOffset, is64Bit);
}

void SigModeRelocator::relocateIDArray(MutableArrayRef<uint8_t> data,
                                        uint32_t idOffset) {
  // ID array: all 32-bit IDs
  for (size_t i = 0; i + 4 <= data.size(); i += 4) {
    relocateID32(data.data() + i, idOffset);
  }
}

void SigModeRelocator::mergeCounter(InputSectionBase *counterSec) {
  // Counter section has 6 uint64_t counters (48 bytes total)
  // Counter section layout:
  // - 6 x uint32_t counters (24 bytes)
  // - 1 x uint32_t reserved entry (4 bytes) - for GOT ID count, do NOT merge
  // Total: 28 bytes per file
  constexpr size_t counterCount = 6;
  constexpr size_t counterSize = sizeof(uint32_t);
  constexpr size_t totalSize = counterCount * counterSize;  // 24 bytes
  
  if (!counterSec || !mergedCounterData)
    return;
  
  ArrayRef<uint8_t> srcData = counterSec->content();
  if (srcData.size() < totalSize)
    return;
  
  // Add each counter value (only the first 6, skip the reserved 7th entry)
  for (size_t i = 0; i < counterCount; ++i) {
    size_t offset = i * counterSize;
    uint32_t srcVal = read32le(srcData.data() + offset);
    uint32_t dstVal = read32le(mergedCounterData + offset);
    write32le(mergedCounterData + offset, srcVal + dstVal);
  }
  // Note: The 7th entry (reserved for GOT ID count) is NOT merged
  // It will be written by convertSigGotInPlace after GOT processing
  
  LLVM_DEBUG(dbgs() << "SigMode: Merged counter from file\n");
}

template <class ELFT>
void SigModeRelocator::relocateIDs(InputSectionBase *sec, uint32_t idOffset) {
  if (idOffset == 0)
    return;
  
  // We need to make a mutable copy of the section data
  // The section's content is const, so we need to allocate new storage
  ArrayRef<uint8_t> origData = sec->content();
  auto *newData = ctx.bAlloc.Allocate<uint8_t>(origData.size());
  memcpy(newData, origData.data(), origData.size());
  
  MutableArrayRef<uint8_t> mutableData(newData, origData.size());
  bool is64Bit = ELFT::Is64Bits;
  
  StringRef name = sec->name;
  
  if (name == sigSectionGOT) {
    relocateGOT(mutableData, idOffset, is64Bit);
  } else if (name == sigSectionHeaderSingle) {
    relocateHeaderSingle(mutableData, idOffset, is64Bit);
  } else if (name == sigSectionHeaderContigSame) {
    relocateHeaderContigSame(mutableData, idOffset, is64Bit);
  } else if (name == sigSectionHeaderSparseSame) {
    relocateHeaderSparseSame(mutableData, idOffset, is64Bit);
  } else if (name == sigSectionIDContigDiff || name == sigSectionIDSparseDiff) {
    relocateIDArray(mutableData, idOffset);
  }
  // Note: ContiguousDiff and SparseDiff headers don't have IDs in them
  // Note: Offset arrays don't have IDs
  // Note: Counter section doesn't need relocation
  
  // Update the section's content pointer
  sec->content_ = newData;
}

template <class ELFT>
void SigModeRelocator::process() {
  // Group input sections by their input file
  // We need to process files in order to accumulate ID counts
  
  // First, collect all SigMode sections grouped by input file
  DenseMap<InputFile *, SmallVector<InputSectionBase *, 16>> fileSections;
  
  for (InputSectionBase *sec : ctx.inputSections) {
    if (!sec || !sec->isLive())
      continue;
    if (!isSigModeSection(sec->name))
      continue;
    fileSections[sec->file].push_back(sec);
  }
  
  if (fileSections.empty())
    return;
  
  LLVM_DEBUG(dbgs() << "SigMode: Processing " << fileSections.size() 
                    << " files with SigMode sections\n");
  
  // Process each file's sections
  // Note: The order matters - we process files in the order they appear in ctx.objectFiles
  accumulatedIDCount = 0;
  
  for (ELFFileBase *file : ctx.objectFiles) {
    auto it = fileSections.find(file);
    if (it == fileSections.end())
      continue;
    
    SmallVector<InputSectionBase *, 16> &sections = it->second;
    
    // Find the counter section to get this file's ID count
    InputSectionBase *counterSec = nullptr;
    for (InputSectionBase *sec : sections) {
      if (sec->name == sigSectionCounter) {
        counterSec = sec;
        break;
      }
    }
    
    assert(counterSec && "Each Collected SigMode file must have a counter section");

    uint32_t fileIDCount = getFileIDCount(counterSec);
    
    LLVM_DEBUG(dbgs() << "SigMode: File " << file->getName() 
                      << " has " << fileIDCount << " IDs, "
                      << "accumulated offset: " << accumulatedIDCount << "\n");
    
    // Handle counter section merging
    if (counterSec) {
      if (!mergedCounterSec) {
        // First file with counter - make a mutable copy
        mergedCounterSec = counterSec;
        ArrayRef<uint8_t> origData = counterSec->content();
        mergedCounterData = ctx.bAlloc.Allocate<uint8_t>(origData.size());
        memcpy(mergedCounterData, origData.data(), origData.size());
        // Update the section to use the mutable data
        counterSec->content_ = mergedCounterData;
      } else {
        // Subsequent file - merge counters and discard this counter section
        mergeCounter(counterSec);
        // Mark this counter section as discarded by setting size to 0
        // The linker will skip empty sections
        counterSec->size = 0;
      }
    }
    
    // Relocate IDs in all sections of this file (except counter)
    if (accumulatedIDCount > 0) {
      for (InputSectionBase *sec : sections) {
        if (sec->name != sigSectionCounter)
          relocateIDs<ELFT>(sec, accumulatedIDCount);
      }
    }
    
    // Accumulate the ID count for the next file
    accumulatedIDCount += fileIDCount;
  }
  
  LLVM_DEBUG(dbgs() << "SigMode: Total ID count after merging: " 
                    << accumulatedIDCount << "\n");
}

} // anonymous namespace

template <class ELFT>
void lld::elf::processSigModeSections(Ctx &ctx) {
  SigModeRelocator relocator(ctx);
  relocator.process<ELFT>();
}

//===----------------------------------------------------------------------===//
// GOT ID In-place Conversion
//===----------------------------------------------------------------------===//
//
// Original .sig_got format: { addr(64), id(64) } = 16 bytes per entry
// Converted format:         { gotIndex(64), id(64) } = 16 bytes per entry
//
// This converts .sig_got entries in-place after GOT addresses are finalized.
// The section size remains unchanged to avoid affecting layout.
//
//===----------------------------------------------------------------------===//

namespace {

// Structure to hold address-to-ID mapping from .sig_got
struct SigGotAddrEntry {
  uint64_t addr;
  uint64_t id;
  
  bool operator<(const SigGotAddrEntry &other) const {
    return addr < other.addr;
  }
};

// Structure for the converted entry
struct SigGotConvertedEntry {
  uint32_t gotIndex;
  uint64_t id;

  bool operator<(const SigGotConvertedEntry &other) const {
    return gotIndex < other.gotIndex;
  }
};

template <class ELFT>
class SigGotConverter {
public:
  SigGotConverter(Ctx &ctx) : ctx(ctx) {}

  void convert();

private:
  Ctx &ctx;
  SmallVector<SigGotAddrEntry, 0> addrEntries;
  SmallVector<SigGotConvertedEntry, 0> convertedEntries;
  size_t totalGotIDCount = 0;

  // Collect all .sig_got entries from input sections
  void collectEntries();

  // Match with GOT and build converted entries
  void matchGotEntries();

  // Write converted data back to sections
  void writeBackToSections();
};

template <class ELFT>
void SigGotConverter<ELFT>::collectEntries() {
  // .sig_got entry: { addr(64), id(64) } = 16 bytes
  constexpr size_t entrySize = 16;

  for (InputSectionBase *isb : ctx.inputSections) {
    if (isb->name != sigSectionGOT)
      continue;

    auto *sec = dyn_cast<InputSection>(isb);
    if (!sec)
      continue;

    ArrayRef<uint8_t> data = sec->content();
    
    // Build a map from offset to relocation for quick lookup
    DenseMap<uint64_t, const Relocation *> relocMap;
    for (const Relocation &rel : sec->relocs()) {
      relocMap[rel.offset] = &rel;
    }

    size_t numEntries = data.size() / entrySize;
    for (size_t i = 0; i < numEntries; ++i) {
      size_t baseOffset = i * entrySize;
      SigGotAddrEntry entry;
      
      // Try to get address from relocation first (needed for PIE)
      auto relIt = relocMap.find(baseOffset);
      
      if (relIt != relocMap.end() && relIt->second->sym) {
        Symbol *sym = relIt->second->sym;
        if (sym->isDefined()) {
          entry.addr = sym->getVA(ctx) + relIt->second->addend;
        } else {
          continue;
        }
      } else {
        entry.addr = read64le(data.data() + baseOffset);
      }
      
      entry.id = read64le(data.data() + baseOffset + 8);
      
      assert(entry.id != sigNullID && entry.id != sigExternalID);
      if (entry.addr != 0) {
        addrEntries.push_back(entry);
      }
    }
  }

  llvm::sort(addrEntries);

  LLVM_DEBUG(dbgs() << "SigMode: Collected " << addrEntries.size()
                    << " entries from .sig_got sections\n");
}

template <class ELFT>
void SigGotConverter<ELFT>::matchGotEntries() {
  if (addrEntries.empty())
    return;

  for (Symbol *sym : ctx.symtab->getSymbols()) {
    if (!sym->isInGot(ctx))
      continue;

    if (!sym->isDefined())
      continue;

    uint64_t symAddr = sym->getVA(ctx);
    
    SigGotAddrEntry searchKey;
    searchKey.addr = symAddr;

    auto it = std::lower_bound(addrEntries.begin(), addrEntries.end(), searchKey);
    
    if (it != addrEntries.end() && it->addr == symAddr) {
      SigGotConvertedEntry converted;
      converted.gotIndex = sym->getGotIdx(ctx);
      converted.id = it->id;
      convertedEntries.push_back(converted);
      
      LLVM_DEBUG(dbgs() << "SigMode: Matched GOT[" << converted.gotIndex 
                        << "] -> ID " << converted.id << "\n");
    }
  }

  llvm::sort(convertedEntries);
  totalGotIDCount = convertedEntries.size();

  LLVM_DEBUG(dbgs() << "SigMode: Matched " << convertedEntries.size()
                    << " GOT entries\n");
}

template <class ELFT>
void SigGotConverter<ELFT>::writeBackToSections() {
  constexpr size_t entrySize = 16;  // { gotIndex(64), id(64) }
  
  // Find the .sig_got OutputSection and get its InputSections in the correct order
  SmallVector<InputSection *, 0> sigGotSections;
  
  // Find the OutputSection for .sig_got
  OutputSection *sigGotOS = nullptr;
  for (OutputSection *os : ctx.outputSections) {
    if (os->name == sigSectionGOT) {
      sigGotOS = os;
      break;
    }
  }
  
  if (sigGotOS) {
    // Use getInputSections to get InputSections in their final layout order
    SmallVector<InputSection *, 0> storage;
    ArrayRef<InputSection *> sections = getInputSections(*sigGotOS, storage);
    for (InputSection *sec : sections) {
      if (sec->name == sigSectionGOT)
        sigGotSections.push_back(sec);
    }
  }
  
  if (sigGotSections.empty())
    return;
  
  // Calculate total capacity across all sections
  size_t totalCapacity = 0;
  for (InputSection *sec : sigGotSections) {
    totalCapacity += sec->content().size() / entrySize;
  }
  
  // Assert that we have enough space
  assert(convertedEntries.size() <= totalCapacity &&
         "Not enough space in .sig_got sections for all converted entries");
  
  // Write converted entries sequentially across sections
  size_t entryIndex = 0;
  size_t totalEntries = convertedEntries.size();
  
  for (InputSection *sec : sigGotSections) {
    ArrayRef<uint8_t> origData = sec->content();
    size_t sectionCapacity = origData.size() / entrySize;
    
    // Allocate new data buffer
    auto *newData = ctx.bAlloc.Allocate<uint8_t>(origData.size());
    
    // Calculate how many entries to write to this section
    size_t entriesToWrite = std::min(sectionCapacity, totalEntries - entryIndex);
    
    // Write entries to this section
    for (size_t i = 0; i < entriesToWrite; ++i) {
      size_t offset = i * entrySize;
      const auto &entry = convertedEntries[entryIndex + i];
      write64le(newData + offset, entry.gotIndex);
      write64le(newData + offset + 8, entry.id);
    }
    
    // Zero out remaining space in this section
    for (size_t offset = entriesToWrite * entrySize; 
         offset + 8 <= origData.size(); offset += 8) {
      write64le(newData + offset, 0);
    }
    
    // Update section
    sec->size = entriesToWrite * entrySize;
    sec->content_ = newData;

        
    // Clear relocations - they are no longer needed as we've already resolved
    // the addresses and converted to GOT indices
    sec->relocations.clear();
    
    entryIndex += entriesToWrite;
    
    // If all entries written, set remaining sections to size 0
    if (entryIndex >= totalEntries) {
      // Continue to process remaining sections and set their size to 0
      continue;
    }
  }
  
  LLVM_DEBUG(dbgs() << "SigMode: Wrote " << totalEntries 
                    << " entries across " << sigGotSections.size() 
                    << " .sig_got sections\n");
}

template <class ELFT>
void SigGotConverter<ELFT>::convert() {
  LLVM_DEBUG(dbgs() << "SigMode: Converting .sig_got entries in-place\n");

  collectEntries();
  matchGotEntries();
  writeBackToSections();
}

} // anonymous namespace

template <class ELFT>
void lld::elf::convertSigGotInPlace(Ctx &ctx) {
  SigGotConverter<ELFT> converter(ctx);
  converter.convert();
}

// Explicit template instantiations
template void lld::elf::processSigModeSections<llvm::object::ELF32LE>(Ctx &);
template void lld::elf::processSigModeSections<llvm::object::ELF32BE>(Ctx &);
template void lld::elf::processSigModeSections<llvm::object::ELF64LE>(Ctx &);
template void lld::elf::processSigModeSections<llvm::object::ELF64BE>(Ctx &);

template void lld::elf::convertSigGotInPlace<llvm::object::ELF32LE>(Ctx &);
template void lld::elf::convertSigGotInPlace<llvm::object::ELF32BE>(Ctx &);
template void lld::elf::convertSigGotInPlace<llvm::object::ELF64LE>(Ctx &);
template void lld::elf::convertSigGotInPlace<llvm::object::ELF64BE>(Ctx &);

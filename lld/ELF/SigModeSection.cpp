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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>

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
  
  // Helper to relocate a single 64-bit ID value (for GOT entries)
  void relocateID64(uint8_t *ptr, uint64_t idOffset);
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

void SigModeRelocator::relocateID64(uint8_t *ptr, uint64_t idOffset) {
  // ID is stored as 64-bit for alignment, but only lower 32 bits are used
  uint64_t id = read64le(ptr);
  
  // Don't relocate special IDs (check lower 32 bits)
  uint32_t id32 = static_cast<uint32_t>(id);
  if (id32 == sigNullID || id32 == sigExternalID)
    return;
  
  // Add offset to lower 32 bits only, keep upper bits as 0
  write64le(ptr, static_cast<uint64_t>(id32 + static_cast<uint32_t>(idOffset)));
}

void SigModeRelocator::relocateGOT(MutableArrayRef<uint8_t> data, 
                                    uint32_t idOffset, bool is64Bit) {
  // GOT entry: { addr(64), PointToID(64)， sym_name_offset(64) }
  size_t entrySize = 24;
  size_t idFieldOffset = 8;   // ID starts after addr
  
  for (size_t i = 0; i + entrySize <= data.size(); i += entrySize) {
    relocateID64(data.data() + i + idFieldOffset, idOffset);
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
  constexpr size_t countersSize = counterCount * counterSize;  // 24 bytes
  constexpr size_t reservedSize = sizeof(uint32_t);            // 4 bytes
  constexpr size_t totalSize = countersSize + reservedSize;    // 28 bytes
  
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
  // Always allocate a mutable copy of the section data
  // This is needed even when idOffset == 0 because external reference resolution
  // may later need to modify the same buffers
  ArrayRef<uint8_t> origData = sec->content();
  auto *newData = ctx.bAlloc.Allocate<uint8_t>(origData.size());
  memcpy(newData, origData.data(), origData.size());
  
  // Update the section's content pointer first
  sec->content_ = newData;
  
  // If no offset, we're done (just allocated mutable buffer)
  if (idOffset == 0)
    return;
  
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
    if (!mergedCounterSec && counterSec) {
      // First file with counter - make a mutable copy
      mergedCounterSec = counterSec;
      ArrayRef<uint8_t> origData = counterSec->content();
      mergedCounterData = ctx.bAlloc.Allocate<uint8_t>(origData.size());
      memcpy(mergedCounterData, origData.data(), origData.size());
      // Update the section to use the mutable data
      counterSec->content_ = mergedCounterData;
    } else if (counterSec) {
      // Subsequent file - merge counters and discard this counter section
      mergeCounter(counterSec);
      // Mark this counter section as discarded by setting size to 0
      // The linker will skip empty sections
      counterSec->size = 0;
    }
    
    // Allocate mutable buffers and relocate IDs for all sections of this file (except counter)
    // Always allocate buffers even when accumulatedIDCount == 0, because external reference
    // resolution (resolveExternalGotReferences) will need to modify these buffers later
    for (InputSectionBase *sec : sections) {
      if (sec->name != sigSectionCounter)
        relocateIDs<ELFT>(sec, accumulatedIDCount);
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
// Original .sig_got format: { addr(64), id(64), sym_name_offset(64) } = 24 bytes per entry
// Converted format:         { gotIndex(64), id(64) } = 16 bytes per entry (packed sequentially)
//
// Entries are written sequentially starting from offset 0.
// The section size remains unchanged, unused space is zero-filled.
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

  // Write Got Number back to Counter Section
  void updateCounterSection();
};

template <class ELFT>
void SigGotConverter<ELFT>::collectEntries() {
  // .sig_got entry: { addr(64), id(64), sym_name_offset(64) } = 24 bytes
  constexpr size_t entrySize = 24;

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
  constexpr size_t outputEntrySize = 16;  // { gotIndex(64), id(64) }
  constexpr size_t inputEntrySize = 24;   // { addr(64), id(64), sym_name_offset(64) }
  
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
  // Note: input format is 24 bytes, output format is 16 bytes
  // Use output entry size for capacity calculation (same section size can hold more output entries)
  size_t totalCapacity = 0;
  for (InputSection *sec : sigGotSections) {
    totalCapacity += sec->content().size() / inputEntrySize;
  }
  
  // Assert that we have enough space
  assert(convertedEntries.size() <= totalCapacity &&
         "global got's number is larger than local global variables' number");
  
  SmallVector<uint64_t, 16> convertedSequence;
  for (const auto &entry : convertedEntries) {
    convertedSequence.push_back(entry.gotIndex);
    convertedSequence.push_back(entry.id);
  }
  
  // Write converted entries sequentially across sections
  size_t entryIndex = 0;
  size_t totalEntries = convertedSequence.size();
  size_t sequenceDataWidth = sizeof(uint64_t);
  
  for (InputSection *sec : sigGotSections) {
    ArrayRef<uint8_t> origData = sec->content();
    size_t sectionCapacity = origData.size() / sequenceDataWidth;
    
    // Allocate new data buffer
    auto *newData = ctx.bAlloc.Allocate<uint8_t>(origData.size());
    
    // Calculate how many entries to write to this section
    size_t entriesToWrite = std::min(sectionCapacity, totalEntries - entryIndex);
    
    // Write entries to this section
    for (size_t i = 0; i < entriesToWrite; ++i) {
      size_t offset = i * sequenceDataWidth;
      const auto &entry = convertedSequence[entryIndex + i];
      write64le(newData + offset, entry);
    }
    
    // Zero out remaining space in this section
    for (size_t offset = entriesToWrite * sequenceDataWidth; 
         offset + 8 <= origData.size(); offset += 8) {
      write64le(newData + offset, 0);
    }
    
    // Update section
    sec->size = entriesToWrite * sequenceDataWidth;
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
void SigGotConverter<ELFT>::updateCounterSection() {
  // Find the merged counter section
  OutputSection *sigCounterOS = nullptr;
  for (OutputSection *os : ctx.outputSections) {
    if (os->name == sigSectionCounter) {
      sigCounterOS = os;
      break;
    }
  }
  
  if (!sigCounterOS)
    return;
  
  // Get the InputSection for the counter
  SmallVector<InputSection *, 0> storage;
  ArrayRef<InputSection *> sections = getInputSections(*sigCounterOS, storage);
  
  if (sections.empty())
    return;
  
  InputSection *counterSec = nullptr;
  for (InputSection *sec : sections) {
    if (sec->size != 0) {
      counterSec = sec;
      break;
    }
  }
  assert(counterSec && "Merged counter section must exist");

  constexpr size_t counterCount = 6;
  constexpr size_t counterSize = sizeof(uint32_t);
  constexpr size_t countersSize = counterCount * counterSize;  // 24 bytes
  
  // Update the GOT ID count
  uint8_t *data = const_cast<uint8_t *>(counterSec->content().data());
  write32le(data + countersSize, static_cast<uint32_t>(totalGotIDCount));
  
  LLVM_DEBUG(dbgs() << "SigMode: Updated GOT ID count to " 
                    << totalGotIDCount << " in counter section\n");
}

template <class ELFT>
void SigGotConverter<ELFT>::convert() {
  LLVM_DEBUG(dbgs() << "SigMode: Converting .sig_got entries in-place\n");

  collectEntries();
  matchGotEntries();
  writeBackToSections();
  updateCounterSection();
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

//===----------------------------------------------------------------------===//
// External GOT Reference Resolution
//===----------------------------------------------------------------------===//
//
// This resolves external GOT references across object files during linking.
//
// Data formats (from compiler):
// - .sig_got:          { addr(64), id(64), sym_name_offset(64) } = 24 bytes
// - .sig_ext_got:      { ext_id(64), sym_name_offset(64), data_offset(64), length(64) } = 32 bytes
// - .sig_ext_fixup_data: { (section_type << 56) | offset } = 8 bytes per entry
// - .sig_symtab:       null-terminated strings
//
// Process:
// 1. Load all .sig_got + .sig_symtab to build global string -> ID mapping
// 2. Load each .sig_ext_got and resolve symbol names to IDs using the mapping
// 3. Update ID values at locations specified in .sig_ext_fixup_data
// 4. Remove .sig_ext_got, .sig_symtab, .sig_ext_fixup_data sections
//
//===----------------------------------------------------------------------===//

namespace {

template <class ELFT>
class ExternalGotResolver {
public:
  ExternalGotResolver(Ctx &ctx) : ctx(ctx) {}

  void resolve();

private:
  Ctx &ctx;
  
  // Global mapping from symbol name to ID (built from all .sig_got sections)
  StringMap<uint64_t> globalSymToId;
  
  // Information about an external reference to resolve
  struct ExtRefInfo {
    uint64_t extId;         // Original external ID
    StringRef symName;      // Symbol name from symtab
    uint64_t dataOffset;    // Offset into fixup data
    uint64_t length;        // Number of fixup locations
  };
  
  // Per-file information
  struct FileInfo {
    InputFile *file;
    InputSection *symtabSec;
    InputSection *extGotSec;
    InputSection *fixupDataSec;
    StringRef symtabContent;
    SmallVector<ExtRefInfo, 0> extRefs;
  };
  
  SmallVector<FileInfo, 0> fileInfos;
  
  // Step 1: Build global string -> ID mapping from all .sig_got + .sig_symtab
  void buildGlobalMapping();
  
  // Step 2: Collect external references from .sig_ext_got
  void collectExternalRefs();
  
  // Step 3: Resolve external references using global mapping
  void resolveExternalRefs();
  
  // Step 4: Update ID values at fixup locations
  void updateIdValues();
  
  // Step 5: Mark temporary sections for removal
  void markSectionsForRemoval();
  
  // Helper: Get symbol name from symtab at offset
  StringRef getSymbolName(StringRef symtab, uint64_t offset);
};

template <class ELFT>
StringRef ExternalGotResolver<ELFT>::getSymbolName(StringRef symtab, uint64_t offset) {
  if (offset >= symtab.size())
    return StringRef();
  
  const char *start = symtab.data() + offset;
  const char *end = symtab.data() + symtab.size();
  const char *nullPos = std::find(start, end, '\0');
  
  return StringRef(start, nullPos - start);
}

template <class ELFT>
void ExternalGotResolver<ELFT>::buildGlobalMapping() {
  // .sig_got entry: { addr(64), id(64), sym_name_offset(64) } = 24 bytes
  constexpr size_t gotEntrySize = 24;
  // UINT64_MAX indicates static symbol (no name in symtab)
  constexpr uint64_t staticSymbolOffset = UINT64_MAX;
  
  // Group sections by file
  DenseMap<InputFile *, InputSection *> fileToSymtab;
  DenseMap<InputFile *, InputSection *> fileToGot;
  
  for (InputSectionBase *isb : ctx.inputSections) {
    auto *sec = dyn_cast<InputSection>(isb);
    if (!sec)
      continue;
    
    if (sec->name == sigSectionSymtab) {
      fileToSymtab[sec->file] = sec;
    } else if (sec->name == sigSectionGOT) {
      fileToGot[sec->file] = sec;
    }
  }
  
  // Build mapping from each file's .sig_got
  for (auto &kv : fileToGot) {
    InputFile *file = kv.first;
    InputSection *gotSec = kv.second;
    
    auto symtabIt = fileToSymtab.find(file);
    if (symtabIt == fileToSymtab.end()) {
      LLVM_DEBUG(dbgs() << "SigMode: No .sig_symtab for file, skipping\n");
      continue;
    }
    InputSection *symtabSec = symtabIt->second;
    
    ArrayRef<uint8_t> symtabData = symtabSec->content();
    StringRef symtab(reinterpret_cast<const char *>(symtabData.data()), symtabData.size());
    
    ArrayRef<uint8_t> gotData = gotSec->content();
    size_t numEntries = gotData.size() / gotEntrySize;
    
    for (size_t i = 0; i < numEntries; ++i) {
      size_t offset = i * gotEntrySize;
      // Skip addr (bytes 0-7)
      uint64_t id = read64le(gotData.data() + offset + 8);
      uint64_t symNameOffset = read64le(gotData.data() + offset + 16);
      
      // Skip static symbols (sym_name_offset == UINT64_MAX)
      if (symNameOffset == staticSymbolOffset) {
        LLVM_DEBUG(dbgs() << "SigMode: Skipping static symbol with id=" << id << "\n");
        continue;
      }
      
      StringRef symName = getSymbolName(symtab, symNameOffset);
      assert (!symName.empty() && "Symbol name must not be empty");
      // Add to global mapping (first definition wins)
      assert (globalSymToId.find(symName) == globalSymToId.end() && "Duplicate symbol name in global mapping");
      globalSymToId[symName] = id;
      LLVM_DEBUG(dbgs() << "SigMode: Global mapping: " << symName 
                        << " -> " << id << "\n");
    }
  }
  
  LLVM_DEBUG(dbgs() << "SigMode: Built global mapping with " 
                    << globalSymToId.size() << " entries\n");
}

template <class ELFT>
void ExternalGotResolver<ELFT>::collectExternalRefs() {
  // .sig_ext_got entry: { ext_id(64), sym_name_offset(64), data_offset(64), length(64) } = 32 bytes
  constexpr size_t extGotEntrySize = 32;
  
  // Group sections by file
  DenseMap<InputFile *, InputSection *> fileToSymtab;
  DenseMap<InputFile *, InputSection *> fileToExtGot;
  DenseMap<InputFile *, InputSection *> fileToFixupData;
  
  for (InputSectionBase *isb : ctx.inputSections) {
    auto *sec = dyn_cast<InputSection>(isb);
    if (!sec)
      continue;
    
    if (sec->name == sigSectionSymtab) {
      fileToSymtab[sec->file] = sec;
    } else if (sec->name == sigSectionExtGOT) {
      fileToExtGot[sec->file] = sec;
    } else if (sec->name == sigSectionExtFixupData) {
      fileToFixupData[sec->file] = sec;
    }
  }
  
  // Collect external references from each file
  for (auto &kv : fileToExtGot) {
    InputFile *file = kv.first;
    InputSection *extGotSec = kv.second;
    
    auto symtabIt = fileToSymtab.find(file);
    auto fixupIt = fileToFixupData.find(file);
    
    if (symtabIt == fileToSymtab.end()) {
      LLVM_DEBUG(dbgs() << "SigMode: No .sig_symtab for file with ext_got\n");
      continue;
    }
    
    FileInfo info;
    info.file = file;
    info.symtabSec = symtabIt->second;
    info.extGotSec = extGotSec;
    info.fixupDataSec = (fixupIt != fileToFixupData.end()) ? fixupIt->second : nullptr;
    
    ArrayRef<uint8_t> symtabData = info.symtabSec->content();
    info.symtabContent = StringRef(reinterpret_cast<const char *>(symtabData.data()), symtabData.size());
    
    ArrayRef<uint8_t> extGotData = extGotSec->content();
    size_t numEntries = extGotData.size() / extGotEntrySize;
    
    for (size_t i = 0; i < numEntries; ++i) {
      size_t offset = i * extGotEntrySize;
      
      ExtRefInfo ref;
      ref.extId = read64le(extGotData.data() + offset);
      uint64_t symNameOffset = read64le(extGotData.data() + offset + 8);
      ref.dataOffset = read64le(extGotData.data() + offset + 16);
      ref.length = read64le(extGotData.data() + offset + 24);
      ref.symName = getSymbolName(info.symtabContent, symNameOffset);
      assert (!ref.symName.empty() && "External reference symbol name must not be empty");
      
      info.extRefs.push_back(ref);
      
      LLVM_DEBUG(dbgs() << "SigMode: ExtRef: extId=" << ref.extId
                        << ", symName=" << ref.symName
                        << ", dataOffset=" << ref.dataOffset
                        << ", length=" << ref.length << "\n");
    }
    
    fileInfos.push_back(std::move(info));
  }
  
  LLVM_DEBUG(dbgs() << "SigMode: Collected external refs from " 
                    << fileInfos.size() << " files\n");
}

template <class ELFT>
void ExternalGotResolver<ELFT>::updateIdValues() {
  // SectionType enum values (from RISCVCollectGlobalPointers.cpp):
  // GOT = 0, HeaderSingle = 1, HeaderContigSame = 2, HeaderContigDiff = 3,
  // HeaderSparseSame = 4, HeaderSparseDiff = 5, IDContigDiff = 6, IDSparseDiff = 7
  
  // Map sectionType to section name
  auto getSectionName = [](uint8_t sectionType) -> const char * {
    switch (sectionType) {
      case 0: return sigSectionGOT;
      case 1: return sigSectionHeaderSingle;
      case 2: return sigSectionHeaderContigSame;
      case 3: return sigSectionHeaderContigDiff;
      case 4: return sigSectionHeaderSparseSame;
      case 5: return sigSectionHeaderSparseDiff;
      case 6: return sigSectionIDContigDiff;
      case 7: return sigSectionIDSparseDiff;
      default: return nullptr;
    }
  };
  
  // GOT section stores ID as 64-bit for alignment (only lower 32 bits used), 
  // other sections use 32-bit IDs directly
  auto isId64Bit = [](uint8_t sectionType) -> bool {
    return sectionType == 0; // GOT = 0
  };
  
  for (FileInfo &info : fileInfos) {
    if (!info.fixupDataSec)
      continue;
    
    // Build a map from section name to section for this file
    DenseMap<StringRef, InputSection *> sectionMap;
    
    for (InputSectionBase *isb : ctx.inputSections) {
      auto *sec = dyn_cast<InputSection>(isb);
      if (!sec || sec->file != info.file)
        continue;
      if (!isSigModeSection(sec->name))
        continue;
      sectionMap[sec->name] = sec;
    }
    
    ArrayRef<uint8_t> fixupData = info.fixupDataSec->content();
    
    for (const ExtRefInfo &ref : info.extRefs) {
      uint32_t newId;
      auto it = globalSymToId.find(ref.symName);
      if (it == globalSymToId.end()) {
        LLVM_DEBUG(dbgs() << "SigMode: Unresolved external symbol: " << ref.symName << "\n");
        continue;
      }
      newId = static_cast<uint32_t>(it->second);
      
      // Process each fixup location
      for (uint64_t i = 0; i < ref.length; ++i) {
        size_t dataIdx = ref.dataOffset + i * 8;
        if (dataIdx + 8 > fixupData.size())
          break;
        
        uint64_t fixupEntry = read64le(fixupData.data() + dataIdx);
        uint8_t sectionType = (fixupEntry >> 56) & 0xFF;
        uint64_t offset = fixupEntry & ((1ULL << 56) - 1);
        
        const char *sectionName = getSectionName(sectionType);
        if (!sectionName) {
          LLVM_DEBUG(dbgs() << "SigMode: Unknown sectionType=" << (int)sectionType << "\n");
          continue;
        }
        
        auto secIt = sectionMap.find(sectionName);
        if (secIt == sectionMap.end()) {
          LLVM_DEBUG(dbgs() << "SigMode: Section " << sectionName << " not found\n");
          continue;
        }
        
        InputSection *targetSec = secIt->second;
        
        // NOTE: This function MUST be called AFTER processSigModeSections(),
        // which already allocated mutable buffers for all sig_* sections.
        // We can directly write to the existing buffer via content_.
        // If processSigModeSections hasn't run, this will corrupt read-only memory!
        
        // Determine if ID is 64-bit or 32-bit based on section type
        bool is64Bit = isId64Bit(sectionType);
        size_t idSize = is64Bit ? 8 : 4;
        
        if (offset + idSize > targetSec->size) {
          LLVM_DEBUG(dbgs() << "SigMode: Offset " << offset << " out of bounds for " << sectionName << "\n");
          continue;
        }
        
        // Get the mutable buffer (already allocated by processSigModeSections)
        uint8_t *buffer = const_cast<uint8_t *>(targetSec->content_);
        
        // Write the ID at the specified offset (64-bit for GOT, 32-bit for others)
        if (is64Bit) {
          write64le(buffer + offset, newId);
        } else {
          write32le(buffer + offset, static_cast<uint32_t>(newId));
        }
        
        LLVM_DEBUG(dbgs() << "SigMode: Updated ID at " << sectionName 
                          << " offset=" << offset 
                          << " newId=" << newId 
                          << " (" << (is64Bit ? "64" : "32") << "-bit)\n");
      }
    }
  }
}

template <class ELFT>
void ExternalGotResolver<ELFT>::markSectionsForRemoval() {
  // Mark .sig_ext_got, .sig_symtab, .sig_ext_fixup_data for removal
  for (InputSectionBase *isb : ctx.inputSections) {
    auto *sec = dyn_cast<InputSection>(isb);
    if (!sec)
      continue;
    
    if (sec->name == sigSectionExtGOT ||
        sec->name == sigSectionSymtab ||
        sec->name == sigSectionExtFixupData) {
      // Mark for removal by setting size to 0
      sec->size = 0;
      LLVM_DEBUG(dbgs() << "SigMode: Marked " << sec->name << " for removal\n");
    }
  }
}

template <class ELFT>
void ExternalGotResolver<ELFT>::resolve() {
  LLVM_DEBUG(dbgs() << "SigMode: Starting external GOT reference resolution\n");
  
  buildGlobalMapping();
  collectExternalRefs();
  updateIdValues();
  markSectionsForRemoval();
  
  LLVM_DEBUG(dbgs() << "SigMode: Completed external GOT reference resolution\n");
}

} // anonymous namespace

template <class ELFT>
void lld::elf::resolveExternalGotReferences(Ctx &ctx) {
  ExternalGotResolver<ELFT> resolver(ctx);
  resolver.resolve();
}

// Explicit template instantiations for resolveExternalGotReferences
template void lld::elf::resolveExternalGotReferences<llvm::object::ELF32LE>(Ctx &);
template void lld::elf::resolveExternalGotReferences<llvm::object::ELF32BE>(Ctx &);
template void lld::elf::resolveExternalGotReferences<llvm::object::ELF64LE>(Ctx &);
template void lld::elf::resolveExternalGotReferences<llvm::object::ELF64BE>(Ctx &);

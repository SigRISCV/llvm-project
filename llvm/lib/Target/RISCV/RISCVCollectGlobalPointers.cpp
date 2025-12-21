//===- RISCVCollectGlobalPointers.cpp - Collect global pointer locations --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass collects all pointer locations in global data sections and builds
// a pointer relationship table for SigMode runtime processing.
//
// Each entry has two IDs:
// - DataID (32-bit): The ID of the data region this entry belongs to
// - PointToID (32-bit): The ID of the data region this entry points to
//
// GOT entries: DataID = 0xFFFFFF (implicit), PointToID = 1, 2, 3, ...
// Output format: { addr(64), PointToID(32) } (DataID is always 0xFFFFFF, not stored)
//
// Pointer entries use Header-Data Separation for compact storage:
// (addr/count/offset are 64-bit, DataID/PointToID are 32-bit)
//
// === 5 Header Sections (.sig_ptr_header_*) ===
//
// Type5 (Single): Single pointer
//   Header: { addr(64), DataID(32), PointToID(32) }
//
// Type2 (ContiguousSame): Contiguous pointers with same PointToID  
//   Header: { addr(64), count(64), DataID(32), PointToID(32) }
//
// Type1 (ContiguousDifferent): Contiguous pointers with different PointToIDs
//   Header: { addr(64), count(64) }
//
// Type4 (SparseSame): Non-contiguous pointers with same PointToID
//   Header: { addr(64), count(64), DataID(32), PointToID(32) }
//
// Type3 (SparseDifferent): Non-contiguous pointers with different PointToIDs
//   Header: { addr(64), count(64) }
//
// === 4 Data Array Sections ===
//
// .sig_id_contig_diff: [DataID(32), PointToID(32)[count]]...  (combined per entry)
// .sig_id_sparse_diff: [DataID(32), PointToID(32)[count]]...  (combined per entry)
// .sig_offset_sparse_same: offset(64)[]  (separate)
// .sig_offset_sparse_diff: offset(64)[]  (separate)
//
// === Count Section (.sig_ptr_header_counter) ===
// All header counts are placed here.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "riscv-collect-global-pointers"
#define RISCV_COLLECT_GLOBAL_POINTERS_NAME "RISC-V Collect Global Pointers"

STATISTIC(NumPointersCollected, "Number of global pointers collected");
STATISTIC(NumGOTEntries, "Number of GOT entries created");
STATISTIC(NumExtGOTEntries, "Number of Ext GOT entries created");
STATISTIC(NumCompressedEntries, "Number of compressed pointer groups");

// Special IDs
static constexpr uint32_t ExternalPointToID = 0xFFFFFF;  // Unknown external
static constexpr uint32_t NullPointToID = 0;
static constexpr uint32_t ExtGOTStartID = 0xFFFFFE;      // External GOT IDs start here and decrease

// Section type enum for external fixup tracking
enum class SigSectionType : uint8_t {
  GOT = 0,
  HeaderSingle = 1,
  HeaderContigSame = 2,
  HeaderContigDiff = 3,
  HeaderSparseSame = 4,
  HeaderSparseDiff = 5,
  IDContigDiff = 6,
  IDSparseDiff = 7,
};

// Pointer group types
enum class PtrGroupType : uint8_t {
  ContiguousDifferent = 1,  // Contiguous pointers, different PointToIDs
  ContiguousSame = 2,       // Contiguous pointers, same PointToID
  SparseDifferent = 3,      // Non-contiguous pointers, different PointToIDs
  SparseSame = 4,           // Non-contiguous pointers, same PointToID
  Single = 5                // Single pointer
};

namespace {

// Structure to represent a single pointer within a global
struct PointerInfo {
  uint64_t Offset;      // Byte offset within the global variable
  uint32_t PointToID;   // ID of the data region this pointer points to
};

// Structure to represent a group of pointers in a global variable
struct PointerGroup {
  GlobalVariable *ContainingGV;       // The global variable
  uint32_t DataID;                    // ID of this data region
  PtrGroupType Type;                  // Type of compression
  SmallVector<PointerInfo, 8> Ptrs;   // Individual pointers (for Type1, Type3, Type5)
  uint32_t CommonPointToID;           // Common PointToID (for Type2, Type4)
  SmallVector<uint64_t, 8> Offsets;   // Offsets only (for Type4)
};

// Structure to represent a GOT entry (local globals)
struct GOTEntry {
  GlobalVariable *GV;     // The global variable
  uint32_t PointToID;     // The assigned PointToID (1, 2, 3, ...)
  uint64_t Size;          // Size of the global
};

// Structure to track PointToID location for external fixup
struct ExtFixupLocation {
  SigSectionType SectionType;  // Which section contains the PointToID
  uint64_t Offset;             // Byte offset within that section
};

// Structure to collect all fixup locations for one external ID
struct ExtFixupInfo {
  uint32_t ExtID;                            // The external ID (0xFFFFFE, 0xFFFFFD, ...)
  const GlobalVariable *GV;                     // The external symbol
  SmallVector<ExtFixupLocation, 4> Locations; // All PointToID locations
};

class RISCVCollectGlobalPointers : public ModulePass {
public:
  static char ID;
  RISCVCollectGlobalPointers() : ModulePass(ID) {}

  StringRef getPassName() const override {
    return RISCV_COLLECT_GLOBAL_POINTERS_NAME;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.setPreservesAll();
  }

  bool runOnModule(Module &M) override;

  // =========== Section Names ===========
  // GOT section
  static constexpr const char *SectionGOT = ".sig_got";
  
  // Counter section (all counts go here)
  static constexpr const char *SectionCounter = ".sig_ptr_header_counter";
  
  // Header sections (5 types)
  static constexpr const char *SectionHeaderSingle = ".sig_ptr_header_single";
  static constexpr const char *SectionHeaderContigSame = ".sig_ptr_header_contig_same";
  static constexpr const char *SectionHeaderContigDiff = ".sig_ptr_header_contig_diff";
  static constexpr const char *SectionHeaderSparseSame = ".sig_ptr_header_sparse_same";
  static constexpr const char *SectionHeaderSparseDiff = ".sig_ptr_header_sparse_diff";
  
  // ID data sections (2 types: contig_diff, sparse_diff)
  static constexpr const char *SectionIDContigDiff = ".sig_id_contig_diff";
  static constexpr const char *SectionIDSparseDiff = ".sig_id_sparse_diff";
  
  // Offset data sections (2 types: sparse_same, sparse_diff)
  static constexpr const char *SectionOffsetSparseSame = ".sig_offset_sparse_same";
  static constexpr const char *SectionOffsetSparseDiff = ".sig_offset_sparse_diff";
  
  // External reference sections
  static constexpr const char *SectionExtFixupHeader = ".sig_ext_got";
  static constexpr const char *SectionExtFixupData = ".sig_ext_fixup_data";
  
  // Symbol name string table section
  static constexpr const char *SectionSymtab = ".sig_symtab";

private:
  // Map from GlobalVariable to its GOT entry (local globals, ID from 1)
  DenseMap<const GlobalVariable *, GOTEntry> GOTMap;
  
  // Map from external GlobalValue to its ExtGOT entry (external globals, ID from 0xFFFFFE decreasing)
  DenseMap<const GlobalVariable *, GOTEntry> ExtGOTMap;
  
  // Reverse map from ExtID to GOTEntry pointer (built after ExtGOTMap is complete)
  DenseMap<uint32_t, const GOTEntry *> ExtGOTIDMap;  // ExtID -> GOTEntry*
  
  // Next external ID to assign (starts at 0xFFFFFE, decreases)
  uint32_t NextExtID = ExtGOTStartID;
  
  // Maximum local ID assigned (the highest ID in GOTMap)
  uint32_t MaxLocalID = 0;
  
  // All pointer groups found
  SmallVector<PointerGroup, 64> PointerGroups;
  
  // External fixup information (collected during section generation)
  DenseMap<uint32_t, ExtFixupInfo> ExtFixupMap;  // ExtID -> ExtFixupInfo
  
  // Symbol name offsets in the .sig_symtab string table
  DenseMap<const GlobalVariable *, uint64_t> SymNameOffsets;

  // Build the GOT map for all global variables
  void buildGOTMap(Module &M, const DataLayout &DL);

  // Check if a type is "pointer-only" (only contains pointers, no other data)
  bool isPointerOnlyType(Type *Ty, const DataLayout &DL, unsigned PtrSize);

  // Get the PointToID for a constant value
  uint32_t getPointToID(Constant *C);

  // Find the PointToID when a pointer points to a global value
  // First checks GOTMap (local), then ExtGOTMap (external), finally returns ExternalPointToID
  uint32_t findGOTPointToID(const GlobalValue *GV);
  
  // Check if an ID is in the external range
  // External IDs are between (MaxLocalID, ExtGOTStartID] excluding ExternalPointToID
  // More simply: ID > MaxLocalID && ID <= ExtGOTStartID
  bool isExtID(uint32_t ID) const {
    return ID > MaxLocalID && ID <= ExtGOTStartID;
  }
  
  // Record a PointToID location for external fixup tracking
  void recordExtFixupLocation(uint32_t ID, SigSectionType SecType, uint64_t Offset);

  // Collect all pointers in a constant, storing them in a flat list
  void collectAllPointers(Constant *C, const DataLayout &DL,
                          uint64_t BaseOffset,
                          SmallVectorImpl<PointerInfo> &Result);

  // Collect pointer offsets from a type (for uninitialized globals)
  // All pointers are treated as PointToID = 0
  void collectPointerOffsetsFromType(Type *Ty, const DataLayout &DL,
                                     uint64_t BaseOffset,
                                     SmallVectorImpl<PointerInfo> &Result);

  // Analyze pointers and create optimized groups
  void analyzeAndGroupPointers(GlobalVariable *GV, const DataLayout &DL,
                               uint32_t DataID,
                               const SmallVectorImpl<PointerInfo> &Ptrs);

  // Generate the pointer table section
  void generatePointerTable(Module &M, const DataLayout &DL);
  
  // Generate external fixup sections
  void generateExtFixupSections(Module &M, const DataLayout &DL);
};

} // end anonymous namespace

char RISCVCollectGlobalPointers::ID = 0;

INITIALIZE_PASS(RISCVCollectGlobalPointers, DEBUG_TYPE,
                RISCV_COLLECT_GLOBAL_POINTERS_NAME, false, false)

ModulePass *llvm::createRISCVCollectGlobalPointersPass() {
  return new RISCVCollectGlobalPointers();
}

bool RISCVCollectGlobalPointers::isPointerOnlyType(Type *Ty, const DataLayout &DL,
                                                    unsigned PtrSize) {
  // Direct pointer type
  if (Ty->isPointerTy())
    return true;

  // Array of pointer-only elements
  if (ArrayType *ATy = dyn_cast<ArrayType>(Ty))
    return isPointerOnlyType(ATy->getElementType(), DL, PtrSize);

  // Struct where all elements are pointers and tightly packed
  if (StructType *STy = dyn_cast<StructType>(Ty)) {
    const StructLayout *SL = DL.getStructLayout(STy);
    uint64_t ExpectedOffset = 0;
    
    for (unsigned I = 0; I < STy->getNumElements(); ++I) {
      Type *ElemTy = STy->getElementType(I);
      
      // Check if element is pointer-only
      if (!isPointerOnlyType(ElemTy, DL, PtrSize))
        return false;
      
      // Check if offset matches expected (no padding)
      if (SL->getElementOffset(I) != ExpectedOffset)
        return false;
      
      ExpectedOffset += DL.getTypeAllocSize(ElemTy);
    }
    
    // Check no trailing padding
    return SL->getSizeInBytes() == ExpectedOffset;
  }

  // Vector of pointers (rare but possible)
  if (auto *VTy = dyn_cast<FixedVectorType>(Ty))
    return isPointerOnlyType(VTy->getElementType(), DL, PtrSize);

  return false;
}

void RISCVCollectGlobalPointers::buildGOTMap(Module &M, const DataLayout &DL) {
  uint32_t NextPointToID = 1;

  for (GlobalVariable &GV : M.globals()) {
    // Skip our own generated tables
    if (GV.getName().starts_with("__sig_"))
      continue;

    // Skip thread-local variables
    if (GV.isThreadLocal())
      continue;

    if (GV.getType()->getAddressSpace() == 100) {
      // Address space 100 doesn't need ID
      continue;
    }

    if (!(GV.getValueType()->containsPointer())) {
      // Only Data Region Doesn't need Unique ID
      continue;
    }

    // Create GOT entry for this global
    // GOT entries: DataID = 0xFFFFFF (implicit), PointToID = 1, 2, 3, ...
    GOTEntry Entry;
    Entry.GV = &GV;
    Entry.Size = DL.getTypeAllocSize(GV.getValueType());


    if (GV.isDeclaration()) {
      Entry.PointToID = NextExtID--;
      ExtGOTMap[&GV] = Entry;
      ++NumExtGOTEntries;
    } else {
      Entry.PointToID = NextPointToID++;
      GOTMap[&GV] = Entry;
      ++NumGOTEntries;
    }

    LLVM_DEBUG(dbgs() << "GOT Entry: " << GV.getName() 
                      << " PointToID=" << Entry.PointToID
                      << " Size=" << Entry.Size << "\n");
  }
  
  // Record the maximum local ID assigned
  MaxLocalID = NextPointToID - 1;
  LLVM_DEBUG(dbgs() << "MaxLocalID = " << MaxLocalID << "\n");
  
  // Build ExtGOTIDMap from ExtGOTMap for quick ExtID -> GOTEntry* lookup
  for (auto &KV : ExtGOTMap) {
    ExtGOTIDMap[KV.second.PointToID] = &KV.second;
  }
}

uint32_t RISCVCollectGlobalPointers::findGOTPointToID(const GlobalValue *GV) {
  // First check GOTMap (local globals)
  const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV);
  auto It = GOTMap.find(GVar);
  if (It != GOTMap.end())
    return It->second.PointToID;
  auto ExtIt = ExtGOTMap.find(GVar);
  if (ExtIt != ExtGOTMap.end())
    return ExtIt->second.PointToID;
  
  // Function pointers or truly unknown - return ExternalPointToID
  return ExternalPointToID;
}

void RISCVCollectGlobalPointers::recordExtFixupLocation(uint32_t ID, 
                                                         SigSectionType SecType,
                                                         uint64_t Offset) {
  // Only record if ID is in the external range
  if (isExtID(ID)) {
    auto &Info = ExtFixupMap[ID];
    if (Info.ExtID == 0) {
      Info.ExtID = ID;
      // Find the external GV from ExtGOTMap
      Info.GV = ExtGOTIDMap[ID]->GV;
    }
    Info.Locations.push_back({SecType, Offset});
  }
}

uint32_t RISCVCollectGlobalPointers::getPointToID(Constant *C) {
  // Null pointer
  if (isa<ConstantPointerNull>(C)) {
    unsigned AS = C->getType()->getPointerAddressSpace();
    if (AS == 100)
      return NullPointToID;
    return ExternalPointToID;
  }

  // Undef value
  if (isa<UndefValue>(C)) {
    return NullPointToID;
  }

  // Check if pointing to address space 100
  if (C->getType()->isPointerTy()) {
    unsigned AS = C->getType()->getPointerAddressSpace();
    if (AS == 100)
      return NullPointToID;
  }

  // Global value (variable or function)
  if (GlobalValue *GV = dyn_cast<GlobalValue>(C)) {
    if (isa<Function>(GV))
      return ExternalPointToID;
    return findGOTPointToID(GV);
  }

  // Constant expression
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    if (CE->getType()->isPointerTy() && 
        CE->getType()->getPointerAddressSpace() == 100)
      return NullPointToID;

    switch (CE->getOpcode()) {
    case Instruction::GetElementPtr:
    case Instruction::BitCast:
    case Instruction::AddrSpaceCast:
    case Instruction::IntToPtr:
      return getPointToID(CE->getOperand(0));
    default:
      break;
    }
  }

  // Block address
  if (isa<BlockAddress>(C))
    return ExternalPointToID;

  return ExternalPointToID;
}

void RISCVCollectGlobalPointers::collectAllPointers(
    Constant *C, const DataLayout &DL, uint64_t BaseOffset,
    SmallVectorImpl<PointerInfo> &Result) {

  Type *Ty = C->getType();

  // If this is a pointer type, record it
  if (Ty->isPointerTy()) {
    PointerInfo Info;
    Info.Offset = BaseOffset;
    Info.PointToID = getPointToID(C);
    Result.push_back(Info);
    ++NumPointersCollected;
    return;
  }

  // Handle aggregate types
  if (ConstantArray *CA = dyn_cast<ConstantArray>(C)) {
    Type *ElemTy = CA->getType()->getElementType();
    uint64_t ElemSize = DL.getTypeAllocSize(ElemTy);
    for (unsigned I = 0; I < CA->getNumOperands(); ++I) {
      collectAllPointers(CA->getOperand(I), DL,
                         BaseOffset + I * ElemSize, Result);
    }
    return;
  }

  if (ConstantStruct *CS = dyn_cast<ConstantStruct>(C)) {
    StructType *STy = CS->getType();
    const StructLayout *SL = DL.getStructLayout(STy);
    for (unsigned I = 0; I < CS->getNumOperands(); ++I) {
      uint64_t ElemOffset = SL->getElementOffset(I);
      collectAllPointers(CS->getOperand(I), DL,
                         BaseOffset + ElemOffset, Result);
    }
    return;
  }

  if (ConstantVector *CV = dyn_cast<ConstantVector>(C)) {
    Type *ElemTy = CV->getType()->getElementType();
    uint64_t ElemSize = DL.getTypeAllocSize(ElemTy);
    for (unsigned I = 0; I < CV->getNumOperands(); ++I) {
      collectAllPointers(CV->getOperand(I), DL,
                         BaseOffset + I * ElemSize, Result);
    }
    return;
  }

  if (ConstantAggregateZero *CAZ = dyn_cast<ConstantAggregateZero>(C)) {
    (void)CAZ;
    if (Ty->containsPointer()) {
      if (ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
        Type *ElemTy = ATy->getElementType();
        uint64_t ElemSize = DL.getTypeAllocSize(ElemTy);
        Constant *ZeroElem = Constant::getNullValue(ElemTy);
        for (uint64_t I = 0; I < ATy->getNumElements(); ++I) {
          collectAllPointers(ZeroElem, DL,
                             BaseOffset + I * ElemSize, Result);
        }
      } else if (StructType *STy = dyn_cast<StructType>(Ty)) {
        const StructLayout *SL = DL.getStructLayout(STy);
        for (unsigned I = 0; I < STy->getNumElements(); ++I) {
          uint64_t ElemOffset = SL->getElementOffset(I);
          Constant *ZeroElem = Constant::getNullValue(STy->getElementType(I));
          collectAllPointers(ZeroElem, DL,
                             BaseOffset + ElemOffset, Result);
        }
      }
    }
    return;
  }

  // ConstantDataSequential, ConstantInt, ConstantFP - no pointers
}

void RISCVCollectGlobalPointers::collectPointerOffsetsFromType(
    Type *Ty, const DataLayout &DL, uint64_t BaseOffset,
    SmallVectorImpl<PointerInfo> &Result) {

  // If this is a pointer type, record it with PointToID = 0 (uninitialized)
  if (Ty->isPointerTy()) {
    PointerInfo Info;
    Info.Offset = BaseOffset;
    Info.PointToID = NullPointToID;  // Uninitialized pointers get PointToID = 0
    Result.push_back(Info);
    ++NumPointersCollected;
    return;
  }

  // Handle array types
  if (ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    Type *ElemTy = ATy->getElementType();
    if (!ElemTy->containsPointer())
      return;
    
    uint64_t ElemSize = DL.getTypeAllocSize(ElemTy);
    for (uint64_t I = 0; I < ATy->getNumElements(); ++I) {
      collectPointerOffsetsFromType(ElemTy, DL, BaseOffset + I * ElemSize, Result);
    }
    return;
  }

  // Handle struct types
  if (StructType *STy = dyn_cast<StructType>(Ty)) {
    const StructLayout *SL = DL.getStructLayout(STy);
    for (unsigned I = 0; I < STy->getNumElements(); ++I) {
      Type *ElemTy = STy->getElementType(I);
      if (ElemTy->containsPointer()) {
        uint64_t ElemOffset = SL->getElementOffset(I);
        collectPointerOffsetsFromType(ElemTy, DL, BaseOffset + ElemOffset, Result);
      }
    }
    return;
  }

  // Handle vector types
  if (auto *VTy = dyn_cast<FixedVectorType>(Ty)) {
    Type *ElemTy = VTy->getElementType();
    if (!(ElemTy->containsPointer()))
      return;
    
    uint64_t ElemSize = DL.getTypeAllocSize(ElemTy);
    for (unsigned I = 0; I < VTy->getNumElements(); ++I) {
      collectPointerOffsetsFromType(ElemTy, DL, BaseOffset + I * ElemSize, Result);
    }
    return;
  }

  // Other types don't contain pointers
}

void RISCVCollectGlobalPointers::analyzeAndGroupPointers(
    GlobalVariable *GV, const DataLayout &DL, uint32_t DataID,
    const SmallVectorImpl<PointerInfo> &Ptrs) {
  
  if (Ptrs.empty())
    return;

  unsigned PtrSize = DL.getPointerSize();

  // Single pointer - Type5
  if (Ptrs.size() == 1) {
    PointerGroup Group;
    Group.ContainingGV = GV;
    Group.DataID = DataID;
    Group.Type = PtrGroupType::Single;
    Group.Ptrs.push_back(Ptrs[0]);
    PointerGroups.push_back(std::move(Group));
    ++NumCompressedEntries;
    
    LLVM_DEBUG(dbgs() << "  Single pointer at offset " << Ptrs[0].Offset
                      << " PointToID=" << Ptrs[0].PointToID << "\n");
    return;
  }

  // Check if pointers are contiguous
  bool IsContiguous = true;
  uint64_t ExpectedOffset = Ptrs[0].Offset;
  for (const auto &P : Ptrs) {
    if (P.Offset != ExpectedOffset) {
      IsContiguous = false;
      break;
    }
    ExpectedOffset += PtrSize;
  }

  // Check if all PointToIDs are the same
  bool AllSamePointToID = true;
  uint32_t FirstPointToID = Ptrs[0].PointToID;
  for (const auto &P : Ptrs) {
    if (P.PointToID != FirstPointToID) {
      AllSamePointToID = false;
      break;
    }
  }

  PointerGroup Group;
  Group.ContainingGV = GV;
  Group.DataID = DataID;

  if (IsContiguous && AllSamePointToID) {
    // Type2: Contiguous pointers with same PointToID
    Group.Type = PtrGroupType::ContiguousSame;
    Group.CommonPointToID = FirstPointToID;
    for (const auto &P : Ptrs)
      Group.Ptrs.push_back(P);
    
    LLVM_DEBUG(dbgs() << "  ContiguousSame: " << Ptrs.size() 
                      << " pointers, PointToID=" << FirstPointToID << "\n");
  } else if (IsContiguous && !AllSamePointToID) {
    // Type1: Contiguous pointers with different PointToIDs
    Group.Type = PtrGroupType::ContiguousDifferent;
    for (const auto &P : Ptrs)
      Group.Ptrs.push_back(P);
    
    LLVM_DEBUG(dbgs() << "  ContiguousDifferent: " << Ptrs.size() << " pointers\n");
  } else if (!IsContiguous && AllSamePointToID) {
    // Type4: Non-contiguous pointers with same PointToID
    Group.Type = PtrGroupType::SparseSame;
    Group.CommonPointToID = FirstPointToID;
    for (const auto &P : Ptrs)
      Group.Offsets.push_back(P.Offset);
    
    LLVM_DEBUG(dbgs() << "  SparseSame: " << Ptrs.size() 
                      << " pointers, PointToID=" << FirstPointToID << "\n");
  } else {
    // Type3: Non-contiguous pointers with different PointToIDs
    Group.Type = PtrGroupType::SparseDifferent;
    for (const auto &P : Ptrs)
      Group.Ptrs.push_back(P);
    
    LLVM_DEBUG(dbgs() << "  SparseDifferent: " << Ptrs.size() << " pointers\n");
  }

  PointerGroups.push_back(std::move(Group));
  ++NumCompressedEntries;
}

void RISCVCollectGlobalPointers::generatePointerTable(Module &M, 
                                                       const DataLayout &DL) {
  LLVMContext &Ctx = M.getContext();
  unsigned PtrSize = DL.getPointerSize();
  
  // Types for table entries
  Type *I8Ty = Type::getInt8Ty(Ctx);
  Type *I32Ty = Type::getInt32Ty(Ctx);
  Type *I64Ty = Type::getInt64Ty(Ctx);
  Type *PtrSizedIntTy = Type::getIntNTy(Ctx, PtrSize * 8);

  // === Build Symbol Name String Table ===
  // Collect all symbol names and their offsets in the string table
  // Only include externally visible symbols (other files may reference them)
  std::string SymtabString;
  SymNameOffsets.clear();  // Clear member variable
  
  // Add externally visible GOTMap entries (for regular .sig_got)
  // Skip static symbols (internal/private linkage) as they can't be referenced externally
  for (auto &KV : GOTMap) {
    const GlobalVariable *GV = KV.first;
    // Only add externally visible symbols to symtab
    if (!GV->hasLocalLinkage()) {
      SymNameOffsets[GV] = SymtabString.size();
      SymtabString += GV->getName().str();
      SymtabString += '\0';  // Null terminator
    }
  }

  // === Generate GOT Table ===
  // New format: { addr(64), id(64), sym_name_offset(64) } = 24 bytes
  // sym_name_offset is the offset of the symbol name in .sig_symtab
  // Sort by PointToID to ensure consistent ordering
  
  SmallVector<const GOTEntry *, 64> SortedGOTEntries;
  for (auto &KV : GOTMap) {
    SortedGOTEntries.push_back(&KV.second);
  }
  llvm::sort(SortedGOTEntries, [](const GOTEntry *A, const GOTEntry *B) {
    return A->PointToID < B->PointToID;
  });
  
  SmallVector<Constant *, 64> GOTTableEntries;
  // { addr(64), id(64), sym_name_offset(64) }
  StructType *GOTEntryTy = StructType::get(Ctx, {PtrSizedIntTy, I64Ty, I64Ty});

  for (const GOTEntry *Entry : SortedGOTEntries) {
    Constant *GVPtr = ConstantExpr::getPtrToInt(Entry->GV, PtrSizedIntTy);
    Constant *PointToIDConst = ConstantInt::get(I64Ty, Entry->PointToID);
    // sym_name_offset is the offset of the symbol name in .sig_symtab
    // Use UINT64_MAX for static symbols (not in symtab) so linker can skip them
    uint64_t SymNameOffset = UINT64_MAX;  // Default: static symbol
    auto SymIt = SymNameOffsets.find(Entry->GV);
    if (SymIt != SymNameOffsets.end()) {
      SymNameOffset = SymIt->second;
    }
    Constant *SymNameOffsetConst = ConstantInt::get(I64Ty, SymNameOffset);

    Constant *GOTEntryConst = ConstantStruct::get(GOTEntryTy, 
                                                   {GVPtr, PointToIDConst, SymNameOffsetConst});
    GOTTableEntries.push_back(GOTEntryConst);
  }

  if (!GOTTableEntries.empty()) {
    ArrayType *GOTTableTy = ArrayType::get(GOTEntryTy, GOTTableEntries.size());
    Constant *GOTTableInit = ConstantArray::get(GOTTableTy, GOTTableEntries);

    GlobalVariable *GOTTableGV = new GlobalVariable(
        M, GOTTableTy, /*isConstant=*/true, GlobalValue::PrivateLinkage,
        GOTTableInit, "");
    GOTTableGV->setSection(SectionGOT);
    GOTTableGV->setAlignment(Align(PtrSize));

    // Count placed in .sig_ptr_header_counter section (all counts in one section)
    GlobalVariable *GOTSizeGV = new GlobalVariable(
        M, I32Ty, /*isConstant=*/true, GlobalValue::PrivateLinkage,
        ConstantInt::get(I32Ty, GOTTableEntries.size()),
        "");
    GOTSizeGV->setSection(SectionCounter);
  } else {
    for (int i = 0; i < 7; i++) {
      GlobalVariable *GOTSizeGV = new GlobalVariable(
          M, I32Ty, /*isConstant=*/true, GlobalValue::PrivateLinkage,
          ConstantInt::get(I32Ty, 0),
          "");
      GOTSizeGV->setSection(SectionCounter);
    }
    return;
  }

  // === Generate Compressed Pointer Tables ===
  // Header-Data Separation Mode:
  // - 5 Header sections (fixed-length entries)
  // - 4 Data array sections (variable-length data)
  //
  // addr/count/offset are 64-bit (ptr-sized), DataID/PointToID are 32-bit
  
  // ========== Header Sections ==========
  
  // Type5 (Single) Header: { addr(64), DataID(32), PointToID(32) }
  SmallVector<Constant *, 32> SingleHeaders;
  StructType *SingleHeaderTy = StructType::get(Ctx, {PtrSizedIntTy, I32Ty, I32Ty});
  
  // Type2 (ContiguousSame) Header: { addr(64), count(64), DataID(32), PointToID(32) }
  SmallVector<Constant *, 32> ContiguousSameHeaders;
  StructType *ContiguousSameHeaderTy = StructType::get(Ctx, 
      {PtrSizedIntTy, PtrSizedIntTy, I32Ty, I32Ty});
  
  // Type1 (ContiguousDifferent) Header: { addr(64), count(64) }
  // Data: ID array = DataID + PointToID[count] (32-bit each, combined)
  SmallVector<Constant *, 32> ContiguousDiffHeaders;
  StructType *ContiguousDiffHeaderTy = StructType::get(Ctx, {PtrSizedIntTy, PtrSizedIntTy});
  SmallVector<Constant *, 256> ContiguousDiffIDs;  // DataID + PointToID[] combined
  
  // Type4 (SparseSame) Header: { addr(64), count(64), DataID(32), PointToID(32) }
  // Data: Offset array (64-bit each)
  SmallVector<Constant *, 32> SparseSameHeaders;
  StructType *SparseSameHeaderTy = StructType::get(Ctx, 
      {PtrSizedIntTy, PtrSizedIntTy, I32Ty, I32Ty});
  SmallVector<Constant *, 256> SparseSameOffsets;  // offset[](64) array
  
  // Type3 (SparseDifferent) Header: { addr(64), count(64) }
  // Data: ID array = DataID + PointToID[count] (32-bit each, combined)
  // Data: Offset array (64-bit each)
  SmallVector<Constant *, 32> SparseDiffHeaders;
  StructType *SparseDiffHeaderTy = StructType::get(Ctx, {PtrSizedIntTy, PtrSizedIntTy});
  SmallVector<Constant *, 256> SparseDiffIDs;      // DataID + PointToID[] combined
  SmallVector<Constant *, 256> SparseDiffOffsets;  // offset[](64) array

  // Track current byte offset in each section for external fixup
  uint64_t SingleHeaderOffset = 0;
  uint64_t ContiguousSameHeaderOffset = 0;
  uint64_t ContiguousDiffIDOffset = 0;
  uint64_t SparseSameHeaderOffset = 0;
  uint64_t SparseDiffIDOffset = 0;
  
  // Size of each header type (in bytes)
  const uint64_t SingleHeaderSize = PtrSize + 4 + 4;  // addr(64) + DataID(32) + PointToID(32)
  const uint64_t ContiguousSameHeaderSize = PtrSize + PtrSize + 4 + 4;  // addr + count + DataID + PointToID
  const uint64_t SparseSameHeaderSize = PtrSize + PtrSize + 4 + 4;  // addr + count + DataID + PointToID
  
  // Offset of PointToID field within each header type (for external fixup tracking)
  // Single: { addr(64), DataID(32), PointToID(32) } -> PointToID at offset addr + DataID
  const uint64_t SinglePointToIDOffset = PtrSize + 4;
  // ContiguousSame: { addr(64), count(64), DataID(32), PointToID(32) } -> PointToID at addr + count + DataID
  const uint64_t ContiguousSamePointToIDOffset = PtrSize + PtrSize + 4;
  // SparseSame: { addr(64), count(64), DataID(32), PointToID(32) } -> same as ContiguousSame
  const uint64_t SparseSamePointToIDOffset = PtrSize + PtrSize + 4;
  
  // For ID arrays (ContiguousDiff, SparseDiff): DataID is first, then PointToID[]
  // PointToID[i] is at offset: DataID(4) + i * sizeof(uint32_t)
  const uint64_t IDArrayDataIDSize = 4;   // DataID is 32-bit
  const uint64_t IDArrayElementSize = 4;  // Each PointToID is 32-bit

  for (const PointerGroup &Group : PointerGroups) {
    Constant *GVPtr = Group.ContainingGV;
    
    switch (Group.Type) {
    case PtrGroupType::Single: {
      // Header: { addr(64), DataID(32), PointToID(32) }
      const PointerInfo &P = Group.Ptrs[0];
      Constant *OffsetConst = ConstantInt::get(PtrSizedIntTy, P.Offset);
      Constant *GEP = ConstantExpr::getInBoundsGetElementPtr(I8Ty, GVPtr, OffsetConst);
      Constant *Addr = ConstantExpr::getPtrToInt(GEP, PtrSizedIntTy);
      
      // Track external fixup location
      if (isExtID(P.PointToID)) {
        recordExtFixupLocation(P.PointToID, SigSectionType::HeaderSingle, 
                               SingleHeaderOffset + SinglePointToIDOffset);
      }
      
      Constant *Header = ConstantStruct::get(SingleHeaderTy,
          {Addr, ConstantInt::get(I32Ty, Group.DataID),
           ConstantInt::get(I32Ty, P.PointToID)});
      SingleHeaders.push_back(Header);
      SingleHeaderOffset += SingleHeaderSize;
      break;
    }
    
    case PtrGroupType::ContiguousSame: {
      // Header: { addr(64), count(64), DataID(32), PointToID(32) }
      const PointerInfo &P = Group.Ptrs[0];
      Constant *OffsetConst = ConstantInt::get(PtrSizedIntTy, P.Offset);
      Constant *GEP = ConstantExpr::getInBoundsGetElementPtr(I8Ty, GVPtr, OffsetConst);
      Constant *Addr = ConstantExpr::getPtrToInt(GEP, PtrSizedIntTy);
      
      // Track external fixup location
      if (isExtID(Group.CommonPointToID)) {
        recordExtFixupLocation(Group.CommonPointToID, SigSectionType::HeaderContigSame,
                               ContiguousSameHeaderOffset + ContiguousSamePointToIDOffset);
      }
      
      Constant *Header = ConstantStruct::get(ContiguousSameHeaderTy,
          {Addr, ConstantInt::get(PtrSizedIntTy, Group.Ptrs.size()),
           ConstantInt::get(I32Ty, Group.DataID),
           ConstantInt::get(I32Ty, Group.CommonPointToID)});
      ContiguousSameHeaders.push_back(Header);
      ContiguousSameHeaderOffset += ContiguousSameHeaderSize;
      break;
    }
    
    case PtrGroupType::ContiguousDifferent: {
      // Header: { addr(64), count(64) }
      const PointerInfo &P = Group.Ptrs[0];
      Constant *OffsetConst = ConstantInt::get(PtrSizedIntTy, P.Offset);
      Constant *GEP = ConstantExpr::getInBoundsGetElementPtr(I8Ty, GVPtr, OffsetConst);
      Constant *Addr = ConstantExpr::getPtrToInt(GEP, PtrSizedIntTy);
      
      Constant *Header = ConstantStruct::get(ContiguousDiffHeaderTy,
          {Addr, ConstantInt::get(PtrSizedIntTy, Group.Ptrs.size())});
      ContiguousDiffHeaders.push_back(Header);
      
      // Data: Combined ID segment = DataID(32) + PointToID[count](32 each)
      ContiguousDiffIDs.push_back(ConstantInt::get(I32Ty, Group.DataID));
      uint64_t PtrIdx = 0;
      for (const PointerInfo &PI : Group.Ptrs) {
        // Track external fixup: PointToID[i] at offset DataID + index * ElementSize
        if (isExtID(PI.PointToID)) {
          recordExtFixupLocation(PI.PointToID, SigSectionType::IDContigDiff,
                                 ContiguousDiffIDOffset + IDArrayDataIDSize + PtrIdx * IDArrayElementSize);
        }
        ContiguousDiffIDs.push_back(ConstantInt::get(I32Ty, PI.PointToID));
        ++PtrIdx;
      }
      // Update offset: DataID + PointToID[count]
      ContiguousDiffIDOffset += IDArrayDataIDSize + Group.Ptrs.size() * IDArrayElementSize;
      break;
    }
    
    case PtrGroupType::SparseSame: {
      // Header: { addr(64), count(64), DataID(32), PointToID(32) }
      Constant *Addr = ConstantExpr::getPtrToInt(GVPtr, PtrSizedIntTy);
      
      // Track external fixup location
      if (isExtID(Group.CommonPointToID)) {
        recordExtFixupLocation(Group.CommonPointToID, SigSectionType::HeaderSparseSame,
                               SparseSameHeaderOffset + SparseSamePointToIDOffset);
      }
      
      Constant *Header = ConstantStruct::get(SparseSameHeaderTy,
          {Addr, ConstantInt::get(PtrSizedIntTy, Group.Offsets.size()),
           ConstantInt::get(I32Ty, Group.DataID),
           ConstantInt::get(I32Ty, Group.CommonPointToID)});
      SparseSameHeaders.push_back(Header);
      SparseSameHeaderOffset += SparseSameHeaderSize;
      
      // Data: offset[count](64 each) - separate offset array for SparseSame
      for (uint64_t Off : Group.Offsets) {
        SparseSameOffsets.push_back(ConstantInt::get(PtrSizedIntTy, Off));
      }
      break;
    }
    
    case PtrGroupType::SparseDifferent: {
      // Header: { addr(64), count(64) }
      Constant *Addr = ConstantExpr::getPtrToInt(GVPtr, PtrSizedIntTy);
      
      Constant *Header = ConstantStruct::get(SparseDiffHeaderTy,
          {Addr, ConstantInt::get(PtrSizedIntTy, Group.Ptrs.size())});
      SparseDiffHeaders.push_back(Header);
      
      // Data: Combined ID segment = DataID(32) + PointToID[count](32 each)
      SparseDiffIDs.push_back(ConstantInt::get(I32Ty, Group.DataID));
      uint64_t PtrIdx = 0;
      for (const PointerInfo &PI : Group.Ptrs) {
        // Track external fixup: PointToID[i] at offset DataID + index * ElementSize
        if (isExtID(PI.PointToID)) {
          recordExtFixupLocation(PI.PointToID, SigSectionType::IDSparseDiff,
                                 SparseDiffIDOffset + IDArrayDataIDSize + PtrIdx * IDArrayElementSize);
        }
        SparseDiffIDs.push_back(ConstantInt::get(I32Ty, PI.PointToID));
        ++PtrIdx;
      }
      // Update offset: DataID + PointToID[count]
      SparseDiffIDOffset += IDArrayDataIDSize + Group.Ptrs.size() * IDArrayElementSize;
      
      // Data: offset[count](64 each) - separate offset array for SparseDiff
      for (const PointerInfo &PI : Group.Ptrs) {
        SparseDiffOffsets.push_back(ConstantInt::get(PtrSizedIntTy, PI.Offset));
      }
      break;
    }
    }
  }

  // Add only ExtFixupMap entries that are actually used (for .sig_ext_fixup)
  // This avoids adding unnecessary entries from ExtGOTMap
  // Uses ExtGOTIDMap (member variable built in buildGOTMap)
  for (auto &KV : ExtFixupMap) {
    uint32_t ExtID = KV.first;
    auto It = ExtGOTIDMap.find(ExtID);
    assert (It != ExtGOTIDMap.end());
    const GlobalVariable *GV = It->second->GV;
    // Only add if not already in SymNameOffsets (from GOTMap)
    assert (SymNameOffsets.find(GV) == SymNameOffsets.end());
    SymNameOffsets[GV] = SymtabString.size();
    SymtabString += GV->getName().str();
    SymtabString += '\0';  // Null terminator
  }
  
  // Create the symbol name string table global variable
  if (!SymtabString.empty()) {
    Constant *SymtabInit = ConstantDataArray::getString(Ctx, SymtabString, false);
    GlobalVariable *SymtabGV = new GlobalVariable(
        M, SymtabInit->getType(), true, GlobalValue::PrivateLinkage,
        SymtabInit, "");
    SymtabGV->setSection(SectionSymtab);
    SymtabGV->setAlignment(Align(1));
  }

  // ========== Generate Global Variables ==========
  // 5 Header sections + 4 Data array sections = 9 sections total
  // All counts go to .sig_ptr_header_counter section
  
  // === Header Section 1: Type5 Single ===
  if (!SingleHeaders.empty()) {
    ArrayType *Ty = ArrayType::get(SingleHeaderTy, SingleHeaders.size());
    Constant *Init = ConstantArray::get(Ty, SingleHeaders);
    GlobalVariable *GV = new GlobalVariable(
        M, Ty, true, GlobalValue::PrivateLinkage, Init, "");
    GV->setSection(SectionHeaderSingle);
    GV->setAlignment(Align(PtrSize));
    
    GlobalVariable *CountGV = new GlobalVariable(M, I32Ty, true, GlobalValue::PrivateLinkage,
        ConstantInt::get(I32Ty, SingleHeaders.size()), "");
    CountGV->setSection(SectionCounter);
  } else {
    GlobalVariable *CountGV = new GlobalVariable(M, I32Ty, true, GlobalValue::PrivateLinkage,
        ConstantInt::get(I32Ty, 0), "");
    CountGV->setSection(SectionCounter);
  }
  
  // === Header Section 2: Type2 ContiguousSame ===
  if (!ContiguousSameHeaders.empty()) {
    ArrayType *Ty = ArrayType::get(ContiguousSameHeaderTy, ContiguousSameHeaders.size());
    Constant *Init = ConstantArray::get(Ty, ContiguousSameHeaders);
    GlobalVariable *GV = new GlobalVariable(
        M, Ty, true, GlobalValue::PrivateLinkage, Init, "");
    GV->setSection(SectionHeaderContigSame);
    GV->setAlignment(Align(PtrSize));
    
    GlobalVariable *CountGV = new GlobalVariable(M, I32Ty, true, GlobalValue::PrivateLinkage,
        ConstantInt::get(I32Ty, ContiguousSameHeaders.size()), "");
    CountGV->setSection(SectionCounter);
  } else {
    GlobalVariable *CountGV = new GlobalVariable(M, I32Ty, true, GlobalValue::PrivateLinkage,
        ConstantInt::get(I32Ty, 0), "");
    CountGV->setSection(SectionCounter);
  }
  
  // === Header Section 3: Type1 ContiguousDifferent ===
  if (!ContiguousDiffHeaders.empty()) {
    ArrayType *Ty = ArrayType::get(ContiguousDiffHeaderTy, ContiguousDiffHeaders.size());
    Constant *Init = ConstantArray::get(Ty, ContiguousDiffHeaders);
    GlobalVariable *GV = new GlobalVariable(
        M, Ty, true, GlobalValue::PrivateLinkage, Init, "");
    GV->setSection(SectionHeaderContigDiff);
    GV->setAlignment(Align(PtrSize));
    
    GlobalVariable *CountGV = new GlobalVariable(M, I32Ty, true, GlobalValue::PrivateLinkage,
        ConstantInt::get(I32Ty, ContiguousDiffHeaders.size()), "");
    CountGV->setSection(SectionCounter);
  } else {
    GlobalVariable *CountGV = new GlobalVariable(M, I32Ty, true, GlobalValue::PrivateLinkage,
        ConstantInt::get(I32Ty, 0), "");
    CountGV->setSection(SectionCounter);
  }
  
  // === Header Section 4: Type4 SparseSame ===
  if (!SparseSameHeaders.empty()) {
    ArrayType *Ty = ArrayType::get(SparseSameHeaderTy, SparseSameHeaders.size());
    Constant *Init = ConstantArray::get(Ty, SparseSameHeaders);
    GlobalVariable *GV = new GlobalVariable(
        M, Ty, true, GlobalValue::PrivateLinkage, Init, "");
    GV->setSection(SectionHeaderSparseSame);
    GV->setAlignment(Align(PtrSize));
    
    GlobalVariable *CountGV = new GlobalVariable(M, I32Ty, true, GlobalValue::PrivateLinkage,
        ConstantInt::get(I32Ty, SparseSameHeaders.size()), "");
    CountGV->setSection(SectionCounter);
  } else {
    GlobalVariable *CountGV = new GlobalVariable(M, I32Ty, true, GlobalValue::PrivateLinkage,
        ConstantInt::get(I32Ty, 0), "");
    CountGV->setSection(SectionCounter);
  }
  
  // === Header Section 5: Type3 SparseDifferent ===
  if (!SparseDiffHeaders.empty()) {
    ArrayType *Ty = ArrayType::get(SparseDiffHeaderTy, SparseDiffHeaders.size());
    Constant *Init = ConstantArray::get(Ty, SparseDiffHeaders);
    GlobalVariable *GV = new GlobalVariable(
        M, Ty, true, GlobalValue::PrivateLinkage, Init, "");
    GV->setSection(SectionHeaderSparseDiff);
    GV->setAlignment(Align(PtrSize));
    
    GlobalVariable *CountGV = new GlobalVariable(M, I32Ty, true, GlobalValue::PrivateLinkage,
        ConstantInt::get(I32Ty, SparseDiffHeaders.size()), "");
    CountGV->setSection(SectionCounter);
  } else {
    GlobalVariable *CountGV = new GlobalVariable(M, I32Ty, true, GlobalValue::PrivateLinkage,
        ConstantInt::get(I32Ty, 0), "");
    CountGV->setSection(SectionCounter);
  }
  
  // === Data Array Section 1: ContiguousDiff IDs (DataID + PointToID[] combined, 32-bit each) ===
  if (!ContiguousDiffIDs.empty()) {
    ArrayType *Ty = ArrayType::get(I32Ty, ContiguousDiffIDs.size());
    Constant *Init = ConstantArray::get(Ty, ContiguousDiffIDs);
    GlobalVariable *GV = new GlobalVariable(
        M, Ty, true, GlobalValue::PrivateLinkage, Init, "");
    GV->setSection(SectionIDContigDiff);
    GV->setAlignment(Align(4));
  }
  
  // === Data Array Section 2: SparseDiff IDs (DataID + PointToID[] combined, 32-bit each) ===
  if (!SparseDiffIDs.empty()) {
    ArrayType *Ty = ArrayType::get(I32Ty, SparseDiffIDs.size());
    Constant *Init = ConstantArray::get(Ty, SparseDiffIDs);
    GlobalVariable *GV = new GlobalVariable(
        M, Ty, true, GlobalValue::PrivateLinkage, Init, "");
    GV->setSection(SectionIDSparseDiff);
    GV->setAlignment(Align(4));
  }
  
  // === Data Array Section 3: SparseSame Offsets (64-bit each) ===
  if (!SparseSameOffsets.empty()) {
    ArrayType *Ty = ArrayType::get(PtrSizedIntTy, SparseSameOffsets.size());
    Constant *Init = ConstantArray::get(Ty, SparseSameOffsets);
    GlobalVariable *GV = new GlobalVariable(
        M, Ty, true, GlobalValue::PrivateLinkage, Init, "");
    GV->setSection(SectionOffsetSparseSame);
    GV->setAlignment(Align(PtrSize));
  }
  
  // === Data Array Section 4: SparseDiff Offsets (64-bit each) ===
  if (!SparseDiffOffsets.empty()) {
    ArrayType *Ty = ArrayType::get(PtrSizedIntTy, SparseDiffOffsets.size());
    Constant *Init = ConstantArray::get(Ty, SparseDiffOffsets);
    GlobalVariable *GV = new GlobalVariable(
        M, Ty, true, GlobalValue::PrivateLinkage, Init, "");
    GV->setSection(SectionOffsetSparseDiff);
    GV->setAlignment(Align(PtrSize));
  }

  LLVM_DEBUG(dbgs() << "Generated GOT table with " << GOTTableEntries.size()
                    << " entries\n");
  LLVM_DEBUG(dbgs() << "Generated pointer tables (Header-Data Separation):\n"
                    << "  Single Headers: " << SingleHeaders.size() << "\n"
                    << "  ContiguousSame Headers: " << ContiguousSameHeaders.size() << "\n"
                    << "  ContiguousDiff Headers: " << ContiguousDiffHeaders.size() << "\n"
                    << "  SparseSame Headers: " << SparseSameHeaders.size() << "\n"
                    << "  SparseDiff Headers: " << SparseDiffHeaders.size() << "\n"
                    << "  ContiguousDiff IDs: " << ContiguousDiffIDs.size() << "\n"
                    << "  SparseDiff IDs: " << SparseDiffIDs.size() << "\n"
                    << "  SparseSame Offsets: " << SparseSameOffsets.size() << "\n"
                    << "  SparseDiff Offsets: " << SparseDiffOffsets.size() << "\n");

  // Generate external fixup sections if there are any external references
  generateExtFixupSections(M, DL);
}

// Generate external fixup sections for external symbol references
// These sections allow the linker to fix up external PointToID values during static linking
void RISCVCollectGlobalPointers::generateExtFixupSections(Module &M, const DataLayout &DL) {
  LLVMContext &Ctx = M.getContext();
  unsigned PtrSize = DL.getPointerSize();
  IntegerType *PtrSizedIntTy = Type::getIntNTy(Ctx, PtrSize * 8);
  IntegerType *I32Ty = Type::getInt32Ty(Ctx);

  if (ExtFixupMap.empty() && ExtGOTMap.empty()) {
    LLVM_DEBUG(dbgs() << "No external references, skipping ext fixup sections\n");
    GlobalVariable *CountGV = new GlobalVariable(M, I32Ty, true, GlobalValue::PrivateLinkage,
        ConstantInt::get(I32Ty, 0), "");
    CountGV->setSection(SectionCounter);
    return;
  }

  // ========== Generate External Fixup Sections ==========
  // Header format: { ext_id(64), sym_index(64), data_offset(64) }
  // Data format: { section_type(8), offset(64) }[] for each location
  if (!ExtFixupMap.empty()) {
    StructType *FixupHeaderTy = StructType::get(Ctx, {
        PtrSizedIntTy,  // ext_id
        PtrSizedIntTy,  // sym_name_offset (offset in .sig_symtab)
        PtrSizedIntTy,  // data_offset
        PtrSizedIntTy   // length (number of locations)
    });
    
    SmallVector<Constant *, 32> FixupHeaders;
    SmallVector<Constant *, 128> FixupDataEntries;
    
    uint64_t DataOffset = 0;
    const uint64_t DataEntrySize = PtrSize;  // offset(8)

    // Use member variable ExtGOTIDMap instead of creating local map
    for (auto &KV : ExtFixupMap) {
      uint32_t ExtID = KV.first;
      const ExtFixupInfo &Info = KV.second;
      GlobalVariable *GV = ExtGOTIDMap[ExtID]->GV;
      
      // Get symbol name offset from SymNameOffsets
      assert(SymNameOffsets.count(GV) && "External fixup GV missing from SymNameOffsets");
      uint64_t SymNameOffset = SymNameOffsets[GV];
      
      // Create header entry
      Constant *Header = ConstantStruct::get(FixupHeaderTy, {
          ConstantInt::get(PtrSizedIntTy, ExtID),
          ConstantInt::get(PtrSizedIntTy, SymNameOffset),
          ConstantInt::get(PtrSizedIntTy, DataOffset),
          ConstantInt::get(PtrSizedIntTy, Info.Locations.size())
      });
      FixupHeaders.push_back(Header);
      
      // Create data entries for all locations
      for (const ExtFixupLocation &Loc : Info.Locations) {
        uint64_t Data = (Loc.Offset & ((1L << 56) -1)) | (static_cast<uint64_t>(Loc.SectionType) << 56);
        Constant *DataEntry = ConstantInt::get(PtrSizedIntTy, Data);
        FixupDataEntries.push_back(DataEntry);
        DataOffset += DataEntrySize;
      }
    }
    
    // Generate fixup header section
    if (!FixupHeaders.empty()) {
      ArrayType *HeaderArrayTy = ArrayType::get(FixupHeaderTy, FixupHeaders.size());
      Constant *HeaderInit = ConstantArray::get(HeaderArrayTy, FixupHeaders);
      GlobalVariable *HeaderGV = new GlobalVariable(
          M, HeaderArrayTy, true, GlobalValue::PrivateLinkage, HeaderInit, "");
      HeaderGV->setSection(SectionExtFixupHeader);
      HeaderGV->setAlignment(Align(PtrSize));
      
      // Count for fixup headers
      GlobalVariable *FixupCountGV = new GlobalVariable(M, I32Ty, true, 
          GlobalValue::PrivateLinkage,
          ConstantInt::get(I32Ty, FixupHeaders.size()), "");
      FixupCountGV->setSection(SectionCounter);
    } else {
      GlobalVariable *FixupCountGV = new GlobalVariable(M, I32Ty, true, 
          GlobalValue::PrivateLinkage,
          ConstantInt::get(I32Ty, 0), "");
      FixupCountGV->setSection(SectionCounter);
    }
    
    // Generate fixup data section
    if (!FixupDataEntries.empty()) {
      ArrayType *DataArrayTy = ArrayType::get(PtrSizedIntTy, FixupDataEntries.size());
      Constant *DataInit = ConstantArray::get(DataArrayTy, FixupDataEntries);
      GlobalVariable *DataGV = new GlobalVariable(
          M, DataArrayTy, true, GlobalValue::PrivateLinkage, DataInit, "");
      DataGV->setSection(SectionExtFixupData);
      DataGV->setAlignment(Align(PtrSize));
    }
    
    LLVM_DEBUG(dbgs() << "Generated external fixup tables:\n"
                      << "  Fixup Headers: " << FixupHeaders.size() << "\n"
                      << "  Fixup Data Entries: " << FixupDataEntries.size() << "\n");
  }
}

bool RISCVCollectGlobalPointers::runOnModule(Module &M) {
  // Check if SigMode is enabled
  const TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
  const TargetMachine &TM = TPC.getTM<TargetMachine>();

  bool SigModeEnabled = false;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    const RISCVSubtarget &ST = TM.getSubtarget<RISCVSubtarget>(F);
    if (ST.isSigModeSupport()) {
      SigModeEnabled = true;
      break;
    }
  }

  if (!SigModeEnabled) {
    LLVM_DEBUG(dbgs() << "SigMode not enabled, skipping\n");
    return false;
  }

  const DataLayout &DL = M.getDataLayout();

  // Clear any previous data
  GOTMap.clear();
  ExtGOTMap.clear();
  ExtGOTIDMap.clear();
  ExtFixupMap.clear();
  SymNameOffsets.clear();
  NextExtID = ExtGOTStartID;  // Reset external ID counter
  MaxLocalID = 0;             // Reset max local ID
  PointerGroups.clear();

  // Step 1: Build GOT map for all global variables
  buildGOTMap(M, DL);

  // Step 2: Iterate over all global variables and collect pointers
  for (GlobalVariable &GV : M.globals()) {
    // Skip declarations
    if (GV.isDeclaration())
      continue;

    // Skip our own generated tables
    if (GV.getName().starts_with("__sig_"))
      continue;

    // Skip thread-local and externally initialized globals
    if (GV.isThreadLocal() || GV.isExternallyInitialized())
      continue;

    // Skip if in address space 100 (sigmode_raw)
    if (GV.getAddressSpace() == 100) {
      LLVM_DEBUG(dbgs() << "Skipping AS100 global: " << GV.getName() << "\n");
      continue;
    }

    // Skip if type doesn't contain pointers
    if (!(GV.getValueType()->containsPointer())) {
      LLVM_DEBUG(dbgs() << "Skipping non-pointer global: " << GV.getName() << "\n");
      continue;
    }

    // Get the DataID for pointers in this global
    auto It = GOTMap.find(&GV);
    if (It == GOTMap.end())
      continue;
    uint32_t DataID = It->second.PointToID;

    LLVM_DEBUG(dbgs() << "Analyzing global: " << GV.getName() 
                      << " DataID=" << DataID << "\n");

    // Collect all pointers
    SmallVector<PointerInfo, 32> Ptrs;
    
    if (GV.hasInitializer()) {
      // Initialized global - collect pointers from initializer
      collectAllPointers(GV.getInitializer(), DL, 0, Ptrs);
    } else {
      // Uninitialized global - collect pointer offsets from type
      // All pointers treated as PointToID = 0
      collectPointerOffsetsFromType(GV.getValueType(), DL, 0, Ptrs);
      LLVM_DEBUG(dbgs() << "  Uninitialized global, " << Ptrs.size() 
                        << " pointers with PointToID=0\n");
    }

    // Analyze and group pointers
    if (!Ptrs.empty())
      analyzeAndGroupPointers(&GV, DL, DataID, Ptrs);
  }

  // Step 3: Generate the tables
  generatePointerTable(M, DL);

  return !PointerGroups.empty() || !GOTMap.empty();
}

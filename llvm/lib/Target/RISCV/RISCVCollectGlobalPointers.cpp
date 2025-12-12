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
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-collect-global-pointers"
#define RISCV_COLLECT_GLOBAL_POINTERS_NAME "RISC-V Collect Global Pointers"

STATISTIC(NumPointersCollected, "Number of global pointers collected");
STATISTIC(NumGOTEntries, "Number of GOT entries created");
STATISTIC(NumCompressedEntries, "Number of compressed pointer groups");

// Special IDs
static constexpr uint32_t ExternalPointToID = 0xFFFFFF;
static constexpr uint32_t NullPointToID = 0;

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

// Structure to represent a GOT entry
struct GOTEntry {
  GlobalVariable *GV;     // The global variable
  uint32_t PointToID;     // The assigned PointToID (1, 2, 3, ...)
  uint64_t Size;          // Size of the global
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

private:
  // Map from GlobalVariable to its GOT entry
  DenseMap<const GlobalVariable *, GOTEntry> GOTMap;
  
  // All pointer groups found
  SmallVector<PointerGroup, 64> PointerGroups;

  // Build the GOT map for all global variables
  void buildGOTMap(Module &M, const DataLayout &DL);

  // Check if a type contains any pointer
  bool containsPointer(Type *Ty);

  // Check if a type is "pointer-only" (only contains pointers, no other data)
  bool isPointerOnlyType(Type *Ty, const DataLayout &DL, unsigned PtrSize);

  // Get the PointToID for a constant value
  uint32_t getPointToID(Constant *C);

  // Find the PointToID when a pointer points to a global value
  uint32_t findGOTPointToID(const GlobalValue *GV);

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
};

} // end anonymous namespace

char RISCVCollectGlobalPointers::ID = 0;

INITIALIZE_PASS(RISCVCollectGlobalPointers, DEBUG_TYPE,
                RISCV_COLLECT_GLOBAL_POINTERS_NAME, false, false)

ModulePass *llvm::createRISCVCollectGlobalPointersPass() {
  return new RISCVCollectGlobalPointers();
}

bool RISCVCollectGlobalPointers::containsPointer(Type *Ty) {
  if (Ty->isPointerTy())
    return true;

  if (ArrayType *ATy = dyn_cast<ArrayType>(Ty))
    return containsPointer(ATy->getElementType());

  if (StructType *STy = dyn_cast<StructType>(Ty)) {
    for (Type *ElemTy : STy->elements())
      if (containsPointer(ElemTy))
        return true;
    return false;
  }

  if (auto *VTy = dyn_cast<FixedVectorType>(Ty))
    return containsPointer(VTy->getElementType());

  return false;
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
    // Skip declarations
    if (GV.isDeclaration())
      continue;

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

    if (!containsPointer(GV.getValueType())) {
      // Only Data Region Doesn't need Unique ID
      continue;
    }

    // Create GOT entry for this global
    // GOT entries: DataID = 0xFFFFFF (implicit), PointToID = 1, 2, 3, ...
    GOTEntry Entry;
    Entry.GV = &GV;
    Entry.PointToID = NextPointToID++;
    Entry.Size = DL.getTypeAllocSize(GV.getValueType());

    GOTMap[&GV] = Entry;
    ++NumGOTEntries;

    LLVM_DEBUG(dbgs() << "GOT Entry: " << GV.getName() 
                      << " PointToID=" << Entry.PointToID
                      << " Size=" << Entry.Size << "\n");
  }
}

uint32_t RISCVCollectGlobalPointers::findGOTPointToID(const GlobalValue *GV) {
  if (const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV)) {
    auto It = GOTMap.find(GVar);
    if (It != GOTMap.end())
      return It->second.PointToID;
  }
  
  // Dummy and external globals get ExternalPointToID
  // Function globals will not enter this function
  return ExternalPointToID;
}

uint32_t RISCVCollectGlobalPointers::getPointToID(Constant *C) {
  // Null pointer
  if (isa<ConstantPointerNull>(C)) {
    unsigned AS = C->getType()->getPointerAddressSpace();
    if (AS == 100) {
      return NullPointToID;
    } else {
      return ExternalPointToID;
    }
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
    if (containsPointer(Ty)) {
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
    if (!containsPointer(ElemTy))
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
      if (containsPointer(ElemTy)) {
        uint64_t ElemOffset = SL->getElementOffset(I);
        collectPointerOffsetsFromType(ElemTy, DL, BaseOffset + ElemOffset, Result);
      }
    }
    return;
  }

  // Handle vector types
  if (auto *VTy = dyn_cast<FixedVectorType>(Ty)) {
    Type *ElemTy = VTy->getElementType();
    if (!containsPointer(ElemTy))
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
  Type *PtrSizedIntTy = Type::getIntNTy(Ctx, PtrSize * 8);

  // === Generate GOT Table ===
  // Format: { addr (ptr-sized), PointToID (i32) }
  // DataID is always 0xFFFFFF, not stored
  // Sort by PointToID to ensure consistent ordering
  
  SmallVector<const GOTEntry *, 64> SortedGOTEntries;
  for (auto &KV : GOTMap) {
    SortedGOTEntries.push_back(&KV.second);
  }
  llvm::sort(SortedGOTEntries, [](const GOTEntry *A, const GOTEntry *B) {
    return A->PointToID < B->PointToID;
  });
  
  SmallVector<Constant *, 64> GOTTableEntries;
  StructType *GOTEntryTy = StructType::get(Ctx, {PtrSizedIntTy, I32Ty});

  for (const GOTEntry *Entry : SortedGOTEntries) {
    Constant *GVPtr = ConstantExpr::getPtrToInt(Entry->GV, PtrSizedIntTy);
    Constant *PointToIDConst = ConstantInt::get(I32Ty, Entry->PointToID);

    Constant *GOTEntryConst = ConstantStruct::get(GOTEntryTy, 
                                                   {GVPtr, PointToIDConst});
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

  for (const PointerGroup &Group : PointerGroups) {
    Constant *GVPtr = Group.ContainingGV;
    
    switch (Group.Type) {
    case PtrGroupType::Single: {
      // Header: { addr(64), DataID(32), PointToID(32) }
      const PointerInfo &P = Group.Ptrs[0];
      Constant *OffsetConst = ConstantInt::get(PtrSizedIntTy, P.Offset);
      Constant *GEP = ConstantExpr::getInBoundsGetElementPtr(I8Ty, GVPtr, OffsetConst);
      Constant *Addr = ConstantExpr::getPtrToInt(GEP, PtrSizedIntTy);
      
      Constant *Header = ConstantStruct::get(SingleHeaderTy,
          {Addr, ConstantInt::get(I32Ty, Group.DataID),
           ConstantInt::get(I32Ty, P.PointToID)});
      SingleHeaders.push_back(Header);
      break;
    }
    
    case PtrGroupType::ContiguousSame: {
      // Header: { addr(64), count(64), DataID(32), PointToID(32) }
      const PointerInfo &P = Group.Ptrs[0];
      Constant *OffsetConst = ConstantInt::get(PtrSizedIntTy, P.Offset);
      Constant *GEP = ConstantExpr::getInBoundsGetElementPtr(I8Ty, GVPtr, OffsetConst);
      Constant *Addr = ConstantExpr::getPtrToInt(GEP, PtrSizedIntTy);
      
      Constant *Header = ConstantStruct::get(ContiguousSameHeaderTy,
          {Addr, ConstantInt::get(PtrSizedIntTy, Group.Ptrs.size()),
           ConstantInt::get(I32Ty, Group.DataID),
           ConstantInt::get(I32Ty, Group.CommonPointToID)});
      ContiguousSameHeaders.push_back(Header);
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
      for (const PointerInfo &PI : Group.Ptrs) {
        ContiguousDiffIDs.push_back(ConstantInt::get(I32Ty, PI.PointToID));
      }
      break;
    }
    
    case PtrGroupType::SparseSame: {
      // Header: { addr(64), count(64), DataID(32), PointToID(32) }
      Constant *Addr = ConstantExpr::getPtrToInt(GVPtr, PtrSizedIntTy);
      
      Constant *Header = ConstantStruct::get(SparseSameHeaderTy,
          {Addr, ConstantInt::get(PtrSizedIntTy, Group.Offsets.size()),
           ConstantInt::get(I32Ty, Group.DataID),
           ConstantInt::get(I32Ty, Group.CommonPointToID)});
      SparseSameHeaders.push_back(Header);
      
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
      for (const PointerInfo &PI : Group.Ptrs) {
        SparseDiffIDs.push_back(ConstantInt::get(I32Ty, PI.PointToID));
      }
      
      // Data: offset[count](64 each) - separate offset array for SparseDiff
      for (const PointerInfo &PI : Group.Ptrs) {
        SparseDiffOffsets.push_back(ConstantInt::get(PtrSizedIntTy, PI.Offset));
      }
      break;
    }
    }
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
    if (!containsPointer(GV.getValueType())) {
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

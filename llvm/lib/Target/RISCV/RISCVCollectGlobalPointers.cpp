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
// - DataID: The ID of the data region this entry belongs to
// - PointToID: The ID of the data region this entry points to
//
// GOT entries: DataID = 0xFFFFFF (implicit), PointToID = 1, 2, 3, ...
// Output format: { addr, PointToID } (DataID is always 0xFFFFFF, not stored)
//
// Pointer entries are stored in compressed formats to reduce table size:
//
// Type1 (ContiguousDifferent): Contiguous pointers with different PointToIDs
//   Format: { type=1, addr, DataID, count, [PointToID...] }
//
// Type2 (ContiguousSame): Contiguous pointers with same PointToID  
//   Format: { type=2, addr, DataID, count, PointToID }
//
// Type3 (SparseDifferent): Non-contiguous pointers with different PointToIDs
//   Format: { type=3, addr, DataID, count, [(offset, PointToID)...] }
//
// Type4 (SparseSame): Non-contiguous pointers with same PointToID
//   Format: { type=4, addr, DataID, count, PointToID, [offset...] }
//
// Type5 (Single): Single pointer
//   Format: { type=5, addr, DataID, PointToID }
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
  
  // Functions and external globals get ExternalPointToID
  return ExternalPointToID;
}

uint32_t RISCVCollectGlobalPointers::getPointToID(Constant *C) {
  // Null pointer
  if (isa<ConstantPointerNull>(C))
    return NullPointToID;

  // Undef value
  if (isa<UndefValue>(C))
    return NullPointToID;

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
        M, GOTTableTy, /*isConstant=*/true, GlobalValue::ExternalLinkage,
        GOTTableInit, "__sig_got_table");
    GOTTableGV->setSection(".sig_got_table");
    GOTTableGV->setAlignment(Align(PtrSize));

    // Count placed in .sig_count section (all counts in one section)
    GlobalVariable *GOTSizeGV = new GlobalVariable(
        M, I32Ty, /*isConstant=*/true, GlobalValue::ExternalLinkage,
        ConstantInt::get(I32Ty, GOTTableEntries.size()),
        "__sig_got_count");
    GOTSizeGV->setSection(".sig_count");
  }

  // === Generate Compressed Pointer Tables ===
  // Each type generates a single byte stream with header+data combined
  
  // Type5 (Single): { addr, DataID, PointToID }
  SmallVector<Constant *, 32> SingleEntries;
  StructType *SingleEntryTy = StructType::get(Ctx, {PtrSizedIntTy, I32Ty, I32Ty});
  
  // Type2 (ContiguousSame): { addr, DataID, count, PointToID }
  SmallVector<Constant *, 32> ContiguousSameEntries;
  StructType *ContiguousSameEntryTy = StructType::get(Ctx, 
      {PtrSizedIntTy, I32Ty, I32Ty, I32Ty});
  
  // Type1 (ContiguousDifferent): Combined as byte stream
  // Each entry: { addr, DataID, count, PointToID[count] }
  SmallVector<Constant *, 256> ContiguousDiffStream;
  uint32_t ContiguousDiffCount = 0;
  
  // Type4 (SparseSame): Combined as byte stream
  // Each entry: { addr, DataID, count, PointToID, offset[count] }
  SmallVector<Constant *, 256> SparseSameStream;
  uint32_t SparseSameCount = 0;
  
  // Type3 (SparseDifferent): Combined as byte stream
  // Each entry: { addr, DataID, count, (offset, PointToID)[count] }
  SmallVector<Constant *, 256> SparseDiffStream;
  uint32_t SparseDiffCount = 0;

  for (const PointerGroup &Group : PointerGroups) {
    Constant *GVPtr = Group.ContainingGV;
    
    switch (Group.Type) {
    case PtrGroupType::Single: {
      const PointerInfo &P = Group.Ptrs[0];
      Constant *OffsetConst = ConstantInt::get(PtrSizedIntTy, P.Offset);
      Constant *GEP = ConstantExpr::getInBoundsGetElementPtr(I8Ty, GVPtr, OffsetConst);
      Constant *Addr = ConstantExpr::getPtrToInt(GEP, PtrSizedIntTy);
      
      Constant *Entry = ConstantStruct::get(SingleEntryTy,
          {Addr, ConstantInt::get(I32Ty, Group.DataID),
           ConstantInt::get(I32Ty, P.PointToID)});
      SingleEntries.push_back(Entry);
      break;
    }
    
    case PtrGroupType::ContiguousSame: {
      const PointerInfo &P = Group.Ptrs[0];
      Constant *OffsetConst = ConstantInt::get(PtrSizedIntTy, P.Offset);
      Constant *GEP = ConstantExpr::getInBoundsGetElementPtr(I8Ty, GVPtr, OffsetConst);
      Constant *Addr = ConstantExpr::getPtrToInt(GEP, PtrSizedIntTy);
      
      Constant *Entry = ConstantStruct::get(ContiguousSameEntryTy,
          {Addr, ConstantInt::get(I32Ty, Group.DataID),
           ConstantInt::get(I32Ty, Group.Ptrs.size()),
           ConstantInt::get(I32Ty, Group.CommonPointToID)});
      ContiguousSameEntries.push_back(Entry);
      break;
    }
    
    case PtrGroupType::ContiguousDifferent: {
      // Format: addr (ptr-sized), DataID (i32), count (i32), PointToID[count] (i32 each)
      // Store as: [addr_low, addr_high (if 64-bit), DataID, count, PointToID...]
      const PointerInfo &P = Group.Ptrs[0];
      Constant *OffsetConst = ConstantInt::get(PtrSizedIntTy, P.Offset);
      Constant *GEP = ConstantExpr::getInBoundsGetElementPtr(I8Ty, GVPtr, OffsetConst);
      Constant *Addr = ConstantExpr::getPtrToInt(GEP, PtrSizedIntTy);
      
      // Store address directly as ptr-sized value
      ContiguousDiffStream.push_back(Addr);
      ContiguousDiffStream.push_back(ConstantInt::get(PtrSizedIntTy, Group.DataID));
      ContiguousDiffStream.push_back(ConstantInt::get(PtrSizedIntTy, Group.Ptrs.size()));
      
      for (const PointerInfo &PI : Group.Ptrs) {
        ContiguousDiffStream.push_back(ConstantInt::get(PtrSizedIntTy, PI.PointToID));
      }
      ++ContiguousDiffCount;
      break;
    }
    
    case PtrGroupType::SparseSame: {
      // Format: addr (ptr-sized), DataID, count, PointToID, offset[count]
      Constant *Addr = ConstantExpr::getPtrToInt(GVPtr, PtrSizedIntTy);
      
      SparseSameStream.push_back(Addr);
      SparseSameStream.push_back(ConstantInt::get(PtrSizedIntTy, Group.DataID));
      SparseSameStream.push_back(ConstantInt::get(PtrSizedIntTy, Group.Offsets.size()));
      SparseSameStream.push_back(ConstantInt::get(PtrSizedIntTy, Group.CommonPointToID));
      
      for (uint64_t Off : Group.Offsets) {
        SparseSameStream.push_back(ConstantInt::get(PtrSizedIntTy, Off));
      }
      ++SparseSameCount;
      break;
    }
    
    case PtrGroupType::SparseDifferent: {
      // Format: addr (ptr-sized), DataID, count, (offset, PointToID)[count]
      Constant *Addr = ConstantExpr::getPtrToInt(GVPtr, PtrSizedIntTy);
      
      SparseDiffStream.push_back(Addr);
      SparseDiffStream.push_back(ConstantInt::get(PtrSizedIntTy, Group.DataID));
      SparseDiffStream.push_back(ConstantInt::get(PtrSizedIntTy, Group.Ptrs.size()));
      
      for (const PointerInfo &PI : Group.Ptrs) {
        SparseDiffStream.push_back(ConstantInt::get(PtrSizedIntTy, PI.Offset));
        SparseDiffStream.push_back(ConstantInt::get(PtrSizedIntTy, PI.PointToID));
      }
      ++SparseDiffCount;
      break;
    }
    }
  }

  // Generate global variables for each table type
  // Each type has its own section for linker merging
  // All counts go to .sig_count section
  
  // Type5 Single entries
  if (!SingleEntries.empty()) {
    ArrayType *Ty = ArrayType::get(SingleEntryTy, SingleEntries.size());
    Constant *Init = ConstantArray::get(Ty, SingleEntries);
    GlobalVariable *GV = new GlobalVariable(
        M, Ty, true, GlobalValue::ExternalLinkage, Init, "__sig_ptr_single");
    GV->setSection(".sig_ptr_single");
    GV->setAlignment(Align(PtrSize));
    
    GlobalVariable *CountGV = new GlobalVariable(M, I32Ty, true, GlobalValue::ExternalLinkage,
        ConstantInt::get(I32Ty, SingleEntries.size()), "__sig_ptr_single_count");
    CountGV->setSection(".sig_count");
  }
  
  // Type2 ContiguousSame entries
  if (!ContiguousSameEntries.empty()) {
    ArrayType *Ty = ArrayType::get(ContiguousSameEntryTy, ContiguousSameEntries.size());
    Constant *Init = ConstantArray::get(Ty, ContiguousSameEntries);
    GlobalVariable *GV = new GlobalVariable(
        M, Ty, true, GlobalValue::ExternalLinkage, Init, "__sig_ptr_contig_same");
    GV->setSection(".sig_ptr_contig_same");
    GV->setAlignment(Align(PtrSize));
    
    GlobalVariable *CountGV = new GlobalVariable(M, I32Ty, true, GlobalValue::ExternalLinkage,
        ConstantInt::get(I32Ty, ContiguousSameEntries.size()), "__sig_ptr_contig_same_count");
    CountGV->setSection(".sig_count");
  }
  
  // Type1 ContiguousDifferent - combined stream (ptr-sized elements)
  if (!ContiguousDiffStream.empty()) {
    ArrayType *Ty = ArrayType::get(PtrSizedIntTy, ContiguousDiffStream.size());
    Constant *Init = ConstantArray::get(Ty, ContiguousDiffStream);
    GlobalVariable *GV = new GlobalVariable(
        M, Ty, true, GlobalValue::ExternalLinkage, Init, "__sig_ptr_contig_diff");
    GV->setSection(".sig_ptr_contig_diff");
    GV->setAlignment(Align(PtrSize));
    
    GlobalVariable *CountGV = new GlobalVariable(M, I32Ty, true, GlobalValue::ExternalLinkage,
        ConstantInt::get(I32Ty, ContiguousDiffCount), "__sig_ptr_contig_diff_count");
    CountGV->setSection(".sig_count");
  }
  
  // Type4 SparseSame - combined stream (ptr-sized elements)
  if (!SparseSameStream.empty()) {
    ArrayType *Ty = ArrayType::get(PtrSizedIntTy, SparseSameStream.size());
    Constant *Init = ConstantArray::get(Ty, SparseSameStream);
    GlobalVariable *GV = new GlobalVariable(
        M, Ty, true, GlobalValue::ExternalLinkage, Init, "__sig_ptr_sparse_same");
    GV->setSection(".sig_ptr_sparse_same");
    GV->setAlignment(Align(PtrSize));
    
    GlobalVariable *CountGV = new GlobalVariable(M, I32Ty, true, GlobalValue::ExternalLinkage,
        ConstantInt::get(I32Ty, SparseSameCount), "__sig_ptr_sparse_same_count");
    CountGV->setSection(".sig_count");
  }
  
  // Type3 SparseDifferent - combined stream (ptr-sized elements)
  if (!SparseDiffStream.empty()) {
    ArrayType *Ty = ArrayType::get(PtrSizedIntTy, SparseDiffStream.size());
    Constant *Init = ConstantArray::get(Ty, SparseDiffStream);
    GlobalVariable *GV = new GlobalVariable(
        M, Ty, true, GlobalValue::ExternalLinkage, Init, "__sig_ptr_sparse_diff");
    GV->setSection(".sig_ptr_sparse_diff");
    GV->setAlignment(Align(PtrSize));
    
    GlobalVariable *CountGV = new GlobalVariable(M, I32Ty, true, GlobalValue::ExternalLinkage,
        ConstantInt::get(I32Ty, SparseDiffCount), "__sig_ptr_sparse_diff_count");
    CountGV->setSection(".sig_count");
  }

  LLVM_DEBUG(dbgs() << "Generated GOT table with " << GOTTableEntries.size()
                    << " entries\n");
  LLVM_DEBUG(dbgs() << "Generated pointer tables:\n"
                    << "  Single: " << SingleEntries.size() << "\n"
                    << "  ContiguousSame: " << ContiguousSameEntries.size() << "\n"
                    << "  ContiguousDiff: " << ContiguousDiffCount << "\n"
                    << "  SparseSame: " << SparseSameCount << "\n"
                    << "  SparseDiff: " << SparseDiffCount << "\n");
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

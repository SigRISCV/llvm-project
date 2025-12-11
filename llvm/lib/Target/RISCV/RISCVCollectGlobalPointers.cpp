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
// GOT entries: DataID = 0xFFFFFF, PointToID = 1, 2, 3, ... (unique per global)
// Pointer entries: DataID = the GOT PointToID of containing global
//                  PointToID rules:
//                    - Pointing to AS100/NULL/undef → 0
//                    - Function pointer (initialized) → 0xFFFFFF  
//                    - Pointing to other global → that global's GOT PointToID
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
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-collect-global-pointers"
#define RISCV_COLLECT_GLOBAL_POINTERS_NAME "RISC-V Collect Global Pointers"

STATISTIC(NumPointersCollected, "Number of global pointers collected");
STATISTIC(NumGOTEntries, "Number of GOT entries created");

// Special IDs
static constexpr uint32_t PC_POINT_TO_ID = 0xFFFFFF;
static constexpr uint32_t NULL_POINT_TO_ID = 0;

namespace {

// Structure to represent a pointer entry in the table
struct PointerEntry {
  GlobalVariable *ContainingGV;  // The global variable containing this pointer
  uint64_t Offset;               // Byte offset within the global variable
  uint32_t DataID;               // ID of the data region this pointer belongs to
  uint32_t PointToID;            // ID of the data region this pointer points to
};

// Structure to represent a GOT entry
struct GOTEntry {
  GlobalVariable *GV;            // The global variable
  uint32_t DataID;               // Always 0xFFFFFF for GOT entries
  uint32_t PointToID;            // The assigned PointToID (1, 2, 3, ...)
  uint64_t StartAddr;            // Start address (for range checking)
  uint64_t Size;                 // Size of the global
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
  
  // All pointer entries found
  SmallVector<PointerEntry, 64> PointerEntries;

  // Build the GOT map for all global variables
  void buildGOTMap(Module &M, const DataLayout &DL);

  // Check if a type contains any pointer
  bool containsPointer(Type *Ty);

  // Get the PointToID for a constant value
  uint32_t getPointToID(Constant *C);

  // Recursively collect pointers in a constant
  void collectPointersInConstant(Constant *C, const DataLayout &DL,
                                 uint64_t BaseOffset, uint32_t DataID,
                                 GlobalVariable *ContainingGV);

  // Find the PointToID when a pointer points to a global value
  uint32_t findGOTPointToID(const GlobalValue *GV);

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
    // GOT entries: DataID = 0xFFFFFF, PointToID = 1, 2, 3, ...
    GOTEntry Entry;
    Entry.GV = &GV;
    Entry.DataID = PC_POINT_TO_ID;  // GOT always has DataID = 0xFFFFFF
    Entry.PointToID = NextPointToID++;     // Sequential PointToID
    Entry.StartAddr = 0;  // Will be resolved at link time
    Entry.Size = DL.getTypeAllocSize(GV.getValueType());

    GOTMap[&GV] = Entry;
    ++NumGOTEntries;

    LLVM_DEBUG(dbgs() << "GOT Entry: " << GV.getName() 
                      << " DataID=" << Entry.DataID 
                      << " PointToID=" << Entry.PointToID
                      << " Size=" << Entry.Size << "\n");
  }
}

// Find the PointToID when a pointer points to a global value
// This returns the GOT PointToID of the target global
uint32_t RISCVCollectGlobalPointers::findGOTPointToID(const GlobalValue *GV) {
  // Check if it's a global variable in our GOT map
  if (const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV)) {
    auto It = GOTMap.find(GVar);
    if (It != GOTMap.end())
      return It->second.PointToID;  // Return the GOT's PointToID
  }
  
  // Functions and external globals get PC_POINT_TO_ID
  return PC_POINT_TO_ID;
}

uint32_t RISCVCollectGlobalPointers::getPointToID(Constant *C) {
  // Null pointer
  if (isa<ConstantPointerNull>(C))
    return NULL_POINT_TO_ID;

  // Undef value
  if (isa<UndefValue>(C))
    return NULL_POINT_TO_ID;

  // Check if pointing to address space 100
  if (C->getType()->isPointerTy()) {
    unsigned AS = C->getType()->getPointerAddressSpace();
    if (AS == 100)
      return NULL_POINT_TO_ID;
  }

  // Global value (variable or function)
  if (GlobalValue *GV = dyn_cast<GlobalValue>(C)) {
    // Function pointer
    if (isa<Function>(GV))
      return PC_POINT_TO_ID;
    
    // Global variable - find its GOT PointToID
    return findGOTPointToID(GV);
  }

  // Constant expression (e.g., getelementptr, bitcast, addrspacecast)
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    // Check if result is in address space 100
    if (CE->getType()->isPointerTy() && 
        CE->getType()->getPointerAddressSpace() == 100)
      return NULL_POINT_TO_ID;

    // For GEP, bitcast, addrspacecast - trace back to the base
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
    return PC_POINT_TO_ID;

  // Default - treat as external
  return PC_POINT_TO_ID;
}

void RISCVCollectGlobalPointers::collectPointersInConstant(
    Constant *C, const DataLayout &DL, uint64_t BaseOffset, uint32_t DataID,
    GlobalVariable *ContainingGV) {

  Type *Ty = C->getType();

  // If this is a pointer type, record it
  if (Ty->isPointerTy()) {
    PointerEntry Entry;
    Entry.ContainingGV = ContainingGV;
    Entry.Offset = BaseOffset;
    Entry.DataID = DataID;
    Entry.PointToID = getPointToID(C);

    PointerEntries.push_back(Entry);
    ++NumPointersCollected;

    LLVM_DEBUG(dbgs() << "  Pointer at offset " << BaseOffset
                      << " DataID=" << DataID
                      << " PointToID=" << Entry.PointToID << "\n");
    return;
  }

  // Handle aggregate types
  if (ConstantArray *CA = dyn_cast<ConstantArray>(C)) {
    Type *ElemTy = CA->getType()->getElementType();
    uint64_t ElemSize = DL.getTypeAllocSize(ElemTy);
    for (unsigned i = 0; i < CA->getNumOperands(); ++i) {
      collectPointersInConstant(CA->getOperand(i), DL,
                                BaseOffset + i * ElemSize, DataID, ContainingGV);
    }
    return;
  }

  if (ConstantStruct *CS = dyn_cast<ConstantStruct>(C)) {
    StructType *STy = CS->getType();
    const StructLayout *SL = DL.getStructLayout(STy);
    for (unsigned i = 0; i < CS->getNumOperands(); ++i) {
      uint64_t ElemOffset = SL->getElementOffset(i);
      collectPointersInConstant(CS->getOperand(i), DL,
                                BaseOffset + ElemOffset, DataID, ContainingGV);
    }
    return;
  }

  if (ConstantVector *CV = dyn_cast<ConstantVector>(C)) {
    Type *ElemTy = CV->getType()->getElementType();
    uint64_t ElemSize = DL.getTypeAllocSize(ElemTy);
    for (unsigned i = 0; i < CV->getNumOperands(); ++i) {
      collectPointersInConstant(CV->getOperand(i), DL,
                                BaseOffset + i * ElemSize, DataID, ContainingGV);
    }
    return;
  }

  if (ConstantAggregateZero *CAZ = dyn_cast<ConstantAggregateZero>(C)) {
    // Zero-initialized - if type contains pointers, they are all null
    if (containsPointer(Ty)) {
      // Recursively handle as if it were explicit zeros
      if (ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
        Type *ElemTy = ATy->getElementType();
        uint64_t ElemSize = DL.getTypeAllocSize(ElemTy);
        Constant *ZeroElem = Constant::getNullValue(ElemTy);
        for (uint64_t i = 0; i < ATy->getNumElements(); ++i) {
          collectPointersInConstant(ZeroElem, DL,
                                    BaseOffset + i * ElemSize, DataID, ContainingGV);
        }
      } else if (StructType *STy = dyn_cast<StructType>(Ty)) {
        const StructLayout *SL = DL.getStructLayout(STy);
        for (unsigned i = 0; i < STy->getNumElements(); ++i) {
          uint64_t ElemOffset = SL->getElementOffset(i);
          Constant *ZeroElem = Constant::getNullValue(STy->getElementType(i));
          collectPointersInConstant(ZeroElem, DL,
                                    BaseOffset + ElemOffset, DataID, ContainingGV);
        }
      }
    }
    return;
  }

  // ConstantDataSequential - contains only primitive data, no pointers
  // ConstantInt, ConstantFP, etc. - not pointers
}

void RISCVCollectGlobalPointers::generatePointerTable(Module &M, 
                                                       const DataLayout &DL) {
  LLVMContext &Ctx = M.getContext();
  unsigned PtrSize = DL.getPointerSize();
  
  // Types for table entries
  Type *I32Ty = Type::getInt32Ty(Ctx);
  Type *PtrSizedIntTy = Type::getIntNTy(Ctx, PtrSize * 8);
  Type *I8Ty = Type::getInt8Ty(Ctx);

  // === Generate GOT Table ===
  // Format: { GlobalPtr, DataID (0xFFFFFF), PointToID (1,2,3...) }
  SmallVector<Constant *, 64> GOTTableEntries;
  
  // Create struct type for GOT entry: { ptr, i32, i32 }
  StructType *GOTEntryTy = StructType::get(Ctx, {PtrSizedIntTy, I32Ty, I32Ty});

  for (auto &KV : GOTMap) {
    const GOTEntry &Entry = KV.second;
    
    // Pointer to the global (as integer)
    Constant *GVPtr = ConstantExpr::getPtrToInt(Entry.GV, PtrSizedIntTy);
    Constant *DataIDConst = ConstantInt::get(I32Ty, Entry.DataID);         // 0xFFFFFF
    Constant *PointToIDConst = ConstantInt::get(I32Ty, Entry.PointToID);   // 1, 2, 3...

    Constant *GOTEntryConst = ConstantStruct::get(GOTEntryTy, 
                                                   {GVPtr, DataIDConst, PointToIDConst});
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

    // GOT table size
    GlobalVariable *GOTSizeGV = new GlobalVariable(
        M, I32Ty, /*isConstant=*/true, GlobalValue::ExternalLinkage,
        ConstantInt::get(I32Ty, GOTTableEntries.size()),
        "__sig_got_count");
    GOTSizeGV->setSection(".sig_got_table");
  }

  // === Generate Pointer Table ===
  // Format: { Address, DataID, PointToID }
  SmallVector<Constant *, 64> PtrTableEntries;

  // Create struct type for pointer entry: { ptr, i32, i32 }
  StructType *PtrEntryTy = StructType::get(Ctx, {PtrSizedIntTy, I32Ty, I32Ty});

  for (const PointerEntry &Entry : PointerEntries) {
    // Calculate address: &ContainingGV + Offset
    Constant *GVPtr = Entry.ContainingGV;
    Constant *OffsetConst = ConstantInt::get(PtrSizedIntTy, Entry.Offset);
    Constant *GEP = ConstantExpr::getInBoundsGetElementPtr(
        I8Ty, GVPtr, OffsetConst);
    Constant *Addr = ConstantExpr::getPtrToInt(GEP, PtrSizedIntTy);

    Constant *DataIDConst = ConstantInt::get(I32Ty, Entry.DataID);
    Constant *PointToIDConst = ConstantInt::get(I32Ty, Entry.PointToID);

    Constant *PtrEntryConst = ConstantStruct::get(PtrEntryTy,
                                                   {Addr, DataIDConst, PointToIDConst});
    PtrTableEntries.push_back(PtrEntryConst);
  }

  if (!PtrTableEntries.empty()) {
    ArrayType *PtrTableTy = ArrayType::get(PtrEntryTy, PtrTableEntries.size());
    Constant *PtrTableInit = ConstantArray::get(PtrTableTy, PtrTableEntries);

    GlobalVariable *PtrTableGV = new GlobalVariable(
        M, PtrTableTy, /*isConstant=*/true, GlobalValue::ExternalLinkage,
        PtrTableInit, "__sig_ptr_table");
    PtrTableGV->setSection(".sig_ptr_table");
    PtrTableGV->setAlignment(Align(PtrSize));

    // Pointer table size
    GlobalVariable *PtrSizeGV = new GlobalVariable(
        M, I32Ty, /*isConstant=*/true, GlobalValue::ExternalLinkage,
        ConstantInt::get(I32Ty, PtrTableEntries.size()),
        "__sig_ptr_count");
    PtrSizeGV->setSection(".sig_ptr_table");
  }

  LLVM_DEBUG(dbgs() << "Generated GOT table with " << GOTTableEntries.size()
                    << " entries\n");
  LLVM_DEBUG(dbgs() << "Generated pointer table with " << PtrTableEntries.size()
                    << " entries\n");
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
  PointerEntries.clear();

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

    // Skip if no initializer
    if (!GV.hasInitializer())
      continue;

    // Skip if type doesn't contain pointers
    if (!containsPointer(GV.getValueType())) {
      LLVM_DEBUG(dbgs() << "Skipping non-pointer global: " << GV.getName() << "\n");
      continue;
    }

    // Get the DataID for pointers in this global
    // Pointer's DataID = the GOT PointToID of the containing global
    auto It = GOTMap.find(&GV);
    if (It == GOTMap.end())
      continue;
    uint32_t DataID = It->second.PointToID;  // Use GOT's PointToID as pointer's DataID

    LLVM_DEBUG(dbgs() << "Analyzing global: " << GV.getName() 
                      << " DataID=" << DataID << "\n");

    // Collect pointers in the initializer
    collectPointersInConstant(GV.getInitializer(), DL, 0, DataID, &GV);
  }

  // Step 3: Generate the tables
  generatePointerTable(M, DL);

  return !PointerEntries.empty() || !GOTMap.empty();
}


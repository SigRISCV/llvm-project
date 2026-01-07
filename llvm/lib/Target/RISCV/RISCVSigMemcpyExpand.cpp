//===- RISCVSigMemcpyExpand.cpp - Expand SigMode memcpy intrinsic ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass expands the @llvm.riscv.xsig.memcpy and @llvm.riscv.xsig.memset
// intrinsics by generating type-specific helper functions. Instead of inlining
// all code at the call site (which causes code bloat), we generate modular
// functions like:
//
//   @__sigmemcpy_sig_sig_StructName(ptr %dest, ptr %src, i64 %len)
//   @__sigmemcpy_raw_sig_StructName(ptr addrspace(100) %dest, ptr %src, i64 %len)
//   @__sigmemset_sig_StructName(ptr %dest, i64 %len)
//
// Each generated function contains a loop over elements and member-wise copy/zero.
// For nested structs/arrays, it calls the corresponding helper function.
// The inliner can later decide whether to inline these based on size.
//
// Function naming convention:
//   @__sigmemcpy_<dest_as>_<src_as>_<type_mangled_name>
//   @__sigmemset_<dest_as>_<type_mangled_name>
//
// For sigmemset, pointer fields are initialized with xsig_setdummyid(null)
// instead of plain zero, ensuring proper ID encoding.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
#include "RISCVTypeRecovery.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/ConstantFold.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "riscv-sig-memcpy-expand"
#define RISCV_SIG_MEMCPY_EXPAND_NAME "RISC-V SigMode Memcpy/Memset Expansion"

STATISTIC(NumMemcpyExpanded, "Number of sigmemcpy intrinsics expanded");
STATISTIC(NumMemsetExpanded, "Number of sigmemset intrinsics expanded");
STATISTIC(NumHelperFuncsCreated, "Number of helper functions created");
STATISTIC(NumMemcpyConverted, "Number of memcpy calls converted to sigmemcpy");
STATISTIC(NumMemsetConverted, "Number of memset calls converted to sigmemset");

namespace {

// Address space constant for raw pointers
constexpr unsigned RawAS = 100;

class RISCVSigMemcpyExpand : public ModulePass {
public:
  static char ID;
  RISCVSigMemcpyExpand() : ModulePass(ID) {}

  StringRef getPassName() const override {
    return RISCV_SIG_MEMCPY_EXPAND_NAME;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
  }

  bool runOnModule(Module &M) override;

private:
  Module *Mod = nullptr;
  const DataLayout *DL = nullptr;
  LLVMContext *Ctx = nullptr;
  
  // Type recovery for inferring memcpy src/dest types
  std::unique_ptr<TypeRecovery> TR;
  
  // Cache of generated helper functions: name -> Function*
  StringMap<Function *> HelperFuncCache;

  //===--------------------------------------------------------------------===//
  // Type Recovery and Memcpy Conversion
  //===--------------------------------------------------------------------===//
  
  // Run type recovery on the module
  void runTypeRecovery(SmallVector<Function*, 0> FunctionsToProcess);
  
  // Check if a call is a memcpy/memmove that needs conversion
  bool isMemcpyCall(CallBase *CB);
  
  // Check if a call is a memset that needs conversion
  bool isMemsetCall(CallBase *CB);
  
  // Get the best type for memcpy conversion from recovered types
  // Returns nullptr if no suitable type found
  Type *selectMemcpyType(Value* I, Value *Dest, Value *Src, uint64_t Size);
  
  // Get the best type for memset conversion from recovered dest type
  // Returns nullptr if no suitable type found
  Type *selectMemsetType(Value* I, Value *Dest, uint64_t Size);

  Type *selectUsedType(Value* I, const TypeRecovery::TypeSet* DestTypes, uint64_t Size, StringRef FuncName);

  SmallPtrSet<MDNode *, 4> filterComplexTypes(
      const SmallPtrSet<MDNode *, 4> &Types);

  SmallPtrSet<MDNode *, 4> filterBareAndArrayPointeeTypes(
      const SmallPtrSet<MDNode *, 4> &Types);

  SmallPtrSet<MDNode *, 4> filterSubSetTypes(
    const SmallPtrSet<MDNode *, 4> &Types);

  std::string getMangledTypeName(Type *Ty);

  //===--------------------------------------------------------------------===//
  // Sigmemcpy functions
  //===--------------------------------------------------------------------===//
  
  // Expand a single sigmemcpy intrinsic call
  bool expandSigMemcpy(CallInst *II);

  // Get or create a helper function for the given type and address spaces
  Function *getOrCreateMemcpyFunc(Type *ElemTy, unsigned DestAS, unsigned SrcAS);

  // Generate the function body for a memcpy helper
  void generateMemcpyFuncBody(Function *F, Type *ElemTy, 
                              unsigned DestAS, unsigned SrcAS);

  // Copy a single element within a generated function body
  // May call other helper functions for nested types
  void copyElementInFunc(IRBuilder<> &Builder, Value *Dest, Value *Src, 
                         Type *Ty, unsigned DestAS, unsigned SrcAS);

  //===--------------------------------------------------------------------===//
  // Sigmemset functions
  //===--------------------------------------------------------------------===//
  
  // Expand a single sigmemset intrinsic call
  bool expandSigMemset(CallInst *II);

  // Get or create a helper function for memset with given type and address space
  Function *getOrCreateMemsetFunc(Type *ElemTy, unsigned DestAS);

  // Generate the function body for a memset helper
  void generateMemsetFuncBody(Function *F, Type *ElemTy, unsigned DestAS);

  // Zero-initialize a single element within a generated function body
  // For pointer fields, uses the pre-computed DummyNull value (xsig_setdummyid(null))
  // DummyNull is created once at function entry and reused for all pointer stores
  // This function is only called for sig address space (AS 0)
  void zeroElementInFunc(IRBuilder<> &Builder, Value *Dest, 
                         Type *Ty, unsigned DestAS, Value *DummyNull);

  //===--------------------------------------------------------------------===//
  // Utility functions
  //===--------------------------------------------------------------------===//

  // Generate a call to llvm.memcpy for types without pointers
  void emitMemcpyCall(IRBuilder<> &Builder, Value *Dest, Value *Src,
                      Value *Size, unsigned DestAS, unsigned SrcAS);

  // Generate a call to llvm.memset for types without pointers
  void emitMemsetCall(IRBuilder<> &Builder, Value *Dest, Value *Size,
                      unsigned DestAS);

  // Get address space name string
  static StringRef getASName(unsigned AS) {
    return AS == RawAS ? "raw" : "sig";
  }

  //===--------------------------------------------------------------------===//
  // Inline expansion for small copies (< 200 bytes)
  //===--------------------------------------------------------------------===//

  // Threshold for inline expansion vs helper function call
  static constexpr uint64_t InlineThreshold = 256;

  // Get all pointer offsets within a type (recursively handles arrays/structs)
  SmallVector<uint64_t, 8> getPointerOffsets(Type *Ty);

  // Get all pointer offsets for a block of 'Num' elements of type 'Ty'
  SmallVector<uint64_t, 8> getBlockPointerOffsets(Type *Ty, uint64_t Num);

  // Generate inline copy code using load/store based on pointer offsets
  void generateBlockCopyByOffset(IRBuilder<> &Builder, Value *Dest, Value *Src,
                                  uint64_t TotalBytes,
                                  const SmallVector<uint64_t, 8> &PtrOffsets,
                                  unsigned DestAS, unsigned SrcAS);

  // High-level function: generate inline block copy for small types
  void generateBlockCopy(IRBuilder<> &Builder, Value *Dest, Value *Src,
                         Type *ElemTy, uint64_t NumElems,
                         unsigned DestAS, unsigned SrcAS);

  // Generate inline memset code using store based on pointer offsets
  void generateBlockZeroByOffset(IRBuilder<> &Builder, Value *Dest,
                                  uint64_t TotalBytes,
                                  const SmallVector<uint64_t, 8> &PtrOffsets,
                                  unsigned DestAS, Value *DummyNull);

  // High-level function: generate inline block zero for small types
  void generateBlockZero(IRBuilder<> &Builder, Value *Dest,
                         Type *ElemTy, uint64_t NumElems,
                         unsigned DestAS, Value *DummyNull);

  //===--------------------------------------------------------------------===//
  // Constant copy optimization (for raw-to-raw without pointers)
  //===--------------------------------------------------------------------===//

  // Get the underlying constant by stripping addrspacecasts
  Constant *getUnderlyingConstant(Value *V);

  // Extract constant data from a Constant value into byte array
  // Returns true if successfully extracted
  bool extractConstantBytes(Constant *C, Type *Ty, SmallVectorImpl<uint8_t> &Bytes);

  // Generate inline stores of constant values
  void generateConstantCopy(IRBuilder<> &Builder, Value *Dest,
                            Constant *SrcConstant, Type *ElemTy, uint64_t NumElems,
                            unsigned DestAS);
};

} // end anonymous namespace

char RISCVSigMemcpyExpand::ID = 0;

INITIALIZE_PASS(RISCVSigMemcpyExpand, DEBUG_TYPE,
                RISCV_SIG_MEMCPY_EXPAND_NAME, false, false)

//===----------------------------------------------------------------------===//
// Type Recovery and Memcpy Conversion
//===----------------------------------------------------------------------===//

void RISCVSigMemcpyExpand::runTypeRecovery(SmallVector<Function*, 0> FunctionsToProcess) {
  TR = std::make_unique<TypeRecovery>(*Mod);
  TR->run(FunctionsToProcess);

  LLVM_DEBUG(dbgs() << "Type recovery completed\n");
  LLVM_DEBUG(TR->dump(dbgs()));
}

/// Get the real address space of a pointer, looking through addrspacecast.
/// For explicit memcpy/memset calls, pointers might be cast from AS(0) to AS(100).
/// Returns the "real" address space based on:
/// 1. If ptr comes from addrspacecast with ".ascast_alloca" suffix,
///    it's an alloca cast to AS100 - real AS is AS100
/// 2. If ptr comes from other addrspacecast: use the source AS
/// 3. Otherwise, use the direct AS
static Value* getRealPtr(Value *Ptr) {
  // Look for addrspacecast
  if (auto *ASC = dyn_cast<AddrSpaceCastInst>(Ptr)) {
    // Check if this is an alloca-generated cast (marked with .ascast_alloca suffix)
    // Pattern: alloca -> addrspacecast with name ".ascast_alloca"
    // This means the real storage is in AS0 (alloca space)
    if (ASC->hasName() && ASC->getName().contains(".ascast_alloca")) {
      // alloca cast to AS100 - real AS is AS100
      return Ptr;
    }
    
    // For other addrspacecasts, use source AS
    return ASC->getOperand(0);
  }
  
  // No cast, use direct AS
  return Ptr;
}

bool RISCVSigMemcpyExpand::isMemcpyCall(CallBase *CB) {
  Function *Callee = CB->getCalledFunction();
  if (!Callee)
    return false;
  
  StringRef Name = Callee->getName();
  // Match memcpy, memmove, llvm.memcpy.*, llvm.memmove.*
  return Name == "memcpy" || Name == "memmove" ||
         Name.starts_with("llvm.memcpy") || 
         Name.starts_with("llvm.memmove");
}

bool RISCVSigMemcpyExpand::isMemsetCall(CallBase *CB) {
  Function *Callee = CB->getCalledFunction();
  if (!Callee)
    return false;
  
  StringRef Name = Callee->getName();
  // Match memset, llvm.memset.*
  return Name == "memset" || Name.starts_with("llvm.memset");
}

/// Check if a type metadata is a bare "ptr undef" (no pointee info)
/// This type is added by TypeRecovery as default for all pointers
/// and should be ignored when selecting concrete types
static bool isBarePointerTypeMD(MDNode *MD) {
  if (!MD || MD->getNumOperands() < 1)
    return false;
  
  auto *CAM = dyn_cast<ConstantAsMetadata>(MD->getOperand(0));
  if (!CAM)
    return false;
  
  auto *Undef = dyn_cast<UndefValue>(CAM->getValue());
  if (!Undef)
    return false;
  
  // It's a pointer type with no pointee metadata (only 1 operand)
  return Undef->getType()->isPointerTy() && MD->getNumOperands() == 1;
}

SmallPtrSet<MDNode *, 4> RISCVSigMemcpyExpand::filterComplexTypes(
      const SmallPtrSet<MDNode *, 4> &Types) {
  static MDNode* PtrVoidMD = nullptr;
  static MDNode* Int8MD = nullptr;
  if (!PtrVoidMD) {
    PtrVoidMD = TR->lookupTypeByString("void*");
    assert(PtrVoidMD && "ptr_to_void type metadata not found");
  }

  if (!Int8MD) {
    Int8MD = TR->lookupTypeByString("i8");
    assert(Int8MD && "i8 type metadata not found");
  }

  SmallPtrSet<MDNode *, 4> SimpleTypes;

  for (MDNode *MD : Types) {
    Type* LLVMType = TR->getLLVMTypeFromMD(MD);
    if (!LLVMType->containsPointer()) {
      SimpleTypes.insert(Int8MD);
    } else if (LLVMType->isPointerOnlyType()) {
      SimpleTypes.insert(PtrVoidMD);
    } else {
      SimpleTypes.insert(MD);
    }
  }

  return SimpleTypes;

}

/// Filter out bare "ptr undef" types from a type set
/// Returns the filtered set
SmallPtrSet<MDNode *, 4> RISCVSigMemcpyExpand::filterBareAndArrayPointeeTypes(
    const SmallPtrSet<MDNode *, 4> &Types) {
  static MDNode* PtrVoidMD = nullptr;
  if (!PtrVoidMD) {
    PtrVoidMD = TR->lookupTypeByString("void*");
    assert(PtrVoidMD && "ptr_to_void type metadata not found");
  }

  SmallPtrSet<MDNode *, 4> Filtered;
  for (MDNode *MD : Types) {
    if (!TR->isPointerTypeMD(MD))
      continue;
    if (isBarePointerTypeMD(MD))
      continue;
    MDNode *PointeeMD = TR->getPointeeTypeMD(MD);
    while (TR->isArrayTypeMD(PointeeMD)) {
      PointeeMD = TR->getArrayElementTypeMD(PointeeMD);
    }
    Filtered.insert(PointeeMD);
  }
  return Filtered;
}

SmallPtrSet<MDNode *, 4> RISCVSigMemcpyExpand::filterSubSetTypes(
    const SmallPtrSet<MDNode *, 4> &Types) {
  SmallVector<MDNode *, 4> TypeList;
  for (MDNode *MD : Types) {
    TypeList.push_back(MD);
  }

  llvm::sort(TypeList, [&](MDNode *A, MDNode *B) {
    Type *TyA = TR->getLLVMTypeFromMD(A);
    Type *TyB = TR->getLLVMTypeFromMD(B);
    if (!TyA || !TyB)
      return TyA > TyB;
    return DL->getTypeAllocSize(TyA) > DL->getTypeAllocSize(TyB);
  });

  SmallPtrSet<MDNode *, 4> ExtendSet;
  for (MDNode *MD : TypeList) {
    MDNode *Current = MD;
    ExtendSet.insert(Current);
    while (TR->isStructTypeMD(Current) || TR->isArrayTypeMD(Current)) {
      if (TR->isArrayTypeMD(Current)) {
        Current = TR->getArrayElementTypeMD(Current);
      } else {
        Current = TR->getStructFieldTypeMD(Current, 0);
      }
      while (TR->isArrayTypeMD(Current)) {
        Current = TR->getArrayElementTypeMD(Current);
      }
      if (!Current) break;
      ExtendSet.insert(Current);
    }
  }

  for (MDNode *MD : TypeList) {
    MDNode *Current = MD;
    while (TR->isStructTypeMD(Current) || TR->isArrayTypeMD(Current)) {
      if (TR->isArrayTypeMD(Current)) {
        Current = TR->getArrayElementTypeMD(Current);
      } else {
        Current = TR->getStructFieldTypeMD(Current, 0);
      }
      while (TR->isArrayTypeMD(Current)) {
        Current = TR->getArrayElementTypeMD(Current);
      }
      if (!Current) break;
      ExtendSet.erase(Current);
    }
  }
  return ExtendSet;
}

Type *RISCVSigMemcpyExpand::selectUsedType(Value* I, const TypeRecovery::TypeSet* DestTypes, uint64_t Size, StringRef FuncName) {
  // Get source location for better diagnostics
  std::string SrcLoc = TypeRecovery::getSourceLocation(I);
  std::string LocStr = SrcLoc.empty() ? "" : " at " + SrcLoc;

  LLVM_DEBUG(dbgs() << "Dest Types for " << FuncName << " " << LocStr << ":\n");
  for (MDNode *MD : *DestTypes) {
    LLVM_DEBUG(dbgs() << "  - " << TR->getTypeString(MD) << "\n");
  }

  // Convert to SmallPtrSet and filter out bare ptr undef
  SmallPtrSet<MDNode *, 4> Types;
  for (MDNode *MD : *DestTypes)
    Types.insert(MD);
  Types = filterBareAndArrayPointeeTypes(Types);
  
  if (Types.empty()) {
    LLVM_DEBUG(dbgs() << "  Memcpy Expand Warning: Only bare ptr types, no concrete type for " << FuncName << "\n");
    errs() << "Memcpy Expand Warning: " << FuncName << " destination has only bare pointer type, no concrete type"
           << LocStr << "\n";
    return nullptr;
  }

  bool no_offset = Size == maxUIntN(uint64_t(64));
  SmallPtrSet<MDNode *, 4> candidateTypes;
  
  for (MDNode *PointeeMD : Types) {
    if (!PointeeMD)
      continue;
    Type *Ty = TR->getLLVMTypeFromMD(PointeeMD);
    if (!Ty)
      continue;
    
    uint64_t TySize = DL->getTypeAllocSize(Ty);
    // Check if Size is a multiple of TySize
    if (no_offset) {
      candidateTypes.insert(PointeeMD);
    } else if (Size % TySize == 0) {
      candidateTypes.insert(PointeeMD);
    }
  }

  LLVM_DEBUG(dbgs() << "Candidate Types for " << FuncName << " " << LocStr << ":\n");
  for (MDNode *MD : candidateTypes) {
    LLVM_DEBUG(dbgs() << "  - " << TR->getTypeString(MD) << "\n");
  }

  SmallPtrSet<MDNode *, 4> NoSubSetTypes;
  if (candidateTypes.size() > 1 && !no_offset) {
    NoSubSetTypes = filterSubSetTypes(candidateTypes);
    candidateTypes = NoSubSetTypes;
  }

  LLVM_DEBUG(dbgs() << "NoSubSet Types for " << FuncName << " " << LocStr << ":\n");
  for (MDNode *MD : NoSubSetTypes) {
    LLVM_DEBUG(dbgs() << "  - " << TR->getTypeString(MD) << "\n");
  }

  SmallPtrSet<MDNode *, 4> SimpleTypes = filterComplexTypes(candidateTypes);

  LLVM_DEBUG(dbgs() << "Simple Types for " << FuncName << " " << LocStr << ":\n");
  for (MDNode *MD : SimpleTypes) {
    LLVM_DEBUG(dbgs() << "  - " << TR->getTypeString(MD) << "\n");
  }

  Type* BestType = nullptr;
  MDNode* OnlyMD = nullptr;
  bool cannot_determined = false;

  for (MDNode *MD : SimpleTypes) {
    Type *Ty = TR->getLLVMTypeFromMD(MD);
    if (!BestType || DL->getTypeAllocSize(Ty) > DL->getTypeAllocSize(BestType)) {
      BestType = Ty;
      OnlyMD = MD;
    }
  }

  if (SimpleTypes.size() > 1) {
    cannot_determined = true;
  } else if (SimpleTypes.size() == 1) {
    if (TR->isUnionTypeMD(OnlyMD)) {
      cannot_determined = true;
    }
  } else {
    cannot_determined = true;
  }
  
  // Multiple types - emit warning with details and select by size
  if (cannot_determined && BestType) {
    LLVM_DEBUG(dbgs() << "  Memcpy Expand Warning: types cannot determined (" << Types.size() 
                      << ") for " << FuncName << ", selecting by size\n");
    errs() << "Memcpy Expand Warning: " << FuncName << " has multiple candidate types or union types (" << Types.size()
          << ")" << LocStr << ":\n";
    for (MDNode *MD : SimpleTypes) {
      errs() << "  - " << TR->getTypeString(MD) << "\n";
    }
    errs() << "More details on candidate types:\n";
    for (MDNode *MD : candidateTypes) {
      errs() << "  - " << TR->getTypeString(MD) << "\n";
    }
    
    if (BestType) {
      errs() << "  Selected type for memset: " << TR->getLLVMTypeString(BestType) << "\n";
    } else {
      errs() << "Memcpy Expand Warning: memset could not determine element type from candidates" 
            << LocStr << "\n";
    }
  } else if (!BestType) {
    LLVM_DEBUG(dbgs() << "  Memcpy Expand Warning: Only bare ptr types, no concrete type for " << FuncName << "\n");
    errs() << "Memcpy Expand Warning: " << FuncName << " destination has only bare pointer type, no concrete type" 
           << LocStr << "\n";
  }
  
  return BestType;
}

Type* RISCVSigMemcpyExpand::selectMemsetType(Value* I, Value *Dest, uint64_t Size) {
  return selectUsedType(I, TR->getTypeSet(Dest), Size, "memset");
}

Type *RISCVSigMemcpyExpand::selectMemcpyType(Value* I, Value *Dest, Value *Src, 
                                              uint64_t Size) {
  
  // Get recovered types for both dest and src
  const TypeRecovery::TypeSet *DestTypes = TR->getTypeSet(Dest);
  const TypeRecovery::TypeSet *SrcTypes = TR->getTypeSet(Src);
  
  // Merge both type sets
  SmallPtrSet<MDNode *, 4> MergedTypes;
  if (DestTypes) {
    for (MDNode *MD : *DestTypes)
      MergedTypes.insert(MD);
  }
  if (SrcTypes) {
    for (MDNode *MD : *SrcTypes)
      MergedTypes.insert(MD);
  }

  return selectUsedType(I, &MergedTypes, Size, "memcpy");
}

//===----------------------------------------------------------------------===//
// Original Functions
//===----------------------------------------------------------------------===//

void RISCVSigMemcpyExpand::emitMemcpyCall(IRBuilder<> &Builder, 
                                           Value *Dest, Value *Src,
                                           Value* Size, 
                                           unsigned DestAS, unsigned SrcAS) {
  // Call llvm.memcpy - in opaque pointer mode, no cast needed
  Builder.CreateMemCpy(Dest, MaybeAlign(1), Src, MaybeAlign(1), Size);
}

void RISCVSigMemcpyExpand::emitMemsetCall(IRBuilder<> &Builder, 
                                           Value *Dest, Value* Size, 
                                           unsigned DestAS) {
  // Call llvm.memset - in opaque pointer mode, no cast needed
  Builder.CreateMemSet(Dest, Builder.getInt8(0), Size, MaybeAlign(1));
}

std::string RISCVSigMemcpyExpand::getMangledTypeName(Type *Ty) {
  std::string Name;
  raw_string_ostream OS(Name);
  
  if (auto *ST = dyn_cast<StructType>(Ty)) {
    if (ST->hasName()) {
      // Use the struct name, replacing dots with underscores
      StringRef StructName = ST->getName();
      for (char C : StructName) {
        OS << (C == '.' ? '_' : C);
      }
    } else {
      // Anonymous struct: use hash of layout
      OS << "anon_struct_" << ST->getNumElements();
      for (unsigned I = 0; I < ST->getNumElements(); I++) {
        OS << "_" << getMangledTypeName(ST->getElementType(I));
      }
    }
  } else if (Ty->isPointerTy()) {
    OS << "ptr";
  } else if (const auto *AT = dyn_cast<ArrayType>(Ty)) {
    assert(false && "Array type should be handled elsewhere");
    // OS << "arr" << AT->getNumElements() << "_" 
    //    << getMangledTypeName(AT->getElementType());
  } else if (Ty->isIntegerTy()) {
    assert(false && "Integer type should be handled elsewhere");
    // OS << "i" << Ty->getIntegerBitWidth();
  } else if (Ty->isFloatTy()) {
    assert(false && "Float type should be handled elsewhere");
    // OS << "f32";
  } else if (Ty->isDoubleTy()) {
    assert(false && "Double type should be handled elsewhere");
    // OS << "f64";
  } else {
    assert(false && "Unknown type");
    // OS << "unknown";
  }
  
  return Name;
}

Function *RISCVSigMemcpyExpand::getOrCreateMemcpyFunc(Type *ElemTy, 
                                                       unsigned DestAS, 
                                                       unsigned SrcAS) {
  // Build function name
  std::string FuncName = "__sigmemcpy_";
  FuncName += getASName(DestAS);
  FuncName += "_";
  FuncName += getASName(SrcAS);
  FuncName += "_";
  FuncName += getMangledTypeName(ElemTy);
  
  // Check cache first
  auto It = HelperFuncCache.find(FuncName);
  if (It != HelperFuncCache.end()) {
    return It->second;
  }
  
  // Check if function already exists in module
  if (Function *ExistingF = Mod->getFunction(FuncName)) {
    HelperFuncCache[FuncName] = ExistingF;
    return ExistingF;
  }
  
  LLVM_DEBUG(dbgs() << "Creating helper function: " << FuncName << "\n");
  
  // Create function type: void (ptr dest, ptr src, i64 len)
  Type *DestPtrTy = PointerType::get(*Ctx, DestAS);
  Type *SrcPtrTy = PointerType::get(*Ctx, SrcAS);
  Type *I64Ty = Type::getInt64Ty(*Ctx);
  
  FunctionType *FT = FunctionType::get(
      Type::getVoidTy(*Ctx),
      {DestPtrTy, SrcPtrTy, I64Ty},
      false);
  
  Function *F = Function::Create(FT, GlobalValue::InternalLinkage, 
                                 FuncName, Mod);
  
  // Set attributes for optimization
  F->addFnAttr(Attribute::NoUnwind);
  F->addFnAttr(Attribute::WillReturn);
  // Note: NoCapture was removed in favor of Captures in newer LLVM
  // F->addParamAttr(0, Attribute::NoCapture);
  // F->addParamAttr(1, Attribute::NoCapture);
  F->addParamAttr(1, Attribute::ReadOnly);
  
  // Name the arguments
  auto *ArgIt = F->arg_begin();
  ArgIt->setName("dest"); ++ArgIt;
  ArgIt->setName("src"); ++ArgIt;
  ArgIt->setName("len");
  
  // Generate function body
  generateMemcpyFuncBody(F, ElemTy, DestAS, SrcAS);
  
  HelperFuncCache[FuncName] = F;
  NumHelperFuncsCreated++;
  
  return F;
}

void RISCVSigMemcpyExpand::generateMemcpyFuncBody(Function *F, Type *ElemTy,
                                                   unsigned DestAS, 
                                                   unsigned SrcAS) {
  auto *ArgIt2 = F->arg_begin();
  Value *Dest = &*ArgIt2++;
  Value *Src = &*ArgIt2++;
  Value *Len = &*ArgIt2;
  
  uint64_t ElemSize = DL->getTypeAllocSize(ElemTy);
  Type *I64Ty = Type::getInt64Ty(*Ctx);
  
  BasicBlock *EntryBB = BasicBlock::Create(*Ctx, "entry", F);
  IRBuilder<> Builder(EntryBB);
  
  // Entry: check if len > 0, if not, skip to exit
  BasicBlock *ExitBB = BasicBlock::Create(*Ctx, "exit", F);
  Value *LenGtZero = Builder.CreateICmpUGT(Len, Builder.getInt64(0), "len.gt.zero");
  
  if (ElemSize <= InlineThreshold) {
    // Small type: use block-based inline expansion
    // BlockLen = InlineThreshold / ElemSize (number of elements per block)
    uint64_t BlockLen = InlineThreshold / ElemSize;
    if (BlockLen < 1) BlockLen = 1;
    
    // Create blocks for main loop (BlockLen elements at a time) and remainder loop
    BasicBlock *MainLoopBB = BasicBlock::Create(*Ctx, "main.loop", F);
    BasicBlock *MainBodyBB = BasicBlock::Create(*Ctx, "main.body", F);
    BasicBlock *RemLoopBB = BasicBlock::Create(*Ctx, "rem.loop", F);
    BasicBlock *RemBodyBB = BasicBlock::Create(*Ctx, "rem.body", F);
    
    Builder.CreateCondBr(LenGtZero, MainLoopBB, ExitBB);
    
    // Main loop: process BlockLen elements at a time
    Builder.SetInsertPoint(MainLoopBB);
    PHINode *MainIdxPhi = Builder.CreatePHI(I64Ty, 2, "main.idx");
    MainIdxPhi->addIncoming(Builder.getInt64(0), EntryBB);
    
    // Calculate remaining count
    Value *Remaining = Builder.CreateSub(Len, MainIdxPhi, "remaining");
    Value *CanDoBlock = Builder.CreateICmpUGE(Remaining, Builder.getInt64(BlockLen), "can.block");
    Builder.CreateCondBr(CanDoBlock, MainBodyBB, RemLoopBB);
    
    // Main body: inline copy BlockLen elements
    Builder.SetInsertPoint(MainBodyBB);
    Value *DestBlock = Builder.CreateGEP(ElemTy, Dest, MainIdxPhi, "dest.block");
    Value *SrcBlock = Builder.CreateGEP(ElemTy, Src, MainIdxPhi, "src.block");
    
    // Generate inline copy for BlockLen elements
    generateBlockCopy(Builder, DestBlock, SrcBlock, ElemTy, BlockLen, DestAS, SrcAS);
    
    Value *MainNextIdx = Builder.CreateAdd(MainIdxPhi, Builder.getInt64(BlockLen), "main.next");
    MainIdxPhi->addIncoming(MainNextIdx, Builder.GetInsertBlock());
    Builder.CreateBr(MainLoopBB);
    
    // Remainder loop: process 1 element at a time
    Builder.SetInsertPoint(RemLoopBB);
    PHINode *RemIdxPhi = Builder.CreatePHI(I64Ty, 2, "rem.idx");
    RemIdxPhi->addIncoming(MainIdxPhi, MainLoopBB);
    
    Value *RemDone = Builder.CreateICmpUGE(RemIdxPhi, Len, "rem.done");
    Builder.CreateCondBr(RemDone, ExitBB, RemBodyBB);
    
    // Remainder body: inline copy 1 element
    Builder.SetInsertPoint(RemBodyBB);
    Value *DestElem = Builder.CreateGEP(ElemTy, Dest, RemIdxPhi, "dest.elem");
    Value *SrcElem = Builder.CreateGEP(ElemTy, Src, RemIdxPhi, "src.elem");
    
    generateBlockCopy(Builder, DestElem, SrcElem, ElemTy, 1, DestAS, SrcAS);
    
    Value *RemNextIdx = Builder.CreateAdd(RemIdxPhi, Builder.getInt64(1), "rem.next");
    RemIdxPhi->addIncoming(RemNextIdx, Builder.GetInsertBlock());
    Builder.CreateBr(RemLoopBB);
    
  } else {
    // Large type: process field by field
    // For large types, we iterate over each element and copy field by field
    BasicBlock *LoopBB = BasicBlock::Create(*Ctx, "loop", F);
    BasicBlock *BodyBB = BasicBlock::Create(*Ctx, "body", F);
    
    Builder.CreateCondBr(LenGtZero, LoopBB, ExitBB);
    
    Builder.SetInsertPoint(LoopBB);
    PHINode *IdxPhi = Builder.CreatePHI(I64Ty, 2, "idx");
    IdxPhi->addIncoming(Builder.getInt64(0), EntryBB);
    Builder.CreateBr(BodyBB);
    
    Builder.SetInsertPoint(BodyBB);
    Value *DestElem = Builder.CreateGEP(ElemTy, Dest, IdxPhi, "dest.elem");
    Value *SrcElem = Builder.CreateGEP(ElemTy, Src, IdxPhi, "src.elem");
    
    // Process large struct field by field
    if (auto *ST = dyn_cast<StructType>(ElemTy)) {
      const StructLayout *SL = DL->getStructLayout(ST);
      
      // Collect consecutive small fields and batch them
      SmallVector<uint64_t, 8> BatchPtrOffsets;
      uint64_t BatchStartOffset = 0;
      uint64_t BatchBytes = 0;
      Value *BatchDestStart = nullptr;
      Value *BatchSrcStart = nullptr;
      
      for (unsigned I = 0; I < ST->getNumElements(); I++) {
        Type *FieldTy = ST->getElementType(I);
        uint64_t FieldOffset = SL->getElementOffset(I);
        uint64_t FieldSize = DL->getTypeAllocSize(FieldTy);
        
        Value *DestField = Builder.CreateStructGEP(ST, DestElem, I, "dest.field");
        Value *SrcField = Builder.CreateStructGEP(ST, SrcElem, I, "src.field");
        
        if (FieldSize >= InlineThreshold) {
          // Flush any pending batch first
          if (BatchBytes > 0) {
            generateBlockCopyByOffset(Builder, BatchDestStart, BatchSrcStart,
                                      BatchBytes, BatchPtrOffsets, DestAS, SrcAS);
            BatchPtrOffsets.clear();
            BatchBytes = 0;
          }
          
          // Large field: recursively call helper function
          if (FieldTy->containsPointer()) {
            if (isa<StructType>(FieldTy)) {
              Function *HelperF = getOrCreateMemcpyFunc(FieldTy, DestAS, SrcAS);
              Builder.CreateCall(HelperF, {DestField, SrcField, Builder.getInt64(1)});
            } else if (auto *AT = dyn_cast<ArrayType>(FieldTy)) {
              Type *ArrElemTy = AT->getElementType();
              uint64_t NumElems = AT->getNumElements();
              Function *HelperF = getOrCreateMemcpyFunc(ArrElemTy, DestAS, SrcAS);
              Builder.CreateCall(HelperF, {DestField, SrcField, Builder.getInt64(NumElems)});
            }
          } else {
            // Large field without pointers: use memcpy
            emitMemcpyCall(Builder, DestField, SrcField, Builder.getInt64(FieldSize), DestAS, SrcAS);
          }
        } else {
          // Small field: accumulate into batch
          if (BatchBytes == 0) {
            BatchStartOffset = FieldOffset;
            BatchDestStart = DestField;
            BatchSrcStart = SrcField;
          }
          
          // Add pointer offsets for this field (relative to batch start)
          SmallVector<uint64_t, 8> FieldPtrOffsets = getPointerOffsets(FieldTy);
          for (uint64_t Off : FieldPtrOffsets) {
            BatchPtrOffsets.push_back(FieldOffset - BatchStartOffset + Off);
          }
          
          BatchBytes = (FieldOffset - BatchStartOffset) + FieldSize;
          
          // Check if we should flush the batch
          bool ShouldFlush = false;
          if (I + 1 < ST->getNumElements()) {
            Type *NextFieldTy = ST->getElementType(I + 1);
            uint64_t NextFieldSize = DL->getTypeAllocSize(NextFieldTy);
            uint64_t NextFieldOffset = SL->getElementOffset(I + 1);
            uint64_t PotentialBatchSize = (NextFieldOffset - BatchStartOffset) + NextFieldSize;
            if (NextFieldSize >= InlineThreshold || PotentialBatchSize > InlineThreshold) {
              ShouldFlush = true;
            }
          } else {
            // Last field
            ShouldFlush = true;
          }
          
          if (ShouldFlush && BatchBytes > 0) {
            generateBlockCopyByOffset(Builder, BatchDestStart, BatchSrcStart,
                                      BatchBytes, BatchPtrOffsets, DestAS, SrcAS);
            BatchPtrOffsets.clear();
            BatchBytes = 0;
          }
        }
      }
    } else {
      // Not a struct - shouldn't happen for large types, but handle it
      copyElementInFunc(Builder, DestElem, SrcElem, ElemTy, DestAS, SrcAS);
    }
    
    Value *NextIdx = Builder.CreateAdd(IdxPhi, Builder.getInt64(1), "idx.next");
    IdxPhi->addIncoming(NextIdx, Builder.GetInsertBlock());
    
    Value *Continue = Builder.CreateICmpULT(NextIdx, Len, "loop.cond");
    Builder.CreateCondBr(Continue, LoopBB, ExitBB);
  }
  
  // Exit block
  Builder.SetInsertPoint(ExitBB);
  Builder.CreateRetVoid();
}

void RISCVSigMemcpyExpand::copyElementInFunc(IRBuilder<> &Builder, 
                                              Value *Dest, Value *Src, 
                                              Type *Ty,
                                              unsigned DestAS, unsigned SrcAS) {
  if (Ty->isPointerTy()) {
    // Pointer type: load from src, cast address space, store to dest
    Type *SrcPtrTy = PointerType::get(*Ctx, SrcAS);
    Value *LoadedPtr = Builder.CreateLoad(SrcPtrTy, Src, "ptr.load");
    
    // Cast to destination address space
    Type *DestPtrTy = PointerType::get(*Ctx, DestAS);
    Value *CastPtr = LoadedPtr;
    if (SrcAS != DestAS) {
      CastPtr = Builder.CreateAddrSpaceCast(LoadedPtr, DestPtrTy, "ptr.cast");
    }
    
    Builder.CreateStore(CastPtr, Dest);
    
  } else if (auto *ST = dyn_cast<StructType>(Ty)) {
    // Struct type: copy each member
    for (unsigned I = 0; I < ST->getNumElements(); I++) {
      Type *MemberTy = ST->getElementType(I);
      Value *DestMember = Builder.CreateStructGEP(ST, Dest, I, "dest.member");
      Value *SrcMember = Builder.CreateStructGEP(ST, Src, I, "src.member");
      
      if (isa<StructType>(MemberTy) || isa<ArrayType>(MemberTy)) {
        // For nested composite types, check if they contain pointers
        if (MemberTy->containsPointer()) {
          // Call helper function for types with pointers
          if (isa<StructType>(MemberTy)) {
            Function *HelperF = getOrCreateMemcpyFunc(MemberTy, DestAS, SrcAS);
            Builder.CreateCall(HelperF, {DestMember, SrcMember, Builder.getInt64(1)});
          } else {
            // Array type
            Type *ElemTy = cast<ArrayType>(MemberTy)->getElementType();
            uint64_t NumElems = cast<ArrayType>(MemberTy)->getNumElements();
            Function *HelperF = getOrCreateMemcpyFunc(ElemTy, DestAS, SrcAS);
            Builder.CreateCall(HelperF, {DestMember, SrcMember, Builder.getInt64(NumElems)});
          }
        } else {
          // Use memcpy for types without pointers
          uint64_t Size = DL->getTypeAllocSize(MemberTy);
          emitMemcpyCall(Builder, DestMember, SrcMember, Builder.getInt64(Size), DestAS, SrcAS);
        }
      } else {
        // Basic type or pointer: copy inline
        copyElementInFunc(Builder, DestMember, SrcMember, MemberTy, DestAS, SrcAS);
      }
    }
    
  } else if (auto *AT = dyn_cast<ArrayType>(Ty)) {
    // Array type
    Type *ElemTy = AT->getElementType();
    uint64_t NumElems = AT->getNumElements();
    
    // Check if element type contains pointers
    if (ElemTy->containsPointer()) {
      // Need to process each element for pointer re-signing
      if (isa<StructType>(ElemTy) || isa<ArrayType>(ElemTy)) {
        // Call helper function for composite element types
        Function *HelperF = getOrCreateMemcpyFunc(ElemTy, DestAS, SrcAS);
        Builder.CreateCall(HelperF, {Dest, Src, Builder.getInt64(NumElems)});
      } else {
        // Array of pointers - process each
        for (uint64_t I = 0; I < NumElems; I++) {
          Value *DestElem = Builder.CreateConstGEP2_64(AT, Dest, 0, I, "dest.arr");
          Value *SrcElem = Builder.CreateConstGEP2_64(AT, Src, 0, I, "src.arr");
          copyElementInFunc(Builder, DestElem, SrcElem, ElemTy, DestAS, SrcAS);
        }
      }
    } else {
      // No pointers in element type - use memcpy for the whole array
      uint64_t Size = DL->getTypeAllocSize(AT);
      emitMemcpyCall(Builder, Dest, Src, Builder.getInt64(Size), DestAS, SrcAS);
    }
    
  } else {
    // Basic type: direct load/store
    Value *Val = Builder.CreateLoad(Ty, Src, "val.load");
    Builder.CreateStore(Val, Dest);
  }
}

bool RISCVSigMemcpyExpand::expandSigMemcpy(CallInst* II) {
  Value *Dest = II->getArgOperand(0);
  Value *Src = II->getArgOperand(1);
  Value *LenVal = II->getArgOperand(2);
  
  // Get address spaces
  Dest = getRealPtr(Dest);
  Src = getRealPtr(Src);
  unsigned DestAS = Dest->getType()->getPointerAddressSpace();
  unsigned SrcAS = Src->getType()->getPointerAddressSpace();

  LLVM_DEBUG(dbgs() << "Expanding sigmemcpy: dest AS=" << DestAS 
                    << ", src AS=" << SrcAS << "\n");

  bool HasPointers = false;
  Type* ElemTy = nullptr;
  bool IsRawToRaw = (DestAS == RawAS && SrcAS == RawAS);

  uint64_t ByteLen = maxUIntN(64);
  if (auto *LenCI = dyn_cast<ConstantInt>(LenVal)) {
    ByteLen = LenCI->getZExtValue();
  }

  if (!IsRawToRaw) {
    ElemTy = selectMemcpyType(II, Dest, Src, ByteLen);
    if (!ElemTy) {
      IsRawToRaw = true;
    } else {
      HasPointers = ElemTy->containsPointer();
    }
  }
  
  IRBuilder<> Builder(II);
  
  if (IsRawToRaw || !HasPointers) {
    // For raw-to-raw or no-pointer cases, check if we can inline small copies
    if (ByteLen != maxUIntN(64) && ByteLen <= InlineThreshold) {
      LLVM_DEBUG(dbgs() << "  Checking raw-to-raw inline expansion for " 
                        << ByteLen << " bytes\n");
      
      // Get the original src before getRealPtr
      Value *OrigSrc = II->getArgOperand(1);
      
      // Try to get underlying constant from src
      Constant *SrcConstant = getUnderlyingConstant(OrigSrc);
      if (SrcConstant) {
        LLVM_DEBUG(dbgs() << "  Found constant source, generating constant copy\n");
        
        // Determine element type for constant copy
        Type *CopyTy = ElemTy;
        uint64_t NumElems = 1;
        
        if (!CopyTy) {
          // No type recovery info, use i8 array
          CopyTy = Type::getInt8Ty(*Ctx);
          NumElems = ByteLen;
        } else {
          uint64_t ElemSize = DL->getTypeAllocSize(CopyTy);
          NumElems = ByteLen / ElemSize;
        }
        
        generateConstantCopy(Builder, Dest, SrcConstant, CopyTy, NumElems, DestAS);
        
        II->eraseFromParent();
        NumMemcpyExpanded++;
        return true;
      } else {
        // Not a constant, generate block copy for raw-to-raw without pointers
        LLVM_DEBUG(dbgs() << "  Non-constant source, generating block copy\n");
        
        // No pointers in this copy
        SmallVector<uint64_t, 8> EmptyOffsets;
        
        generateBlockCopyByOffset(Builder, Dest, Src, ByteLen, EmptyOffsets, 
                                  DestAS, SrcAS);
        
        II->eraseFromParent();
        NumMemcpyExpanded++;
        return true;
      }
    }
    
    LLVM_DEBUG(dbgs() << "  Using llvm.memcpy (raw-to-raw=" << IsRawToRaw 
                      << ", has_pointers=" << HasPointers << ")\n");
    return false;
  }
  
  // Calculate number of elements
  uint64_t ElemSize = DL->getTypeAllocSize(ElemTy);
  uint64_t NumElems = (ByteLen != maxUIntN(64)) ? (ByteLen / ElemSize) : 1;
  
  // Check if we should inline or use helper function
  if (ByteLen != maxUIntN(64) && ByteLen <= InlineThreshold) {
    // Small copy: inline expansion
    LLVM_DEBUG(dbgs() << "  Inline expansion for " << ByteLen << " bytes\n");
    generateBlockCopy(Builder, Dest, Src, ElemTy, NumElems, DestAS, SrcAS);
    
    II->eraseFromParent();
    NumMemcpyExpanded++;
    return true;
  }
  
  // Large copy: use helper function
  LLVM_DEBUG(dbgs() << "  Using sigmemcpy helper function for type: " << *ElemTy << "\n");
  
  if (auto *LenCI = dyn_cast<ConstantInt>(LenVal)) {
    LenVal = Builder.getInt64(NumElems);
  } else {
    Value *ElemSizeVal = Builder.getInt64(ElemSize);
    LenVal = Builder.CreateUDiv(LenVal, ElemSizeVal, "elem.len");
  }
  Function *HelperF = getOrCreateMemcpyFunc(ElemTy, DestAS, SrcAS);
  Builder.CreateCall(HelperF, {Dest, Src, LenVal});

  // Remove the original intrinsic call
  II->eraseFromParent();
  NumMemcpyExpanded++;
  return true;
}

//===----------------------------------------------------------------------===//
// Sigmemset expansion
//===----------------------------------------------------------------------===//

Function *RISCVSigMemcpyExpand::getOrCreateMemsetFunc(Type *ElemTy, 
                                                       unsigned DestAS) {
  // Build function name
  std::string FuncName = "__sigmemset_";
  FuncName += getASName(DestAS);
  FuncName += "_";
  FuncName += getMangledTypeName(ElemTy);
  
  // Check cache first
  auto It = HelperFuncCache.find(FuncName);
  if (It != HelperFuncCache.end()) {
    return It->second;
  }
  
  // Check if function already exists in module
  if (Function *ExistingF = Mod->getFunction(FuncName)) {
    HelperFuncCache[FuncName] = ExistingF;
    return ExistingF;
  }
  
  LLVM_DEBUG(dbgs() << "Creating memset helper function: " << FuncName << "\n");
  
  // Create function type: void (ptr dest, i64 len)
  Type *DestPtrTy = PointerType::get(*Ctx, DestAS);
  Type *I64Ty = Type::getInt64Ty(*Ctx);
  
  FunctionType *FT = FunctionType::get(
      Type::getVoidTy(*Ctx),
      {DestPtrTy, I64Ty},
      false);
  
  Function *F = Function::Create(FT, GlobalValue::InternalLinkage, 
                                 FuncName, Mod);
  
  // Set attributes for optimization
  F->addFnAttr(Attribute::NoUnwind);
  F->addFnAttr(Attribute::WillReturn);
  
  // Name the arguments
  auto *ArgIt = F->arg_begin();
  ArgIt->setName("dest"); ++ArgIt;
  ArgIt->setName("len");
  
  // Generate function body
  generateMemsetFuncBody(F, ElemTy, DestAS);
  
  HelperFuncCache[FuncName] = F;
  NumHelperFuncsCreated++;
  
  return F;
}

void RISCVSigMemcpyExpand::generateMemsetFuncBody(Function *F, Type *ElemTy,
                                                   unsigned DestAS) {
  // This function is only called for sig address space (AS 0)
  // because raw AS or types without pointers use regular memset
  assert(DestAS == 0 && "generateMemsetFuncBody should only be called for sig AS");
  
  auto *ArgIt = F->arg_begin();
  Value *Dest = &*ArgIt++;
  Value *Len = &*ArgIt;
  
  uint64_t ElemSize = DL->getTypeAllocSize(ElemTy);
  Type *I64Ty = Type::getInt64Ty(*Ctx);
  
  BasicBlock *EntryBB = BasicBlock::Create(*Ctx, "entry", F);
  IRBuilder<> Builder(EntryBB);
  
  // Pre-compute xsig_setdummyid(null) once at function entry
  Type *DefaultPtrTy = PointerType::get(*Ctx, 0);
  Value *NullPtr = ConstantPointerNull::get(cast<PointerType>(DefaultPtrTy));
  FunctionType *SetDummyIdTy = FunctionType::get(DefaultPtrTy, {DefaultPtrTy}, false);
  FunctionCallee SetDummyIdFn = Mod->getOrInsertFunction(
      "llvm.riscv.xsig.setdummyid", SetDummyIdTy);
  Value *DummyNull = Builder.CreateCall(SetDummyIdFn, {NullPtr}, "dummy.null");
  
  // Entry: check if len > 0, if not, skip to exit
  BasicBlock *ExitBB = BasicBlock::Create(*Ctx, "exit", F);
  Value *LenGtZero = Builder.CreateICmpUGT(Len, Builder.getInt64(0), "len.gt.zero");
  
  if (ElemSize <= InlineThreshold) {
    // Small type: use block-based inline expansion
    uint64_t BlockLen = InlineThreshold / ElemSize;
    if (BlockLen < 1) BlockLen = 1;
    
    BasicBlock *MainLoopBB = BasicBlock::Create(*Ctx, "main.loop", F);
    BasicBlock *MainBodyBB = BasicBlock::Create(*Ctx, "main.body", F);
    BasicBlock *RemLoopBB = BasicBlock::Create(*Ctx, "rem.loop", F);
    BasicBlock *RemBodyBB = BasicBlock::Create(*Ctx, "rem.body", F);
    
    Builder.CreateCondBr(LenGtZero, MainLoopBB, ExitBB);
    
    // Main loop
    Builder.SetInsertPoint(MainLoopBB);
    PHINode *MainIdxPhi = Builder.CreatePHI(I64Ty, 2, "main.idx");
    MainIdxPhi->addIncoming(Builder.getInt64(0), EntryBB);
    
    Value *Remaining = Builder.CreateSub(Len, MainIdxPhi, "remaining");
    Value *CanDoBlock = Builder.CreateICmpUGE(Remaining, Builder.getInt64(BlockLen), "can.block");
    Builder.CreateCondBr(CanDoBlock, MainBodyBB, RemLoopBB);
    
    // Main body
    Builder.SetInsertPoint(MainBodyBB);
    Value *DestBlock = Builder.CreateGEP(ElemTy, Dest, MainIdxPhi, "dest.block");
    generateBlockZero(Builder, DestBlock, ElemTy, BlockLen, DestAS, DummyNull);
    
    Value *MainNextIdx = Builder.CreateAdd(MainIdxPhi, Builder.getInt64(BlockLen), "main.next");
    MainIdxPhi->addIncoming(MainNextIdx, Builder.GetInsertBlock());
    Builder.CreateBr(MainLoopBB);
    
    // Remainder loop
    Builder.SetInsertPoint(RemLoopBB);
    PHINode *RemIdxPhi = Builder.CreatePHI(I64Ty, 2, "rem.idx");
    RemIdxPhi->addIncoming(MainIdxPhi, MainLoopBB);
    
    Value *RemDone = Builder.CreateICmpUGE(RemIdxPhi, Len, "rem.done");
    Builder.CreateCondBr(RemDone, ExitBB, RemBodyBB);
    
    // Remainder body
    Builder.SetInsertPoint(RemBodyBB);
    Value *DestElem = Builder.CreateGEP(ElemTy, Dest, RemIdxPhi, "dest.elem");
    generateBlockZero(Builder, DestElem, ElemTy, 1, DestAS, DummyNull);
    
    Value *RemNextIdx = Builder.CreateAdd(RemIdxPhi, Builder.getInt64(1), "rem.next");
    RemIdxPhi->addIncoming(RemNextIdx, Builder.GetInsertBlock());
    Builder.CreateBr(RemLoopBB);
    
  } else {
    // Large type: process field by field
    BasicBlock *LoopBB = BasicBlock::Create(*Ctx, "loop", F);
    BasicBlock *BodyBB = BasicBlock::Create(*Ctx, "body", F);
    
    Builder.CreateCondBr(LenGtZero, LoopBB, ExitBB);
    
    Builder.SetInsertPoint(LoopBB);
    PHINode *IdxPhi = Builder.CreatePHI(I64Ty, 2, "idx");
    IdxPhi->addIncoming(Builder.getInt64(0), EntryBB);
    Builder.CreateBr(BodyBB);
    
    Builder.SetInsertPoint(BodyBB);
    Value *DestElem = Builder.CreateGEP(ElemTy, Dest, IdxPhi, "dest.elem");
    
    if (auto *ST = dyn_cast<StructType>(ElemTy)) {
      const StructLayout *SL = DL->getStructLayout(ST);
      
      SmallVector<uint64_t, 8> BatchPtrOffsets;
      uint64_t BatchStartOffset = 0;
      uint64_t BatchBytes = 0;
      Value *BatchDestStart = nullptr;
      
      for (unsigned I = 0; I < ST->getNumElements(); I++) {
        Type *FieldTy = ST->getElementType(I);
        uint64_t FieldOffset = SL->getElementOffset(I);
        uint64_t FieldSize = DL->getTypeAllocSize(FieldTy);
        
        Value *DestField = Builder.CreateStructGEP(ST, DestElem, I, "dest.field");
        
        if (FieldSize >= InlineThreshold) {
          // Flush pending batch
          if (BatchBytes > 0) {
            generateBlockZeroByOffset(Builder, BatchDestStart, BatchBytes,
                                      BatchPtrOffsets, DestAS, DummyNull);
            BatchPtrOffsets.clear();
            BatchBytes = 0;
          }
          
          // Large field
          if (FieldTy->containsPointer()) {
            if (isa<StructType>(FieldTy)) {
              Function *HelperF = getOrCreateMemsetFunc(FieldTy, DestAS);
              Builder.CreateCall(HelperF, {DestField, Builder.getInt64(1)});
            } else if (auto *AT = dyn_cast<ArrayType>(FieldTy)) {
              Type *ArrElemTy = AT->getElementType();
              uint64_t NumElems = AT->getNumElements();
              Function *HelperF = getOrCreateMemsetFunc(ArrElemTy, DestAS);
              Builder.CreateCall(HelperF, {DestField, Builder.getInt64(NumElems)});
            }
          } else {
            emitMemsetCall(Builder, DestField, Builder.getInt64(FieldSize), DestAS);
          }
        } else {
          // Small field: accumulate
          if (BatchBytes == 0) {
            BatchStartOffset = FieldOffset;
            BatchDestStart = DestField;
          }
          
          SmallVector<uint64_t, 8> FieldPtrOffsets = getPointerOffsets(FieldTy);
          for (uint64_t Off : FieldPtrOffsets) {
            BatchPtrOffsets.push_back(FieldOffset - BatchStartOffset + Off);
          }
          
          BatchBytes = (FieldOffset - BatchStartOffset) + FieldSize;
          
          bool ShouldFlush = false;
          if (I + 1 < ST->getNumElements()) {
            Type *NextFieldTy = ST->getElementType(I + 1);
            uint64_t NextFieldSize = DL->getTypeAllocSize(NextFieldTy);
            uint64_t NextFieldOffset = SL->getElementOffset(I + 1);
            uint64_t PotentialBatchSize = (NextFieldOffset - BatchStartOffset) + NextFieldSize;
            if (NextFieldSize >= InlineThreshold || PotentialBatchSize > InlineThreshold) {
              ShouldFlush = true;
            }
          } else {
            ShouldFlush = true;
          }
          
          if (ShouldFlush && BatchBytes > 0) {
            generateBlockZeroByOffset(Builder, BatchDestStart, BatchBytes,
                                      BatchPtrOffsets, DestAS, DummyNull);
            BatchPtrOffsets.clear();
            BatchBytes = 0;
          }
        }
      }
    } else {
      zeroElementInFunc(Builder, DestElem, ElemTy, DestAS, DummyNull);
    }
    
    Value *NextIdx = Builder.CreateAdd(IdxPhi, Builder.getInt64(1), "idx.next");
    IdxPhi->addIncoming(NextIdx, Builder.GetInsertBlock());
    
    Value *Continue = Builder.CreateICmpULT(NextIdx, Len, "loop.cond");
    Builder.CreateCondBr(Continue, LoopBB, ExitBB);
  }
  
  // Exit block
  Builder.SetInsertPoint(ExitBB);
  Builder.CreateRetVoid();
}

void RISCVSigMemcpyExpand::zeroElementInFunc(IRBuilder<> &Builder, 
                                              Value *Dest, Type *Ty,
                                              unsigned DestAS, Value *DummyNull) {
  // This function is only called for sig address space (AS 0)
  // Raw address space uses regular memset, no helper function needed
  assert(DestAS == 0 && "zeroElementInFunc should only be called for sig AS");
  assert(DummyNull && "DummyNull must be pre-computed for sig AS");
  
  if (Ty->isPointerTy()) {
    // Pointer type: store the pre-computed DummyNull (xsig_setdummyid(null))
    Builder.CreateStore(DummyNull, Dest);
  } else if (auto *ST = dyn_cast<StructType>(Ty)) {
    // Struct type: zero each member
    for (unsigned I = 0; I < ST->getNumElements(); I++) {
      Type *MemberTy = ST->getElementType(I);
      Value *DestMember = Builder.CreateStructGEP(ST, Dest, I, "dest.member");
      
      if (isa<StructType>(MemberTy) || isa<ArrayType>(MemberTy)) {
        // For nested composite types, check if they contain pointers
        if (MemberTy->containsPointer()) {
          // Call helper function for types with pointers
          if (isa<StructType>(MemberTy)) {
            Function *HelperF = getOrCreateMemsetFunc(MemberTy, DestAS);
            Builder.CreateCall(HelperF, {DestMember, Builder.getInt64(1)});
          } else {
            // Array type
            Type *ElemTy = cast<ArrayType>(MemberTy)->getElementType();
            uint64_t NumElems = cast<ArrayType>(MemberTy)->getNumElements();
            Function *HelperF = getOrCreateMemsetFunc(ElemTy, DestAS);
            Builder.CreateCall(HelperF, {DestMember, Builder.getInt64(NumElems)});
          }
        } else {
          // Use memset for types without pointers
          uint64_t Size = DL->getTypeAllocSize(MemberTy);
          emitMemsetCall(Builder, DestMember, Builder.getInt64(Size), DestAS);
        }
      } else {
        // Basic type or pointer: zero inline
        zeroElementInFunc(Builder, DestMember, MemberTy, DestAS, DummyNull);
      }
    }
    
  } else if (auto *AT = dyn_cast<ArrayType>(Ty)) {
    // Array type
    Type *ElemTy = AT->getElementType();
    uint64_t NumElems = AT->getNumElements();
    
    // Check if element type contains pointers
    if (ElemTy->containsPointer()) {
      // Need to process each element for proper pointer zeroing
      if (isa<StructType>(ElemTy) || isa<ArrayType>(ElemTy)) {
        // Call helper function for composite element types
        Function *HelperF = getOrCreateMemsetFunc(ElemTy, DestAS);
        Builder.CreateCall(HelperF, {Dest, Builder.getInt64(NumElems)});
      } else {
        // Array of pointers - process each
        for (uint64_t I = 0; I < NumElems; I++) {
          Value *DestElem = Builder.CreateConstGEP2_64(AT, Dest, 0, I, "dest.arr");
          zeroElementInFunc(Builder, DestElem, ElemTy, DestAS, DummyNull);
        }
      }
    } else {
      // No pointers in element type - use memset for the whole array
      uint64_t Size = DL->getTypeAllocSize(AT);
      emitMemsetCall(Builder, Dest, Builder.getInt64(Size), DestAS);
    }
    
  } else {
    // Basic type: store zero
    Value *Zero = Constant::getNullValue(Ty);
    Builder.CreateStore(Zero, Dest);
  }
}

bool RISCVSigMemcpyExpand::expandSigMemset(CallInst *II) {
  Value *Dest = II->getArgOperand(0);
  Value *LenVal = II->getArgOperand(2);
  
  // Get address space
  Dest = getRealPtr(Dest);
  unsigned DestAS = Dest->getType()->getPointerAddressSpace();

  LLVM_DEBUG(dbgs() << "Expanding sigmemset: dest AS=" << DestAS << "\n");
  
  bool HasPointers = false;
  Type* ElemTy = nullptr;
  bool IsRawToRaw = (DestAS == RawAS);

  uint64_t ByteLen = maxUIntN(64);
  if (auto *LenCI = dyn_cast<ConstantInt>(LenVal)) {
    ByteLen = LenCI->getZExtValue();
  }

  if (!IsRawToRaw) {
    ElemTy = selectMemsetType(II, Dest, ByteLen);
    if (!ElemTy) {
      IsRawToRaw = true;
    } else {
      HasPointers = ElemTy->containsPointer();
    }
  }
  
  IRBuilder<> Builder(II);
  
  if (IsRawToRaw || !HasPointers) {
    LLVM_DEBUG(dbgs() << "  Using llvm.memset (raw=" << IsRawToRaw
                      << ", has_pointers=" << HasPointers << ")\n");
    return false;
  } else {
    // Calculate number of elements
    uint64_t ElemSize = DL->getTypeAllocSize(ElemTy);
    uint64_t NumElems = (ByteLen != maxUIntN(64)) ? (ByteLen / ElemSize) : 1;
    
    // Check if we should inline or use helper function
    if (ByteLen != maxUIntN(64) && ByteLen <= InlineThreshold) {
      // Small memset: inline expansion
      LLVM_DEBUG(dbgs() << "  Inline expansion for " << ByteLen << " bytes\n");
      
      // Pre-compute xsig_setdummyid(null) for pointer fields
      Type *DefaultPtrTy = PointerType::get(*Ctx, 0);
      Value *NullPtr = ConstantPointerNull::get(cast<PointerType>(DefaultPtrTy));
      FunctionType *SetDummyIdTy = FunctionType::get(DefaultPtrTy, {DefaultPtrTy}, false);
      FunctionCallee SetDummyIdFn = Mod->getOrInsertFunction(
          "llvm.riscv.xsig.setdummyid", SetDummyIdTy);
      Value *DummyNull = Builder.CreateCall(SetDummyIdFn, {NullPtr}, "dummy.null");
      
      generateBlockZero(Builder, Dest, ElemTy, NumElems, DestAS, DummyNull);
    } else {
      // Large memset: use helper function
      LLVM_DEBUG(dbgs() << "  Using sigmemset helper function for type: " << *ElemTy << "\n");
      
      if (auto *LenCI = dyn_cast<ConstantInt>(LenVal)) {
        LenVal = Builder.getInt64(NumElems);
      } else {
        Value *ElemSizeVal = Builder.getInt64(ElemSize);
        LenVal = Builder.CreateUDiv(LenVal, ElemSizeVal, "elem.len");
      }
      Function *HelperF = getOrCreateMemsetFunc(ElemTy, DestAS);
      Builder.CreateCall(HelperF, {Dest, LenVal});
    }
  }
  
  // Remove the original intrinsic call
  II->eraseFromParent();
  NumMemsetExpanded++;
  
  return true;
}

bool RISCVSigMemcpyExpand::runOnModule(Module &M) {
  // Check if we should run this pass
  auto &TPC = getAnalysis<TargetPassConfig>();
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
  
  Mod = &M;
  DL = &M.getDataLayout();
  Ctx = &M.getContext();
  HelperFuncCache.clear();
  
  bool Changed = false;
  
  //===--------------------------------------------------------------------===//
  // Phase 1: Type Recovery and Memcpy/Memset Conversion
  //===--------------------------------------------------------------------===//
  
  LLVM_DEBUG(dbgs() << "=== Phase 1: Type Recovery ===\n");
  
  //===--------------------------------------------------------------------===//
  // Phase 2: Collect and Expand SigMemcpy/SigMemset Intrinsics
  //===--------------------------------------------------------------------===//
  
  LLVM_DEBUG(dbgs() << "=== Phase 2: Expand SigMemcpy/SigMemset ===\n");
  
  // Collect all sigmemcpy and sigmemset calls (including newly converted ones)
  SmallVector<CallInst *, 16> SigMemcpyCalls;
  SmallVector<CallInst *, 16> SigMemsetCalls;
  
  SmallVector<Function*, 0> FunctionsToProcess;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    
    bool HasSigMemcpyOrMemset = false;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *II = dyn_cast<CallInst>(&I)) {
          // Check by intrinsic name since it's an overloaded intrinsic
          Function* call_func = II->getCalledFunction();
          if (!call_func)
            continue;
          StringRef Name = call_func->getName();
          if (Name.starts_with("llvm.memcpy") || Name == "memcpy") {
            SigMemcpyCalls.push_back(II);
            HasSigMemcpyOrMemset = true;
          } else if (Name.starts_with("llvm.memset") || Name == "memset") {
            SigMemsetCalls.push_back(II);
            HasSigMemcpyOrMemset = true;
          }
        }
      }
    }

    if (HasSigMemcpyOrMemset) {
      FunctionsToProcess.push_back(&F);
    }
  }
  
  LLVM_DEBUG(dbgs() << "Found " << SigMemcpyCalls.size() << " sigmemcpy calls\n");
  LLVM_DEBUG(dbgs() << "Found " << SigMemsetCalls.size() << " sigmemset calls\n");

  if (SigMemcpyCalls.empty() && SigMemsetCalls.empty()) {
    LLVM_DEBUG(dbgs() << "No sigmemcpy or sigmemset calls found, skipping expansion\n");
    return false;
  }

  // Run type recovery to infer types for all values
  runTypeRecovery(FunctionsToProcess);
  
  // Expand each sigmemcpy call
  for (CallInst *II : SigMemcpyCalls) {
    Changed |= expandSigMemcpy(II);
  }
  
  // Expand each sigmemset call
  for (CallInst *II : SigMemsetCalls) {
    Changed |= expandSigMemset(II);
  }
  
  // Clean up type recovery
  TR.reset();
  
  return Changed;
}

ModulePass *llvm::createRISCVSigMemcpyExpandPass() {
  return new RISCVSigMemcpyExpand();
}

//===----------------------------------------------------------------------===//
// Inline Expansion for Small Copies
//===----------------------------------------------------------------------===//

SmallVector<uint64_t, 8> RISCVSigMemcpyExpand::getPointerOffsets(Type *Ty) {
  SmallVector<uint64_t, 8> Offsets;
  
  if (Ty->isPointerTy()) {
    // Single pointer at offset 0
    Offsets.push_back(0);
  } else if (auto *ST = dyn_cast<StructType>(Ty)) {
    // Struct: get offsets of all pointer fields
    const StructLayout *SL = DL->getStructLayout(ST);
    for (unsigned I = 0; I < ST->getNumElements(); I++) {
      Type *FieldTy = ST->getElementType(I);
      uint64_t FieldOffset = SL->getElementOffset(I);
      
      // Recursively get pointer offsets within this field
      SmallVector<uint64_t, 8> FieldOffsets = getPointerOffsets(FieldTy);
      for (uint64_t Off : FieldOffsets) {
        Offsets.push_back(FieldOffset + Off);
      }
    }
  } else if (auto *AT = dyn_cast<ArrayType>(Ty)) {
    // Array: get offsets for each element
    Type *ElemTy = AT->getElementType();
    uint64_t ElemSize = DL->getTypeAllocSize(ElemTy);
    uint64_t NumElems = AT->getNumElements();
    
    SmallVector<uint64_t, 8> ElemOffsets = getPointerOffsets(ElemTy);
    for (uint64_t I = 0; I < NumElems; I++) {
      for (uint64_t Off : ElemOffsets) {
        Offsets.push_back(I * ElemSize + Off);
      }
    }
  }
  // Basic types (int, float, etc.) have no pointer offsets
  
  return Offsets;
}

SmallVector<uint64_t, 8> RISCVSigMemcpyExpand::getBlockPointerOffsets(
    Type *Ty, uint64_t Num) {
  SmallVector<uint64_t, 8> Offsets;
  
  uint64_t ElemSize = DL->getTypeAllocSize(Ty);
  SmallVector<uint64_t, 8> SingleOffsets = getPointerOffsets(Ty);
  
  for (uint64_t I = 0; I < Num; I++) {
    for (uint64_t Off : SingleOffsets) {
      Offsets.push_back(I * ElemSize + Off);
    }
  }
  
  return Offsets;
}

void RISCVSigMemcpyExpand::generateBlockCopyByOffset(
    IRBuilder<> &Builder, Value *Dest, Value *Src,
    uint64_t TotalBytes, const SmallVector<uint64_t, 8> &PtrOffsets,
    unsigned DestAS, unsigned SrcAS) {
  
  Type *I8Ty = Builder.getInt8Ty();
  Type *I16Ty = Builder.getInt16Ty();
  Type *I32Ty = Builder.getInt32Ty();
  Type *I64Ty = Builder.getInt64Ty();
  Type *SrcPtrTy = PointerType::get(*Ctx, SrcAS);
  Type *DestPtrTy = PointerType::get(*Ctx, DestAS);

  uint64_t DataIdx = 0;
  size_t PtrIdx = 0;
  
  // Phase 1: Copy 8-byte chunks, handling pointers specially
  while (DataIdx + 8 <= TotalBytes) {
    Value *SrcPtr = Builder.CreateConstGEP1_64(I8Ty, Src, DataIdx, "src.byte");
    Value *DestPtr = Builder.CreateConstGEP1_64(I8Ty, Dest, DataIdx, "dest.byte");
    
    // Check if current position is a pointer offset
    if (PtrIdx < PtrOffsets.size() && DataIdx == PtrOffsets[PtrIdx]) {
      // Load pointer from source (opaque pointers - no cast needed)
      Value *LoadedPtr = Builder.CreateLoad(SrcPtrTy, SrcPtr, "ptr.load");
      
      // Cast to destination address space if needed
      Value *StoredPtr = LoadedPtr;
      if (SrcAS != DestAS) {
        StoredPtr = Builder.CreateAddrSpaceCast(LoadedPtr, DestPtrTy, "ptr.cast");
      }
      
      Builder.CreateStore(StoredPtr, DestPtr);
      
      PtrIdx++;
    } else {
      Value *Val = Builder.CreateLoad(I64Ty, SrcPtr, "i64.load");
      Builder.CreateStore(Val, DestPtr);
    }
    
    DataIdx += 8;
  }
  
  // Phase 2: Copy remaining 4-byte chunks
  while (DataIdx + 4 <= TotalBytes) {
    Value *SrcPtr = Builder.CreateConstGEP1_64(I8Ty, Src, DataIdx, "src.byte");
    Value *DestPtr = Builder.CreateConstGEP1_64(I8Ty, Dest, DataIdx, "dest.byte");

    Value *Val = Builder.CreateLoad(I32Ty, SrcPtr, "i32.load");
    Builder.CreateStore(Val, DestPtr);

    DataIdx += 4;
  }
  
  // Phase 3: Copy remaining 2-byte chunks
  while (DataIdx + 2 <= TotalBytes) {
    Value *SrcPtr = Builder.CreateConstGEP1_64(I8Ty, Src, DataIdx, "src.byte");
    Value *DestPtr = Builder.CreateConstGEP1_64(I8Ty, Dest, DataIdx, "dest.byte");

    Value *Val = Builder.CreateLoad(I16Ty, SrcPtr, "i16.load");
    Builder.CreateStore(Val, DestPtr);

    DataIdx += 2;
  }
  
  // Phase 4: Copy remaining 1-byte chunks
  while (DataIdx + 1 <= TotalBytes) {
    Value *SrcPtr = Builder.CreateConstGEP1_64(I8Ty, Src, DataIdx, "src.byte");
    Value *DestPtr = Builder.CreateConstGEP1_64(I8Ty, Dest, DataIdx, "dest.byte");
    
    Value *Val = Builder.CreateLoad(I8Ty, SrcPtr, "i8.load");
    Builder.CreateStore(Val, DestPtr);
    
    DataIdx += 1;
  }
}

void RISCVSigMemcpyExpand::generateBlockCopy(
    IRBuilder<> &Builder, Value *Dest, Value *Src,
    Type *ElemTy, uint64_t NumElems,
    unsigned DestAS, unsigned SrcAS) {
  
  uint64_t ElemSize = DL->getTypeAllocSize(ElemTy);
  uint64_t TotalBytes = ElemSize * NumElems;
  
  SmallVector<uint64_t, 8> PtrOffsets = getBlockPointerOffsets(ElemTy, NumElems);
  generateBlockCopyByOffset(Builder, Dest, Src, TotalBytes, PtrOffsets, 
                            DestAS, SrcAS);
}

void RISCVSigMemcpyExpand::generateBlockZeroByOffset(
    IRBuilder<> &Builder, Value *Dest,
    uint64_t TotalBytes, const SmallVector<uint64_t, 8> &PtrOffsets,
    unsigned DestAS, Value *DummyNull) {
  
  Type *I8Ty = Builder.getInt8Ty();
  Type *I16Ty = Builder.getInt16Ty();
  Type *I32Ty = Builder.getInt32Ty();
  Type *I64Ty = Builder.getInt64Ty();
  
  uint64_t DataIdx = 0;
  size_t PtrIdx = 0;
  
  // Phase 1: Zero 8-byte chunks, handling pointers specially
  while (DataIdx + 8 <= TotalBytes) {
    Value *DestPtr = Builder.CreateConstGEP1_64(I8Ty, Dest, DataIdx, "dest.byte");
    
    // Check if current position is a pointer offset
    if (PtrIdx < PtrOffsets.size() && DataIdx == PtrOffsets[PtrIdx]) {
      // Store DummyNull (xsig_setdummyid(null)) for pointer fields
      Builder.CreateStore(DummyNull, DestPtr);
      PtrIdx++;
    } else {
      Builder.CreateStore(ConstantInt::get(I64Ty, 0), DestPtr);
    }
    
    DataIdx += 8;
  }
  
  // Phase 2: Zero remaining 4-byte chunks
  while (DataIdx + 4 <= TotalBytes) {
    Value *DestPtr = Builder.CreateConstGEP1_64(I8Ty, Dest, DataIdx, "dest.byte");
    Builder.CreateStore(ConstantInt::get(I32Ty, 0), DestPtr);
    DataIdx += 4;
  }
  
  // Phase 3: Zero remaining 2-byte chunks
  while (DataIdx + 2 <= TotalBytes) {
    Value *DestPtr = Builder.CreateConstGEP1_64(I8Ty, Dest, DataIdx, "dest.byte");
    Builder.CreateStore(ConstantInt::get(I16Ty, 0), DestPtr);
    DataIdx += 2;
  }
  
  // Phase 4: Zero remaining 1-byte chunks
  while (DataIdx + 1 <= TotalBytes) {
    Value *DestPtr = Builder.CreateConstGEP1_64(I8Ty, Dest, DataIdx, "dest.byte");
    Builder.CreateStore(ConstantInt::get(I8Ty, 0), DestPtr);
    DataIdx += 1;
  }
}
void RISCVSigMemcpyExpand::generateBlockZero(
    IRBuilder<> &Builder, Value *Dest,
    Type *ElemTy, uint64_t NumElems,
    unsigned DestAS, Value *DummyNull) {
  
  uint64_t ElemSize = DL->getTypeAllocSize(ElemTy);
  uint64_t TotalBytes = ElemSize * NumElems;
  
  SmallVector<uint64_t, 8> PtrOffsets = getBlockPointerOffsets(ElemTy, NumElems);
  generateBlockZeroByOffset(Builder, Dest, TotalBytes, PtrOffsets, 
                            DestAS, DummyNull);
}

//===----------------------------------------------------------------------===//
// Constant Copy Optimization (for raw-to-raw without pointers)
//===----------------------------------------------------------------------===//

Constant *RISCVSigMemcpyExpand::getUnderlyingConstant(Value *V) {
  // Strip addrspacecasts to find the underlying constant
  while (auto *ASC = dyn_cast<AddrSpaceCastOperator>(V)) {
    V = ASC->getOperand(0);
  }
  
  // Check if it's a GEP from a global constant
  if (auto *GEP = dyn_cast<GEPOperator>(V)) {
    Value *Base = GEP->getPointerOperand();
    if (auto *GV = dyn_cast<GlobalVariable>(Base)) {
      // A global marked as constant is guaranteed not to be modified
      if (GV->isConstant() && GV->hasInitializer()) {
        // Try to get the element at the GEP index
        Constant *Init = GV->getInitializer();
        SmallVector<Value *, 4> Indices;
        for (auto *Idx = GEP->idx_begin(); Idx != GEP->idx_end(); ++Idx) {
          Indices.push_back(*Idx);
        }
        if (Constant *Elem = ConstantFoldGetElementPtr(
                Init->getType(), Init, std::nullopt, Indices)) {
          return Elem;
        }
      }
    }
  }
  
  // Direct global variable reference
  if (auto *GV = dyn_cast<GlobalVariable>(V)) {
    if (GV->isConstant() && GV->hasInitializer()) {
      return GV->getInitializer();
    }
  }
  
  // Direct constant (e.g., ConstantAggregateZero, ConstantExpr)
  // These are always safe as they're not backed by mutable storage
  if (auto *C = dyn_cast<Constant>(V)) {
    // But skip GlobalVariables (handled above) and Functions
    if (!isa<GlobalValue>(C)) {
      return C;
    }
  }
  
  return nullptr;
}

bool RISCVSigMemcpyExpand::extractConstantBytes(Constant *C, Type *Ty,
                                                 SmallVectorImpl<uint8_t> &Bytes) {
  uint64_t TySize = DL->getTypeAllocSize(Ty);
  
  if (isa<ConstantAggregateZero>(C) || C->isNullValue()) {
    // All zeros
    Bytes.resize(TySize, 0);
    return true;
  }
  
  if (auto *CDS = dyn_cast<ConstantDataSequential>(C)) {
    // ConstantDataArray or ConstantDataVector
    StringRef RawData = CDS->getRawDataValues();
    Bytes.assign(RawData.begin(), RawData.end());
    // Pad to full type size if needed
    if (Bytes.size() < TySize) {
      Bytes.resize(TySize, 0);
    }
    return true;
  }
  
  if (auto *CI = dyn_cast<ConstantInt>(C)) {
    // Integer constant
    uint64_t Val = CI->getZExtValue();
    uint64_t BitWidth = CI->getBitWidth();
    uint64_t ByteWidth = (BitWidth + 7) / 8;
    Bytes.resize(TySize, 0);
    for (uint64_t I = 0; I < ByteWidth && I < TySize; I++) {
      Bytes[I] = static_cast<uint8_t>((Val >> (I * 8)) & 0xFF);
    }
    return true;
  }
  
  if (auto *CFP = dyn_cast<ConstantFP>(C)) {
    // Floating point constant
    APInt IntVal = CFP->getValueAPF().bitcastToAPInt();
    uint64_t Val = IntVal.getZExtValue();
    uint64_t BitWidth = IntVal.getBitWidth();
    uint64_t ByteWidth = (BitWidth + 7) / 8;
    Bytes.resize(TySize, 0);
    for (uint64_t I = 0; I < ByteWidth && I < TySize; I++) {
      Bytes[I] = static_cast<uint8_t>((Val >> (I * 8)) & 0xFF);
    }
    return true;
  }
  
  if (auto *CA = dyn_cast<ConstantAggregate>(C)) {
    // ConstantArray, ConstantStruct, or ConstantVector
    Bytes.resize(TySize, 0);
    
    if (auto *ST = dyn_cast<StructType>(Ty)) {
      const StructLayout *SL = DL->getStructLayout(ST);
      for (unsigned I = 0; I < CA->getNumOperands(); I++) {
        Constant *Elem = CA->getOperand(I);
        Type *ElemTy = ST->getElementType(I);
        uint64_t Offset = SL->getElementOffset(I);
        
        SmallVector<uint8_t, 64> ElemBytes;
        if (!extractConstantBytes(Elem, ElemTy, ElemBytes)) {
          return false;
        }
        
        for (uint64_t J = 0; J < ElemBytes.size() && Offset + J < TySize; J++) {
          Bytes[Offset + J] = ElemBytes[J];
        }
      }
      return true;
    }
    
    if (auto *AT = dyn_cast<ArrayType>(Ty)) {
      Type *ElemTy = AT->getElementType();
      uint64_t ElemSize = DL->getTypeAllocSize(ElemTy);
      
      for (unsigned I = 0; I < CA->getNumOperands(); I++) {
        Constant *Elem = CA->getOperand(I);
        uint64_t Offset = I * ElemSize;
        
        SmallVector<uint8_t, 64> ElemBytes;
        if (!extractConstantBytes(Elem, ElemTy, ElemBytes)) {
          return false;
        }
        
        for (uint64_t J = 0; J < ElemBytes.size() && Offset + J < TySize; J++) {
          Bytes[Offset + J] = ElemBytes[J];
        }
      }
      return true;
    }
    
    return false;
  }
  
  // Undef - treat as zero
  if (isa<UndefValue>(C)) {
    Bytes.resize(TySize, 0);
    return true;
  }
  
  // Cannot extract bytes from this constant type
  return false;
}

void RISCVSigMemcpyExpand::generateConstantCopy(
    IRBuilder<> &Builder, Value *Dest,
    Constant *SrcConstant, Type *ElemTy, uint64_t NumElems,
    unsigned DestAS) {
  
  uint64_t ElemSize = DL->getTypeAllocSize(ElemTy);
  uint64_t TotalBytes = ElemSize * NumElems;
  
  // Build array type for the full copy
  Type *FullTy = (NumElems == 1) ? ElemTy : ArrayType::get(ElemTy, NumElems);
  
  // Extract constant bytes
  SmallVector<uint8_t, 256> Bytes;
  if (!extractConstantBytes(SrcConstant, FullTy, Bytes)) {
    // Fall back to regular block copy if extraction fails
    LLVM_DEBUG(dbgs() << "  Constant extraction failed, falling back to block copy\n");
    // For raw-to-raw without pointers, use memcpy
    emitMemcpyCall(Builder, Dest, SrcConstant, Builder.getInt64(TotalBytes), DestAS, DestAS);
    return;
  }
  
  // Ensure bytes array is padded to 8-byte alignment for our algorithm
  uint64_t PaddedSize = ((TotalBytes + 7) / 8) * 8;
  Bytes.resize(PaddedSize, 0);
  
  // Convert bytes to 64-bit words (little-endian)
  SmallVector<uint64_t, 32> Words;
  for (uint64_t I = 0; I < PaddedSize; I += 8) {
    uint64_t Word = 0;
    for (uint64_t J = 0; J < 8 && I + J < Bytes.size(); J++) {
      Word |= static_cast<uint64_t>(Bytes[I + J]) << (J * 8);
    }
    Words.push_back(Word);
  }
  
  // Generate stores
  Type *I8Ty = Builder.getInt8Ty();
  Type *I16Ty = Builder.getInt16Ty();
  Type *I32Ty = Builder.getInt32Ty();
  Type *I64Ty = Builder.getInt64Ty();
  
  uint64_t DataIdx = 0;
  
  // Phase 1: Store 8-byte chunks
  while (DataIdx + 8 <= TotalBytes) {
    Value *DestPtr = Builder.CreateConstGEP1_64(I8Ty, Dest, DataIdx, "dest.byte");
    uint64_t WordIdx = DataIdx / 8;
    uint64_t Val = Words[WordIdx];
    Builder.CreateStore(ConstantInt::get(I64Ty, Val), DestPtr);
    DataIdx += 8;
  }
  
  // Get the last partial word for remaining bytes
  uint64_t Remain = 0;
  if (DataIdx < TotalBytes) {
    uint64_t WordIdx = DataIdx / 8;
    if (WordIdx < Words.size()) {
      Remain = Words[WordIdx];
    }
  }
  
  // Phase 2: Store remaining 4-byte chunks
  while (DataIdx + 4 <= TotalBytes) {
    Value *DestPtr = Builder.CreateConstGEP1_64(I8Ty, Dest, DataIdx, "dest.byte");
    uint32_t Val = static_cast<uint32_t>(Remain & 0xFFFFFFFF);
    Builder.CreateStore(ConstantInt::get(I32Ty, Val), DestPtr);
    DataIdx += 4;
    Remain >>= 32;
  }
  
  // Phase 3: Store remaining 2-byte chunks
  while (DataIdx + 2 <= TotalBytes) {
    Value *DestPtr = Builder.CreateConstGEP1_64(I8Ty, Dest, DataIdx, "dest.byte");
    uint16_t Val = static_cast<uint16_t>(Remain & 0xFFFF);
    Builder.CreateStore(ConstantInt::get(I16Ty, Val), DestPtr);
    DataIdx += 2;
    Remain >>= 16;
  }
  
  // Phase 4: Store remaining 1-byte chunks
  while (DataIdx + 1 <= TotalBytes) {
    Value *DestPtr = Builder.CreateConstGEP1_64(I8Ty, Dest, DataIdx, "dest.byte");
    uint8_t Val = static_cast<uint8_t>(Remain & 0xFF);
    Builder.CreateStore(ConstantInt::get(I8Ty, Val), DestPtr);
    DataIdx += 1;
    Remain >>= 8;
  }
}


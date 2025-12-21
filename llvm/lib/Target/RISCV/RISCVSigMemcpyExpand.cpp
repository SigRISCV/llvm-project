//===- RISCVSigMemcpyExpand.cpp - Expand SigMode memcpy intrinsic ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass expands the @llvm.riscv.xsig.memcpy intrinsic into explicit
// member-by-member copy operations. For structures/arrays containing pointers,
// it generates setnewid calls to properly re-sign the pointers.
//
// The intrinsic has the signature:
//   void @llvm.riscv.xsig.memcpy(ptr %dest, ptr %src, metadata !type, i64 %len)
//
// Expansion rules:
// - Basic types (i8, i16, i32, i64, float, double): direct load/store
// - Pointer types: load + setnewid + store (or setrawid based on dest addr space)
// - Struct types: recursively copy each member
// - Array types: loop over elements and copy each
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "RISCVTargetMachine.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-sig-memcpy-expand"
#define RISCV_SIG_MEMCPY_EXPAND_NAME "RISC-V SigMode Memcpy Expansion"

STATISTIC(NumMemcpyExpanded, "Number of sigmemcpy intrinsics expanded");
STATISTIC(NumPointersResigned, "Number of pointers re-signed during copy");

namespace {

// Address space constants
constexpr unsigned AS_SIG = 0;    // Normal address space (sig pointers)
constexpr unsigned AS_RAW = 100;  // Raw address space

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
  const DataLayout *DL = nullptr;
  LLVMContext *Ctx = nullptr;

  // Expand a single sigmemcpy intrinsic call
  bool expandSigMemcpy(IntrinsicInst *II);

  // Copy a single element of the given type
  void copyElement(IRBuilder<> &Builder, Value *Dest, Value *Src, 
                   Type *Ty, unsigned DestAS, unsigned SrcAS);

  // Create a loop to copy array elements
  void copyArrayElements(IRBuilder<> &Builder, Value *Dest, Value *Src,
                         ArrayType *AT, unsigned DestAS, unsigned SrcAS);


  // Get the set ID function based on destination address space
  Function *getSetIDFunction(unsigned DestAS);
};

} // end anonymous namespace

char RISCVSigMemcpyExpand::ID = 0;

INITIALIZE_PASS(RISCVSigMemcpyExpand, DEBUG_TYPE,
                RISCV_SIG_MEMCPY_EXPAND_NAME, false, false)

void RISCVSigMemcpyExpand::copyElement(IRBuilder<> &Builder, Value *Dest, 
                                        Value *Src, Type *Ty,
                                        unsigned DestAS, unsigned SrcAS) {
  if (Ty->isPointerTy()) {
    // get Pointer in src addrspace
    Type* ElementTy = PointerType::get(*Ctx, SrcAS);
    Value *LoadedPtr = Builder.CreateLoad(ElementTy, Src, "ptr.load");
    
    // addrspace cast, setrawid and setdummyid will be used in ISelLowering stage
    Type *GenericPtrTy = PointerType::get(*Ctx, DestAS);
    Value *CastPtr = LoadedPtr;
    if (LoadedPtr->getType() != GenericPtrTy) {
      CastPtr = Builder.CreatePointerCast(LoadedPtr, GenericPtrTy);
    }
    
    // store Pointer in dest addrspace
    Builder.CreateStore(CastPtr, Dest);
  } else if (auto *ST = dyn_cast<StructType>(Ty)) {
    // Struct type: recursively copy each member
    for (unsigned i = 0; i < ST->getNumElements(); i++) {
      Type *ElemTy = ST->getElementType(i);
      Value *DestField = Builder.CreateStructGEP(ST, Dest, i, "dest.field");
      Value *SrcField = Builder.CreateStructGEP(ST, Src, i, "src.field");
      copyElement(Builder, DestField, SrcField, ElemTy, DestAS, SrcAS);
    }
    
  } else if (auto *AT = dyn_cast<ArrayType>(Ty)) {
    // Array type: copy each element
    copyArrayElements(Builder, Dest, Src, AT, DestAS, SrcAS);
  } else {
    // Basic type: direct load/store
    Value *Val = Builder.CreateLoad(Ty, Src, "val.load");
    Builder.CreateStore(Val, Dest);
  }
}

void RISCVSigMemcpyExpand::copyArrayElements(IRBuilder<> &Builder, 
                                              Value *Dest, Value *Src,
                                              ArrayType *AT,
                                              unsigned DestAS, unsigned SrcAS) {
  Type *ElemTy = AT->getElementType();
  uint64_t NumElements = AT->getNumElements();
  
  // For small arrays, unroll the loop
  constexpr uint64_t UnrollThreshold = 8;
  
  if (NumElements <= UnrollThreshold) {
    // Unrolled copy
    for (uint64_t i = 0; i < NumElements; i++) {
      Value *Idx = Builder.getInt64(i);
      Value *DestElem = Builder.CreateGEP(AT, Dest, {Builder.getInt64(0), Idx}, 
                                          "dest.elem");
      Value *SrcElem = Builder.CreateGEP(AT, Src, {Builder.getInt64(0), Idx}, 
                                         "src.elem");
      copyElement(Builder, DestElem, SrcElem, ElemTy, DestAS, SrcAS);
    }
  } else {
    // Generate a loop for larger arrays
    Function *F = Builder.GetInsertBlock()->getParent();
    BasicBlock *PreheaderBB = Builder.GetInsertBlock();
    BasicBlock *LoopBB = BasicBlock::Create(*Ctx, "copy.loop", F);
    BasicBlock *ExitBB = BasicBlock::Create(*Ctx, "copy.exit", F);
    
    // Branch to loop
    Builder.CreateBr(LoopBB);
    
    // Loop body
    Builder.SetInsertPoint(LoopBB);
    PHINode *IdxPhi = Builder.CreatePHI(Builder.getInt64Ty(), 2, "idx");
    IdxPhi->addIncoming(Builder.getInt64(0), PreheaderBB);
    
    // Copy element at current index
    Value *DestElem = Builder.CreateGEP(AT, Dest, {Builder.getInt64(0), IdxPhi}, 
                                        "dest.elem");
    Value *SrcElem = Builder.CreateGEP(AT, Src, {Builder.getInt64(0), IdxPhi}, 
                                       "src.elem");
    copyElement(Builder, DestElem, SrcElem, ElemTy, DestAS, SrcAS);
    
    // Increment index
    Value *NextIdx = Builder.CreateAdd(IdxPhi, Builder.getInt64(1), "idx.next");
    IdxPhi->addIncoming(NextIdx, LoopBB);
    
    // Check loop condition
    Value *Cond = Builder.CreateICmpULT(NextIdx, Builder.getInt64(NumElements), 
                                        "loop.cond");
    Builder.CreateCondBr(Cond, LoopBB, ExitBB);
    
    // Continue in exit block
    Builder.SetInsertPoint(ExitBB);
  }
}

bool RISCVSigMemcpyExpand::expandSigMemcpy(IntrinsicInst *II) {
  Value *Dest = II->getArgOperand(0);
  Value *Src = II->getArgOperand(1);
  // Operand 2 is metadata containing type info
  Value *LenVal = II->getArgOperand(3);
  
  // Get address spaces
  unsigned DestAS = Dest->getType()->getPointerAddressSpace();
  unsigned SrcAS = Src->getType()->getPointerAddressSpace();
  
  LLVM_DEBUG(dbgs() << "Expanding sigmemcpy: dest AS=" << DestAS 
                    << ", src AS=" << SrcAS << "\n");
  
  // Get the element type from metadata
  auto *TypeMD = cast<MetadataAsValue>(II->getArgOperand(2));
  auto *TypeNode = cast<ValueAsMetadata>(TypeMD->getMetadata());
  Type *ElemTy = TypeNode->getType();
  
  if (!ElemTy) {
    LLVM_DEBUG(dbgs() << "  Could not determine element type, skipping\n");
    return false;
  }
  
  LLVM_DEBUG(dbgs() << "  Element type: " << *ElemTy << "\n");
  
  // If the type doesn't contain pointers, we could just use regular memcpy
  // But for safety, we still do member-wise copy
  
  IRBuilder<> Builder(II);
  
  // Get number of elements
  if (auto *LenCI = dyn_cast<ConstantInt>(LenVal)) {
    uint64_t NumElems = LenCI->getZExtValue();
    
    // Handle multiple elements
    if (NumElems == 1) {
      // Single element copy
      copyElement(Builder, Dest, Src, ElemTy, DestAS, SrcAS);
    } else {
      // Multiple elements - create array copy
      ArrayType *AT = ArrayType::get(ElemTy, NumElems);
      copyArrayElements(Builder, Dest, Src, AT, DestAS, SrcAS);
    }
  } else {
    // Dynamic length - need to generate a runtime loop
    // For now, emit a warning and fall back to per-element loop
    LLVM_DEBUG(dbgs() << "  Dynamic length not fully supported yet\n");
    
    // Generate a runtime loop
    Function *F = II->getFunction();
    BasicBlock *OrigBB = II->getParent();
    BasicBlock *LoopBB = BasicBlock::Create(*Ctx, "sigmemcpy.loop", F);
    BasicBlock *ExitBB = OrigBB->splitBasicBlock(II, "sigmemcpy.exit");
    
    // Remove the unconditional branch added by splitBasicBlock
    OrigBB->getTerminator()->eraseFromParent();
    
    Builder.SetInsertPoint(OrigBB);
    Builder.CreateBr(LoopBB);
    
    // Loop body
    Builder.SetInsertPoint(LoopBB);
    PHINode *IdxPhi = Builder.CreatePHI(Builder.getInt64Ty(), 2, "idx");
    IdxPhi->addIncoming(Builder.getInt64(0), OrigBB);
    
    // Calculate element addresses
    uint64_t ElemSize = DL->getTypeAllocSize(ElemTy);
    Value *Offset = Builder.CreateMul(IdxPhi, Builder.getInt64(ElemSize));
    Value *DestElem = Builder.CreateGEP(Builder.getInt8Ty(), Dest, Offset);
    Value *SrcElem = Builder.CreateGEP(Builder.getInt8Ty(), Src, Offset);
    
    // Cast to proper pointer types
    PointerType *ElemPtrTy = PointerType::get(ElemTy, DestAS);
    DestElem = Builder.CreatePointerCast(DestElem, ElemPtrTy);
    ElemPtrTy = PointerType::get(ElemTy, SrcAS);
    SrcElem = Builder.CreatePointerCast(SrcElem, ElemPtrTy);
    
    copyElement(Builder, DestElem, SrcElem, ElemTy, DestAS, SrcAS);
    
    // Increment and check
    Value *NextIdx = Builder.CreateAdd(IdxPhi, Builder.getInt64(1));
    IdxPhi->addIncoming(NextIdx, LoopBB);
    
    Value *Cond = Builder.CreateICmpULT(NextIdx, LenVal);
    Builder.CreateCondBr(Cond, LoopBB, ExitBB);
    
    Builder.SetInsertPoint(&*ExitBB->getFirstInsertionPt());
  }
  
  // Remove the original intrinsic call
  II->eraseFromParent();
  NumMemcpyExpanded++;
  
  return true;
}

bool RISCVSigMemcpyExpand::runOnModule(Module &M) {
  // Check if we should run this pass
  auto &TPC = getAnalysis<TargetPassConfig>();
  const RISCVSubtarget *ST = 
      &TPC.getTM<RISCVTargetMachine>().getSubtarget<RISCVSubtarget>(
          *M.begin());
  
  if (!ST->isSigModeSupport())
    return false;
  
  DL = &M.getDataLayout();
  Ctx = &M.getContext();
  
  bool Changed = false;
  
  // Collect all sigmemcpy calls first (to avoid iterator invalidation)
  SmallVector<IntrinsicInst *, 16> SigMemcpyCalls;
  
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
          if (II->getIntrinsicID() == Intrinsic::riscv_xsig_memcpy) {
            SigMemcpyCalls.push_back(II);
          }
        }
      }
    }
  }
  
  // Expand each sigmemcpy call
  for (IntrinsicInst *II : SigMemcpyCalls) {
    Changed |= expandSigMemcpy(II);
  }
  
  return Changed;
}

ModulePass *llvm::createRISCVSigMemcpyExpandPass() {
  return new RISCVSigMemcpyExpand();
}

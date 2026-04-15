//===- RISCVSigModeIDIsolation.cpp - ID Isolation for SigMode -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass ensures proper ID isolation for SigMode by:
//
// 1. Replacing usage of `ptr null` (ConstantPointerNull) with setnewid calls
//    to ensure null pointers get a proper SigMode ID.
//
// 2. Adding setnewid calls after stack allocations (alloca) to give each
//    stack object a unique SigMode ID.
//
// Transformations:
//
// For null pointers:
//   Before: store ptr null, ptr %addr
//   After:  %0 = inttoptr i64 0 to ptr
//           %1 = call ptr @llvm.riscv.xsig.setnewid(ptr %0)
//           store ptr %1, ptr %addr
//
// For stack allocations:
//   Before: %p = alloca [10 x i32]
//           ... use %p ...
//   After:  %p = alloca [10 x i32]
//           %q = call ptr @llvm.riscv.xsig.setnewid(ptr %p)
//           ... use %q instead of %p ...
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-sigmode-id-isolation"
#define RISCV_SIGMODE_ID_ISOLATION_NAME "RISC-V SigMode ID Isolation"

STATISTIC(NumNullPtrsReplaced, "Number of null pointers replaced with setnewid");
STATISTIC(NumAllocasProcessed, "Number of allocas processed with setnewid");

namespace {

class RISCVSigModeIDIsolation : public ModulePass,
                                 public InstVisitor<RISCVSigModeIDIsolation> {
public:
  static char ID;
  RISCVSigModeIDIsolation() : ModulePass(ID) {}

  StringRef getPassName() const override {
    return RISCV_SIGMODE_ID_ISOLATION_NAME;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.setPreservesCFG();
  }

  bool runOnModule(Module &M) override;

  // Visitor for collecting allocas
  void visitAllocaInst(AllocaInst &AI) { Allocas.push_back(&AI); }

private:
  SmallVector<AllocaInst *, 16> Allocas;
  
  // Process a single function
  bool runOnFunction(Function &F, Module &M);
  
  // Replace ptr null usage in a single function
  bool replaceNullPointers(Function &F, Function *SetNewIDFn, 
                           Type *PtrTy, Type *I64Ty);
  
  // SetNewId for Heap Allocator
  bool processHeapAllocators(Function &F, Function *SetNewIDFn, Type *PtrTy);
  
  // Add setnewid to allocas in a single function
  bool processAllocas(Function &F, Function *SetNewIDFn, Type *PtrTy);
};

} // end anonymous namespace

char RISCVSigModeIDIsolation::ID = 0;

INITIALIZE_PASS(RISCVSigModeIDIsolation, DEBUG_TYPE,
                RISCV_SIGMODE_ID_ISOLATION_NAME, false, false)

ModulePass *llvm::createRISCVSigModeIDIsolationPass() {
  return new RISCVSigModeIDIsolation();
}

bool RISCVSigModeIDIsolation::runOnModule(Module &M) {
  // Check if SigMode is enabled for any function
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
    LLVM_DEBUG(dbgs() << "SigMode not enabled, skipping ID isolation\n");
    return false;
  }
  
  bool Changed = false;
  
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    
    Changed |= runOnFunction(F, M);
  }
  
  return Changed;
}

bool RISCVSigModeIDIsolation::runOnFunction(Function &F, Module &M) {
  bool Changed = false;

  // Get the setnewid intrinsic
  Function *SetDummyIDFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::riscv_xsig_setdummyid);
  LLVMContext &Ctx = M.getContext();
  
  // Get types
  Type *PtrTy = PointerType::get(Ctx, 0);  // ptr in address space 0
  Type *I64Ty = Type::getInt64Ty(Ctx);
  
  // 1. Replace null pointers
  Changed |= replaceNullPointers(F, SetDummyIDFn, PtrTy, I64Ty);

  Function *SetNewIDFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::riscv_xsig_setnewid);
  
  Changed |= processHeapAllocators(F, SetNewIDFn, PtrTy);
  
  // 2. Process allocas - collect them first
  Allocas.clear();
  visit(F);
  
  if (!Allocas.empty()) {
    Changed |= processAllocas(F, SetNewIDFn, PtrTy);
  }
  
  return Changed;
}

bool RISCVSigModeIDIsolation::processHeapAllocators(Function &F, Function *SetNewIDFn, Type *PtrTy) {

  static const char* heap_allocator_list[] = {"malloc", "calloc", "realloc"};

  bool Changed = false;
  
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      // Look for calls to malloc/calloc/realloc
      if (auto *Call = dyn_cast<CallInst>(&I)) {
        Function *Callee = Call->getCalledFunction();
        if (!Callee)
          continue;
        
        StringRef CalleeName = Callee->getName();
        bool is_heap_allocator = false;
        for (const char* name : heap_allocator_list) {
          if (CalleeName == name) {
            is_heap_allocator = true;
            break;
          }
        }
        if (is_heap_allocator) {
          LLVM_DEBUG(dbgs() << "Processing heap allocator call: " << *Call << "\n");

          SmallVector<User *, 4> AllocUsers;
          AddrSpaceCastInst *MatchedASC = nullptr;
          int addrspace_num = 0;
          for (User *U : Call->users()) {
            AllocUsers.push_back(U);
            auto *ASC = dyn_cast<AddrSpaceCastInst>(U);
            if (!ASC)
              continue;
            addrspace_num ++;
            assert(addrspace_num <= 1 &&
                   "heap allocator has multiple addrspacecast users");
            if (ASC->getSrcAddressSpace() != 100 || ASC->getDestAddressSpace() != 0)
              continue;
            MatchedASC = ASC;
          }

          if (!MatchedASC) {
            LLVM_DEBUG(dbgs() << "  Skipping allocator without AS100 to AS0 cast\n");
            continue;
          }

          IRBuilder<> Builder(MatchedASC->getNextNode());
          Builder.SetCurrentDebugLocation(MatchedASC->getDebugLoc());

          Value *NewPtr = Builder.CreateCall(SetNewIDFn, {MatchedASC});
          Value *NewPtrAS100 = Builder.CreateAddrSpaceCast(NewPtr, Call->getType());

          SmallVector<User *, 8> ASCUsers(MatchedASC->users());
          for (User *U : ASCUsers)
            if (U != NewPtr && U != NewPtrAS100)
              U->replaceUsesOfWith(MatchedASC, NewPtr);

          for (User *U : AllocUsers) {
            if (U == MatchedASC || U == NewPtr || U == NewPtrAS100)
              continue;
            U->replaceUsesOfWith(Call, NewPtrAS100);
          }

          if (MatchedASC->use_empty())
            MatchedASC->eraseFromParent();

          Changed = true;
          NumAllocasProcessed++;

          LLVM_DEBUG(dbgs() << "Replaced allocator call result with setnewid: "
                            << *NewPtr << "\n");
        }
      }
    }
  }
  
  return Changed;
}

bool RISCVSigModeIDIsolation::replaceNullPointers(Function &F, 
                                                   Function *SetDummyIDFn,
                                                   Type *PtrTy, Type *I64Ty) {
  bool Changed = false;
  
  // Collect all uses of ConstantPointerNull in this function
  // We collect first to avoid modifying the IR while iterating
  SmallVector<std::pair<Instruction *, unsigned>, 16> NullUses;
  
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      for (unsigned OpIdx = 0; OpIdx < I.getNumOperands(); ++OpIdx) {
        if (isa<ConstantPointerNull>(I.getOperand(OpIdx))) {
          // Skip phi nodes - they require special handling
          // (incoming values must be defined at end of predecessor blocks)
          if (isa<PHINode>(&I))
            continue;
          ConstantPointerNull *NullPtr =  cast<ConstantPointerNull>(I.getOperand(OpIdx));
          Type *OrigPtrTy = NullPtr->getType();
          if (OrigPtrTy->getPointerAddressSpace() == 0) {
            NullUses.push_back({&I, OpIdx});
          }
        }
      }
    }
  }
  
  // Replace each use
  for (auto &[Inst, OpIdx] : NullUses) {
    ConstantPointerNull *NullPtr = 
        cast<ConstantPointerNull>(Inst->getOperand(OpIdx));
    Type *OrigPtrTy = NullPtr->getType();
    
    // Create: %tmp = inttoptr i64 0 to ptr
    IRBuilder<> Builder(Inst);
    
    // Create: %newptr = call ptr @llvm.riscv.xsig.setdummyid(ptr %tmp)
    Value *NewPtr = Builder.CreateCall(SetDummyIDFn, {NullPtr});
    
    // If original type was different pointer type (e.g., different address space),
    // cast back to the original type
    if (OrigPtrTy != PtrTy) {
      NewPtr = Builder.CreateBitOrPointerCast(NewPtr, OrigPtrTy);
    }
    
    // Replace the operand
    Inst->setOperand(OpIdx, NewPtr);
    Changed = true;
    NumNullPtrsReplaced++;
    
    LLVM_DEBUG(dbgs() << "Replaced ptr null in: " << *Inst << "\n");
  }
  
  return Changed;
}

// Check if the alloca's address is cast to address space 100 in any of its uses.
// If so, it's already handled by the address space cast mechanism.
static bool isUsedInAddrSpace100(AllocaInst *AI) {
  for (User *U : AI->users()) {
    if (auto *ASCI = dyn_cast<AddrSpaceCastInst>(U)) {
      if (ASCI->getDestAddressSpace() == 100)
        return true;
    }
    // Also check through bitcasts
    if (auto *BC = dyn_cast<BitCastInst>(U)) {
      for (User *BCU : BC->users()) {
        if (auto *ASCI = dyn_cast<AddrSpaceCastInst>(BCU)) {
          if (ASCI->getDestAddressSpace() == 100)
            return true;
        }
      }
    }
  }
  return false;
}

// Check if the allocated type is a struct, union (also represented as struct in LLVM),
// or array type that needs ID isolation
static bool needsIDIsolation(AllocaInst *AI) {
  Type *AllocatedTy = AI->getAllocatedType();

  if (!(AllocatedTy->containsPointer()))
    return false;

  // Array types need isolation
  if (AllocatedTy->isArrayTy())
    return true;
  
  // Struct types (including unions) need isolation
  if (AllocatedTy->isStructTy())
    return true;
  
  if (ConstantInt *CI = dyn_cast<ConstantInt>(AI->getArraySize())) {
    if (CI->getZExtValue() > 1) {
      return true;
    }
  } else {
    // Non-constant array size with pointer element type
    return true;
  }

    // Scalar types (int, float, ptr, etc.) don't need isolation
  return false;
}

bool RISCVSigModeIDIsolation::processAllocas(Function &F, Function *SetNewIDFn,
                                              Type *PtrTy) {
  bool Changed = false;
  
  for (AllocaInst *AI : Allocas) {
    // Skip allocas with no uses
    if (AI->use_empty())
      continue;
    
    // Only process struct/union/array types
    if (!needsIDIsolation(AI)) {
      LLVM_DEBUG(dbgs() << "Skipping non-aggregate alloca: " << *AI << "\n");
      continue;
    }
    
    // Skip if address is cast to address space 100
    // (already handled by addrspace cast mechanism)
    if (isUsedInAddrSpace100(AI)) {
      LLVM_DEBUG(dbgs() << "Skipping alloca with addrspace(100) cast: " << *AI << "\n");
      continue;
    }
    
    LLVM_DEBUG(dbgs() << "Processing alloca: " << *AI << "\n");
    
    // Insert setnewid call immediately after the alloca
    IRBuilder<> Builder(AI->getNextNode());
    Builder.SetCurrentDebugLocation(AI->getDebugLoc());
    
    // If alloca type differs from generic ptr, we need to cast
    Value *AllocaPtr = AI;
    
    // Create: %bounded = call ptr @llvm.riscv.xsig.setnewid(ptr %alloca)
    Value *BoundedPtr = Builder.CreateCall(SetNewIDFn, {AllocaPtr});
    
    // Replace all uses of the alloca with the bounded pointer
    // But we need to skip the setnewid call itself and any casts we just created
    SmallVector<Use *, 16> UsesToReplace;
    for (Use &U : AI->uses()) {
      Instruction *User = cast<Instruction>(U.getUser());
      // Skip the instructions we just inserted
      if (User == AllocaPtr || User == BoundedPtr)
        continue;
      if (isa<CallInst>(User)) {
        CallInst *CI = cast<CallInst>(User);
        if (!CI->getCalledFunction())
          continue;
        if (CI->getCalledFunction()->getName().starts_with("llvm.lifetime.start"))
          continue;
        if (CI->getCalledFunction()->getName().starts_with("llvm.lifetime.end"))
          continue;
      }
      UsesToReplace.push_back(&U);
    }
    
    for (Use *U : UsesToReplace) {
      U->set(BoundedPtr);
    }
    
    Changed = true;
    NumAllocasProcessed++;
    
    LLVM_DEBUG(dbgs() << "  Bounded alloca with setnewid: " << *BoundedPtr << "\n");
  }
  
  return Changed;
}

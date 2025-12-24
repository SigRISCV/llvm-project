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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-sig-memcpy-expand"
#define RISCV_SIG_MEMCPY_EXPAND_NAME "RISC-V SigMode Memcpy/Memset Expansion"

STATISTIC(NumMemcpyExpanded, "Number of sigmemcpy intrinsics expanded");
STATISTIC(NumMemsetExpanded, "Number of sigmemset intrinsics expanded");
STATISTIC(NumHelperFuncsCreated, "Number of helper functions created");

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
  
  // Cache of generated helper functions: name -> Function*
  StringMap<Function *> HelperFuncCache;

  //===--------------------------------------------------------------------===//
  // Sigmemcpy functions
  //===--------------------------------------------------------------------===//
  
  // Expand a single sigmemcpy intrinsic call
  bool expandSigMemcpy(IntrinsicInst *II);

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
  bool expandSigMemset(IntrinsicInst *II);

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

  // Get mangled name for a type
  std::string getMangledTypeName(Type *Ty);

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
};

} // end anonymous namespace

char RISCVSigMemcpyExpand::ID = 0;

INITIALIZE_PASS(RISCVSigMemcpyExpand, DEBUG_TYPE,
                RISCV_SIG_MEMCPY_EXPAND_NAME, false, false)

void RISCVSigMemcpyExpand::emitMemcpyCall(IRBuilder<> &Builder, 
                                           Value *Dest, Value *Src,
                                           Value* Size, 
                                           unsigned DestAS, unsigned SrcAS) {
  // Get the appropriate pointer types for the address spaces
  Type *DestPtrTy = PointerType::get(*Ctx, DestAS);
  Type *SrcPtrTy = PointerType::get(*Ctx, SrcAS);
  
  // Cast pointers if needed
  Value *DestCast = Builder.CreatePointerCast(Dest, DestPtrTy);
  Value *SrcCast = Builder.CreatePointerCast(Src, SrcPtrTy);
  
  // Call llvm.memcpy.p<dest>.p<src>.i64
  Builder.CreateMemCpy(DestCast, MaybeAlign(1), SrcCast, MaybeAlign(1), Size);
}

void RISCVSigMemcpyExpand::emitMemsetCall(IRBuilder<> &Builder, 
                                           Value *Dest, Value* Size, 
                                           unsigned DestAS) {
  // Get the appropriate pointer type for the address space
  Type *DestPtrTy = PointerType::get(*Ctx, DestAS);
  
  // Cast pointer if needed
  Value *DestCast = Builder.CreatePointerCast(Dest, DestPtrTy);
  
  // Call llvm.memset.p<dest>.i64 with zero
  Builder.CreateMemSet(DestCast, Builder.getInt8(0), Size, MaybeAlign(1));
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
  auto ArgIt = F->arg_begin();
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
  BasicBlock *EntryBB = BasicBlock::Create(*Ctx, "entry", F);
  BasicBlock *LoopBB = BasicBlock::Create(*Ctx, "loop", F);
  BasicBlock *BodyBB = BasicBlock::Create(*Ctx, "body", F);
  BasicBlock *ExitBB = BasicBlock::Create(*Ctx, "exit", F);
  
  auto ArgIt = F->arg_begin();
  Value *Dest = &*ArgIt++;
  Value *Src = &*ArgIt++;
  Value *Len = &*ArgIt;
  
  IRBuilder<> Builder(EntryBB);
  
  // Entry: check if len > 0, if not, skip to exit
  Value *LenGtZero = Builder.CreateICmpUGT(Len, Builder.getInt64(0), "len.gt.zero");
  Builder.CreateCondBr(LenGtZero, LoopBB, ExitBB);
  
  // Loop header with PHI for index
  Builder.SetInsertPoint(LoopBB);
  PHINode *IdxPhi = Builder.CreatePHI(Builder.getInt64Ty(), 2, "idx");
  IdxPhi->addIncoming(Builder.getInt64(0), EntryBB);
  Builder.CreateBr(BodyBB);
  
  // Loop body: copy one element
  Builder.SetInsertPoint(BodyBB);
  
  // Calculate element addresses using GEP
  Value *DestElem = Builder.CreateGEP(ElemTy, Dest, IdxPhi, "dest.elem");
  Value *SrcElem = Builder.CreateGEP(ElemTy, Src, IdxPhi, "src.elem");
  
  // Copy the element (may generate calls to other helper functions)
  copyElementInFunc(Builder, DestElem, SrcElem, ElemTy, DestAS, SrcAS);
  
  // Increment index and check loop condition
  Value *NextIdx = Builder.CreateAdd(IdxPhi, Builder.getInt64(1), "idx.next");
  IdxPhi->addIncoming(NextIdx, Builder.GetInsertBlock());
  
  Value *Continue = Builder.CreateICmpULT(NextIdx, Len, "loop.cond");
  Builder.CreateCondBr(Continue, LoopBB, ExitBB);
  
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

/// Look up type from module-level named metadata by type ID.
/// The named metadata format: !sigmemcpy.types = !{!0, !1, ...}
///   !0 = !{i64 0, %struct.Type undef}
/// Returns nullptr if not found.
static Type *getTypeFromNamedMetadata(Module *M, StringRef MetadataName, 
                                       uint64_t TypeId) {
  NamedMDNode *TypeRegistry = M->getNamedMetadata(MetadataName);
  if (!TypeRegistry)
    return nullptr;
  
  for (unsigned i = 0; i < TypeRegistry->getNumOperands(); ++i) {
    MDNode *Entry = TypeRegistry->getOperand(i);
    if (Entry->getNumOperands() >= 2) {
      // Entry format: !{i64 ID, %Type undef}
      if (auto *IdMD = dyn_cast<ConstantAsMetadata>(Entry->getOperand(0))) {
        if (auto *IdConst = dyn_cast<ConstantInt>(IdMD->getValue())) {
          if (IdConst->getZExtValue() == TypeId) {
            // Found the entry with matching ID
            if (auto *TypeMD = dyn_cast<ValueAsMetadata>(Entry->getOperand(1))) {
              return TypeMD->getType();
            }
          }
        }
      }
    }
  }
  return nullptr;
}

bool RISCVSigMemcpyExpand::expandSigMemcpy(IntrinsicInst *II) {
  Value *Dest = II->getArgOperand(0);
  Value *Src = II->getArgOperand(1);
  Value *LenVal = II->getArgOperand(2);
  Value *TypeIdVal = II->getArgOperand(3);  // New: type ID parameter
  
  // Get address spaces
  unsigned DestAS = Dest->getType()->getPointerAddressSpace();
  unsigned SrcAS = Src->getType()->getPointerAddressSpace();
  
  LLVM_DEBUG(dbgs() << "Expanding sigmemcpy: dest AS=" << DestAS 
                    << ", src AS=" << SrcAS << "\n");
  
  // Get the element type from module-level named metadata using type ID
  auto *TypeIdConst = dyn_cast<ConstantInt>(TypeIdVal);
  assert(TypeIdConst && "llvm.riscv.xsig.memcpy type_id must be constant");
  uint64_t TypeId = TypeIdConst->getZExtValue();
  
  Type *ElemTy = getTypeFromNamedMetadata(Mod, "sigmemcpy.types", TypeId);
  assert(ElemTy && "Could not find type in !sigmemcpy.types metadata");
  
  LLVM_DEBUG(dbgs() << "  Element type (ID=" << TypeId << "): " << *ElemTy << "\n");
  
  IRBuilder<> Builder(II);
  
  // Optimization 1: For raw-to-raw copy, just use regular memcpy
  // Optimization 2: For types without pointers, just use regular memcpy
  bool IsRawToRaw = (DestAS == RawAS && SrcAS == RawAS);
  bool HasPointers = ElemTy->containsPointer();
  
  if (IsRawToRaw || !HasPointers) {
    LLVM_DEBUG(dbgs() << "  Using llvm.memcpy (raw-to-raw=" << IsRawToRaw 
                      << ", has_pointers=" << HasPointers << ")\n");
    
    uint64_t ElemSize = DL->getTypeAllocSize(ElemTy);
    
    // Calculate total size
    if (auto *LenCI = dyn_cast<ConstantInt>(LenVal)) {
      uint64_t TotalSize = ElemSize * LenCI->getZExtValue();
      emitMemcpyCall(Builder, Dest, Src, Builder.getInt64(TotalSize), DestAS, SrcAS);
    } else {
      // Dynamic length: compute size at runtime
      Value *TotalSize = Builder.CreateMul(LenVal, Builder.getInt64(ElemSize));
      emitMemcpyCall(Builder, Dest, Src, TotalSize, DestAS, SrcAS);
    }
  } else {
    // Need to generate helper function for pointer re-signing
    Function *HelperF = getOrCreateMemcpyFunc(ElemTy, DestAS, SrcAS);
    Builder.CreateCall(HelperF, {Dest, Src, LenVal});
  }
  
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
  
  BasicBlock *EntryBB = BasicBlock::Create(*Ctx, "entry", F);
  BasicBlock *LoopBB = BasicBlock::Create(*Ctx, "loop", F);
  BasicBlock *BodyBB = BasicBlock::Create(*Ctx, "body", F);
  BasicBlock *ExitBB = BasicBlock::Create(*Ctx, "exit", F);
  
  auto *ArgIt = F->arg_begin();
  Value *Dest = &*ArgIt++;
  Value *Len = &*ArgIt;
  
  IRBuilder<> Builder(EntryBB);
  
  // Pre-compute xsig_setdummyid(null) once at function entry
  // This value will be reused for all pointer field initializations
  Type *DefaultPtrTy = PointerType::get(*Ctx, 0);
  Value *NullPtr = ConstantPointerNull::get(cast<PointerType>(DefaultPtrTy));
  
  // Call xsig_setdummyid(null) once
  FunctionType *SetDummyIdTy = FunctionType::get(DefaultPtrTy, {DefaultPtrTy}, false);
  FunctionCallee SetDummyIdFn = Mod->getOrInsertFunction(
      "llvm.riscv.xsig.setdummyid", SetDummyIdTy);
  Value *DummyNull = Builder.CreateCall(SetDummyIdFn, {NullPtr}, "dummy.null");
  
  // Entry: check if len > 0, if not, skip to exit
  Value *LenGtZero = Builder.CreateICmpUGT(Len, Builder.getInt64(0), "len.gt.zero");
  Builder.CreateCondBr(LenGtZero, LoopBB, ExitBB);
  
  // Loop header with PHI for index
  Builder.SetInsertPoint(LoopBB);
  PHINode *IdxPhi = Builder.CreatePHI(Builder.getInt64Ty(), 2, "idx");
  IdxPhi->addIncoming(Builder.getInt64(0), EntryBB);
  Builder.CreateBr(BodyBB);
  
  // Loop body: zero one element
  Builder.SetInsertPoint(BodyBB);
  
  // Calculate element address using GEP
  Value *DestElem = Builder.CreateGEP(ElemTy, Dest, IdxPhi, "dest.elem");
  
  // Zero the element (may generate calls to other helper functions)
  zeroElementInFunc(Builder, DestElem, ElemTy, DestAS, DummyNull);
  
  // Increment index and check loop condition
  Value *NextIdx = Builder.CreateAdd(IdxPhi, Builder.getInt64(1), "idx.next");
  IdxPhi->addIncoming(NextIdx, Builder.GetInsertBlock());
  
  Value *Continue = Builder.CreateICmpULT(NextIdx, Len, "loop.cond");
  Builder.CreateCondBr(Continue, LoopBB, ExitBB);
  
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

bool RISCVSigMemcpyExpand::expandSigMemset(IntrinsicInst *II) {
  Value *Dest = II->getArgOperand(0);
  Value *LenVal = II->getArgOperand(1);
  Value *TypeIdVal = II->getArgOperand(2);  // New: type ID parameter
  
  // Get address space
  unsigned DestAS = Dest->getType()->getPointerAddressSpace();
  
  LLVM_DEBUG(dbgs() << "Expanding sigmemset: dest AS=" << DestAS << "\n");
  
  // Get the element type from module-level named metadata using type ID
  auto *TypeIdConst = dyn_cast<ConstantInt>(TypeIdVal);
  assert(TypeIdConst && "llvm.riscv.xsig.memset type_id must be constant");
  uint64_t TypeId = TypeIdConst->getZExtValue();
  
  Type *ElemTy = getTypeFromNamedMetadata(Mod, "sigmemset.types", TypeId);
  assert(ElemTy && "Could not find type in !sigmemset.types metadata");
  
  LLVM_DEBUG(dbgs() << "  Element type (ID=" << TypeId << "): " << *ElemTy << "\n");
  
  IRBuilder<> Builder(II);
  
  // Optimization 1: For raw address space, just use regular memset
  // Optimization 2: For types without pointers, just use regular memset
  bool IsRaw = (DestAS == RawAS);
  bool HasPointers = ElemTy->containsPointer();
  
  if (IsRaw || !HasPointers) {
    LLVM_DEBUG(dbgs() << "  Using llvm.memset (raw=" << IsRaw 
                      << ", has_pointers=" << HasPointers << ")\n");
    
    uint64_t ElemSize = DL->getTypeAllocSize(ElemTy);
    
    // Calculate total size
    if (auto *LenCI = dyn_cast<ConstantInt>(LenVal)) {
      uint64_t TotalSize = ElemSize * LenCI->getZExtValue();
      emitMemsetCall(Builder, Dest, Builder.getInt64(TotalSize), DestAS);
    } else {
      // Dynamic length: compute size at runtime
      Value *TotalSize = Builder.CreateMul(LenVal, Builder.getInt64(ElemSize));
      emitMemsetCall(Builder, Dest, TotalSize, DestAS);
    }
  } else {
    // Need to generate helper function for proper pointer zeroing
    Function *HelperF = getOrCreateMemsetFunc(ElemTy, DestAS);
    Builder.CreateCall(HelperF, {Dest, LenVal});
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
  
  // Collect all sigmemcpy and sigmemset calls first (to avoid iterator invalidation)
  SmallVector<IntrinsicInst *, 16> SigMemcpyCalls;
  SmallVector<IntrinsicInst *, 16> SigMemsetCalls;
  
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *II = dyn_cast<IntrinsicInst>(&I)) {
          // Check by intrinsic name since it's an overloaded intrinsic
          StringRef Name = II->getCalledFunction()->getName();
          if (Name.starts_with("llvm.riscv.xsig.memcpy")) {
            SigMemcpyCalls.push_back(II);
          } else if (Name.starts_with("llvm.riscv.xsig.memset")) {
            SigMemsetCalls.push_back(II);
          }
        }
      }
    }
  }
  
  // Expand each sigmemcpy call
  for (IntrinsicInst *II : SigMemcpyCalls) {
    Changed |= expandSigMemcpy(II);
  }
  
  // Expand each sigmemset call
  for (IntrinsicInst *II : SigMemsetCalls) {
    Changed |= expandSigMemset(II);
  }
  
  return Changed;
}

ModulePass *llvm::createRISCVSigMemcpyExpandPass() {
  return new RISCVSigMemcpyExpand();
}

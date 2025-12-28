//===- RISCVTypeRecovery.cpp - Type Recovery for SigMode Memcpy -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVTypeRecovery.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "riscv-type-recovery"

using namespace llvm;

//===----------------------------------------------------------------------===//
// Type String Generation
//===----------------------------------------------------------------------===//

std::string TypeRecovery::getTypeString(MDNode *MD) {
  if (!MD || MD->getNumOperands() < 1)
    return "";
  
  // Check cache first
  auto CacheIt = MDToTypeString.find(MD);
  if (CacheIt != MDToTypeString.end())
    return CacheIt->second;
  
  // Compute the type string
  std::string Result;
  raw_string_ostream OS(Result);
  
  // Get the LLVM type from first operand
  Type *Ty = getLLVMTypeFromMD(MD);
  if (!Ty) {
    OS << "unknown";
    MDToTypeString[MD] = Result;
    return Result;
  }
  
  // Build type string based on type kind
  if (Ty->isPointerTy()) {
    OS << "ptr";
    assert(Ty->getPointerAddressSpace() == 0);
    // Include pointee info if available
    if (MD->getNumOperands() >= 2) {
      if (auto *PointeeMD = dyn_cast<MDNode>(MD->getOperand(1))) {
        OS << "_to_" << getTypeString(PointeeMD);
      }
    }
  } else if (auto *ST = dyn_cast<StructType>(Ty)) {
    if (ST->hasName()) {
      // Use struct name, sanitize it
      StringRef Name = ST->getName();
      for (char C : Name) {
        OS << (isalnum(C) ? C : '_');
      }
    } else {
      OS << "anon_struct_" << ST->getNumElements();
    }
  } else if (auto *AT = dyn_cast<ArrayType>(Ty)) {
    OS << "arr" << AT->getNumElements();
    // Include element info if available
    if (MD->getNumOperands() >= 2) {
      if (auto *ElemMD = dyn_cast<MDNode>(MD->getOperand(1))) {
        OS << "_of_" << getTypeString(ElemMD);
      }
    }
  } else if (Ty->isIntegerTy()) {
    OS << "i" << Ty->getIntegerBitWidth();
  } else if (Ty->isFloatTy()) {
    OS << "f32";
  } else if (Ty->isDoubleTy()) {
    OS << "f64";
  } else if (Ty->isVoidTy()) {
    OS << "void";
  } else {
    Ty->print(OS);
  }
  
  // Store in cache
  MDToTypeString[MD] = Result;
  
  return Result;
}

void TypeRecovery::registerType(MDNode *MD) {
  if (!MD)
    return;
  
  std::string TypeStr = getTypeString(MD);
  if (TypeStr.empty())
    return;
  
  // Only register if not already present
  if (TypeStringToMD.find(TypeStr) == TypeStringToMD.end()) {
    TypeStringToMD[TypeStr] = MD;
    LLVM_DEBUG(dbgs() << "  Registered type: " << TypeStr << "\n");
  }
}

void TypeRecovery::collectMetadataRecursive(MDNode *MD) {
  if (!MD)
    return;

  if (MDToTypeString.find(MD) != MDToTypeString.end())
    return; // Already collected
  
  // Register this metadata
  registerType(MD);
  
  // Recursively collect from operands
  for (unsigned I = 0; I < MD->getNumOperands(); ++I) {
    if (auto *ChildMD = dyn_cast<MDNode>(MD->getOperand(I))) {
      collectMetadataRecursive(ChildMD);
    }
  }
}

//===----------------------------------------------------------------------===//
// MDNode Helper Functions
//===----------------------------------------------------------------------===//

bool TypeRecovery::isPointerTypeMD(MDNode *MD) {
  if (!MD || MD->getNumOperands() < 1)
    return false;
  
  auto *CAM = dyn_cast<ConstantAsMetadata>(MD->getOperand(0));
  if (!CAM)
    return false;
  
  auto *Undef = dyn_cast<UndefValue>(CAM->getValue());
  if (!Undef)
    return false;
  
  return Undef->getType()->isPointerTy();
}

MDNode *TypeRecovery::getPointeeTypeMD(MDNode *MD) {
  if (!isPointerTypeMD(MD))
    return nullptr;
  
  // Pointer metadata format: !{ptr undef, !pointee_type}
  if (MD->getNumOperands() < 2)
    return nullptr;
  
  return dyn_cast<MDNode>(MD->getOperand(1));
}

MDNode *TypeRecovery::getOrCreatePtrToTypeMD(MDNode *PointeeMD) {
  if (!PointeeMD)
    return nullptr;
  
  // Build the type string for lookup
  std::string TypeStr = "ptr_to_" + getTypeString(PointeeMD);
  
  // Check if already exists
  if (MDNode *Existing = lookupTypeByString(TypeStr))
    return Existing;
  
  // Create new ptr metadata: !{ptr undef, PointeeMD}
  Type *PtrTy = PointerType::get(Ctx, 0);
  UndefValue *PtrUndef = UndefValue::get(PtrTy);
  
  Metadata *Ops[] = {
    ConstantAsMetadata::get(PtrUndef),
    PointeeMD
  };
  
  MDNode *NewMD = MDNode::get(Ctx, Ops);
  TypeStringToMD[TypeStr] = NewMD;
  
  LLVM_DEBUG(dbgs() << "  Created ptr type: " << TypeStr << "\n");
  
  return NewMD;
}

bool TypeRecovery::isStructTypeMD(MDNode *MD) {
  if (!MD || MD->getNumOperands() < 1)
    return false;
  
  auto *CAM = dyn_cast<ConstantAsMetadata>(MD->getOperand(0));
  if (!CAM)
    return false;
  
  auto *Undef = dyn_cast<UndefValue>(CAM->getValue());
  if (!Undef)
    return false;
  
  return Undef->getType()->isStructTy();
}

bool TypeRecovery::isArrayTypeMD(MDNode *MD) {
  if (!MD || MD->getNumOperands() < 1)
    return false;
  
  auto *CAM = dyn_cast<ConstantAsMetadata>(MD->getOperand(0));
  if (!CAM)
    return false;
  
  auto *Undef = dyn_cast<UndefValue>(CAM->getValue());
  if (!Undef)
    return false;
  
  return Undef->getType()->isArrayTy();
}

MDNode *TypeRecovery::getStructFieldTypeMD(MDNode *StructMD, unsigned FieldIdx) {
  if (!isStructTypeMD(StructMD))
    return nullptr;
  
  // Struct metadata format: !{%struct undef, !{field0, field1, ...}}
  if (StructMD->getNumOperands() < 2)
    return nullptr;
  
  auto *FieldsMD = dyn_cast<MDNode>(StructMD->getOperand(1));
  if (!FieldsMD || FieldIdx >= FieldsMD->getNumOperands())
    return nullptr;
  
  return dyn_cast<MDNode>(FieldsMD->getOperand(FieldIdx));
}

MDNode *TypeRecovery::getArrayElementTypeMD(MDNode *ArrayMD) {
  if (!isArrayTypeMD(ArrayMD))
    return nullptr;
  
  // Array metadata format: !{[N x T] undef, !element_type}
  if (ArrayMD->getNumOperands() < 2)
    return nullptr;
  
  return dyn_cast<MDNode>(ArrayMD->getOperand(1));
}

Type *TypeRecovery::getLLVMTypeFromMD(MDNode *MD) {
  if (!MD || MD->getNumOperands() < 1)
    return nullptr;
  
  auto *CAM = dyn_cast<ConstantAsMetadata>(MD->getOperand(0));
  if (!CAM)
    return nullptr;
  
  return CAM->getValue()->getType();
}

//===----------------------------------------------------------------------===//
// Main Entry Point
//===----------------------------------------------------------------------===//

void TypeRecovery::run() {
  LLVM_DEBUG(dbgs() << "=== TypeRecovery: Starting type recovery ===\n");
  
  // Phase 1: Build type string -> MDNode map from all metadata
  buildTypeMap();
  LLVM_DEBUG(dbgs() << "Type map has " << TypeStringToMD.size() << " entries\n");
  
  // Phase 2: Single-pass initialization to collect all typed values
  initialize();
  LLVM_DEBUG(dbgs() << "After initialization: " << Worklist.size() 
                    << " values in worklist\n");
  
  // Phase 3: Iterative propagation
  propagate();
  LLVM_DEBUG(dbgs() << "=== TypeRecovery: Completed ===\n");
  LLVM_DEBUG(dump(dbgs()));
}

//===----------------------------------------------------------------------===//
// Type Map Building
//===----------------------------------------------------------------------===//

void TypeRecovery::buildTypeMap() {
  LLVM_DEBUG(dbgs() << "Building type map from metadata...\n");
  
  // Collect from global variables
  for (GlobalVariable &GV : Mod.globals()) {
    if (MDNode *MD = GV.getMetadata("sigmode.type")) {
      collectMetadataRecursive(MD);
    }
  }
  
  // Collect from functions
  for (Function &F : Mod) {
    if (MDNode *FuncMD = F.getMetadata("sigmode.func")) {
      collectMetadataRecursive(FuncMD);
    }
    
    // Also collect from instructions within functions
    if (F.isDeclaration())
      continue;
    
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (MDNode *MD = I.getMetadata("sigmode.type")) {
          collectMetadataRecursive(MD);
        }
      }
    }
  }
  
  // Collect from named metadata (sigmemcpy.types, etc.)
  if (NamedMDNode *TypeRegistry = Mod.getNamedMetadata("sigmemcpy.types")) {
    for (unsigned I = 0; I < TypeRegistry->getNumOperands(); ++I) {
      collectMetadataRecursive(TypeRegistry->getOperand(I));
    }
  }
}

//===----------------------------------------------------------------------===//
// Single-Pass Initialization
//===----------------------------------------------------------------------===//

void TypeRecovery::initialize() {
  LLVM_DEBUG(dbgs() << "Single-pass initialization...\n");
  
  // Process globals
  for (GlobalVariable &GV : Mod.globals()) {
    if (MDNode *MD = GV.getMetadata("sigmode.type")) {
      addType(&GV, MD);
      Worklist.insert(&GV);
    }
  }
  
  // Process functions (in a single pass)
  for (Function &F : Mod) {
    // Process function metadata for arguments
    if (MDNode *FuncMD = F.getMetadata("sigmode.func")) {
      if (FuncMD->getNumOperands() >= 2) {
        auto *TypesMD = dyn_cast<MDNode>(FuncMD->getOperand(1));
        if (TypesMD) {
          // Skip index 0 (return type), process argument types
          unsigned ArgIdx = 0;
          for (Argument &Arg : F.args()) {
            unsigned MDIdx = ArgIdx + 1;
            if (MDIdx < TypesMD->getNumOperands()) {
              if (auto *ArgTypeMD = dyn_cast<MDNode>(TypesMD->getOperand(MDIdx))) {
                addType(&Arg, ArgTypeMD);
                Worklist.insert(&Arg);
              }
            }
            ArgIdx++;
          }
        }
      }
    }
    
    if (F.isDeclaration())
      continue;
    
    // Single pass through all instructions
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        // Check for sigmode.type metadata (alloca, etc.)
        if (MDNode *MD = I.getMetadata("sigmode.type")) {
          addType(&I, MD);
          Worklist.insert(&I);
          continue;
        }
        
        // GEP can get type info from struct/array metadata
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          Worklist.insert(GEP);
          continue;
        }
        
        // Calls can get type info from sigmode.func metadata or func pointer
        if (auto *CB = dyn_cast<CallBase>(&I)) {
          Worklist.insert(CB);
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Iterative Propagation
//===----------------------------------------------------------------------===//

void TypeRecovery::propagate() {
  unsigned Iteration = 0;
  
  // Compute max iterations based on module size
  // Use number of values in TypeMap as a proxy for complexity
  // Each value can be visited at most a few times before converging
  unsigned MaxIterations = std::max(100U, 
                                    static_cast<unsigned>(TypeMap.size() * 2));
  
  while (!Worklist.empty()) {
    Iteration++;
    LLVM_DEBUG(dbgs() << "Iteration " << Iteration 
                      << ": processing " << Worklist.size() << " values\n");
    
    // Process each value in current worklist
    for (Value *V : Worklist) {
      processValue(V);
    }
    
    // Swap worklists
    Worklist = std::move(NextWorklist);
    NextWorklist.clear();
    
    // Safety check to prevent infinite loops
    if (Iteration > MaxIterations) {
      LLVM_DEBUG(dbgs() << "Warning: TypeRecovery exceeded " << MaxIterations
                        << " iterations (TypeMap size: " << TypeMap.size() << ")\n");
      break;
    }
  }
  
  LLVM_DEBUG(dbgs() << "Converged after " << Iteration << " iterations\n");
}

bool TypeRecovery::processValue(Value *V) {
  bool Changed = false;
  
  // Forward propagation: compute types for users
  for (User *U : V->users()) {
    if (auto *I = dyn_cast<Instruction>(U)) {
      TypeSet NewTypes;
      computeForwardTypes(I, NewTypes);
      
      for (MDNode *MD : NewTypes) {
        if (addType(I, MD)) {
          Changed = true;
          NextWorklist.insert(I);
        }
      }
    }
  }
  
  // Backward propagation
  if (isa<Instruction>(V)) {
    propagateBackward(V);
  }
  
  return Changed;
}

bool TypeRecovery::addType(Value *V, MDNode *MD) {
  if (!MD)
    return false;
  
  auto &TS = TypeMap[V];
  return TS.insert(MD).second;  // Returns true if newly inserted
}

bool TypeRecovery::unionTypes(Value *V, const TypeSet &Other) {
  bool Changed = false;
  for (MDNode *MD : Other) {
    if (addType(V, MD))
      Changed = true;
  }
  return Changed;
}

//===----------------------------------------------------------------------===//
// Forward Propagation
//===----------------------------------------------------------------------===//

void TypeRecovery::computeForwardTypes(Instruction *I, TypeSet &Result) {
  if (auto *LI = dyn_cast<LoadInst>(I)) {
    handleLoad(LI, Result);
  } else if (auto *CI = dyn_cast<CastInst>(I)) {
    handleCast(CI, Result);
  } else if (auto *PHI = dyn_cast<PHINode>(I)) {
    handlePHI(PHI, Result);
  } else if (auto *SI = dyn_cast<SelectInst>(I)) {
    handleSelect(SI, Result);
  } else if (auto *CB = dyn_cast<CallBase>(I)) {
    handleCall(CB, Result);
  } else if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
    handleGEP(GEP, Result);
  }
}

void TypeRecovery::handleLoad(LoadInst *LI, TypeSet &Result) {
  // load ptr, ptr %p  =>  result is pointee of %p's pointee type
  // If %p has type "ptr to T", then load result has type "T"
  // But if T is also a pointer, we need its pointee info
  
  Value *Ptr = LI->getPointerOperand();
  const TypeSet *PtrTypes = getTypeSet(Ptr);
  if (!PtrTypes)
    return;
  
  for (MDNode *PtrMD : *PtrTypes) {
    // PtrMD should be the type of %p (a pointer type)
    // We need the pointee type
    if (MDNode *PointeeMD = getPointeeTypeMD(PtrMD)) {
      Result.insert(PointeeMD);
    }
  }
}

void TypeRecovery::handleCast(CastInst *CI, TypeSet &Result) {
  // BitCast, AddrSpaceCast: inherit source types
  if (!CI->getType()->isPointerTy())
    return;
  
  Value *Src = CI->getOperand(0);
  const TypeSet *SrcTypes = getTypeSet(Src);
  if (SrcTypes) {
    for (MDNode *MD : *SrcTypes) {
      Result.insert(MD);
    }
  }
}

void TypeRecovery::handlePHI(PHINode *PHI, TypeSet &Result) {
  // Union of all incoming values
  for (Value *Inc : PHI->incoming_values()) {
    const TypeSet *IncTypes = getTypeSet(Inc);
    if (IncTypes) {
      for (MDNode *MD : *IncTypes) {
        Result.insert(MD);
      }
    }
  }
}

void TypeRecovery::handleSelect(SelectInst *SI, TypeSet &Result) {
  // Union of true and false values
  Value *TrueVal = SI->getTrueValue();
  Value *FalseVal = SI->getFalseValue();
  
  const TypeSet *TrueTypes = getTypeSet(TrueVal);
  const TypeSet *FalseTypes = getTypeSet(FalseVal);
  
  if (TrueTypes) {
    for (MDNode *MD : *TrueTypes)
      Result.insert(MD);
  }
  if (FalseTypes) {
    for (MDNode *MD : *FalseTypes)
      Result.insert(MD);
  }
}

void TypeRecovery::handleCall(CallBase *CB, TypeSet &Result) {
  // Get return type from function's !sigmode.func metadata
  Function *Callee = CB->getCalledFunction();
  if (!Callee)
    return;
  
  MDNode *FuncMD = Callee->getMetadata("sigmode.func");
  if (!FuncMD || FuncMD->getNumOperands() < 2)
    return;
  
  // Format: !{ptr undef, !{ret_type, param_types...}}
  auto *TypesMD = dyn_cast<MDNode>(FuncMD->getOperand(1));
  if (!TypesMD || TypesMD->getNumOperands() < 1)
    return;
  
  // First element is return type
  if (auto *RetTypeMD = dyn_cast<MDNode>(TypesMD->getOperand(0))) {
    Result.insert(RetTypeMD);
  }
}

void TypeRecovery::handleGEP(GetElementPtrInst *GEP, TypeSet &Result) {
  // GEP computes address of element/field
  // If base is struct*, GEP with constant indices gives field pointer
  
  Value *BasePtr = GEP->getPointerOperand();
  const TypeSet *BaseTypes = getTypeSet(BasePtr);
  if (!BaseTypes)
    return;
  
  for (MDNode *BaseMD : *BaseTypes) {
    // BaseMD is the type of the base pointer (ptr to something)
    MDNode *PointeeMD = getPointeeTypeMD(BaseMD);
    if (!PointeeMD)
      continue;
    
    // For struct GEP with constant indices, compute field type
    if (isStructTypeMD(PointeeMD) && GEP->getNumIndices() >= 2) {
      // First index skips the pointer, second index is struct field
      auto *IdxIt = GEP->idx_begin();
      ++IdxIt;  // Skip first index
      
      if (auto *FieldIdxCI = dyn_cast<ConstantInt>(*IdxIt)) {
        unsigned FieldIdx = FieldIdxCI->getZExtValue();
        if (MDNode *FieldMD = getStructFieldTypeMD(PointeeMD, FieldIdx)) {
          // Result is pointer to field type
          MDNode *ResultMD = getOrCreatePtrToTypeMD(FieldMD);
          if (ResultMD)
            Result.insert(ResultMD);
        }
      }
    } else if (isArrayTypeMD(PointeeMD)) {
      // Array element access
      if (MDNode *ElemMD = getArrayElementTypeMD(PointeeMD)) {
        MDNode *ResultMD = getOrCreatePtrToTypeMD(ElemMD);
        if (ResultMD)
          Result.insert(ResultMD);
      }
    } else {
      // For other cases, result is still ptr to same pointee type
      MDNode *ResultMD = getOrCreatePtrToTypeMD(PointeeMD);
      if (ResultMD)
        Result.insert(ResultMD);
    }
  }
}

//===----------------------------------------------------------------------===//
// Backward Propagation
//===----------------------------------------------------------------------===//

void TypeRecovery::propagateBackward(Value *V) {
  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return;
  
  if (auto *LI = dyn_cast<LoadInst>(I)) {
    backpropLoad(LI);
  } else if (auto *SI = dyn_cast<StoreInst>(I)) {
    backpropStore(SI);
  } else if (auto *PHI = dyn_cast<PHINode>(I)) {
    backpropPHI(PHI);
  } else if (auto *Sel = dyn_cast<SelectInst>(I)) {
    backpropSelect(Sel);
  } else if (auto *CB = dyn_cast<CallBase>(I)) {
    // Check if it's memcpy
    Function *Callee = CB->getCalledFunction();
    if (Callee && (Callee->getName().contains("memcpy") ||
                   Callee->getName().contains("memmove"))) {
      backpropMemcpy(CB);
    } else if (Callee && Callee->getName().contains("memset")) {
      backpropMemset(CB);
    }
  }
}

void TypeRecovery::backpropLoad(LoadInst *LI) {
  // %q = load ptr, ptr %p
  // If %q's type is known (T), then %p's type is ptr-to-T
  
  const TypeSet *ResultTypes = getTypeSet(LI);
  if (!ResultTypes || ResultTypes->empty())
    return;
  
  Value *Ptr = LI->getPointerOperand();
  
  for (MDNode *ResultMD : *ResultTypes) {
    // Create ptr-to-ResultMD
    MDNode *PtrTypeMD = getOrCreatePtrToTypeMD(ResultMD);
    if (PtrTypeMD && addType(Ptr, PtrTypeMD)) {
      NextWorklist.insert(Ptr);
    }
  }
}

void TypeRecovery::backpropStore(StoreInst *SI) {
  // store ptr %val, ptr %dest
  // If %val's type is T, then %dest's type is ptr-to-T
  // If %dest's type is ptr-to-T, then %val's type is T
  
  Value *Val = SI->getValueOperand();
  Value *Dest = SI->getPointerOperand();
  
  // Forward: val type -> dest type
  const TypeSet *ValTypes = getTypeSet(Val);
  if (ValTypes) {
    for (MDNode *ValMD : *ValTypes) {
      MDNode *DestMD = getOrCreatePtrToTypeMD(ValMD);
      if (DestMD && addType(Dest, DestMD)) {
        NextWorklist.insert(Dest);
      }
    }
  }
  
  // Backward: dest type -> val type
  const TypeSet *DestTypes = getTypeSet(Dest);
  if (DestTypes) {
    for (MDNode *DestMD : *DestTypes) {
      if (MDNode *PointeeMD = getPointeeTypeMD(DestMD)) {
        if (addType(Val, PointeeMD)) {
          NextWorklist.insert(Val);
        }
      }
    }
  }
}

void TypeRecovery::backpropPHI(PHINode *PHI) {
  // If PHI result type is known, propagate to all incoming values
  const TypeSet *PHITypes = getTypeSet(PHI);
  if (!PHITypes || PHITypes->empty())
    return;
  
  for (Value *Inc : PHI->incoming_values()) {
    if (unionTypes(Inc, *PHITypes)) {
      NextWorklist.insert(Inc);
    }
  }
}

void TypeRecovery::backpropSelect(SelectInst *SI) {
  // If select result type is known, propagate to true/false values
  const TypeSet *SelTypes = getTypeSet(SI);
  if (!SelTypes || SelTypes->empty())
    return;
  
  Value *TrueVal = SI->getTrueValue();
  Value *FalseVal = SI->getFalseValue();
  
  if (unionTypes(TrueVal, *SelTypes)) {
    NextWorklist.insert(TrueVal);
  }
  if (unionTypes(FalseVal, *SelTypes)) {
    NextWorklist.insert(FalseVal);
  }
}

void TypeRecovery::backpropMemcpy(CallBase *CB) {
  // memcpy(dest, src, len)
  // dest and src should have the same pointee type
  // Merge their types bidirectionally
  
  if (CB->arg_size() < 2)
    return;
  
  Value *Dest = CB->getArgOperand(0);
  Value *Src = CB->getArgOperand(1);
  
  const TypeSet *DestTypes = getTypeSet(Dest);
  const TypeSet *SrcTypes = getTypeSet(Src);
  
  // Merge src types into dest
  if (SrcTypes) {
    if (unionTypes(Dest, *SrcTypes)) {
      NextWorklist.insert(Dest);
    }
  }
  
  // Merge dest types into src
  if (DestTypes) {
    if (unionTypes(Src, *DestTypes)) {
      NextWorklist.insert(Src);
    }
  }
}

void TypeRecovery::backpropMemset(CallBase *CB) {
  // memset(dest, val, len)
  // dest type is useful for expansion
  // No backward propagation needed, just mark for processing
  if (CB->arg_size() < 1)
    return;
  
  Value *Dest = CB->getArgOperand(0);
  const TypeSet *DestTypes = getTypeSet(Dest);
  
  // If we have dest type info, mark the memset for later processing
  if (DestTypes && !DestTypes->empty()) {
    LLVM_DEBUG(dbgs() << "  Memset dest has type info\n");
  }
}

//===----------------------------------------------------------------------===//
// Debug Output
//===----------------------------------------------------------------------===//

void TypeRecovery::dump(raw_ostream &OS) const {
  OS << "=== Type Recovery Results ===\n";
  OS << "Total values with types: " << TypeMap.size() << "\n\n";
  
  for (const auto &Entry : TypeMap) {
    Value *V = Entry.first;
    const TypeSet &TS = Entry.second;
    
    OS << "  ";
    if (V->hasName())
      OS << V->getName();
    else
      V->printAsOperand(OS, false);
    
    OS << " : {";
    bool First = true;
    for (MDNode *MD : TS) {
      if (!First)
        OS << ", ";
      First = false;
      
      // Print the LLVM type from metadata
      if (Type *T = getLLVMTypeFromMD(MD)) {
        T->print(OS);
      } else {
        OS << "?";
      }
    }
    OS << "}\n";
  }
}

void TypeRecovery::dumpFunction(Function &F, raw_ostream &OS) const {
  OS << "=== Type Recovery for " << F.getName() << " ===\n";
  
  // Arguments
  for (Argument &Arg : F.args()) {
    const TypeSet *TS = getTypeSet(&Arg);
    OS << "  Arg " << Arg.getArgNo() << " (";
    Arg.printAsOperand(OS, false);
    OS << "): ";
    
    if (TS && !TS->empty()) {
      OS << "{";
      bool First = true;
      for (MDNode *MD : *TS) {
        if (!First) OS << ", ";
        First = false;
        if (Type *T = getLLVMTypeFromMD(MD))
          T->print(OS);
      }
      OS << "}";
    } else {
      OS << "unknown";
    }
    OS << "\n";
  }
  
  // Instructions
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      const TypeSet *TS = getTypeSet(&I);
      if (!TS || TS->empty())
        continue;
      
      OS << "  ";
      I.printAsOperand(OS, false);
      OS << " (";
      OS << I.getOpcodeName();
      OS << "): {";
      
      bool First = true;
      for (MDNode *MD : *TS) {
        if (!First) OS << ", ";
        First = false;
        if (Type *T = getLLVMTypeFromMD(MD))
          T->print(OS);
      }
      OS << "}\n";
    }
  }
}

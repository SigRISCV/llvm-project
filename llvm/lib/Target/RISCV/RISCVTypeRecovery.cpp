//===- RISCVTypeRecovery.cpp - Type Recovery for SigMode Memcpy -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <filesystem>
#include "RISCVTypeRecovery.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "riscv-type-recovery"

using namespace llvm;

//===----------------------------------------------------------------------===//
// Type String Generation
//===----------------------------------------------------------------------===//
static std::string debugMD(MDNode* MD) {
  std::string Result;
  raw_string_ostream OS(Result);
  OS << "{";
  for (unsigned i = 0; i < MD->getNumOperands(); ++i) {
    if (i > 0) OS << ", ";
    if (auto *SubMD = dyn_cast<MDNode>(MD->getOperand(i))) {
      OS << debugMD(SubMD);
    } else if (auto *MDC = dyn_cast<ConstantAsMetadata>(MD->getOperand(i))) {
      MDC->getValue()->print(OS);
    } else {
      MD->getOperand(i)->print(OS);
    }
  }
  OS << "}";
  return OS.str();
}


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
    OS << getTypeString(dyn_cast<MDNode>(MD->getOperand(0)));
    OS << " func(";
    for (int i = 1; i < MD->getNumOperands(); ++i) {
      if (i > 1) OS << ", ";
      OS << getTypeString(dyn_cast<MDNode>(MD->getOperand(i)));
    }
    OS << ")";
    MDToTypeString[MD] = Result;
    return Result;
  }
  
  // Build type string based on type kind
  if (Ty->isPointerTy()) {
    assert(Ty->getPointerAddressSpace() == 0);
    // Include pointee info if available
    if (MD->getNumOperands() >= 2) {
      if (auto *PointeeMD = dyn_cast<MDNode>(MD->getOperand(1))) {
        OS << getTypeString(PointeeMD);
      }
    } else {
      OS << "void";
    }
    OS << "*";
  } else if (auto *ST = dyn_cast<StructType>(Ty)) {
    if (ST->hasName()) {
      // Use struct name, sanitize it
      StringRef Name = ST->getName();
      for (char C : Name) {
        OS << (isalnum(C) ? C : ' ');
      }
    } else {
      OS << "anon_struct_" << ST->getNumElements();
    }
  } else if (auto *AT = dyn_cast<ArrayType>(Ty)) {
    // Include element info if available
    if (MD->getNumOperands() >= 2) {
      if (auto *ElemMD = dyn_cast<MDNode>(MD->getOperand(1))) {
        OS <<  getTypeString(ElemMD);
      }
    }

    OS << "[" << AT->getNumElements() << "]";
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

std::string TypeRecovery::getLLVMTypeString(Type* Ty) {
  auto CacheIt = LLVMTypeToTypeString.find(Ty);
  if (CacheIt != LLVMTypeToTypeString.end())
    return CacheIt->second;
  
  // Compute the type string
  std::string Result;
  raw_string_ostream OS(Result);
  
  // Get the LLVM type from first operand

  if (auto FT = dyn_cast<FunctionType>(Ty)) {
    OS << getLLVMTypeString(FT->getReturnType());
    OS << " func(";
    for (int i = 0; i < FT->getNumParams(); ++i) {
      if (i > 0) OS << ", ";
      OS << getLLVMTypeString(FT->getParamType(i));
    }
    OS << ")";
    LLVMTypeToTypeString[Ty] = Result;
    return Result;
  }
  
  // Build type string based on type kind
  if (Ty->isPointerTy()) {
    OS << "void*";
  } else if (auto *ST = dyn_cast<StructType>(Ty)) {
    if (ST->hasName()) {
      // Use struct name, sanitize it
      StringRef Name = ST->getName();
      for (char C : Name) {
        OS << (isalnum(C) ? C : ' ');
      }
    } else {
      OS << "anon_struct_" << ST->getNumElements();
    }
  } else if (auto *AT = dyn_cast<ArrayType>(Ty)) {
    // Include element info if available
    OS << getLLVMTypeString(AT->getElementType());
    OS << "[" << AT->getNumElements() << "]";
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
  LLVMTypeToTypeString[Ty] = Result;
  
  return Result;
}

MDNode* TypeRecovery::getMDFromLLVMTType(Type* Ty) {
  auto CacheIt = LLVMTypeToMD.find(Ty);
  if (CacheIt != LLVMTypeToMD.end())
    return CacheIt->second;
  
  std::string TypeStr = getLLVMTypeString(Ty);
  MDNode *MD = lookupTypeByString(TypeStr);;
  if (MD) {
    LLVMTypeToMD[Ty] = MD;
  }
  return MD;
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

  std::string TypeStr = getTypeString(MD);
  if (TypeStringToMD.find(TypeStr) != TypeStringToMD.end())
    return; // Already registered
  
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

bool TypeRecovery::isVoidTypeMD(MDNode *MD) {
  Type* Ty = getLLVMTypeFromMD(MD);
  if (!Ty)
    return false;

  return Ty->isVoidTy();
}

bool TypeRecovery::isUnionTypeMD(MDNode *MD) {
  Type* Ty = getLLVMTypeFromMD(MD);
  if (!Ty)
    return false;

  if (!Ty->isStructTy())
    return false;
  StringRef StructName = Ty->getStructName();
  std::string Name = StructName.str();
  if (Name.substr(0, 6) == "union.")
    return true;
  return false;
}

bool TypeRecovery::isFunctionPointerTypeMD(MDNode *MD) {

  MDNode *PointeeMD = getPointeeTypeMD(MD);
  if (!PointeeMD)
    return false;

  if (PointeeMD->getNumOperands() < 1)
    return false;

  Type* Ty = getLLVMTypeFromMD(PointeeMD);
  if (!Ty)
    return true;

  return false;
}

MDNode* TypeRecovery::getFunctionTypeMD(MDNode *MD) {
  if (!isFunctionPointerTypeMD(MD))
    return nullptr;

  return getPointeeTypeMD(MD);
}

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
  std::string TypeStr = getTypeString(PointeeMD) + "*";
  
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

MDNode *TypeRecovery::getOrCreateArrayOfTypeMD(MDNode *ElementMD, uint64_t NumElements) {
  if (!ElementMD)
    return nullptr;
  
  // Build the type string for lookup
  std::string TypeStr = getTypeString(ElementMD) + "[" + std::to_string(NumElements) + "]";
  
  // Check if already exists
  if (MDNode *Existing = lookupTypeByString(TypeStr))
    return Existing;
  
  // Create new array metadata: !{[N x T] undef, ElementMD}
  Type *ElemTy = getLLVMTypeFromMD(ElementMD);
  if (!ElemTy)
    return nullptr;
  
  ArrayType *ArrTy = ArrayType::get(ElemTy, NumElements);
  UndefValue *ArrUndef = UndefValue::get(ArrTy);
  
  Metadata *Ops[] = {
    ConstantAsMetadata::get(ArrUndef),
    ElementMD
  };
  
  MDNode *NewMD = MDNode::get(Ctx, Ops);
  TypeStringToMD[TypeStr] = NewMD;
  
  LLVM_DEBUG(dbgs() << "  Created array type: " << TypeStr << "\n");
  
  return NewMD;
}

void TypeRecovery::getFieldMDNodeFromOffset(MDNode* MD, uint64_t Offset, TypeSet &typeset) {
  if (!MD)
    return;

  Type* Ty = getLLVMTypeFromMD(MD);
  if (!Ty)
    return;

  uint64_t TypeSize = Mod.getDataLayout().getTypeAllocSize(Ty);
  if (TypeSize == 0)
    return;
  
  Offset = Offset % TypeSize;
  if (Offset == 0) {
    typeset.insert(MD);
  }
  if (Ty->isStructTy()) {
    StructType* ST = dyn_cast<StructType>(Ty);
    const DataLayout& DL = Mod.getDataLayout();
    uint64_t last_field_end = 0;
    for (unsigned i = 0; i < ST->getNumElements(); ++i) {
      Type* FieldTy = ST->getElementType(i);
      uint64_t FieldOffset = DL.getStructLayout(ST)->getElementOffset(i);
      if (Offset >= FieldOffset && Offset < FieldOffset + DL.getTypeAllocSize(FieldTy)) {
        MDNode* FieldMD = getStructFieldTypeMD(MD, i);
        if (FieldMD) {
          getFieldMDNodeFromOffset(FieldMD, Offset - FieldOffset, typeset);
        }
      } else if (Offset >= last_field_end && Offset < FieldOffset) {
        // Padding between last field and this field
        MDNode* Int8MD = lookupTypeByString("i8");
        MDNode* PaddingMD = getOrCreateArrayOfTypeMD(Int8MD, FieldOffset - last_field_end);
        if (PaddingMD) {
          typeset.insert(PaddingMD);
        }
      }
      last_field_end = FieldOffset + DL.getTypeAllocSize(FieldTy);
    }
  } else if (Ty->isArrayTy()) {
    ArrayType* AT = dyn_cast<ArrayType>(Ty);
    const DataLayout& DL = Mod.getDataLayout();
    Type* ElemTy = AT->getElementType();
    uint64_t ElemSize = DL.getTypeAllocSize(ElemTy);
    uint64_t offset = Offset % ElemSize;
    MDNode* ElemMD = getArrayElementTypeMD(MD);
    if (ElemMD) {
      getFieldMDNodeFromOffset(ElemMD, Offset, typeset);
    }
  }
}

Type *TypeRecovery::getLLVMTypeFromMD(MDNode *MD) {
  if (!MD || MD->getNumOperands() < 1)
    return nullptr;
  
  auto *CAM = dyn_cast<ConstantAsMetadata>(MD->getOperand(0));
  if (!CAM)
    return nullptr;

  Constant* Value = CAM->getValue();
  if (Value->isNullValue()) {
    return llvm::Type::getVoidTy(Ctx);
  }
  
  return CAM->getValue()->getType();
}

//===----------------------------------------------------------------------===//
// Main Entry Point
//===----------------------------------------------------------------------===//

void TypeRecovery::run(SmallVector<Function*, 0> FunctionsToProcess) {
  LLVM_DEBUG(dbgs() << "=== TypeRecovery: Starting type recovery ===\n");
  
  // Phase 1: Build type string -> MDNode map from all metadata
  buildTypeMap();
  LLVM_DEBUG(dbgs() << "Type map has " << TypeStringToMD.size() << " entries\n");
  
  // Phase 2: Single-pass initialization to collect all typed values
  int max_iter = initialize(FunctionsToProcess);
  LLVM_DEBUG(dbgs() << "After initialization: " << Worklist.size() 
                    << " values in worklist\n");
  LLVM_DEBUG(dbgs() << "Max iterations: " << max_iter << "\n");

  // Phase 3: Iterative propagation
  propagate(max_iter);
  LLVM_DEBUG(dbgs() << "=== TypeRecovery: Completed ===\n");
  LLVM_DEBUG(dump(dbgs()));
}

//===----------------------------------------------------------------------===//
// Type Map Building
//===----------------------------------------------------------------------===//

/// Create and register a basic type metadata: !{type undef}
MDNode *TypeRecovery::createBasicTypeMD(Type *Ty) {
  UndefValue *Undef = UndefValue::get(Ty);
  Metadata *Ops[] = {ConstantAsMetadata::get(Undef)};
  MDNode *MD = MDNode::get(Ctx, Ops);
  registerType(MD);
  return MD;
}

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
  // if (NamedMDNode *TypeRegistry = Mod.getNamedMetadata("sigmemcpy.types")) {
  //   for (unsigned I = 0; I < TypeRegistry->getNumOperands(); ++I) {
  //     collectMetadataRecursive(TypeRegistry->getOperand(I));
  //   }
  // }
  
  // Register basic types that may not be annotated by frontend
  // These are needed for type propagation through primitive operations
  LLVM_DEBUG(dbgs() << "Registering basic types...\n");
  
  // Integer types: i1, i8, i16, i32, i64
  createBasicTypeMD(Type::getInt1Ty(Ctx));
  createBasicTypeMD(Type::getInt8Ty(Ctx));
  createBasicTypeMD(Type::getInt16Ty(Ctx));
  createBasicTypeMD(Type::getInt32Ty(Ctx));
  createBasicTypeMD(Type::getInt64Ty(Ctx));
  
  // Floating point types: float, double
  createBasicTypeMD(Type::getFloatTy(Ctx));
  createBasicTypeMD(Type::getDoubleTy(Ctx));
  
  // Void pointer (ptr without pointee info)
  createBasicTypeMD(PointerType::get(Ctx, 0));
  
  LLVM_DEBUG(dbgs() << "Type map now has " << TypeStringToMD.size() << " entries\n");
}

//===----------------------------------------------------------------------===//
// Single-Pass Initialization
//===----------------------------------------------------------------------===//

int TypeRecovery::initialize(SmallVector<Function*, 0> FunctionsToProcess) {
  LLVM_DEBUG(dbgs() << "Single-pass initialization...\n");
  
  // Process globals
  for (GlobalVariable &GV : Mod.globals()) {
    if (MDNode *MD = GV.getMetadata("sigmode.type")) {
      addTypeInitStage(&GV, MD);
      Worklist.insert(&GV);
    } else {
      addTypeInitStage(&GV, nullptr);
    }
  }
  
  int max_func_IR_num = 0;
  int func_IR_num = 0;
  // Process functions (in a single pass)
  for (Function* F : FunctionsToProcess) {
    // Process function metadata for arguments
    if (MDNode *FuncMD = F->getMetadata("sigmode.func")) {
      if (FuncMD->getNumOperands() >= 2) {
        auto *TypesMD = dyn_cast<MDNode>(FuncMD->getOperand(1));
        if (TypesMD) {
          // Skip index 0 (return type), process argument types
          unsigned ArgIdx = 0;
          for (Argument &Arg : F->args()) {
            unsigned MDIdx = ArgIdx + 1;
            if (MDIdx < TypesMD->getNumOperands()) {
              if (auto *ArgTypeMD = dyn_cast<MDNode>(TypesMD->getOperand(MDIdx))) {
                addTypeInitStage(&Arg, ArgTypeMD);
                Worklist.insert(&Arg);
              }
            }
            ArgIdx++;
          }
        }
      }
    }
    
    if (F->isDeclaration())
      continue;
    
    // Single pass through all instructions
    for (BasicBlock &BB : *F) {
      for (Instruction &I : BB) {
        if (dyn_cast<CallBase>(&I) || dyn_cast<CastInst>(&I) ||
          dyn_cast<GetElementPtrInst>(&I) || dyn_cast<AllocaInst>(&I) ||
          dyn_cast<LoadInst>(&I) || dyn_cast<StoreInst>(&I) ||
          dyn_cast<PHINode>(&I) || dyn_cast<SelectInst>(&I) ||
          dyn_cast<AddrSpaceCastInst>(&I)) {
          func_IR_num++;
        }

        // Check for sigmode.type metadata (alloca, etc.)
        if (MDNode *MD = I.getMetadata("sigmode.type")) {
          addTypeInitStage(&I, MD);
          Worklist.insert(&I);
          continue;
        }
        
        // For any instruction with an output, initialize with its basic type
        // This ensures every pointer has at least "ptr undef" type
        if (!I.getType()->isVoidTy()) {
          Type *OutputTy = I.getType();
          if (OutputTy->isPointerTy() || OutputTy->isIntegerTy() ||
              OutputTy->isFloatTy() || OutputTy->isDoubleTy()) {
            // Look up or create basic type metadata
            std::string TypeStr = getLLVMTypeString(OutputTy);
            
            MDNode *BasicMD = lookupTypeByString(TypeStr);
            assert(BasicMD && "Failed to find basic type metadata");
            addTypeInitStage(&I, BasicMD);
          } else {
            addTypeInitStage(&I, nullptr); // Non-pointer/non-primitive types
          }
        }
        
        // GEP can get type info from struct/array metadata
        if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          Worklist.insert(GEP);
        } else if (auto *CB = dyn_cast<CallBase>(&I)) {
          Worklist.insert(CB);
        } else if (auto *LD = dyn_cast<LoadInst>(&I)) {
          Worklist.insert(LD);
        } else if (auto *SD = dyn_cast<StoreInst>(&I)) {
          Worklist.insert(SD);
        } else if (auto *ASC = dyn_cast<AddrSpaceCastInst>(&I)) {
          Worklist.insert(ASC);
        }
      }
    }
    if (func_IR_num > max_func_IR_num) {
      max_func_IR_num = func_IR_num;
    }
    func_IR_num = 0;
  }

  return std::max(max_func_IR_num * 2, 100); // set iteration times based on largest function IR size
}

//===----------------------------------------------------------------------===//
// Iterative Propagation
//===----------------------------------------------------------------------===//

void TypeRecovery::propagate(int max_iteration_time) {
  unsigned Iteration = 0;
  
  // Compute max iterations based on module size
  // Use number of values in TypeMap as a proxy for complexity
  // Each value can be visited at most a few times before converging
  unsigned MaxIterations = max_iteration_time;
  
  uint64_t random = std::rand();
  dumpIterationToFile(random);

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
    
    // Dump iteration state to file
    dumpIterationToFile(random);
    
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

  // Backward propagation
  if (isa<Instruction>(V)) {
    Changed |= propagateBackward(V);
  }
  
  // Forward propagation: compute types for users
  if (V->hasUseList()) {
    for (User *U : V->users()) {
      if (auto *I = dyn_cast<Instruction>(U)) {
        // Forward propagation is now done directly in the handlers
        // Each handler calls addType and inserts into NextWorklist internally
        Changed |= computeForwardTypes(I);
      }
    }
  }
  
  return Changed;
}

bool TypeRecovery::addType(Value *V, MDNode *MD) {
  assert(V->getType()->isVoidTy() == false && "Cannot add type to void value");

  if (!MD)
    return false;

  if (!V->getType()->isPointerTy())
    return false;

  if (!isPointerTypeMD(MD))
    return false;

  if (dyn_cast<Constant>(V)) {
    return false;
  }

  auto TSor = TypeMap.find(V);
  if (TSor == TypeMap.end()) {
    return false;
  }

  if (!isPointerTypeMD(MD)) {
    return false;
  } else {
    MDNode *PointeeMD = getPointeeTypeMD(MD);
    if (PointeeMD && PointeeMD != lookupTypeByString("void*")) {
      if (TSor->second.count(PointeeMD) > 0) {
        return false;
      }
    }
  }

  return TSor->second.insert(MD).second;  // Returns true if newly inserted
}

bool TypeRecovery::addTypeInitStage(Value *V, MDNode *MD) {
  auto TSor = TypeMap.find(V);
  if (TSor == TypeMap.end()) {
    TypeMap[V] = TypeSet();
  }
  if (!MD)
    return false;
  return TypeMap[V].insert(MD).second;  // Returns true if newly inserted
}

bool TypeRecovery::unionTypes(Value *V, const TypeSet &Other) {
  if (!V->getType()->isPointerTy())
    return false;

  bool Changed = false;
  for (MDNode *MD : Other) {
    Changed |= addType(V, MD);
  }
  return Changed;
}

//===----------------------------------------------------------------------===//
// Forward Propagation
//===----------------------------------------------------------------------===//

bool TypeRecovery::computeForwardTypes(Instruction *I) {
  bool changed = false;
  if (auto *LI = dyn_cast<LoadInst>(I)) {
    changed = handleLoad(LI);
  } else if (auto *SI = dyn_cast<StoreInst>(I)) {
    changed = handleStore(SI);
  } else if (auto *CI = dyn_cast<CastInst>(I)) {
    changed = handleCast(CI);
  } else if (auto *PHI = dyn_cast<PHINode>(I)) {
    changed = handlePHI(PHI);
  } else if (auto *SI = dyn_cast<SelectInst>(I)) {
    changed = handleSelect(SI);
  } else if (auto *CB = dyn_cast<CallBase>(I)) {
    Function *Callee = CB->getCalledFunction();
    if (Callee && (Callee->getName().contains("xsig.setnewid") ||
                         Callee->getName().contains("xsig.setnewid") || 
                         Callee->getName().contains("xsig.setrawid"))) {
      changed = handleSetIDCall(CB);
    } else if(!Callee) {
      changed = handleCall(CB);
    }
  } else if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
    changed = handleGEP(GEP);
  } else if (auto *ASC = dyn_cast<AddrSpaceCastInst>(I)) {
    changed = handleAddrSpaceCast(ASC);
  }
  return changed;
}

bool TypeRecovery::handleLoad(LoadInst *LI) {
  // load ptr, ptr %p  =>  result is pointee of %p's pointee type
  // If %p has type "ptr to T", then load result has type "T"
  
  if (!LI->getType()->isPointerTy())
    return false;

  Value *Ptr = LI->getPointerOperand();
  const TypeSet *PtrTypes = getTypeSet(Ptr);
  if (!PtrTypes)
    return false;
  
  bool Changed = false;
  for (MDNode *PtrMD : *PtrTypes) {
    // PtrMD should be the type of %p (a pointer type)
    // We need the pointee type
    if (MDNode *PointeeMD = getPointeeTypeMD(PtrMD)) {
      if (!isPointerTypeMD(PointeeMD)) {
        continue;
      }
      if (addType(LI, PointeeMD)) {
        NextWorklist.insert(LI);
        Changed = true;
      }
    }
  }
  return Changed;
}

bool TypeRecovery::handleStore(StoreInst *SI) {
  NextWorklist.insert(SI);
  return true;
}

bool TypeRecovery::handleCast(CastInst *CI) {
  // BitCast, AddrSpaceCast: inherit source types
  if (!CI->getType()->isPointerTy())
    return false;
  
  Value *Src = CI->getOperand(0);
  if (!Src->getType()->isPointerTy())
    return false;

  const TypeSet *SrcTypes = getTypeSet(Src);
  if (SrcTypes) {
    if (unionTypes(CI, *SrcTypes)) {
      NextWorklist.insert(CI);
      return true;
    }
  }
  return false;
}

bool TypeRecovery::handlePHI(PHINode *PHI) {
  // Union of all incoming values
  bool Changed = false;
  for (Value *Inc : PHI->incoming_values()) {
    const TypeSet *IncTypes = getTypeSet(Inc);
    if (IncTypes) {
      if (unionTypes(PHI, *IncTypes)) {
        Changed = true;
      }
    }
  }
  if (Changed) {
    NextWorklist.insert(PHI);
  }
  return Changed;
}

bool TypeRecovery::handleSelect(SelectInst *SI) {
  // Union of true and false values
  Value *TrueVal = SI->getTrueValue();
  Value *FalseVal = SI->getFalseValue();
  
  const TypeSet *TrueTypes = getTypeSet(TrueVal);
  const TypeSet *FalseTypes = getTypeSet(FalseVal);
  
  bool Changed = false;
  if (TrueTypes) {
    if (unionTypes(SI, *TrueTypes))
      Changed = true;
  }
  if (FalseTypes) {
    if (unionTypes(SI, *FalseTypes))
      Changed = true;
  }
  if (Changed) {
    NextWorklist.insert(SI);
  }
  return Changed;
}

bool TypeRecovery::handleCall(CallBase *CB) {
  // memcpy(dest, src, size)
  NextWorklist.insert(CB);
  return true;
}

bool TypeRecovery::handleSetIDCall(CallBase *CB) {
  const TypeSet* SrcTypes = getTypeSet(CB->getArgOperand(0));
  Value* Dest = CB;

  bool Changed = false;
  if (unionTypes(Dest, *SrcTypes)) {
    Changed = true;
  }
  return Changed;
}

bool TypeRecovery::handleGEP(GetElementPtrInst *GEP) {
  // GEP computes address of element/field
  // If base is struct*, GEP with constant indices gives field pointer
  
  Value *BasePtr = GEP->getPointerOperand();
  Type *SourceElemTy = GEP->getSourceElementType();
  const TypeSet *BaseTypes = getTypeSet(BasePtr);

  if (!BaseTypes)
    return false;
  
  // Special case: i8 GEP (byte-level pointer arithmetic)
  // output = getelementptr i8, ptr %p, i64 %offset
  // The result inherits the base pointer's type
  if (SourceElemTy->isIntegerTy(8)) {
    if (BaseTypes) {
      auto Iter = GEP->idx_begin();
      auto Offset = dyn_cast<ConstantInt>(*Iter);
      if (!Offset) {
        if (unionTypes(GEP, *BaseTypes)) {
          NextWorklist.insert(GEP);
          return true;
        }
      } else {
        // If Offset is a constant, we need to handle it
        bool changed = false;
        for (MDNode* BaseMD : *BaseTypes) {
          TypeSet FieldTypes;
          MDNode* BasePointeeMD = getPointeeTypeMD(BaseMD);
          if (!BasePointeeMD)
            continue;
          getFieldMDNodeFromOffset(BasePointeeMD, Offset->getZExtValue(), FieldTypes);
          for (MDNode* FT : FieldTypes) {
            MDNode* PtrToFT = getOrCreatePtrToTypeMD(FT);
            if (PtrToFT) {
              if (addType(GEP, PtrToFT)) {
                changed = true;
              }
            }
          }
        }
        return changed;
      }
    }
    return false;
  }
  
  bool Changed = false;

  if (SourceElemTy->isStructTy()) {
    if (auto *ST = dyn_cast<StructType>(SourceElemTy)) {
      MDNode* StructMD = getMDFromLLVMTType(ST);
      MDNode* PtrStructMD = getOrCreatePtrToTypeMD(StructMD);
      if (StructMD && PtrStructMD) {
        if (GEP->getNumIndices() >= 2) {
          auto *IdxIt = GEP->idx_begin();
          ++IdxIt;  // Skip first index
          
          if (auto *FieldIdxCI = dyn_cast<ConstantInt>(*IdxIt)) {
            unsigned FieldIdx = FieldIdxCI->getZExtValue();
            if (MDNode *FieldMD = getStructFieldTypeMD(StructMD, FieldIdx)) {
              // GEP result is a pointer to the field type
              MDNode *ResultMD = getOrCreatePtrToTypeMD(FieldMD);
              if (ResultMD && addType(GEP, ResultMD)) {
                Changed = true;
              }
            }
          }
        } else {
          TypeSet FieldTypes;
          getFieldMDNodeFromOffset(StructMD, 0, FieldTypes);
          for (MDNode* FT : FieldTypes) {
            MDNode* PtrToFT = getOrCreatePtrToTypeMD(FT);
            if (PtrToFT) {
              if (addType(GEP, PtrToFT)) {
                Changed = true;
              }
            }
          }
        }
      }
    }
  } else if (SourceElemTy->isArrayTy()) {
    if (auto *AT = dyn_cast<ArrayType>(SourceElemTy)) {
      for (MDNode *BaseMD : *BaseTypes) {
        // BaseMD is the type of the base pointer (ptr to something)
        MDNode *PointeeMD = getPointeeTypeMD(BaseMD);
        if (PointeeMD && isArrayTypeMD(PointeeMD) &&
            SourceElemTy == getLLVMTypeFromMD(PointeeMD)) {
          // Array element access
          if (GEP->getNumIndices() >= 2) {
            if (MDNode *ElemMD = getArrayElementTypeMD(PointeeMD)) {
              MDNode *ResultMD = getOrCreatePtrToTypeMD(ElemMD);
              if (ResultMD && addType(GEP, ResultMD)) {
                Changed = true;
              }
            }
          } else {
            TypeSet FieldTypes;
            getFieldMDNodeFromOffset(PointeeMD, 0, FieldTypes);
            for (MDNode* FT : FieldTypes) {
              MDNode* PtrToFT = getOrCreatePtrToTypeMD(FT);
              if (PtrToFT) {
                if (addType(GEP, PtrToFT)) {
                  Changed = true;
                }
              }
            }
          }
        }
      }
    }
  }
  
  if (Changed) {
    NextWorklist.insert(GEP);
  }
  return Changed;
}

bool TypeRecovery::handleAddrSpaceCast(AddrSpaceCastInst *ASC) {
  // AddrSpaceCast: inherit source types
  if (!ASC->getType()->isPointerTy())
    return false;
  
  Value *Src = ASC->getOperand(0);
  if (!Src->getType()->isPointerTy())
    return false;

  const TypeSet *SrcTypes = getTypeSet(Src);
  if (SrcTypes) {
    if (unionTypes(ASC, *SrcTypes)) {
      NextWorklist.insert(ASC);
      return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Backward Propagation
//===----------------------------------------------------------------------===//

bool TypeRecovery::propagateBackward(Value *V) {
  auto *I = dyn_cast<Instruction>(V);
  if (!I)
    return false;
  
  bool changed = false;
  if (auto *LI = dyn_cast<LoadInst>(I)) {
    changed = backpropLoad(LI);
  } else if (auto *SI = dyn_cast<StoreInst>(I)) {
    changed = backpropStore(SI);
  } else if (auto *PHI = dyn_cast<PHINode>(I)) {
    changed = backpropPHI(PHI);
  } else if (auto *Sel = dyn_cast<SelectInst>(I)) {
    changed = backpropSelect(Sel);
  } else if (auto *CI = dyn_cast<CastInst>(I)) {
    // Handle addrspacecast and bitcast
    changed = backpropCast(CI);
  } else if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
    // Handle GEP backward propagation
    changed = backpropGEP(GEP);
  } else if (auto *CB = dyn_cast<CallBase>(I)) {
    // Check if it's memcpy
    Function *Callee = CB->getCalledFunction();
    if (Callee && (Callee->getName().contains("memcpy") ||
                   Callee->getName().contains("memmove"))) {
      changed = backpropMemcpy(CB);
    } else if (Callee && (Callee->getName().contains("xsig.setnewid") ||
                         Callee->getName().contains("xsig.setnewid") || 
                         Callee->getName().contains("xsig.setrawid"))) {
      changed = backpropSetIDCall(CB);
    } else {
      changed = backpropCall(CB);
    }
  } else if (auto *ASC = dyn_cast<AddrSpaceCastInst>(I)) {
    changed = backpropAddrSpaceCast(ASC);
  }

  return changed;
}

bool TypeRecovery::backpropLoad(LoadInst *LI) {
  // %q = load ptr, ptr %p
  // If %q's type is known (T), then %p's type is ptr-to-T
  
  const TypeSet *ResultTypes = getTypeSet(LI);
  if (!ResultTypes || ResultTypes->empty())
    return false;
  
  Value *Ptr = LI->getPointerOperand();
  bool Changed = false;
  
  for (MDNode *ResultMD : *ResultTypes) {
    // Create ptr-to-ResultMD
    MDNode *PtrTypeMD = getOrCreatePtrToTypeMD(ResultMD);
    if (PtrTypeMD && addType(Ptr, PtrTypeMD)) {
      NextWorklist.insert(Ptr);
      Changed = true;
    }
  }
  return Changed;
}

bool TypeRecovery::backpropStore(StoreInst *SI) {
  // store ptr %val, ptr %dest
  // If %val's type is T, then %dest's type is ptr-to-T
  // If %dest's type is ptr-to-T, then %val's type is T
  
  Value *Val = SI->getValueOperand();
  Value *Dest = SI->getPointerOperand();
  bool Changed = false;
  if (Val == Dest)
    return false;

  if (!Val->getType()->isPointerTy()) {
    return false;
  }
  
  // Forward: val type -> dest type
  const TypeSet *ValTypes = getTypeSet(Val);
  if (ValTypes) {
    for (MDNode *ValMD : *ValTypes) {
      MDNode *DestMD = getOrCreatePtrToTypeMD(ValMD);
      if (DestMD && addType(Dest, DestMD)) {
        NextWorklist.insert(Dest);
        Changed = true;
      }
    }
  }
  
  // Backward: dest type -> val type
  const TypeSet *DestTypes = getTypeSet(Dest);
  if (DestTypes) {
    for (MDNode *DestMD : *DestTypes) {
      if (MDNode *PointeeMD = getPointeeTypeMD(DestMD)) {
        if (!isPointerTypeMD(PointeeMD)) {
          continue;
        }
        if (addType(Val, PointeeMD)) {
          NextWorklist.insert(Val);
          Changed = true;
        }
      }
    }
  }
  return Changed;
}

bool TypeRecovery::backpropPHI(PHINode *PHI) {
  // If PHI result type is known, propagate to all incoming values
  const TypeSet *PHITypes = getTypeSet(PHI);
  if (!PHITypes || PHITypes->empty())
    return false;
  
  bool Changed = false;
  for (Value *Inc : PHI->incoming_values()) {
    if (unionTypes(Inc, *PHITypes)) {
      NextWorklist.insert(Inc);
      Changed = true;
    }
  }
  return Changed;
}

bool TypeRecovery::backpropSelect(SelectInst *SI) {
  // If select result type is known, propagate to true/false values
  const TypeSet *SelTypes = getTypeSet(SI);
  if (!SelTypes || SelTypes->empty())
    return false;
  
  Value *TrueVal = SI->getTrueValue();
  Value *FalseVal = SI->getFalseValue();
  
  bool Changed = false;
  if (unionTypes(TrueVal, *SelTypes)) {
    NextWorklist.insert(TrueVal);
    Changed = true;
  }
  if (unionTypes(FalseVal, *SelTypes)) {
    NextWorklist.insert(FalseVal);
    Changed = true;
  }
  return Changed;
}

bool TypeRecovery::backpropCall(CallBase *CB) {
  // Get function metadata - either from direct callee or indirect call's func ptr
  SmallVector<MDNode *, 0> FuncMDList;
  Function *Callee = CB->getCalledFunction();
  
  if (Callee) {
    // Direct call: get metadata from callee function
    MDNode* MD = Callee->getMetadata("sigmode.func");
    if (MD && isFunctionPointerTypeMD(MD)) {
      FuncMDList.push_back(getFunctionTypeMD(MD));
    }
  } else {
    // Indirect call: try to get metadata from function pointer's type
    Value *CalledValue = CB->getCalledOperand();
    const TypeSet *FuncPtrTypes = getTypeSet(CalledValue);
    if (FuncPtrTypes && !FuncPtrTypes->empty()) {
      // Function pointer should have type "ptr to func"
      // The metadata format is !{ptr undef, !{ret_type, param_types...}}
      for (MDNode *MD : *FuncPtrTypes) {
        MDNode *PointeeMD = getPointeeTypeMD(MD);
        if (PointeeMD && isFunctionPointerTypeMD(PointeeMD)) {
          FuncMDList.push_back(getFunctionTypeMD(PointeeMD));
        }
      }
    }
  }

  bool Changed = false;

  for (MDNode* FuncMD : FuncMDList) {
    // Format: !{ptr undef, !{ret_type, param_types...}} for func ptr
    // Or direct !{ret_type, param_types...} for func type
    MDNode *TypesMD = FuncMD;
    
    if (!TypesMD || TypesMD->getNumOperands() < 1)
      return false;
    
    // First element is return type
    if (auto *RetTypeMD = dyn_cast<MDNode>(TypesMD->getOperand(0))) {
      if (!isVoidTypeMD(RetTypeMD) && !CB->getType()->isVoidTy() && addType(CB, RetTypeMD)) {
        NextWorklist.insert(CB);
        Changed = true;
      }
    }
    
    // Propagate parameter types to arguments
    // Skip the first element (return type), remaining are parameter types
    unsigned NumParams = TypesMD->getNumOperands() - 1;
    unsigned NumArgs = CB->arg_size();
    
    // Only propagate for non-variadic part (min of formal params and actual args)
    unsigned PropCount = std::min(NumParams, NumArgs);
    
    for (unsigned I = 0; I < PropCount; ++I) {
      unsigned MDIdx = I + 1;  // +1 to skip return type
      if (auto *ParamTypeMD = dyn_cast<MDNode>(TypesMD->getOperand(MDIdx))) {
        Value *Arg = CB->getArgOperand(I);
        if (addType(Arg, ParamTypeMD)) {
          NextWorklist.insert(Arg);
          Changed = true;
        }
      }
    }
  }
  
  return Changed;
}

bool TypeRecovery::backpropMemcpy(CallBase *CB) {
  // memcpy(dest, src, len)
  // dest and src should have the same pointee type
  // Merge their types bidirectionally
  
  if (CB->arg_size() < 2)
    return false;
  
  Value *Dest = CB->getArgOperand(0);
  Value *Src = CB->getArgOperand(1);
  
  const TypeSet *DestTypes = getTypeSet(Dest);
  const TypeSet *SrcTypes = getTypeSet(Src);
  
  bool Changed = false;
  
  // Merge src types into dest
  if (SrcTypes) {
    if (unionTypes(Dest, *SrcTypes)) {
      NextWorklist.insert(Dest);
      Changed = true;
    }
  }
  
  // Merge dest types into src
  if (DestTypes) {
    if (unionTypes(Src, *DestTypes)) {
      NextWorklist.insert(Src);
      Changed = true;
    }
  }
  return Changed;
}

bool TypeRecovery::backpropSetIDCall(CallBase *CB) {
  const TypeSet* DestTypes = getTypeSet(CB);
  Value* Src = CB->getArgOperand(0);

  bool Changed = false;
  if (unionTypes(Src, *DestTypes)) {
    NextWorklist.insert(Src);
    Changed = true;
  }
  return Changed;
}

bool TypeRecovery::backpropCast(CastInst *CI) {
  // output = addrspacecast/bitcast input
  // If output's type is known, propagate to input
  
  if (!CI->getType()->isPointerTy())
    return false;
  
  const TypeSet *ResultTypes = getTypeSet(CI);
  if (!ResultTypes || ResultTypes->empty())
    return false;
  
  Value *Src = CI->getOperand(0);
  if (Src->getType()->isPointerTy() == false)
    return false;

  if (unionTypes(Src, *ResultTypes)) {
    NextWorklist.insert(Src);
    return true;
  }
  return false;
}

bool TypeRecovery::backpropGEP(GetElementPtrInst *GEP) {
    // GEP computes address of element/field
  // If base is struct*, GEP with constant indices gives field pointer
  
  Value *BasePtr = GEP->getPointerOperand();
  Type *SourceElemTy = GEP->getSourceElementType();
  const TypeSet *GEPSet = getTypeSet(GEP);

  if (!GEPSet)
    return false;
  
  // Special case: i8 GEP (byte-level pointer arithmetic)
  // output = getelementptr i8, ptr %p, i64 %offset
  // The result inherits the base pointer's type
  if (SourceElemTy->isIntegerTy(8)) {
    if (GEPSet) {
      auto Iter = GEP->idx_begin();
      auto Offset = dyn_cast<ConstantInt>(*Iter);
      if (!Offset) {
        if (unionTypes(BasePtr, *GEPSet)) {
          NextWorklist.insert(BasePtr);
          return true;
        }
      } else {
        // bool changed = false;
        // for (MDNode* GEPMD : *GEPSet) {
        //   MDNode* BasePointeeMD = getPointeeTypeMD(GEPMD);
        //   if (!BasePointeeMD) continue;
        //   Type* BasePointeeTy = getLLVMTypeFromMD(BasePointeeMD);
        //   if (!BasePointeeTy) continue;
        //   uint64_t typesize = Mod.getDataLayout().getTypeAllocSize(BasePointeeTy);
        //   if (typesize == 0) continue;
        //   if (Offset->getZExtValue() % typesize != 0) continue;
        //   if (addType(BasePtr, GEPMD)) {
        //     NextWorklist.insert(BasePtr);
        //     changed = true;
        //   }
        // }
        // return changed;
      }
    }
    return false;
  }
  
  bool Changed = false;

  if (SourceElemTy->isStructTy()) {
    if (auto *ST = dyn_cast<StructType>(SourceElemTy)) {
      MDNode* StructMD = getMDFromLLVMTType(ST);
      MDNode* PtrStructMD = getOrCreatePtrToTypeMD(StructMD);
      if (StructMD && PtrStructMD) {
        if (addType(BasePtr, PtrStructMD)) {
          NextWorklist.insert(BasePtr);
        }
      }
    }
  } else if (SourceElemTy->isArrayTy()) {
    if (auto *AT = dyn_cast<ArrayType>(SourceElemTy)) {
      for (MDNode *BaseMD : *GEPSet) {
        // BaseMD is the type of the base pointer (ptr to something)
        MDNode *PointeeMD = getPointeeTypeMD(BaseMD);
        if (GEP->getNumIndices() >= 2) {
          Type* ElemTy = AT->getElementType();
          if (ElemTy && PointeeMD && getLLVMTypeFromMD(PointeeMD) == ElemTy) {
            MDNode *ArrayMDNode = getOrCreateArrayOfTypeMD(PointeeMD, AT->getNumElements());
            MDNode *PtrArrayMD = getOrCreatePtrToTypeMD(ArrayMDNode);
            if (PtrArrayMD && addType(BasePtr, PtrArrayMD)) {
              NextWorklist.insert(BasePtr);
            }
          }
        } else {
          if (PointeeMD && getLLVMTypeFromMD(PointeeMD) == SourceElemTy) {
            if (addType(BasePtr, BaseMD)) {
              NextWorklist.insert(BasePtr);
            }
          }
        }
      }
    }
  }

  return Changed;

}

bool TypeRecovery::backpropAddrSpaceCast(AddrSpaceCastInst *ASC) {
  // AddrSpaceCast: inherit source types
  if (!ASC->getType()->isPointerTy())
    return false;
  
  const TypeSet *ResultTypes = getTypeSet(ASC);
  if (!ResultTypes || ResultTypes->empty())
    return false;
  
  Value *Src = ASC->getOperand(0);
  if (!Src->getType()->isPointerTy())
    return false;

  if (unionTypes(Src, *ResultTypes)) {
    NextWorklist.insert(Src);
    return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Formatting Helpers
//===----------------------------------------------------------------------===//

std::string TypeRecovery::getSourceLocation(const Instruction *I) {
  if (!I)
    return "";
  
  if (const DILocation *Loc = I->getDebugLoc()) {
    std::string Result;
    raw_string_ostream OS(Result);
    StringRef Filename = Loc->getFilename();
    // Get just the filename, not the full path
    size_t LastSlash = Filename.rfind('/');
    if (LastSlash != StringRef::npos)
      Filename = Filename.substr(LastSlash + 1);
    OS << Filename << ":" << Loc->getLine();
    return Result;
  }
  return "";
}

std::string TypeRecovery::getSourceLocation(const Value *V) {
  if (const auto *I = dyn_cast<Instruction>(V))
    return getSourceLocation(I);
  return "";
}

//===----------------------------------------------------------------------===//
// Iteration Dump
//===----------------------------------------------------------------------===//

void TypeRecovery::dumpIterationToFile(uint64_t random) {
  ++IterationCount;

  std::filesystem::path dir_path = "recovery_iter";
  if (!std::filesystem::exists(dir_path)) {
    std::filesystem::create_directory(dir_path);
  }
  
  std::string Filename = (dir_path / ("type_recovery_iter_" + std::to_string(random) + "_" + std::to_string(IterationCount) + ".txt")).string();
  std::error_code EC;
  raw_fd_ostream File(Filename, EC, sys::fs::OF_Text);
  
  if (EC) {
    errs() << "Warning: could not create iteration dump file: " << Filename 
           << " (" << EC.message() << ")\n";
    return;
  }
  
  errs() << "TypeRecovery: dumping iteration " << IterationCount 
         << " to file: " << Filename << "\n";
  
  // Use unified dump function
  dump(File);
}

//===----------------------------------------------------------------------===//
// Debug Output (IR-style format)
//===----------------------------------------------------------------------===//

/// Print type set as space-separated C-style type strings
void TypeRecovery::printTypeSet(const TypeSet &TS, raw_ostream &OS) {
  bool First = true;
  for (MDNode *MD : TS) {
    if (!First)
      OS << " ";
    First = false;
    OS << getTypeString(MD);
  }
}

void TypeRecovery::dump(raw_ostream &OS) {
  OS << "; === Type Recovery Results (Iteration " << IterationCount << ") ===\n";
  OS << "; Total values with types: " << TypeMap.size() << "\n\n";
  
  // 1. Global variables
  OS << "; --- Global Variables ---\n";
  for (GlobalVariable &GV : Mod.globals()) {
    // Print global declaration
    GV.print(OS);
    OS << "\n";
    
    // Print type if available
    const TypeSet *TS = getTypeSet(&GV);
    if (TS && !TS->empty()) {
      OS << "  ; types: ";
      printTypeSet(*TS, OS);
      OS << "\n";
    }
    OS << "\n";
  }
  
  // 2. Functions
  for (Function &F : Mod) {
    if (F.isDeclaration())
      continue;
    
    dumpFunction(F, OS);
  }
}

void TypeRecovery::dumpFunction(Function &F, raw_ostream &OS) {
  // Print function header
  OS << "define ";
  F.getReturnType()->print(OS);
  OS << " @" << F.getName() << "(";
  
  // Print arguments
  bool FirstArg = true;
  for (Argument &Arg : F.args()) {
    if (!FirstArg)
      OS << ", ";
    FirstArg = false;
    Arg.getType()->print(OS);
    if (Arg.hasName())
      OS << " %" << Arg.getName();
    else
      OS << " %" << Arg.getArgNo();
  }
  OS << ") {\n";
  
  // Print argument types
  for (Argument &Arg : F.args()) {
    const TypeSet *TS = getTypeSet(&Arg);
    if (TS && !TS->empty()) {
      OS << "  ; arg ";
      if (Arg.hasName())
        OS << "%" << Arg.getName();
      else
        OS << "%" << Arg.getArgNo();
      OS << " types: ";
      printTypeSet(*TS, OS);
      OS << "\n";
    }
  }
  
  // Print each basic block
  for (BasicBlock &BB : F) {
    // Print basic block label
    if (BB.hasName())
      OS << BB.getName() << ":\n";
    else
      OS << "; <label>:\n";
    
    // Print each instruction
    for (Instruction &I : BB) {
      // Print the instruction
      OS << "  ";
      I.print(OS);
      OS << "\n";
      
      // Print type if available (only for instructions that produce values)
      if (!I.getType()->isVoidTy()) {
        const TypeSet *TS = getTypeSet(&I);
        if (TS && !TS->empty()) {
          OS << "    ; types: ";
          printTypeSet(*TS, OS);
          OS << "\n";
        }
      }
    }
    OS << "\n";
  }
  
  OS << "}\n\n";
}


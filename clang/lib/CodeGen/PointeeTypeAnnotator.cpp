//===--- PointeeTypeAnnotator.cpp - Pointee Type Annotation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PointeeTypeAnnotator.h"
#include "CodeGenModule.h"
#include "CodeGenTypes.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/DeclCXX.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"

using namespace clang;
using namespace CodeGen;

PointeeTypeAnnotator::PointeeTypeAnnotator(CodeGenModule &CGM)
    : CGM(CGM), Ctx(CGM.getLLVMContext()), M(CGM.getModule()) {}

bool PointeeTypeAnnotator::isEnabled() const {
  return CGM.getContext().getTargetInfo().isSigModeSupported();
}

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

const void *PointeeTypeAnnotator::getTypeCacheKey(QualType Ty) const {
  return Ty.getCanonicalType().getAsOpaquePtr();
}

llvm::Type *PointeeTypeAnnotator::getLLVMType(QualType Ty) const {
  return CGM.getTypes().ConvertType(Ty);
}

bool PointeeTypeAnnotator::containsPointers(QualType Ty) const {
  return Ty->isContainPointer();
}

//===----------------------------------------------------------------------===//
// Metadata Creation
//===----------------------------------------------------------------------===//

llvm::MDNode *PointeeTypeAnnotator::getTypeMetadata(QualType Ty) {
  if (!isEnabled())
    return nullptr;

  // Handle void type - return nullptr
  if (Ty->isVoidType())
    return nullptr;

  Ty = Ty.getCanonicalType();

  // Check cache first
  const void *Key = getTypeCacheKey(Ty);
  auto It = TypeMetadataCache.find(Key);
  if (It != TypeMetadataCache.end())
    return It->second;

  // Check for cycles (self-referential types)
  if (TypesInProgress.count(Key)) {
    // Return a simple type marker for recursive reference
    llvm::Type *LLVMTy = getLLVMType(Ty);
    return createBasicTypeMD(LLVMTy);
  }

  // Mark this type as being processed
  TypesInProgress.insert(Key);

  llvm::MDNode *Result = nullptr;

  if (const auto *RT = Ty->getAs<RecordType>()) {
    Result = createRecordTypeMD(RT);
  } else if (Ty->isPointerType()) {
    Result = createPointerTypeMD(Ty);
  } else if (const auto *AT = dyn_cast<ArrayType>(Ty.getTypePtr())) {
    Result = createArrayTypeMD(AT, Ty);
  } else if (const auto *FT = Ty->getAs<FunctionType>()) {
    Result = createFunctionTypeMD(FT);
  } else {
    // Basic type (int, char, float, etc.)
    llvm::Type *LLVMTy = getLLVMType(Ty);
    Result = createBasicTypeMD(LLVMTy);
  }

  // Done processing this type
  TypesInProgress.erase(Key);

  // Cache the result
  if (Result)
    TypeMetadataCache[Key] = Result;

  return Result;
}

llvm::MDNode *PointeeTypeAnnotator::getPointeeMetadata(QualType Ty) {
  if (!Ty->isPointerType())
    return nullptr;

  QualType PointeeTy = Ty->getPointeeType();
  if (PointeeTy->isVoidType())
    return nullptr;

  return getTypeMetadata(PointeeTy);
}

//===----------------------------------------------------------------------===//
// Basic Type Metadata
//===----------------------------------------------------------------------===//

llvm::MDNode *PointeeTypeAnnotator::createBasicTypeMD(llvm::Type *LLVMTy) {
  // Format: !{llvm_type undef}
  llvm::UndefValue *Undef = llvm::UndefValue::get(LLVMTy);
  llvm::Metadata *Ops[] = {
      llvm::ConstantAsMetadata::get(Undef)
  };
  return llvm::MDNode::get(Ctx, Ops);
}

//===----------------------------------------------------------------------===//
// Pointer Type Metadata
//===----------------------------------------------------------------------===//

llvm::MDNode *PointeeTypeAnnotator::createPointerTypeMD(QualType Ty) {
  assert(Ty->isPointerType());

  QualType PointeeTy = Ty->getPointeeType();

  // Get pointer type - addrspace is preserved in the LLVM type
  llvm::Type *PtrTy = getLLVMType(Ty);
  llvm::UndefValue *PtrUndef = llvm::UndefValue::get(PtrTy);

  // void* case - just the pointer, no pointee info
  if (PointeeTy->isVoidType()) {
    llvm::Metadata *Ops[] = {
        llvm::ConstantAsMetadata::get(PtrUndef)
    };
    return llvm::MDNode::get(Ctx, Ops);
  }

  // Get pointee type metadata recursively
  llvm::MDNode *PointeeMD = getTypeMetadata(PointeeTy);

  // Format: !{ptr [addrspace(N)] undef, !pointee_type}
  llvm::Metadata *Ops[] = {
      llvm::ConstantAsMetadata::get(PtrUndef),
      PointeeMD
  };
  return llvm::MDNode::get(Ctx, Ops);
}

//===----------------------------------------------------------------------===//
// Array Type Metadata
//===----------------------------------------------------------------------===//

llvm::MDNode *PointeeTypeAnnotator::createArrayTypeMD(const ArrayType *AT,
                                                       QualType OrigTy) {
  QualType ElemTy = AT->getElementType();

  // Get LLVM array type
  llvm::Type *LLVMTy = getLLVMType(OrigTy);
  llvm::UndefValue *ArrayUndef = llvm::UndefValue::get(LLVMTy);

  // Get element type metadata recursively
  llvm::MDNode *ElemMD = getTypeMetadata(ElemTy);

  // Format: !{[N x T] undef, !element_type}
  llvm::Metadata *Ops[] = {
      llvm::ConstantAsMetadata::get(ArrayUndef),
      ElemMD
  };
  return llvm::MDNode::get(Ctx, Ops);
}

//===----------------------------------------------------------------------===//
// Record (Struct/Union/Class) Type Metadata
//===----------------------------------------------------------------------===//

llvm::MDNode *PointeeTypeAnnotator::createRecordTypeMD(const RecordType *RT) {
  const RecordDecl *RD = RT->getDecl()->getDefinition();
  if (!RD)
    RD = RT->getDecl();

  // Get LLVM struct type from RecordType
  QualType RecordQTy(RT, 0);
  llvm::Type *LLVMTy = getLLVMType(RecordQTy);
  llvm::UndefValue *StructUndef = llvm::UndefValue::get(LLVMTy);

  // Collect field type metadata
  llvm::SmallVector<llvm::Metadata *, 16> FieldMDs;

  for (const auto *FD : RD->fields()) {
    // Skip anonymous structs/unions - their fields are flattened
    if (FD->isAnonymousStructOrUnion()) {
      if (const auto *FieldRT = FD->getType()->getAs<RecordType>()) {
        // Recursively add anonymous struct/union fields
        const RecordDecl *AnonymousRD = FieldRT->getDecl();
        for (const auto *AnonymousFD : AnonymousRD->fields()) {
          llvm::MDNode *FieldMD = getTypeMetadata(AnonymousFD->getType());
          if (FieldMD)
            FieldMDs.push_back(FieldMD);
        }
      }
      continue;
    }

    llvm::MDNode *FieldMD = getTypeMetadata(FD->getType());
    if (FieldMD)
      FieldMDs.push_back(FieldMD);
  }

  // Create field list metadata
  llvm::MDNode *FieldsMD = FieldMDs.empty() ?
      nullptr : llvm::MDNode::get(Ctx, FieldMDs);

  // Format: !{%struct.Name undef, !{field_types...}}
  if (FieldsMD) {
    llvm::Metadata *Ops[] = {
        llvm::ConstantAsMetadata::get(StructUndef),
        FieldsMD
    };
    return llvm::MDNode::get(Ctx, Ops);
  } else {
    // No fields or empty struct
    llvm::Metadata *Ops[] = {
        llvm::ConstantAsMetadata::get(StructUndef)
    };
    return llvm::MDNode::get(Ctx, Ops);
  }
}

//===----------------------------------------------------------------------===//
// Function Type Metadata
//===----------------------------------------------------------------------===//

llvm::MDNode *PointeeTypeAnnotator::createFunctionTypeMD(const FunctionType *FT) {
  // For function types, we create metadata for the function pointer
  // Format: !{ptr undef, !{return_type, param_types...}}

  llvm::Type *PtrTy = llvm::PointerType::get(Ctx, 0);
  llvm::UndefValue *PtrUndef = llvm::UndefValue::get(PtrTy);

  // Collect return type and parameter types
  llvm::SmallVector<llvm::Metadata *, 8> TypeMDs;

  // Return type
  QualType RetTy = FT->getReturnType();
  if (!RetTy->isVoidType()) {
    if (llvm::MDNode *RetMD = getTypeMetadata(RetTy))
      TypeMDs.push_back(RetMD);
  }

  // Parameter types
  if (const auto *FPT = dyn_cast<FunctionProtoType>(FT)) {
    for (QualType ParamTy : FPT->param_types()) {
      if (llvm::MDNode *ParamMD = getTypeMetadata(ParamTy))
        TypeMDs.push_back(ParamMD);
    }
  }

  llvm::MDNode *TypesMD = TypeMDs.empty() ?
      nullptr : llvm::MDNode::get(Ctx, TypeMDs);

  if (TypesMD) {
    llvm::Metadata *Ops[] = {
        llvm::ConstantAsMetadata::get(PtrUndef),
        TypesMD
    };
    return llvm::MDNode::get(Ctx, Ops);
  } else {
    llvm::Metadata *Ops[] = {
        llvm::ConstantAsMetadata::get(PtrUndef)
    };
    return llvm::MDNode::get(Ctx, Ops);
  }
}

//===----------------------------------------------------------------------===//
// Annotation Functions
//===----------------------------------------------------------------------===//

void PointeeTypeAnnotator::annotateGlobalVariable(llvm::GlobalVariable *GV,
                                                   QualType Ty) {
  if (!isEnabled())
    return;

  QualType globalType = CGM.getContext().getPointerType(Ty);

  // Get type metadata
  llvm::MDNode *TypeMD = getTypeMetadata(globalType);
  if (TypeMD) {
    GV->setMetadata("sigmode.type", TypeMD);
  }

  // // For pointer types, also annotate the pointee
  // if (Ty->isPointerType()) {
  //   llvm::MDNode *PointeeMD = getPointeeMetadata(Ty);
  //   if (PointeeMD) {
  //     GV->setMetadata("sigmode.pointee", PointeeMD);
  //   }
  // }
}

void PointeeTypeAnnotator::annotateAlloca(llvm::AllocaInst *AI, QualType Ty) {
  if (!isEnabled())
    return;

  QualType allocType = CGM.getContext().getPointerType(Ty);

  // Get type metadata
  llvm::MDNode *TypeMD = getTypeMetadata(allocType);
  if (TypeMD) {
    AI->setMetadata("sigmode.type", TypeMD);
  }

  // // For pointer types, also annotate the pointee
  // if (Ty->isPointerType()) {
  //   llvm::MDNode *PointeeMD = getPointeeMetadata(Ty);
  //   if (PointeeMD) {
  //     AI->setMetadata("sigmode.pointee", PointeeMD);
  //   }
  // }
}

void PointeeTypeAnnotator::annotateFunctionArg(llvm::Function *F,
                                                unsigned ArgIdx,
                                                QualType OriginalTy,
                                                bool IsIndirect) {
  if (!isEnabled())
    return;

  if (ArgIdx >= F->arg_size())
    return;

  llvm::MDNode *TypeMD = getTypeMetadata(OriginalTy);
  if (!TypeMD)
    return;

  // Format: !{i32 arg_index, i1 is_indirect, !type}
  llvm::Metadata *Ops[] = {
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), ArgIdx)),
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt1Ty(Ctx), IsIndirect)),
      TypeMD
  };

  std::string MDName = "sigmode.arg." + std::to_string(ArgIdx);
  F->setMetadata(MDName, llvm::MDNode::get(Ctx, Ops));
}

void PointeeTypeAnnotator::annotateFunctionReturn(llvm::Function *F,
                                                   QualType RetTy,
                                                   bool IsIndirect) {
  if (!isEnabled())
    return;

  if (RetTy->isVoidType())
    return;

  llvm::MDNode *TypeMD = getTypeMetadata(RetTy);
  if (!TypeMD)
    return;

  // Format: !{i1 is_indirect, !type}
  llvm::Metadata *Ops[] = {
      llvm::ConstantAsMetadata::get(
          llvm::ConstantInt::get(llvm::Type::getInt1Ty(Ctx), IsIndirect)),
      TypeMD
  };

  F->setMetadata("sigmode.ret", llvm::MDNode::get(Ctx, Ops));
}

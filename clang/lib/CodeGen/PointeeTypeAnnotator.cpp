//===--- PointeeTypeAnnotator.cpp - Pointee Type Annotation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PointeeTypeAnnotator.h"
#include "ABIInfo.h"
#include "CodeGenModule.h"
#include "CodeGenTypes.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/TypeBase.h"
#include "clang/CodeGen/CGFunctionInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

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
  if (Ty->isVoidType()) {
    return llvm::Type::getVoidTy(Ctx);
  }
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

  Ty = CGM.getContext().getNoRawQual(Ty);
  Ty = Ty.getCanonicalType();

  // Check cache first
  const void *Key = getTypeCacheKey(Ty);
  auto It = TypeMetadataCache.find(Key);
  if (It != TypeMetadataCache.end())
    return It->second;

  // Create a temporary placeholder to handle cycles
  llvm::TempMDTuple Placeholder = llvm::MDNode::getTemporary(Ctx, {});
  TypeMetadataCache[Key] = Placeholder.get();

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

  // Replace the temporary placeholder with the actual result
  Placeholder->replaceAllUsesWith(Result);

  // Update cache with the permanent result
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
    return llvm::MDNode::getDistinct(Ctx, Ops);
  }

  // No fields or empty struct
  llvm::Metadata *Ops[] = {
      llvm::ConstantAsMetadata::get(StructUndef)
  };
  return llvm::MDNode::getDistinct(Ctx, Ops);
}

//===----------------------------------------------------------------------===//
// Function Type Metadata
//===----------------------------------------------------------------------===//

llvm::MDNode *PointeeTypeAnnotator::createFunctionTypeMD(const FunctionType *FT) {
  // For function types, we create metadata for the function pointer
  // Format: !{ptr undef, !{return_type, param_types...}}
  // We use CGFunctionInfo to get accurate ABI-level argument representation

  llvm::Type *PtrTy = llvm::PointerType::get(Ctx, 0);
  llvm::UndefValue *PtrUndef = llvm::UndefValue::get(PtrTy);

  // Get CGFunctionInfo through arrangeFreeFunctionType
  const CGFunctionInfo *FI = nullptr;
  if (const auto *FPT = dyn_cast<FunctionProtoType>(FT)) {
    FI = &CGM.getTypes().arrangeFreeFunctionType(
        CanQual<FunctionProtoType>::CreateUnsafe(QualType(FPT, 0)));
  } else if (const auto *FNPT = dyn_cast<FunctionNoProtoType>(FT)) {
    FI = &CGM.getTypes().arrangeFreeFunctionType(
        CanQual<FunctionNoProtoType>::CreateUnsafe(QualType(FNPT, 0)));
  }

  if (!FI) {
    // Fallback: just create a basic function pointer metadata
    llvm::Metadata *Ops[] = {llvm::ConstantAsMetadata::get(PtrUndef)};
    return llvm::MDNode::getDistinct(Ctx, Ops);
  }

  // Collect return type and parameter types using ABI info
  llvm::SmallVector<llvm::Metadata *, 16> TypeMDs;

  // Return type
  QualType RetTy = FT->getReturnType();
  const ABIArgInfo &RetInfo = FI->getReturnInfo();
  
  // If return is indirect (sret), first element is void, then sret pointer type
  if (RetInfo.isIndirect() || RetInfo.getKind() == ABIArgInfo::Ignore) {
    TypeMDs.push_back(getTypeMetadata(CGM.getContext().VoidTy));
  }
  annotateFunctionArg(TypeMDs, RetTy, RetInfo);

  // Parameter types
  if (const auto *FPT = dyn_cast<FunctionProtoType>(FT)) {
    CGFunctionInfo::const_arg_iterator info_it = FI->arg_begin();
    for (QualType ParamTy : FPT->param_types()) {
      if (info_it != FI->arg_end()) {
        const ABIArgInfo &ArgI = info_it->info;
        annotateFunctionArg(TypeMDs, ParamTy, ArgI);
        ++info_it;
      } else {
        // Fallback for variadic or mismatched args
        ABIArgInfo DirectInfo = ABIArgInfo::getDirect();
        annotateFunctionArg(TypeMDs, ParamTy, DirectInfo);
      }
    }
  }

  llvm::MDNode *TypesMD = TypeMDs.empty() ?
      nullptr : llvm::MDNode::get(Ctx, TypeMDs);

  if (TypesMD) {
    llvm::Metadata *Ops[] = {
        llvm::ConstantAsMetadata::get(PtrUndef),
        TypesMD
    };
    return llvm::MDNode::getDistinct(Ctx, Ops);
  }

  llvm::Metadata *Ops[] = {
      llvm::ConstantAsMetadata::get(PtrUndef)
  };
  return llvm::MDNode::getDistinct(Ctx, Ops);
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

void PointeeTypeAnnotator::annotateFunctionArg(llvm::SmallVector<llvm::Metadata *, 16>& ArgsMDs,
  QualType argtype, const ABIArgInfo &arginfo) {
  argtype = argtype.getCanonicalType();
  
  switch (arginfo.getKind()) {
    case ABIArgInfo::Indirect:
    case ABIArgInfo::IndirectAliased: {
      // Indirect: passed as a pointer to the actual type
      QualType pointerType = CGM.getContext().getPointerType(argtype);
      llvm::MDNode *NodeMD = getTypeMetadata(pointerType);
      if (NodeMD)
        ArgsMDs.push_back(NodeMD);
      break;
    }
    case ABIArgInfo::Direct:
    case ABIArgInfo::Extend: {
      // Direct/Extend: passed directly (possibly with sign/zero extension)
      // Check if there's a coercion type
      llvm::Type *CoerceTy = arginfo.getCoerceToType();
      if (CoerceTy && CoerceTy != getLLVMType(argtype)) {
        // There's a coercion - create metadata for the coerced LLVM type
        llvm::MDNode *NodeMD = createBasicTypeMD(CoerceTy);
        if (NodeMD)
          ArgsMDs.push_back(NodeMD);
      } else {
        llvm::MDNode *NodeMD = getTypeMetadata(argtype);
        if (NodeMD)
          ArgsMDs.push_back(NodeMD);
      }
      break;
    }
    case ABIArgInfo::InAlloca: {
      // InAlloca: argument is in a struct allocated on the stack
      // For annotation, treat as a pointer to the type
      QualType pointerType = CGM.getContext().getPointerType(argtype);
      llvm::MDNode *NodeMD = getTypeMetadata(pointerType);
      if (NodeMD)
        ArgsMDs.push_back(NodeMD);
      break;
    }
    case ABIArgInfo::Expand: {
      // Expand: struct is expanded into individual fields
      if (const auto *ArrayType = argtype->getAsArrayTypeUnsafe()) {
        QualType ElemTy = ArrayType->getElementType();
        uint64_t NumElements = 1;
        if (const auto *CAT = dyn_cast<ConstantArrayType>(ArrayType)) {
          NumElements = CAT->getSize().getZExtValue();
        }
        for (uint64_t i = 0; i < NumElements; ++i) {
          // Expanded array elements are passed directly
          ABIArgInfo DirectInfo = ABIArgInfo::getDirect();
          annotateFunctionArg(ArgsMDs, ElemTy, DirectInfo);
        }
      } else if (const auto *RecordTy = argtype->getAs<RecordType>()) {
        const RecordDecl *RD = RecordTy->getDecl()->getDefinition();
        if (!RD)
          RD = RecordTy->getDecl();

        for (const auto *FD : RD->fields()) {
          QualType FieldTy = FD->getType();
          // Recursively expand nested structs/arrays, otherwise direct
          if (FieldTy->isRecordType() || FieldTy->isArrayType()) {
            ABIArgInfo ExpandInfo = ABIArgInfo::getExpand();
            annotateFunctionArg(ArgsMDs, FieldTy, ExpandInfo);
          } else {
            ABIArgInfo DirectInfo = ABIArgInfo::getDirect();
            annotateFunctionArg(ArgsMDs, FieldTy, DirectInfo);
          }
        }
      } else {
        // Fallback to direct annotation for scalars
        llvm::MDNode *NodeMD = getTypeMetadata(argtype);
        if (NodeMD)
          ArgsMDs.push_back(NodeMD);
      }
      break;
    }
    case ABIArgInfo::CoerceAndExpand: {
      // CoerceAndExpand: the coerce-to type is a struct that gets expanded
      // We can get the exact types from ABIArgInfo
      llvm::StructType *CoerceTy = arginfo.getCoerceAndExpandType();
      if (CoerceTy) {
        // Get the unpacked elements (skip padding elements)
        llvm::ArrayRef<llvm::Type *> UnpackedTypes = arginfo.getCoerceAndExpandTypeSequence();
        for (llvm::Type *ElemTy : UnpackedTypes) {
          llvm::MDNode *NodeMD = createBasicTypeMD(ElemTy);
          if (NodeMD)
            ArgsMDs.push_back(NodeMD);
        }
      } else {
        // Fallback: expand struct fields
        if (const auto *RecordTy = argtype->getAs<RecordType>()) {
          const RecordDecl *RD = RecordTy->getDecl()->getDefinition();
          if (!RD)
            RD = RecordTy->getDecl();

          for (const auto *FD : RD->fields()) {
            QualType FieldTy = FD->getType();
            ABIArgInfo DirectInfo = ABIArgInfo::getDirect();
            annotateFunctionArg(ArgsMDs, FieldTy, DirectInfo);
          }
        } else {
          llvm::MDNode *NodeMD = getTypeMetadata(argtype);
          if (NodeMD)
            ArgsMDs.push_back(NodeMD);
        }
      }
      break;
    }
    case ABIArgInfo::TargetSpecific: {
      // TargetSpecific: handled by target-specific hooks
      // For now, treat as direct
      llvm::MDNode *NodeMD = getTypeMetadata(argtype);
      if (NodeMD)
        ArgsMDs.push_back(NodeMD);
      break;
    }
    case ABIArgInfo::Ignore:
      // Ignore: argument is not passed (void, empty structs)
      // No metadata needed
      break;
  }
}

void PointeeTypeAnnotator::annotateFunction(llvm::Function *F, QualType FuncTy, 
  const CGFunctionInfo &FI) {

  FuncTy = FuncTy.getCanonicalType();
  // Check cache first
  const void *Key = getTypeCacheKey(FuncTy);
  auto It = TypeMetadataCache.find(Key);
  llvm::MDNode *Result = nullptr;
  if (It != TypeMetadataCache.end()) {
    Result = It->second;
  } else {
    // Create a temporary placeholder to handle cycles
    llvm::TempMDTuple Placeholder = llvm::MDNode::getTemporary(Ctx, {});
    TypeMetadataCache[Key] = Placeholder.get();

    // the first metadata is for return
    // the other is for arg list in IR
    llvm::SmallVector<llvm::Metadata *, 16> ArgsMDs;
    
    QualType RetTy = FuncTy->getAs<FunctionType>()->getReturnType();
    const ABIArgInfo &RetInfo = FI.getReturnInfo();
    if (RetInfo.isIndirect() || RetInfo.getKind() == ABIArgInfo::Ignore) {
      // sret: return value is passed as an indirect pointer argument
      QualType voidtype = CGM.getContext().VoidTy;
      ArgsMDs.push_back(getTypeMetadata(voidtype));
    }
    DEBUG_FILE << F->getName() << "\n";
    DEBUG_FILE << RetTy.getAsString() << "\n";
    DEBUG_FILE << (uint64_t)(RetInfo.getKind()) << "\n";

    // Process return type
    annotateFunctionArg(ArgsMDs, RetTy, RetInfo);
    
    // Process all arguments using FI's argument info
    for (const auto &ArgInfo : FI.arguments()) {
      annotateFunctionArg(ArgsMDs, ArgInfo.type, ArgInfo.info);
    }

    llvm::Type *PtrTy = llvm::PointerType::get(Ctx, 0);
    llvm::UndefValue *PtrUndef = llvm::UndefValue::get(PtrTy);
    llvm::MDNode *TypesMD = llvm::MDNode::getDistinct(Ctx, ArgsMDs);
    llvm::Metadata *Ops[] = {
      llvm::ConstantAsMetadata::get(PtrUndef),
      TypesMD
    };
    Result = llvm::MDNode::getDistinct(Ctx, Ops);

    Placeholder->replaceAllUsesWith(Result);
    // Update cache with the permanent result
    TypeMetadataCache[Key] = Result;
  }

  std::string MDName = "sigmode.func";
  F->setMetadata(MDName, Result);
}

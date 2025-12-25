//===--- PointeeTypeAnnotator.h - Pointee Type Annotation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities for annotating pointer types with their
// pointee type information using LLVM IR types directly.
//
// Metadata format uses actual LLVM types:
//   - Basic type: !{i32 undef}
//   - Pointer: !{ptr undef, !pointee_type}
//   - Raw pointer: !{ptr addrspace(100) undef, !pointee_type}
//   - Array [N x T]: !{[N x T] undef, !element_type}
//   - Struct: !{%struct.Name undef, !{field_type0, field_type1, ...}}
//
// Examples:
//   int         -> !{i32 undef}
//   int*        -> !{ptr undef, !{i32 undef}}
//   int**       -> !{ptr undef, !{ptr undef, !{i32 undef}}}
//   int[3]      -> !{[3 x i32] undef, !{i32 undef}}
//   struct Node -> !{%struct.Node undef, !{...fields...}}
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_POINTEETYPEANNOTATOR_H
#define LLVM_CLANG_LIB_CODEGEN_POINTEETYPEANNOTATOR_H

#include "clang/AST/Type.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/GlobalVariable.h"

namespace clang {
namespace CodeGen {

class CodeGenModule;

/// Utility class for annotating pointers with pointee type information.
/// Uses actual LLVM IR types in metadata for precise type representation.
class PointeeTypeAnnotator {
public:
  explicit PointeeTypeAnnotator(CodeGenModule &CGM);

  /// Check if annotation is enabled.
  bool isEnabled() const;

  //===--------------------------------------------------------------------===//
  // Type Metadata Creation
  //===--------------------------------------------------------------------===//

  /// Create metadata node representing a Clang type using LLVM IR types.
  /// Format: !{llvm_type undef, element_info} or !{llvm_type undef}
  /// Returns nullptr for void type.
  llvm::MDNode *getTypeMetadata(QualType Ty);

  /// Get the pointee type metadata for a pointer type.
  /// Returns nullptr if Ty is not a pointer or if pointee is void.
  llvm::MDNode *getPointeeMetadata(QualType Ty);

  //===--------------------------------------------------------------------===//
  // Annotation Functions
  //===--------------------------------------------------------------------===//

  /// Annotate a global variable with its type metadata.
  void annotateGlobalVariable(llvm::GlobalVariable *GV, QualType Ty);

  /// Annotate an alloca instruction with type metadata.
  void annotateAlloca(llvm::AllocaInst *AI, QualType Ty);

  /// Annotate function argument with type metadata.
  /// Handles both direct and indirect (byval/sret) arguments.
  void annotateFunctionArg(llvm::Function *F, unsigned ArgIdx,
                           QualType OriginalTy, bool IsIndirect);

  /// Annotate function return type with type metadata.
  void annotateFunctionReturn(llvm::Function *F, QualType RetTy, bool IsIndirect);

  //===--------------------------------------------------------------------===//
  // Query Functions
  //===--------------------------------------------------------------------===//

  /// Check if a type contains any pointer fields (recursive).
  bool containsPointers(QualType Ty) const;

private:
  CodeGenModule &CGM;
  llvm::LLVMContext &Ctx;
  llvm::Module &M;

  /// Cache of type metadata to avoid duplicate creation.
  /// Maps canonical QualType to MDNode.
  llvm::DenseMap<const void *, llvm::MDNode *> TypeMetadataCache;

  /// Set of types currently being processed to detect cycles.
  llvm::SmallPtrSet<const void *, 8> TypesInProgress;

  /// Create metadata for basic types (int, char, float, etc.)
  /// Format: !{llvm_type undef}
  llvm::MDNode *createBasicTypeMD(llvm::Type *LLVMTy);

  /// Create metadata for pointer types.
  /// Format: !{ptr undef, !pointee_type}
  llvm::MDNode *createPointerTypeMD(QualType Ty);

  /// Create metadata for array types.
  /// Format: !{[N x T] undef, !element_type}
  llvm::MDNode *createArrayTypeMD(const ArrayType *AT, QualType OrigTy);

  /// Create metadata for record (struct/union/class) types.
  /// Format: !{%struct.Name undef, !{field_types...}}
  llvm::MDNode *createRecordTypeMD(const RecordType *RT);

  /// Create metadata for function pointer types.
  llvm::MDNode *createFunctionTypeMD(const FunctionType *FT);

  /// Get canonical key for type caching.
  const void *getTypeCacheKey(QualType Ty) const;

  /// Get LLVM type for a QualType.
  llvm::Type *getLLVMType(QualType Ty) const;
};

} // namespace CodeGen
} // namespace clang

#endif // LLVM_CLANG_LIB_CODEGEN_POINTEETYPEANNOTATOR_H

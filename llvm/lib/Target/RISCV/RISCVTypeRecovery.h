//===- RISCVTypeRecovery.h - Type Recovery for SigMode Memcpy -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the TypeRecovery class that recovers pointee type 
// information for Values in IR by propagating metadata from annotated sources
// (globals, allocas, functions) through the dataflow graph.
//
// The algorithm uses a three-set iterative worklist approach:
//   A = all instructions (constant)
//   B = current worklist (instructions to process this iteration)
//   C = next worklist (instructions affected by changes)
//
// Metadata format from frontend (PointeeTypeAnnotator):
//   - !sigmode.type on alloca/global: !{llvm_type undef, !pointee_info}
//   - !sigmode.func on functions: !{ptr undef, !{ret_type, param_types...}}
//
// For pointer types, the second operand contains the pointee type metadata.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVTYPERECOVERY_H
#define LLVM_LIB_TARGET_RISCV_RISCVTYPERECOVERY_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

/// TypeRecovery - Recovers type metadata for IR values through dataflow analysis.
///
/// The recovery process:
/// 1. Initialize: Build type string -> MDNode map from all metadata
/// 2. Collect: Single pass to collect all typed values into worklist
/// 3. Propagate: Use worklist algorithm to propagate types through IR
/// 4. Query: Provide type information for memcpy/memset expansion
///
class TypeRecovery {
public:
  /// Type set for a value - may have multiple possible types
  using TypeSet = SmallPtrSet<MDNode *, 4>;

  explicit TypeRecovery(Module &M) : Mod(M), Ctx(M.getContext()) {}

  /// Run type recovery on the module
  void run();

  /// Get the recovered type set for a value
  const TypeSet *getTypeSet(Value *V) const {
    auto It = TypeMap.find(V);
    return It != TypeMap.end() ? &It->second : nullptr;
  }

  /// Check if a value has exactly one recovered type
  bool hasUniqueType(Value *V) const {
    const TypeSet *TS = getTypeSet(V);
    return TS && TS->size() == 1;
  }

  /// Get the unique type if there's exactly one, otherwise nullptr
  MDNode *getUniqueType(Value *V) const {
    const TypeSet *TS = getTypeSet(V);
    if (TS && TS->size() == 1)
      return *TS->begin();
    return nullptr;
  }

  /// Debug: Dump all recovered types to the given stream (IR-style format)
  void dump(raw_ostream &OS);

  /// Debug: Dump types for a specific function (IR-style format)
  void dumpFunction(Function &F, raw_ostream &OS);

  /// Helper: Print a type set as space-separated type strings
  void printTypeSet(const TypeSet &TS, raw_ostream &OS);

  //===--------------------------------------------------------------------===//
  // Type String and Metadata Map Operations
  //===--------------------------------------------------------------------===//

  /// Get a unique string representation of a type metadata
  /// Uses cache for efficiency - computes once, stores in MDToTypeString
  std::string getTypeString(MDNode *MD);
  std::string getLLVMTypeString(Type* type);
  MDNode* getMDFromLLVMTType(Type* type);

  /// Look up metadata by type string, returns nullptr if not found
  MDNode *lookupTypeByString(StringRef TypeStr) const {
    auto It = TypeStringToMD.find(TypeStr);
    return It != TypeStringToMD.end() ? It->second : nullptr;
  }

  /// Register a new type metadata with its string representation
  void registerType(MDNode *MD);

  /// Create a basic type metadata: !{type undef}
  /// Used for registering primitive types (i1, i8, i32, float, etc.)
  MDNode *createBasicTypeMD(Type *Ty);

  //===--------------------------------------------------------------------===//
  // MDNode Helper Functions (using the type map)
  //===--------------------------------------------------------------------===//

  /// Check if a metadata node represents a pointer type
  /// Format: !{ptr undef, !pointee_type}
  static bool isPointerTypeMD(MDNode *MD);

  /// Get the pointee type metadata from a pointer type metadata
  /// Uses the type map for lookup
  MDNode *getPointeeTypeMD(MDNode *MD);

  /// Create or get a pointer-to-T metadata given T's metadata
  /// Looks up existing in map first, creates if not found
  MDNode *getOrCreatePtrToTypeMD(MDNode *PointeeMD);

  /// Get field type from a struct metadata at given index
  /// Struct format: !{%struct undef, !{field0, field1, ...}}
  MDNode *getStructFieldTypeMD(MDNode *StructMD, unsigned FieldIdx);

  /// Get array element type metadata
  /// Array format: !{[N x T] undef, !element_type}
  MDNode *getArrayElementTypeMD(MDNode *ArrayMD);

  /// Create or get an array of N elements of type T metadata
  /// Looks up existing in map first, creates if not found
  MDNode *getOrCreateArrayOfTypeMD(MDNode *ElementMD, uint64_t NumElements);

  /// Check if metadata represents a struct type
  static bool isStructTypeMD(MDNode *MD);

  /// Check if metadata represents an array type
  static bool isArrayTypeMD(MDNode *MD);

  /// Get the LLVM Type from a type metadata node
  /// The type is stored as the first operand: undef of that type
  Type *getLLVMTypeFromMD(MDNode *MD);

  //===--------------------------------------------------------------------===//
  // Formatting Helpers for Diagnostics
  //===--------------------------------------------------------------------===//

  /// Get source location string from an Instruction's debug info
  /// Returns "file:line" or empty string if no debug info
  static std::string getSourceLocation(const Instruction *I);

  /// Get source location string from a Value (if it's an Instruction)
  static std::string getSourceLocation(const Value *V);

private:
  Module &Mod;
  LLVMContext &Ctx;

  /// Map from Value to its possible types
  DenseMap<Value *, TypeSet> TypeMap;

  /// Map from type string to MDNode for efficient lookup
  /// Key is generated by getTypeString()
  StringMap<MDNode *> TypeStringToMD;

  /// Cache: Map from MDNode to its type string
  /// Avoids recomputing type string for the same metadata
  DenseMap<MDNode *, std::string> MDToTypeString;
  DenseMap<Type *, std::string> LLVMTypeToTypeString;
  DenseMap<Type*, MDNode *> LLVMTypeToMD;

  /// Current worklist (B)
  DenseSet<Value *> Worklist;

  /// Next worklist (C)
  DenseSet<Value *> NextWorklist;

  /// Iteration counter for dump files
  unsigned IterationCount = 0;

  //===--------------------------------------------------------------------===//
  // Initialization (Single Pass)
  //===--------------------------------------------------------------------===//

  /// Build type string -> MDNode map from all metadata in module
  void buildTypeMap();

  /// Collect metadata from a single MDNode and register it
  void collectMetadataRecursive(MDNode *MD);

  /// Single-pass initialization: collect all typed values into worklist
  int initialize();

  //===--------------------------------------------------------------------===//
  // Propagation
  //===--------------------------------------------------------------------===//

  /// Run the iterative propagation until fixpoint
  void propagate(int iteration_time = 100);

  /// Dump current type map to a numbered file (iteration_N.txt)
  void dumpIterationToFile(uint64_t random);

  /// Process a single value, returns true if types changed
  bool processValue(Value *V);

  /// Add a type to a value's type set, returns true if new
  bool addType(Value *V, MDNode *MD);
  bool addTypeInitStage(Value *V, MDNode *MD);

  /// Union another type set into a value's set, returns true if changed
  bool unionTypes(Value *V, const TypeSet &Other);

  //===--------------------------------------------------------------------===//
  // Forward Propagation Rules
  // Each handler directly calls addType and inserts into NextWorklist
  // Returns true if any types were added
  //===--------------------------------------------------------------------===//

  /// Compute forward propagated types for an instruction
  bool computeForwardTypes(Instruction *I);

  /// Load: result type is pointee of operand's pointee
  bool handleLoad(LoadInst *LI);

  /// Store: bidirectional - value and dest inform each other
  bool handleStore(StoreInst *SI);

  /// BitCast/AddrSpaceCast: inherit source types
  bool handleCast(CastInst *CI);

  /// PHI: union of all incoming values
  bool handlePHI(PHINode *PHI);

  /// Select: union of true and false values
  bool handleSelect(SelectInst *SI);

  /// GEP: compute field type from base struct/array metadata
  bool handleGEP(GetElementPtrInst *GEP);

  /// setid func: llvm.riscv.xsig.setdummyid/setrawid/setnewid
  bool handleSetIDCall(CallBase *CI);

  bool handleCall(CallBase *CI);

  /// AddrSpaceCast: inherit source types
  bool handleAddrSpaceCast(AddrSpaceCastInst *ASC);

  //===--------------------------------------------------------------------===//
  // Backward Propagation Rules
  // Returns true if any types were added
  //===--------------------------------------------------------------------===//

  /// Propagate types backward from uses to defs
  bool propagateBackward(Value *V);

  /// Load: if result type known, infer operand type (ptr to result)
  bool backpropLoad(LoadInst *LI);

  /// Store: bidirectional - value and dest inform each other
  bool backpropStore(StoreInst *SI);

  /// PHI/Select: propagate to incoming values
  bool backpropPHI(PHINode *PHI);
  bool backpropSelect(SelectInst *SI);

  /// Memcpy/Memmove: merge src and dest types bidirectionally
  bool backpropMemcpy(CallBase *CI);

  /// setid func: propagate pointer type to argument
  bool backpropSetIDCall(CallBase *CI);

  /// Cast (addrspacecast/bitcast): propagate result type to source
  bool backpropCast(CastInst *CI);

  /// GEP: propagate result type to base pointer
  bool backpropGEP(GetElementPtrInst *GEP);

  /// Call: return type from !sigmode.func metadata
  bool backpropCall(CallBase *CI);

  /// AddrSpaceCast: propagate result type to source
  bool backpropAddrSpaceCast(AddrSpaceCastInst *ASC);
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVTYPERECOVERY_H

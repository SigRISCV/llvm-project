//===- IntrinsicsRISCVXCV.td - CORE-V intrinsics -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the Sig vendor intrinsics for RISC-V.
//
//===----------------------------------------------------------------------===//


let TargetPrefix = "riscv" in {

def int_riscv_xsig_setrawid : Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty], [IntrNoMem, IntrWillReturn]>,
  ClangBuiltin<"__builtin_riscv_xsig_setrawid">;

def int_riscv_xsig_setdummyid : Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty], [IntrNoMem, IntrWillReturn]>,
  ClangBuiltin<"__builtin_riscv_xsig_setdummyid">;

def int_riscv_xsig_setnewid : Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty], [IntrNoMem, IntrWillReturn]>,
  ClangBuiltin<"__builtin_riscv_xsig_setnewid">;

def int_riscv_xsig_break_recover : Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty], [IntrNoMem, IntrWillReturn]>,
  ClangBuiltin<"__builtin_riscv_xsig_break_recover">;

def int_riscv_xsig_load_sig : Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty], [IntrWillReturn, IntrReadMem, IntrArgMemOnly]>,
  ClangBuiltin<"__builtin_riscv_xsig_load_sig">;

def int_riscv_xsig_store_sig : Intrinsic<[], [llvm_ptr_ty, llvm_ptr_ty], [IntrWriteMem, IntrArgMemOnly]>,
  ClangBuiltin<"__builtin_riscv_xsig_store_sig">;

def int_riscv_xsig_load_dyn : Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty], [IntrWillReturn, IntrReadMem, IntrArgMemOnly]>,
  ClangBuiltin<"__builtin_riscv_xsig_load_dyn">;

def int_riscv_xsig_store_dyn : Intrinsic<[], [llvm_ptr_ty, llvm_ptr_ty], [IntrWriteMem, IntrArgMemOnly]>,
  ClangBuiltin<"__builtin_riscv_xsig_store_dyn">;

def int_riscv_xsig_debug_sig : Intrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>,
  ClangBuiltin<"__builtin_riscv_xsig_debug_sig">;

def int_riscv_xsig_debug_chr : Intrinsic<[], [llvm_i64_ty], [IntrNoMem, IntrHasSideEffects]>,
  ClangBuiltin<"__builtin_riscv_xsig_debug_chr">;

def int_riscv_xsig_debug_int : Intrinsic<[], [llvm_i64_ty], [IntrNoMem, IntrHasSideEffects]>,
  ClangBuiltin<"__builtin_riscv_xsig_debug_int">;

def int_riscv_xsig_debug_ptr : Intrinsic<[], [llvm_ptr_ty], [IntrNoMem, IntrHasSideEffects]>,
  ClangBuiltin<"__builtin_riscv_xsig_debug_ptr">;

def int_riscv_xsig_debug_csr : Intrinsic<[], [llvm_i64_ty], [IntrNoMem, IntrHasSideEffects]>,
  ClangBuiltin<"__builtin_riscv_xsig_debug_csr">;

def int_riscv_xsig_debug_exit : Intrinsic<[], [], [IntrNoMem, IntrHasSideEffects]>,
  ClangBuiltin<"__builtin_riscv_xsig_debug_exit">;

// SigMode memcpy intrinsic for copying structures/arrays with pointer members
// Uses llvm_anyptr_ty to accept any address space (0 for sig, 100 for raw)
// Type information is passed via type_metadata_id parameter:
//   - The ID references a named metadata "!sigmemcpy.types" in the module
//   - Format: !sigmemcpy.types = !{!0, !1, ...}
//             !0 = !{i64 0, %struct.MyStruct undef}  ; ID 0 -> MyStruct
//   - This approach is more stable than instruction-level metadata during optimization
// Usage: call void @llvm.riscv.xsig.memcpy.p0.p0(ptr %d, ptr %s, i64 %n, i64 %type_id)
def int_riscv_xsig_memcpy : Intrinsic<[],
  [llvm_anyptr_ty,     // dest pointer (sig or raw)
   llvm_anyptr_ty,     // src pointer (sig or raw)
   llvm_i64_ty,        // number of elements
   llvm_i64_ty],       // type metadata ID (index into !sigmemcpy.types)
  [IntrArgMemOnly, IntrWillReturn,
   NoCapture<ArgIndex<0>>, NoCapture<ArgIndex<1>>,
   WriteOnly<ArgIndex<0>>, ReadOnly<ArgIndex<1>>,
   ImmArg<ArgIndex<3>>]>;  // type_id must be immediate constant

// SigMode memset intrinsic for zero-initializing structures/arrays with pointer members
// For pointer fields, stores xsig_setdummyid(null) instead of plain zero
// Uses llvm_anyptr_ty to accept any address space (0 for sig, 100 for raw)
// Type information is passed via type_metadata_id parameter:
//   - The ID references a named metadata "!sigmemset.types" in the module
//   - Format: !sigmemset.types = !{!0, !1, ...}
//             !0 = !{i64 0, %struct.MyStruct undef}  ; ID 0 -> MyStruct
// Usage: call void @llvm.riscv.xsig.memset.p0(ptr %d, i64 %n, i64 %type_id)
def int_riscv_xsig_memset : Intrinsic<[],
  [llvm_anyptr_ty,     // dest pointer (sig or raw)
   llvm_i64_ty,        // number of elements
   llvm_i64_ty],       // type metadata ID (index into !sigmemset.types)
  [IntrArgMemOnly, IntrWillReturn,
   NoCapture<ArgIndex<0>>, WriteOnly<ArgIndex<0>>,
   ImmArg<ArgIndex<2>>]>;  // type_id must be immediate constant

} // TargetPrefix = "riscv"

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

} // TargetPrefix = "riscv"

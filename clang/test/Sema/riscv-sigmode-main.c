// RUN: %clang_cc1 -triple riscv64 -target-feature +experimental-xsig -target-feature +sig-mode -fsyntax-only -verify %s

void takes_raw(__raw char *__raw *);

int main(int argc, char **argv, char **envp) {
  _Static_assert(
      __builtin_types_compatible_p(__typeof__(argv), __raw char *__raw *), "");
  _Static_assert(
      __builtin_types_compatible_p(__typeof__(envp), __raw char *__raw *), "");
  takes_raw(argv);
  takes_raw(envp);
  return argc;
}

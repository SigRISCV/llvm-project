let Predicates = [HasVendorXSig] in {

def LS : Load_ri<0b111, "ls">, Sched<[WriteLDW, ReadMemBase]>;
def SS : Store_rri<0b100, "ss">, Sched<[WriteSTD, ReadStoreData, ReadMemBase]>;

}
from pysb import *
Model()

# Init Monomers
Monomer('A19', ['a19_3', 'a19_2', 'a19_15', 'a19_35', 'a19_40'])
Monomer('A3', ['a3_19', 'a3_2', 'a3_35', 'a3_18'])
Monomer('A2', ['a2_19', 'a2_3', 'a2_15'])
Monomer('A15', ['a15_19', 'a15_2', 'a15_40'])
Monomer('A35', ['a35_19', 'a35_3'])
Monomer('A40', ['a40_19', 'a40_15'])
Monomer('A18', ['a18_3'])

# init Parameters
Parameter('kon', 1e-5)
Parameter('koff', 1e-3)
Parameter('kon_trimer', 1e-1)
Parameter('koff_trimer', 1e-6)
Parameter('all_init', 1000)

#init species
Initial(A19(a19_3=None, a19_2=None, a19_15=None, a19_35=None, a19_40=None), all_init)
Initial(A3(a3_19=None, a3_2=None, a3_35=None, a3_18=None), all_init)
Initial(A2(a2_19=None, a2_3=None, a2_15=None), all_init)
Initial(A15(a15_19=None, a15_2=None, a15_40=None), all_init)
Initial(A35(a35_19=None, a35_3=None), all_init)
Initial(A40(a40_19=None, a40_15=None), all_init)
Initial(A18(a18_3=None), all_init)

#init observable
#Observable('obsA19', A19(a19_3=None, a19_2=None, a19_15=None, a19_35=None, a19_40=None))
Observable('obsA19_A3', A19() % A3())
#Observable('obsComplex', A19() % A3() % A2() % A15() % A35() % A40() % A18())
Observable('obs_A19_A3_A2_lin', A19(a19_3=1) %
                         A3(a3_19=1, a3_2=6) %
                         A2(a2_3=6))

Observable('obs_A19_A3_A2_trimer', A19(a19_3=1, a19_2=2) %
                         A3(a3_19=1, a3_2=6) %
                         A2(a2_3=6, a2_19=2))
# Observable('obsComplex_only', A19(a19_3=1, a19_2=2, a19_15=3, a19_35=4, a19_40=5) %
#                               A3(a3_19=1, a3_2=6, a3_35=7, a3_18=8) %
#                               A2(a2_19=2, a2_3=6, a2_15=9) %
#                               A15(a15_19=3, a15_2=9, a15_40=10) %
#                               A35(a35_19=4, a35_3=7) %
#                               A40(a40_19=5, a40_15=10) %
#                               A18(a18_3=8))
#Observable('obsMisinteraction', A19() % A3() % A19())

# rules
Rule('A19_A3_std', A19(a19_3=None) + A3(a3_19=None) | A19(a19_3=1) % A3(a3_19=1), *[kon, koff])
Rule('A19_A2_std', A19(a19_2=None) + A2(a2_19=None) | A19(a19_2=2) % A2(a2_19=2), *[kon, koff])
Rule('A19_A15_std', A19(a19_15=None) + A15(a15_19=None) | A19(a19_15=3) % A15(a15_19=3), *[kon, koff])
Rule('A19_A35_std', A19(a19_35=None) + A35(a35_19=None) | A19(a19_35=4) % A35(a35_19=4), *[kon, koff])
Rule('A19_A40_std', A19(a19_40=None) + A40(a40_19=None) | A19(a19_40=5) % A40(a40_19=5), *[kon, koff])
Rule('A3_A2_std', A3(a3_2=None) + A2(a2_3=None) | A3(a3_2=6) % A2(a2_3=6), *[kon, koff])
Rule('A3_A35_std', A3(a3_35=None) + A35(a35_3=None) | A3(a3_35=7) % A35(a35_3=7), *[kon, koff])
Rule('A3_A18_std', A3(a3_18=None) + A18(a18_3=None) | A3(a3_18=8) % A18(a18_3=8), *[kon, koff])
Rule('A2_A15_std', A2(a2_15=None) + A15(a15_2=None) | A2(a2_15=9) % A15(a15_2=9), *[kon, koff])
Rule('A15_A40_std', A15(a15_40=None) + A40(a40_15=None) | A15(a15_40=10) % A40(a40_15=10), *[kon, koff])

#trimer snapping rules (3 choose 2 for each trimer)
Rule('A19_A3_A2_trimerize', A19(a19_3=1, a19_2=None) % A3(a3_19=1, a3_2=6) % A2(a2_3=6, a2_19=None) >>
     A19(a19_3=1, a19_2=2) % A3(a3_19=1, a3_2=6) % A2(a2_3=6, a2_19=2), *[kon_trimer])

Rule('A2_A19_A3_trimerize', A2(a2_19=2, a2_3=None) % A19(a19_2=2, a19_3=6) % A3(a3_19=6, a3_2=None) >>
     A19(a19_3=1, a19_2=2) % A3(a3_19=1, a3_2=6) % A2(a2_3=6, a2_19=2), *[kon_trimer])

Rule('A19_A2_A3_trimerize', A19(a19_2=2, a19_3=None) % A2(a2_19=2, a2_3=6) % A3(a3_2=6, a3_19=None) >>
     A19(a19_3=1, a19_2=2) % A3(a3_19=1, a3_2=6) % A2(a2_3=6, a2_19=2), *[kon_trimer])


Rule('A19_A15_A2_trimerize', A19(a19_15=3, a19_2=None) % A15(a15_19=3, a15_2=9) % A2(a2_15=9, a2_19=None) >>
     A19(a19_15=3, a19_2=2) % A15(a15_19=3, a15_2=9) % A2(a2_15=9, a2_19=2), *[kon_trimer])

Rule('A2_A19_A15_trimerize', A2(a2_19=2, a2_15=None) % A19(a19_2=2, a19_15=3) % A3(a15_19=3, a15_2=None) >>
     A19(a19_15=3, a19_2=2) % A15(a15_19=3, a15_2=9) % A2(a2_15=9, a2_19=2), *[kon_trimer])

Rule('A19_A2_A15_trimerize', A19(a19_2=2, a19_15=None) % A2(a2_19=2, a2_15=9) % A3(a15_2=9, a15_19=None) >>
     A19(a19_15=3, a19_2=2) % A15(a15_19=3, a15_2=9) % A2(a2_15=9, a2_19=2), *[kon_trimer])


Rule('A19_A15_A40_trimerize', A19(a19_15=3, a19_2=None) % A15(a15_19=3, a15_2=9) % A2(a2_15=9, a2_19=None) >>
     A19(a19_15=3, a19_2=2) % A15(a15_19=3, a15_2=9) % A2(a2_15=9, a2_19=2), *[kon_trimer])

Rule('A40_A19_A15_trimerize', A2(a2_19=2, a2_15=None) % A19(a19_2=2, a19_15=3) % A3(a15_19=3, a15_2=None) >>
     A19(a19_15=3, a19_2=2) % A15(a15_19=3, a15_2=9) % A2(a2_15=9, a2_19=2), *[kon_trimer])

Rule('A19_A40_A15_trimerize', A19(a19_2=2, a19_15=None) % A2(a2_19=2, a2_15=9) % A3(a15_2=9, a15_19=None) >>
     A19(a19_15=3, a19_2=2) % A15(a15_19=3, a15_2=9) % A2(a2_15=9, a2_19=2), *[kon_trimer])
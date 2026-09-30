"""Wire MaxHerding / UHerding / WassersteinIP into the 5 WASSAL drivers.

Adds, to each driver:
  1. import line after the modern_al import
  2. three dispatch elif blocks after the alfamargin elif
  3. three ("AL", "<method>") entries after ("AL", "alfamargin"),

Idempotent: skips drivers already containing 'sota_al'.
"""
import re
import sys

DRIVERS = [
    "tutorials/All_Wassal/wassal_cifar10_multiclass_vanilla2.py",
    "tutorials/All_Wassal/wassal_svhn_multiclass_vanilla.py",
    "tutorials/All_Wassal/wassal_stl10.py",
    "tutorials/All_Wassal/wassal_caltech_multiclass_vanilla2.py",
    "tutorials/All_Wassal/wassal_pneumonia_multiclass_vanilla.py",
]

IMPORT_LINE = "from trust.strategies.modern_al import TypiClust, ProbCover, DCoM, ALFAMargin"
IMPORT_NEW = IMPORT_LINE + "\nfrom trust.strategies.sota_al import MaxHerding, UHerding, WassersteinIP"

DISPATCH_ANCHOR = (
    '        elif sf == "alfamargin":\n'
    "            strategy_sel = ALFAMargin(\n"
    "                train_set, unlabeled_lake_set, model, num_cls, strategy_args\n"
    "            )\n"
)
DISPATCH_NEW = DISPATCH_ANCHOR + (
    '        elif sf == "maxherding":\n'
    "            strategy_sel = MaxHerding(\n"
    "                train_set, unlabeled_lake_set, model, num_cls, strategy_args\n"
    "            )\n"
    '        elif sf == "uherding":\n'
    "            strategy_sel = UHerding(\n"
    "                train_set, unlabeled_lake_set, model, num_cls, strategy_args\n"
    "            )\n"
    '        elif sf == "wassersteinip":\n'
    "            strategy_sel = WassersteinIP(\n"
    "                train_set, unlabeled_lake_set, model, num_cls, strategy_args\n"
    "            )\n"
)

LIST_ANCHOR = '    ("AL", "alfamargin"),\n'
LIST_NEW = LIST_ANCHOR + (
    '    ("AL", "maxherding"),\n'
    '    ("AL", "uherding"),\n'
    '    ("AL", "wassersteinip"),\n'
)


def main():
    ok = True
    for path in DRIVERS:
        with open(path) as f:
            src = f.read()
        if "sota_al" in src:
            print(f"SKIP {path} (already wired)")
            continue
        n_imp = src.count(IMPORT_LINE)
        n_disp = src.count(DISPATCH_ANCHOR)
        n_list = src.count(LIST_ANCHOR)
        if (n_imp, n_disp, n_list) != (1, 1, 1):
            print(f"ERROR {path}: anchors found import={n_imp} dispatch={n_disp} list={n_list}")
            ok = False
            continue
        out = src.replace(IMPORT_LINE, IMPORT_NEW)
        out = out.replace(DISPATCH_ANCHOR, DISPATCH_NEW)
        out = out.replace(LIST_ANCHOR, LIST_NEW)
        with open(path, "w") as f:
            f.write(out)
        print(f"OK   {path}")
    if not ok:
        sys.exit(1)
    print("done")


if __name__ == "__main__":
    main()

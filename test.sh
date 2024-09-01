echo "Testing Dot product Ops On CPU"
TRITON_INTERPRET=1 TRITON_CPU_BACKEND=1 python3 ops/dot.py 

echo "Testing Vector Add Ops On CPU"
TRITON_INTERPRET=1 TRITON_CPU_BACKEND=1 python3 ops/vctadd.py

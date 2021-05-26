import os
import numpy as np


total = 0
nums = 0
with open("output.txt", "r") as fp:
    for v in fp:
        v = int(v)
        if v > 50:
            continue
        if v < 0:
            v = 0
        total += v
        nums += 1

print("result:", total / nums)
print("Hello world")
import math

def Mergesort(Array):
    if len(Array) == 1:
        return Array

    midpoint = math.floor(len(Array)/2)

    AL = Array[:midpoint]
    AR = Array[midpoint:]
    L = Mergesort(AL)
    R = Mergesort(AR)

    B = Merge(L, R)
    return B

def Merge(L, R):
    i = j = k = 0
    B = []
    if not L:
        return R
    elif not R:
        return L
    else:
        length = len(L) + len(R)

    while k < length:
        if j >= len(R):
            B.extend(L[i:])
            return B
            
        if i >= len(L):
            B.extend(R[j:])
            return B

        if L[i] < R[j]:
            B.append(L[i])
            i += 1
        else:
            B.append(R[j])
            j += 1
        k += 1
    return B


array = [4, 3, 10, 9, 5, 6, 3, 1, 2, 8, 7]
Mergesort(array)
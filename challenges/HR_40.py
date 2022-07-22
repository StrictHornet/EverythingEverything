#!/bin/python3

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    s = input()
    
    dict = {}
    count = 0
    for alphabet in sorted(s):
        if alphabet not in dict:
            count = s.count(alphabet)
            dict[alphabet] = count
            
    sorted_vals = sorted(dict.values(), reverse=True)
    #print(sorted_vals)
    sorted_dict = {}
    
    for k in sorted_vals:
        for key in dict:
            if dict[key] == k:
                sorted_dict[key] = k
            
   # print(sorted_dict)
    
    i = 0
    for item in sorted_dict:
        i += 1
        print(item, dict.get(item))
        if i == 3:
            break
        

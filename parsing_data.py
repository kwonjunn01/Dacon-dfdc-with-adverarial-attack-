#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 21:26:25 2020

@author: diml
"""

with open('train_list_fake_2.txt', 'r') as f:
    lines = f.readlines()
fake = []
real = []
for line in lines[208750:]:
    if line[0] == 'f':
        fake.append(line)

        
print(len(lines))
# with open('train_list_fake_2.txt', 'w') as f:
#       f.writelines(fake)
    
# with open('train_list_real.txt', 'w') as f:
#     f.writelines(real) 173177
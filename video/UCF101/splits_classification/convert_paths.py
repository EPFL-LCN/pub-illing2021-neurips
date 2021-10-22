# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 12:33:53 2020

@author: Jean
"""

import os

if __name__ == '__main__':
    
    files = os.listdir('./')[1:]
    print(files)
    
    for file_name in files:
        with open(file_name, 'r') as stream:
            paths = stream.readlines()
            print(paths[0])
            for path in paths:
                path.replace('\\','/')
        print(paths[0])
        with open(file_name, 'w') as stream:
            stream.writelines(paths)
        
                             
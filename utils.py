import numpy as np

def cumulate(l):
    c_rs = []
    for i in range(len(l)):
        c_rs.append(
            sum(l[:i])
        )
    return c_rs

if __name__ =="__main__":
    l = [1,1,1,1,1,1,1]
    c_l = cumulate(l)
    print(c_l)
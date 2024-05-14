
def print_obj_file(p1, p2):
    for i in range(len(p1)):
        print("v %f %f %f" % (p1[i][0],  p1[i][2],  p1[i][1]))
        print("v %f %f %f" % (p2[i][0],  p2[i][2],  p2[i][1]))
        print("l %d %d" % (1+i*2, 2+i*2))
    

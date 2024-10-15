comp_base_map={'A':'T','T':'A','C':'G','G':'C','[':']', ']':'['}

def split_list(l,n=100):
    i=0    
    chunk = l[i*n:(i+1)*n]
    while chunk:
        yield chunk
        i+=1
        chunk = l[i*n:(i+1)*n]
        
        
def revcomp(s):
    return ''.join(comp_base_map[x] for x in s[::-1])
import numpy as np

##理论上house1应当优于house算法
##但是采用house1的话误差矩阵与原矩阵二范数比值接近1
##采用house则可以稳定在1e-14至1e-15量级

def house(x,tol=1e-16):
    e=np.zeros_like(x)
    e[0]=1
    k=-np.sign(x[0])*np.linalg.norm(x)
    v=x-k*e
    norm=np.linalg.norm(v)
    if norm>tol:
        v/=norm
    else:
        v=np.zeros_like(v)
    return v,k #k是无效值 忽略
def house1(x,tol=1e-16):
    n=len(x)
    v=np.zeros_like(x)
    ypsilon=np.linalg.norm(x,np.inf)
    x/=ypsilon
    cita=np.dot(x[1:n],x[1:n])
    v[1:n]=x[1:n]
    if cita==0:
        beta=0
    else:
        alpha=np.sqrt(x[0]**2+cita)
        if x[0]<=0:
            v[0]=x[0]-alpha
        else:
            v[0]=-cita/(x[0]+alpha)
        beta=2*v[0]*v[0]/(cita+v[0]**2)
        v/=v[0]
    return v,beta

def delete(A,tol=1e-16):
    #将满足条件“np.abs(A[i,i-1])<=(np.abs(A[i,i])+np.abs(A[i-1,i-1]))*tol”的副对角元素置0
    #tol取机器精度
    n=A.shape[0]
    for i in range(1,n):
        if np.abs(A[i,i-1])<=(np.abs(A[i,i])+np.abs(A[i-1,i-1]))*tol:
            A[i,i-1]=0
    return A

def upper_hessenberg(A):
    n=A.shape[0]
    A=np.array(A)
    Q=np.eye(n)
    for i in range(n-2):
        v,k=house(A[i+1:,i])
        Hk=np.eye(n-i-1)-2*np.outer(v,v)
        #HK=np.eye(n)
        #HK[i+1:,i+1:]=Hk
        #Q=Q@HK
        Q[:,i+1:]=Q[:,i+1:]@Hk
        #A=Hk@A@Hk
        A[i+1:,:]=Hk@A[i+1:,:]
        A[:,i+1:]=A[:,i+1:]@Hk
    return Q,A

def is_diagonal(A,tol=1e-16):#判别是否有大于2*2的对角块
    n=A.shape[0]
    for i in range(n-2):
        if np.abs(A[i+1,i])>tol:
            if np.abs(A[i+2,i+1])>tol:
                return False
    print("矩阵已经收敛。")
    return True

def is_reducible(A,tol=1e-16):#判别“大于2*2的对角块”是不是原矩阵本身
    #如果是的话返回false 直接迭代整个矩阵
    #若不是返回true 对非原矩阵本身而是其大于2*2子块作迭代
    n=A.shape[0]
    for i in range(n-2):
        if np.abs(A[i+1,i])<tol:
            return False
    return True

##这个程序假定了end-start>3 这一假设的可行性是由cycle中的检验函数保证的
def Francis_doubleshift_QR_ren(Q,A,start,end,tol):#end比实际末行大1
    A=delete(A)
    n=A.shape[0]
   
    s=A[end-1,end-1]+A[end-2,end-2]
    t=A[end-1,end-1]*A[end-2,end-2]-A[end-2,end-1]*A[end-1,end-2]
    x=A[start,start]*A[start,start]+A[start,start+1]*A[start+1,start]-s*A[start,start]+t
    y=A[start+1,start]*(A[start,start]+A[start+1,start+1]-s)
    z=A[start+1,start]*A[start+2,start+1]

    v,k=house(np.array([x,y,z]))
    Hk=np.eye(3)-2*np.outer(v,v)
    #HK=np.eye(n)
    #HK[:3,:3]=Hk
    #Q=Q@HK
    Q[start:,start:start+3]=Q[start:,start:start+3]@Hk
    #A=HK@A@HK
    A[start:start+3,start:]=Hk@A[start:start+3,start:]
    A[start:start+4,start:start+3]=A[start:start+4,start:start+3]@Hk
    for i in range(start+1,end-2):
        x=A[i,i-1]
        y=A[i+1,i-1]
        z=A[i+2,i-1]
        v,k=house(np.array([x,y,z]))
        Hk=np.eye(3)-2*np.outer(v,v)
        #HK=np.eye(n)
        #HK[i:(i+3),i:(i+3)]=Hk
        #Q=Q@HK
        Q[:,i:(i+3)]=Q[:,i:(i+3)]@Hk
        #A=HK@A@HK
        r=min(i+4,n)
        A[i:(i+3),(i-1):]=Hk@A[i:(i+3),(i-1):]
        A[:r,i:(i+3)]=A[:r,i:(i+3)]@Hk
        

    x=A[end-2,end-3]
    y=A[end-1,end-3]
    v,k=house(np.array([x,y]))
    Hk=np.eye(2)-2*np.outer(v,v)
    #HK=np.eye(n)
    #HK[(end-2):end,(end-2):end]=Hk
    #Q=Q@HK
    Q[:,(end-2):end]=Q[:,(end-2):end]@Hk
    #A=HK@A@HK
    A[(end-2):end,(end-3):]=Hk@A[(end-2):end,(end-3):]
    A[:,(end-2):end]=A[:,(end-2):end]@Hk

    #此处print为了展现每一步迭代具体情况
    #可根据需求删去
    print(A)
    return Q,A
        
def search(A,n,tol):#调用search的时候一定有大于2*2的 非原矩阵本身的对角块
    Zero_list=[]
    for i in range(1,n):
        if np.abs(A[i,i-1])<tol:
            Zero_list.append(i)
    if (n-Zero_list[-1])>2:
        end=n
        start=Zero_list[-1]
        return start,end
    if Zero_list[0]>2:
        start=0
        end=Zero_list[0]
        return start,end
    for j in range(len(Zero_list)-1):
        if (Zero_list[j+1]-Zero_list[j])>2:
            end=Zero_list[j+1]
            start=Zero_list[j]
    return start,end

    
   
def Francis_cycle(Q,M,max_iter,tol1=1e-16,tol2=1e-5):
    #tol1:判别是否删除接近0副对角元素的容忍度（取机器精度）
    #tol2:判别矩阵是否收敛到拟上三角阵的容忍度（可根据需求修改）
    n=M.shape[0]
    iteration_times=0
    while not is_diagonal(M,tol2):
        iteration_times+=1
        print(f"第{iteration_times}次迭代：")
        if not is_reducible(M,tol1):
            start,end=search(M,n,tol1)
            print(f"<start,end>=<{start},{end}>")
            Q,M=Francis_doubleshift_QR_ren(Q,M,start,end,tol1)
        else:
            print("无需分块。")
            print(f"<start,end>=<0,{n}>")
            Q,M=Francis_doubleshift_QR_ren(Q,M,0,n,tol1)
        if iteration_times>max_iter:
            print("超出最大迭代次数，")
            return Q,M
    print(f"迭代次数为{iteration_times}.")
    return Q,M
    

max_iter=500
dim=50
A=np.zeros((dim,dim))
#这里提供一个测试矩阵样例A 元素为-10到10整数
#请根据dim自行增大max_iter
#注意：这种生成方式可能生成非常病态的矩阵导致迭代次数远超预期
for i in range(dim):
    for j in range(dim):
        A[i,j]=np.random.randint(-10,10)
#A=np.random.rand(dim,dim)也可以作为一种生成方法
print("原矩阵为：")
print(A)
copyA=A.copy()
Q,H=upper_hessenberg(A)
print("上hessenberg阵为：")
print(H)
Q,H=Francis_cycle(Q,H,max_iter)
print("拟上三角阵为：")
print(H)
print("正交误差为：")
print(np.linalg.norm(Q@Q.T-np.eye(dim),2))
print("误差矩阵与原矩阵的二范数比值为：")
print(np.linalg.norm(Q@H@Q.T-copyA,2)/np.linalg.norm(copyA,2))




import numpy as np
from numpy import linalg
from scipy import signal
from scipy import linalg as lnlg

def Controllability(A,B):
    if (np.shape(A)[0]==np.shape(A)[1]):
        ctrb=B
        for i in range(1,np.shape(A)[0]):
            ctrb=np.hstack((ctrb,A**i*B)) 
        return linalg.matrix_rank(ctrb)==np.shape(A)[0]

def Observability(A,C):
    if (np.shape(A)[0]==np.shape(A)[1]):
        obsv=(C)
        for i in range(1,np.shape(A)[0]):
            obsv=np.vstack((obsv,C*A**i)) 
        return linalg.matrix_rank(obsv)==np.shape(A)[0]


def LTI_LQR(A,B,Q,R):
    #solve dat riccatti
    X = np.matrix(lnlg.solve_continuous_are(A,B,Q,R))
    
    #solve dat gain
    K = np.matrix(lnlg.inv(R)*B.T*X)

    eigVals, eigVects = lnlg.eig(A-B*K)

    return K, X, eigVals

def LTI_DLQR(A,B,Q,R):
    #solve dat riccatti
    X = np.matrix(lnlg.solve_discrete_are(A,B,Q,R))
    
    #solve dat gain
    K = np.matrix(lnlg.inv(B.T*X*B+R)*(B.T*X*A))

    eigVals, eigVects = lnlg.eig(A-B*K)

    return K, X, eigVals

def scaling_1in(A,B,C,D,K):
    size=np.shape(A)[0]
    Z=np.matrix(np.block([[np.zeros((1,size)), 1]]))
    N=np.matrix(lnlg.inv(np.matrix(np.block([[A,B],[C,D]]))))*Z.T
    Nx=N[0:size]
    Nu=N[size]
    return (Nu)+(K)*(Nx)

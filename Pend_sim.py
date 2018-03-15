import numpy as np
from numpy import linalg
from scipy import signal
from matplotlib import pyplot
import filterpy
import Control_Utils as cu

#I used the university of michigan example at http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling


#pendulum on a cart in 2-d
#equations are:
#(M+m)*\ddot{x}+b*\dot{x}+m*l*\ddot{\theta}*\cos(\theta)-m*l*(\dot{\theta})^2\sin(\theta)=F
#(I+m*l^2)\ddot{\theta}+m*g*l*\sin(\theta)=-m*l*\ddot{x}*cos(\theta)

#linearize about \pi with small angle approximation (aka first order taylor expansion about \pi)
#thus \cos(\theta+\pi) \approx -1 and \sin(\theta+\pi) \approx -\theta
#this gives
#(I+m*l^2)*\ddot{\theta}-m*g*l*\theta=m*l*\ddot{x}
#(M+m)*\ddot{x}+b*\dot{x}-m*l*\ddot{\theta}=u

M=0.5 #kg
m=0.05 #kg
b=0.1 #forget the unit
I=0.006 #forget the unit
g=9.8 #m/s^2
l=0.3 #m

A=np.matrix([[0, 1, 0, 0],
            [0, (-(I+m*l**2)*b)/(I*(M+m)+M*m*l**2), (m**2*g*l**2)/(I*(M+m)+M*m*l**2), 0],
            [0, 0, 0, 1],
            [0, (-m*l*b)/(I*(M+m)+M*m*l**2), (m*g*l*(M+m))/(I*(M+m)+M*m*l**2), 0]])

B=np.matrix([[0],
            [(I+m*l**2)/(I*(M+m)+M*m*l**2)],
            [0],
            [(m*l)/(I*(M+m)+M*m*l**2)]])

C=np.matrix([[1, 0, 0, 0],
            [0, 0, 1, 0]])

D=np.matrix([[0],[0]])

Q= C.T*C
Q[0,0]=5000
Q[2,2]=100
R= np.matrix([[1]])


def main():
    base=signal.StateSpace(A,B,C[1,:],D[1,:])

    #w, mag, phase = signal.bode(base)
    #pyplot.figure()
    #pyplot.semilogx(w,mag)
    #pyplot.figure()
    #pyplot.semilogx(w,phase)
    #pyplot.show()

    if (cu.Controllability(A,B)):
        print ("Matrix is controllable")
    
    if (cu.Observability(A,C)):
        print ("Matrix is observable")

    (Ku, X, eigVals)=cu.LTI_LQR(A,B,Q,R)

    Nlqr=cu.scaling_1in(A,B,C[0,:],0,Ku)

    print (Nlqr)
    Alqr=A-B*Ku
    Blqr=B
    Clqr=C
    Dlqr=D



    t=np.linspace(0,5,num=5001)
    r=0.2*np.ones(np.size(t))

    tout, yout, xout =signal.lsim((Alqr,Blqr*Nlqr,Clqr,Dlqr),r,t)

    pyplot.figure()
    pyplot.plot(tout,yout[:,0],tout,yout[:,1])

    pyplot.show()


if __name__ == "__main__":
    main()

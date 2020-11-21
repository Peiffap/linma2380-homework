A=[10.5 -9.5 -6.5 4.5; 13 -12 -7.5 5.5; 9 -6 -6 2; 13 -10 -7.5 3.5]; %A1
%A=[14.5 -13.5 -8.5 6.5; 16 -15 -9 7; 13 -10 -8 4; 16 -13 -9 5]; %A2
%A=[16 -15 -10 8; 18 -17 -11 9; 14 -11 -9 5; 18 -15 -11 7]; %A3

%A=[-1 2 -3; 0 -2 3; 0 0 -3]; % works with this matrix

eig(A)

I=eye(size(A));

Q=I;
% random positive definite matrices
%Z=rand(4);
%Q=Z*Z'
 
[U,S]=schur(A); %-> S upper triangular and A=U*S*U'

% O(n^4) algo
V=kron(conj(U),U);
C=kron(I,S')+kron(S.',I); %-> C lower triangular
D=-V'*Q(:);

X=C\D;
vecP=V*X;
P=reshape(vecP,size(A))
A'*P+P*A %supposed to be -Q

% supposed to be equal but isn't :
C=V*(kron(I,S')+kron(S.',I))*V';
C2=kron(I,A')+kron(A.',I);
%C-C2

% O(n^6) algo
C2=kron(I,A')+kron(A.',I);
D2=-Q(:);
vecP2=C2\D2;
P2=reshape(vecP2,size(A))
A'*P2+P2*A; % = -I

%
P3=lyap(A',A,Q)
A'*P3+P3*A; % =-I

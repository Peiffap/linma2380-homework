clear all
close all
load('y.mat', 'y');
yc = y(1:78);%55 = 0.7 * 78 ///// 23 = 0.3*78
yc_train = yc(1:55);
yc_val = yc(56:end);
ysd = y(78:139);% 43 = 0.7*62 ////// 19 = 0.3*62
ysd_train = ysd(1:43);
ysd_val = ysd(44:end);

%---------------------------------
% Mode Confinement
%--------------------------------
N = length(yc_train)-1;
errorc_train = zeros(30,1);
errorc_val = zeros(30,1);
for p = 30:30
    
[alpha, yc_train_p] = solveLeastSquares(yc_train, p,N);

errorc_train(p) = norm(yc_train-yc_train_p);

Nval = length(yc_val);
ytemp = yc(N+2-p : end);
for t = p+1 : length(ytemp)
   ytemp(t) = alpha(1);
   for i = 1:p
      ytemp(t) = ytemp(t)+alpha(i+1)*ytemp(t-i);
   end   
end
yc_val_p = ytemp(p+1:end);
errorc_val(p) = norm(yc_val-yc_val_p);

%plot(yc_train); hold on ; plot(yc_train_p);
plot(yc_val); hold on ; plot(yc_val_p);
end
figure
plot(1:30, errorc_train);
figure
plot(1:30, errorc_val);



function [alpha, y_p] = solveLeastSquares(y,p,N)
alpha = zeros(p+1,1);
A = zeros(N+1-p,p+1);
A(:,1) = 1;
for j = 1:p
    A(:,p+2-j) = y(j:j+(N-p));
end
b=y(p+1:end);

%[Q,R] = qr(A);
%alpha = R\(Q'*b);
alpha = A\b;

y_p = zeros(N+1,1);
y_p(1:p) = y(1:p);
y_p(p+1:end)=A*alpha;
end

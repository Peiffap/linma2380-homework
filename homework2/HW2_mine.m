clear all
%close all

load('y.mat', 'y');

yc = y(1:78); % confinement
yc_train = yc(1:55);
yc_val = yc(56:end);

% ysd = y(78:139); % social distancing
% yc_train = ysd(1:43);
% yc_val = ysd(44:end);

%---------------------------------
% Mode Confinement
%--------------------------------

N = length(yc_train)-1;
errorc_train = zeros(30,1);
errorc_val = zeros(30,1);

figure
for p = 1:20

    % solve the least squares
    [alpha, yc_train_p] = solveLeastSquares(yc_train,p,N);
    
    % compute the error on the training set
    errorc_train(p) = norm(yc_train-yc_train_p);

    Nval = length(yc_val);
    
    y_previous=yc_train_p(end-p+1:end); % values of yc_train_p useful for the computation of yc_val_p
    
    yc_val_p=alpha(1)*ones(Nval,1);
    
    for i=1:Nval
        for j=1:p
            yc_val_p(i)=yc_val_p(i)+alpha(j+1)*y_previous(p+i-j);
        end
        y_previous=[y_previous;yc_val_p(i)];
    end

    % compute the error on the validation set
    errorc_val(p) = norm(yc_val-yc_val_p);

    plot(yc_train_p); hold on ;
    %plot(yc_val_p); hold on ;
end
plot(yc_train,'LineWidth',3);
title('Estimates of y on the training set')
%plot(yc_val,'LineWidth',3);
%title('Estimates of y on the validation set')

figure
plot(1:30, errorc_train);
title('Error on the training set')
figure
plot(1:30, errorc_val);
title('Error on the validation set')



function [alpha, y_p] = solveLeastSquares(y,p,N)
y_p=zeros(length(y),1);

A = zeros(N+1,p+1);
A(:,1) = 1;
for j=1:p
    A(j+1:end,j+1)=y(1:N-j+1);
end
b=y(p+1:end);
A=A(p+1:end,:);

[Q,R] = qr(A);
Q_hat=Q(:,1:p+1);
R=R(1:p+1,1:p+1);
alpha = R\(Q_hat'*b);

y_p(1:p)=y(1:p);
%y_p(p+1:end)=A*alpha;

y_p(p+1:end)=alpha(1)*ones(length(y_p(p+1:end)),1);
for i=p+1:length(y)
    for j=1:p
        y_p(i)=y_p(i)+alpha(j+1)*y_p(i-j);
    end
end
end
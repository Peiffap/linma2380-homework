%=====================================================================
% PartCtest.m
%-training error = error between real data y and ŷ(t) = f(ŷ(t-1)...ŷ(t-p))
%-validation error = error between real data y and ŷ(t) = f(y(t-1)...y(t-p))
%=======================================================================

clear all
close all
load('y.mat', 'y');
% Confinement
yc = y(1:78);%55 = 0.7 * 78 ///// 23 = 0.3*78
yc_train = yc(1:55);
yc_val = yc(56:end);
%Social distancing
ysd = y(79:139);% 43 = 0.7*62 ////// 19 = 0.3*62
ysd_train = ysd(1:43);
ysd_val = ysd(44:end);
%% 
%---------------------------------
% Confinement Mode
%--------------------------------
errorc_train = zeros(30,1);
errorc_val = zeros(30,1);
plot([yc_train;yc_val]); hold on;
for p = 1:30
[errorc_val(p), errorc_train(p)] = error(yc,yc_train, yc_val, p);
end
title('Confinement Mode Fitting and Prediction');
figure
plot(1:30, errorc_train);
title('Confinement Mode - Training error in function of p');
figure
plot(1:30, errorc_val);
title('Confinement Mode - Validation error in function of p');

%% 
%---------------------------------
% Social distancing Mode
%--------------------------------
errorsd_train = zeros(30,1);
errorsd_val = zeros(30,1);
plot([ysd_train;ysd_val]); hold on;
for p = 1:30
[errorsd_val(p), errorsd_train(p)] = error(ysd,ysd_train, ysd_val, p);
end
title('Social Distancing Mode Fitting and Prediction');

figure
plot(1:30, errorsd_train);
title('Social Distancing Mode - Training error in function of p');

figure
plot(1:30, errorsd_val);
title('Social Distancing Mode - Validation error in function of p');

%%
%-----------------------------------------
% Useful functions
%------------------------------------------
function [error_val, error_train] = error(y,y_train, y_val, p)
N = length(y_train)-1;

% Solve least squares prob
[alpha, y_train_p] = solveLeastSquares(y_train, p,N);

%Training Error
error_train = norm(y_train-y_train_p);

%Validation Error
ytemp = y(N+2-p:end);
ytemp2 = ytemp;
for t = p+1 : length(ytemp)
   ytemp(t) = alpha(1);
   for i = 2:p+1
      ytemp(t) = ytemp(t)+alpha(i)*ytemp2(t-i+1);
   end   
end
y_val_p = ytemp(p+1:end);
error_val = norm(y_val-y_val_p);

plot([y_train_p;y_val_p], 'DisplayName',strcat('p=', num2str(p))); hold on;
end


function [alpha, y_p] = solveLeastSquares(y,p,N)
A = zeros(N+1,p+1);
A(:,1) = 1;
for j = 1:p
    A(j+1:end,j+1) = y(1:end-j);
end
b=y;

%QR decomp
[Q,R] = qr(A);
Qprim = Q(:,1:p+1);
Rprim = R(1:p+1,:);
%TODO : backward substitution
alpha = Rprim\(Qprim'*b);

%model
%y_p = A*alpha;
y_p = pred([zeros(p,1);y], alpha, p);
end

function y_p = pred(ytemp, alpha, p)
for t = p+1 : length(ytemp)
   ytemp(t) = alpha(1);
   for i = 2:p+1
      ytemp(t) = ytemp(t)+alpha(i)*ytemp(t-i+1);
   end   
end
y_p = ytemp(p+1:end);
end
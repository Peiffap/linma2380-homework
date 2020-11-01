%=====================================================================
% PartCtest.m
%-training error = error between real data y and ŷ(t) = f(y(t-1)...y(t-p))
%-validation error = error between real data y and ŷ(t) = f(ŷ(t-1)...ŷ(t-p))
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
pval = 1:26;
errorc_train = zeros(length(pval),1);
errorc_val = zeros(length(pval),1);
figure
plot(yc, '.-', 'LineWidth',3); hold on;
for p = 1:length(pval)
[errorc_val(p), errorc_train(p)] = error(yc,yc_train, yc_val, pval(p));
end
title('Confinement Mode Fitting and Prediction');
figure
plot(pval, errorc_train);
title('Confinement Mode - Training error in function of p');
figure
plot(pval, errorc_val);
title('Confinement Mode - Validation error in function of p');

%% 
%---------------------------------
% Social distancing Mode
%--------------------------------
pval = 1:20;
errorsd_train = zeros(length(pval),1);
errorsd_val = zeros(length(pval),1);
figure
plot(ysd, '.-', 'LineWidth',3); hold on;
for p = 1:length(pval)
[errorsd_val(p), errorsd_train(p)] = error(ysd,ysd_train, ysd_val, pval(p));
end
title('Social Distancing Mode - AR models of nb of hospitalizations - Training set');

figure
plot(pval, errorsd_train);
title('Social Distancing Mode - Training error in function of p');

figure
plot(pval, errorsd_val);
title('Social Distancing Mode - Validation error in function of p');

%%
%-----------------------------------------
% Useful functions
%------------------------------------------
function [error_val, error_train] = error(y,y_train, y_val, p)
N = length(y_train)-1;

% Solve least squares prob
alpha = solveLeastSquares(y_train, p,N);

y_train_p = pred(y_train, alpha, p);
%Training Error
error_train = norm(y_train(p+1:end)-y_train_p);

%Validation Error
ytemp = y(N+2-p:end);
y_val_p = pred(ytemp, alpha, p);
error_val = norm(y_val-y_val_p);

%plot([y_train(1:p);y_train_p], 'DisplayName',strcat('p=', num2str(p))); hold on;
plot([y_train;y_val_p], 'DisplayName',strcat('p=', num2str(p))); hold on;

end

function alpha = solveLeastSquares(y,p,N)
A = zeros(N-p+1,p+1);
A(:,1) = 1;
for j = 1:p
    A(:,p+2-j) = y(j:j+(N-p));
end
b=y(p+1:end);

if (N-p)>= p+1
    %QR decomp
    [Q,R] = qr(A);
    Qprim = Q(:,1:p+1);
    Rprim = R(1:p+1,:);
    %TODO : backward substitution
    alpha = Rprim\(Qprim'*b);
else
    alpha = A\b;
end

%model
%y_p = A*alpha;
%y_p = pred([zeros(p,1);y], alpha, p);
end

function y_p = pred(ytemp, alpha, p)
ytemp2 = ytemp;
for t = p+1 : length(ytemp)
   ytemp(t) = alpha(1);
   for i = 2:p+1
      ytemp(t) = ytemp(t)+alpha(i)*ytemp2(t-i+1);
   end
end
y_p = ytemp(p+1:end);
end


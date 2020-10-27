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
for p = 2:2
[errorc_val(p), errorc_train(p)] = error(yc,yc_train, yc_val, p);
end
figure
plot(1:30, errorc_train);
figure
plot(1:30, errorc_val);

%% 
%---------------------------------
% Social distancing Mode
%--------------------------------
errorsd_train = zeros(30,1);
errorsd_val = zeros(30,1);
plot([ysd_train;ysd_val]); hold on;
for p = 2:2
[errorsd_val(p), errorsd_train(p)] = error(ysd,ysd_train, ysd_val, p);
end
figure
plot(1:30, errorsd_train);
figure
plot(1:30, errorsd_val);

%%
%-----------------------------------------
% Useful functions
%------------------------------------------
function [error_val, error_train] = error(y,y_train, y_val, p)
N = length(y_train)-1;
[alpha, y_train_p] = solveLeastSquares(y_train, p,N);
error_train = norm(y_train-y_train_p);
ytemp = y(N+2-p:end);
for t = p+1 : length(ytemp)
   ytemp(t) = alpha(1);
   for i = 2:p+1
      ytemp(t) = ytemp(t)+alpha(i)*ytemp(t-i+1);
   end   
   test = ytemp(t)
   test1 = ytemp(t-i+1)
end
y_val_p = ytemp(p+1:end);
error_val = norm(y_val-y_val_p);
plot([y_train_p;y_val_p], 'DisplayName',strcat('p=', num2str(p))); hold on;
y_val_p(1)
end

function [alpha, y_p] = solveLeastSquares(y,p,N)
A = zeros(N+1,p+1);
A(:,1) = 1;
for j = 1:p
    A(j+1:p,j+1) = y(1:p-j);
    A(p+1:end,p+2-j) = y(j:j+(N-p));
end
b=y;

[Q,R] = qr(A);
Qprim = Q(:,1:p+1);
Rprim = R(1:p+1,:);
alpha = Rprim\(Qprim'*b);

y_p = A*alpha;
end

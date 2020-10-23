%clear; clc; 
load('outputset1.mat');
%clc; close all;

A = states(100, :, 1);
B = states(100, :, 5);

n1 = normalize(A, 'zscore');
n2 = normalize(B, 'zscore');

% n1 = normalize(A, 'range', [-1 1]);
% n2 = normalize(B, 'range', [-1 1]);

figure(3)
subplot(2,1,1)
plot(A)
subplot(2,1,2)
plot(n1)

figure(4)
subplot(2,1,1)
plot(B)
subplot(2,1,2)
plot(n2)


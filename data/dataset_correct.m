clear;clc;

load('outputset1_new.mat');

s1 = states;
c1 = controls;

c = 0
for i=1:400
    
    a = sum(s1(i,:,:));
    if a< 0.1
        i
        c = c + 1 
        plot(s1(i,:,1))
    end
    
end
%save('outputset1_new.mat', 'states', 'controls')
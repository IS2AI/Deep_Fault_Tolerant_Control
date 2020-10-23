clear;

% read files
f1  = load('outputset1.mat');
f2  = load('outputset2.mat');
f3  = load('outputset3.mat');
f4  = load('outputset4.mat');
f5  = load('outputset6.mat');
f6  = load('outputset7.mat');
f7  = load('outputset8.mat');
f8  = load('outputset9.mat');

ind = 6;

feat1 = f1.states(:,:,ind);
feat2 = f2.states(:,:,ind);
feat3 = f3.states(:,:,ind);
feat4 = f4.states(:,:,ind);
feat5 = f5.states(:,:,ind);
feat6 = f6.states(:,:,ind);
feat7 = f7.states(:,:,ind);
feat8 = f8.states(:,:,ind);


fs1 = feat1(:);
fs2 = feat2(:);
fs3 = feat3(:);
fs4 = feat4(:);
fs5 = feat5(:);
fs6 = feat6(:);
fs7 = feat7(:);
fs8 = feat8(:);


FF = [fs1;fs2;fs3;fs4;fs5;fs6;fs7;fs8];

m = mean(FF)

s = std(FF)
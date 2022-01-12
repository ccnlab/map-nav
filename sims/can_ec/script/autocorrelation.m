M = csvread('/Users/zycyc/go/src/github.com/emer/attractorEC/train_trial_act2.csv');
m = reshape(M,[100,100]);
c = xcorr2(m);

subplot(1,2,1)
h1 = heatmap(m)
subplot(1,2,2)
h2 = heatmap(c)

a = [];
tx = cnn.layers{1}.a{1};
for i = 1 : 48
t = tx(:,:,i)';
a = [a t(:)];
end
displayData(a');


k1 = [];
tx = cnn.layers{2}.k{1};
for i = 1 : 6
    t = tx{i}';
    k1 = [k1 t(:)];
end
k1 = k1';
displayData(k1);

figure(70);
om = [];
for j = 1:6
tx = cnn.layers{2}.a{j};
for i = 1 : 48
t = tx(:,:,i)';
om = [om t(:)];
end
end
displayData(om');

figure(70);
om3 = [];
for j = 1:6
tx = cnn.layers{3}.a{j};
for i = 1 : 48
t = tx(:,:,i)';
om3 = [om3 t(:)];
end
end
displayData(om3');

figure(60);
k2 = [];
for j = 1:6
tx = cnn.layers{4}.k{j};
for i = 1 : 12
t = tx{i}';
k2 = [k2 t(:)];
end
end
displayData(k2');

figure(80);
om2 = [];
for j = 1:12
tx = cnn.layers{4}.a{j};
for i = 1 : 48
t = tx(:,:,i)';
om2 = [om2 t(:)];
end
end
displayData(om2');


figure(100);
om5 = [];
for j = 1:12
tx = cnn.layers{5}.a{j};
for i = 1 : 48
t = tx(:,:,i)';
om5 = [om5 t(:)];
end
end
displayData(om5');
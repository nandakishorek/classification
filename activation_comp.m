h = bar([0.1296; 0.0506; 0.0861]);
grid on
l = cell(1,3);
l{1}='Sigmoid'; l{2}='Relu'; l{3}='Tanh';
set(gca,'xticklabel', l)
set(gcf,'DefaultTextColor','red')
ylabel('classification error', 'Color', 'r');

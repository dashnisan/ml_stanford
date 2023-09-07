function plotData(x, y)
%PLOTDATA Plots the data points x and y into a new figure 
%   PLOTDATA(x,y) plots the data points and gives the figure axes labels of
%   population and profit.

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the training data into a figure using the 
%               "figure" and "plot" commands. Set the axes labels using
%               the "xlabel" and "ylabel" commands. Assume the 
%               population and revenue data have been passed in
%               as the x and y arguments of this function.
%
% Hint: You can use the 'rx' option with plot to have the markers
%       appear as red crosses. Furthermore, you can make the
%       markers larger by using plot(..., 'rx', 'MarkerSize', 10);
close all
figure; % open a new figure window
plot(x,y, 'r+', 'MarkerSize', 11)

xlabel('City X population / 10^{4}')
ylabel('Profit in the city X / 10^{4}')
title('fx:plotData')
legend('revenue')

minxv=min(x)*(1-0.2);
maxxv=max(x)*(1+0.2);
display([num2str(minxv), ' ', num2str(maxxv)])
xlim([minxv maxxv])
set(gca,'XMinorTick','on')
set(gca,'YMinorTick','on')
%set(gca, 'XAxisLocation', 'origin') % does not work in octave

grid on
grid minor
box off



% ============================================================

end

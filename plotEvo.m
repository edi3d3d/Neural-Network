function plotEvo(res, color)

    x = (1:length(res.loss)) * 1000;

    % Loss
    subplot(2,1,1);
    hold on;
    plot(x, res.loss, 'Color', color, 'LineWidth', 0.5); 
    hold off;
    
    title('Training Loss');
    xlabel('Iteration');
    ylabel('Loss');
    grid on;
    
    % Gradient norm
    subplot(2,1,2);
    hold on;
    semilogy(x, res.grad, 'Color', color, 'LineWidth', 0.5); 
    hold off;

    title('Gradient Norm');
    xlabel('Iteration');
    ylabel('||grad||');
    grid on;
end
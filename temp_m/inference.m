function accuracy = inference(c_iter, dim, gain, power, eval, repeat)
% perform inference task
    accuracy_sum = 0;
    for i = 1:repeat
        factor = 1 ./ sum(c_iter);
        z_hat = factor .* (squeeze(sum(reshape(c_iter, [1 size(c_iter)]) .* eval.z, 2)) + randn(1, dim.N) * sqrt(power.L_0 * gain.delta_0));
        predicted = predict(eval.Mdl, z_hat);
        accuracy_sum = accuracy_sum + sum(predicted == eval.label_test) / eval.num_test_samples;
    end
    accuracy = accuracy_sum / repeat;
end


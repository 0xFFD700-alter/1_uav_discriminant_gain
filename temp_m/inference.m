function acc = inference(c_iter, dim, gain, infer, repeat)
% perform inference task
    rng(2020);
    acc_sum = 0;
    for i = 1:repeat
        factor = 1 ./ sum(c_iter);
        z_hat = factor .* (squeeze(sum(reshape(c_iter, [1 size(c_iter)]) .* infer.z, 2)) + randn(1, dim.N) * sqrt(gain.delta_0));
        pred = predict(infer.Mdl, z_hat);
        acc_sum = acc_sum + sum(pred == infer.label_test) / infer.num_test_samples;
    end
    acc = acc_sum / repeat;
end


function gen_distorted_data(K, x)
    rng(2022);
    N = size(x, 2);
    samples = size(x, 1);
    delta = 16 * rand([K N]);
    z = reshape(x, [samples, 1, N]) + reshape(sqrt(delta), [1 K N]) .* randn([1 K N]);
    save('./data/inference/data_test_noise.mat', 'z');
    save('./data/inference/var_sensor.mat', 'delta');
end
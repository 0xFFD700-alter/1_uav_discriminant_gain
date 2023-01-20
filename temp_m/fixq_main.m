function [acc_1, acc_2] = fixq_main(peak_p, model_1, model_2, fixq)
    num_a = evalin('base', 'num_a');
    num_b = evalin('base', 'num_b');
    dim = evalin('base', 'dim');
    power = evalin('base', 'power');
    gain = evalin('base', 'gain');
    sca = evalin('base', 'sca');
    infer = evalin('base', 'infer');
    repeat = evalin('base', 'repeat');
    scale_c = evalin('base', 'scale_c');
    init_scale = evalin('base', 'init_scale');
    
    % data and model to perform inference tasks
    load(model_1(1), model_1(2));
    load(model_2(1), model_2(2));

    % power constraints
    P_list = [peak_p(1) * ones(num_a, 1); peak_p(2) * ones(num_b, 1)] * 1e-3;
    power.P = P_list * ones(1, dim.N);                      % peak power constraints
    power.P_bar = P_list .* power.ratio;                    % average power constraints
    
    [c_iter, ~] = solve_c_alter(fixq, dim, power, gain, sca, scale_c, init_scale, 1);
    infer.Mdl = eval(model_1(2));
    acc_1 = inference(c_iter, dim, gain, infer, repeat);
    infer.Mdl = eval(model_2(2));
    acc_2 = inference(c_iter, dim, gain, infer, repeat);
end
function [dis, acc] = fixq_main(peak_p, model, fixq)
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
    load(model(1), model(2));
    infer.Mdl = eval(model(2));

    % power constraints
    P_list = [peak_p(1) * ones(num_a, 1); peak_p(2) * ones(num_b, 1)] * 1e-3;
    power.P = P_list * ones(1, dim.N);                      % peak power constraints
    power.P_bar = P_list .* power.ratio;                    % average power constraints
    
    [c_iter, a_iter] = solve_c_alter(fixq, dim, power, gain, sca, scale_c, init_scale, 1);
    acc = inference(c_iter, dim, gain, infer, repeat);
    dis = sum(a_iter);
end
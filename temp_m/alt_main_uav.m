function q_iter = alt_main_uav(peak_p, model_1, model_2)
    num_a = evalin('base', 'num_a');
    num_b = evalin('base', 'num_b');
    dim = evalin('base', 'dim');
    power = evalin('base', 'power');
    gain = evalin('base', 'gain');
    sca = evalin('base', 'sca');
    infer = evalin('base', 'infer');
    uav = evalin('base', 'uav');
    q_iter = evalin('base', 'q_iter');
    epsilon = evalin('base', 'epsilon');
    repeat = evalin('base', 'repeat');
    scale_c = evalin('base', 'scale_c');
    scale_q = evalin('base', 'scale_q');
    init_scale = evalin('base', 'init_scale');

    % data and model to perform inference tasks
    load(model_1(1), model_1(2));
    load(model_2(1), model_2(2));

    % power constraints
    P_list = [peak_p(1) * ones(num_a, 1); peak_p(2) * ones(num_b, 1)] * 1e-3;
    power.P = P_list * ones(1, dim.N);                      % peak power constraints
    power.P_bar = P_list .* power.ratio;                    % average power constraints

    % auxiliary variables for alternating opt
    acc_list_1 = [];
    acc_list_2 = [];
    dis_iter = 0;

    for iii = 1:100

        [c_iter, a_iter] = solve_c_alter(q_iter, dim, power, gain, sca, scale_c, init_scale, 1);
        
        infer.Mdl = eval(model_1(2));
        acc_1 = inference(c_iter, dim, gain, infer, repeat);
        infer.Mdl = eval(model_2(2));
        acc_2 = inference(c_iter, dim, gain, infer, repeat);
        acc_list_1 = [acc_list_1 acc_1];
        acc_list_2 = [acc_list_2 acc_2];

        dis_opt = sum(a_iter);

        if abs(dis_opt - dis_iter) < epsilon
            break;
        end
        dis_iter = dis_opt;

        q_iter = solve_q(c_iter, dim, power, uav, scale_q, 1);

    end

end
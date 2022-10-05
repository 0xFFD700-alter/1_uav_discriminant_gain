function q_iter = solve_q(c_iter, dim, power, uav)
% solve trajectory q
    cvx_begin
        cvx_solver mosek
        variable q(dim.N, 2)
        expression vstack(dim.K, dim.N)
        expression q_norm(dim.N, 1)
        for k = 1: dim.K
            vstack(k, :) = sum((q - squeeze(power.w(k, :, :))) .^ 2, 2);
        end
        minimize sum(sum((vstack + power.H ^ 2).* c_iter .^ 2 .* power.E))
    
        subject to
            (vstack + power.H ^ 2) .* c_iter .^ 2 .* power.E <= power.P * power.L_0; 
            sum((vstack + power.H ^ 2) .* c_iter .^ 2 .* power.E, 2) <= power.P_bar * power.L_0 * dim.N;
            q_norm(1) = norm(q(1, :) - uav.q_init);
            for n = 2: dim.N
                q_norm(n) = norm(q(n, :) - q(n - 1, :));
            end
            q_norm <= uav.slot * uav.Vm * ones(dim.N, 1);
    cvx_end
    
    assert(strcmp(cvx_status, 'Solved'));
    q_iter = q;
end


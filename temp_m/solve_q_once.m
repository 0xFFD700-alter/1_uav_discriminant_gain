function q_iter = solve_q_once(c_iter, dim, power, uav, verbose)
    cvx_begin
        if verbose == 0 || verbose == 1
            cvx_quiet true
        end

        cvx_solver mosek
        variable q(dim.N, 2) nonnegative
%         variable eta nonnegative
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

    if verbose == 1
        fprintf('cvx_status: %s cvx_optval: %f \n', [cvx_status ',' blanks(18 - length(cvx_status))], cvx_optval);
    end
    assert(strcmp(cvx_status, 'Solved'));
    q_iter = q;
end

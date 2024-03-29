function [c_iter, a_iter] = solve_c_alter(q_iter, dim, power, gain, sca, scale, init_scale, verbose)
% solve precoding strength c with SCA
    fprintf(['\n' repmat('*', 1, 10) 'solve precoding strength c with SCA' repmat('*', 1, 10) '\n']);
    [c_iter, a_iter] = init_sca(q_iter, dim, power, gain, init_scale);
    gain_iter = sum(a_iter);
    patience_count = 0;
    iter_count = 0;
    while 1
        iter_count = iter_count + 1;
        cvx_begin
            if verbose == 0 || verbose == 1
                cvx_quiet true
            end

            cvx_solver mosek
            variable c(dim.K, dim.N) nonnegative
            variable a(1, dim.N) nonnegative
            maximize sum(a)
    
            subject to
                vstack = zeros(dim.K, dim.N);
                for k = 1: dim.K
                    vstack(k, :) = sum((q_iter - squeeze(power.w(k, :, :))) .^ 2, 2);
                end
                c .^ 2 <= power.P * power.L_0 ./ ((vstack + power.H ^ 2) .* power.E) * scale;
                sum(c .^ 2 .* (vstack + power.H ^ 2) .* power.E, 2) <= power.P_bar * power.L_0 * dim.N * scale;
        
                function_g = sum(c_iter) .^ 2 ./ a_iter;
                first_order_g = 2 * sum(c_iter) ./ a_iter .* sum(c - c_iter) ...
                                - (sum(c_iter) ./ a_iter) .^ 2 .* (a - a_iter);
                taylor_g = function_g + first_order_g;
                sum(c .^ 2 .* gain.delta) + sum(c) .^ 2 .* gain.sigma + scale * gain.delta_0 <= taylor_g .* gain.u;
        cvx_end
        
        if verbose == 1
            fprintf('%3d   cvx_status: %s cvx_optval: %f \n', iter_count, [cvx_status ',' blanks(18 - length(cvx_status))], cvx_optval);
        end
        assert(strcmp(cvx_status, 'Solved'));
        
        c_iter = sca.momentum * c_iter + (1 - sca.momentum) * c;
        a_iter = sca.momentum * a_iter + (1 - sca.momentum) * a;
        
        if abs(cvx_optval - gain_iter) <= sca.epsilon
            patience_count = patience_count + 1;
            if patience_count > sca.patience
                break
            end
        else
            patience_count = 0;
        end
        gain_iter = cvx_optval;
    end
    c_iter = c_iter / sqrt(scale);
end

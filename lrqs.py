import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import TruncatedSVD

def get_q_obs(X, Y, G=10, B=10):
    
    N = len(X)
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    Y_sorted = Y[sort_idx]
    
    bins = np.array_split(np.arange(N), G)
    Q_obs = np.zeros((G, B))
    
    quantiles = np.linspace(1/(2*B), 1 - 1/(2*B), B)
    
    for i, b_idx in enumerate(bins):
        y_bin = Y_sorted[b_idx]
        if len(y_bin) == 0:
            if i > 0:
                Q_obs[i, :] = Q_obs[i-1, :]
            else:
                Q_obs[i, :] = 0.0
        else:
            Q_obs[i, :] = np.quantile(y_bin, quantiles)
            
    return Q_obs

def fit_pnl_lrqs(X, Y, K=1, G=10, B=10, n_iter=10, n_inner=5, n_init=5, fix_g=False): 

    Q_obs = get_q_obs(X, Y, G=G, B=B)
    
    # fix-g
    if fix_g:
        Z = Q_obs.copy()
        for _ in range(n_inner):
            a = np.mean(Z, axis=1, keepdims=True)
            R = Z - a
            
            if K < min(R.shape):
                svd = TruncatedSVD(n_components=K)
                R_approx = svd.fit_transform(R) @ svd.components_
            else:
                R_approx = R
                
            Z = R_approx + a
            
            ir_row = IsotonicRegression(out_of_bounds='clip')
            x_dummy = np.arange(B)
            for i in range(G):
                Z[i, :] = ir_row.fit_transform(x_dummy, Z[i, :])
                
        score = np.linalg.norm(Q_obs - Z)
        return score

    # est-g
    Q_obs_flat = Q_obs.flatten()
    ir_g = IsotonicRegression(out_of_bounds='clip')
    best_score = float('inf') 
    
    for init_idx in range(n_init):
        if init_idx == 0:
            Z = Q_obs.copy() 
        else:
            Z = Q_obs.copy() + np.random.normal(0, 0.5, size=Q_obs.shape)
            
        for t in range(n_iter):
            Z_flat = Z.flatten()
            
            # 1. Update g^(t)
            sort_idx = np.argsort(Z_flat)
            Z_sorted = Z_flat[sort_idx]
            Q_obs_sorted = Q_obs_flat[sort_idx]
            
            g_pred = ir_g.fit_transform(Z_sorted, Q_obs_sorted)
            
            # 2. Update Z^(t) by the inverse of g^(t)
            g_pred_strict = g_pred + np.linspace(0, 1e-8, len(g_pred))
            Z_new_flat = np.interp(Q_obs_flat, g_pred_strict, Z_sorted)
            Z = Z_new_flat.reshape(Q_obs.shape)
            
            # 3. Inner loop
            for _ in range(n_inner):
                a = np.mean(Z, axis=1, keepdims=True)
                R = Z - a
                
                if K < min(R.shape):
                    svd = TruncatedSVD(n_components=K)
                    R_approx = svd.fit_transform(R) @ svd.components_
                else:
                    R_approx = R
                    
                Z = R_approx + a
                
                ir_row = IsotonicRegression(out_of_bounds='clip')
                x_dummy = np.arange(B)
                for i in range(G):
                    Z[i, :] = ir_row.fit_transform(x_dummy, Z[i, :])
                    
                mean_Z = np.mean(Z)
                var_Z = np.var(Z)
                if var_Z > 1e-8:
                    Z = (Z - mean_Z) / np.sqrt(var_Z)
                else:
                    Z = Z - mean_Z
                    
        Z_flat = Z.flatten()
        sort_idx = np.argsort(Z_flat)
        Z_sorted = Z_flat[sort_idx]
        Q_obs_sorted = Q_obs_flat[sort_idx]
        
        g_final_pred = ir_g.fit_transform(Z_sorted, Q_obs_sorted)
        
        reverse_idx = np.argsort(sort_idx)
        g_Z = g_final_pred[reverse_idx]
        
        score = np.linalg.norm(Q_obs_flat - g_Z)
        
        if score < best_score:
            best_score = score

    return best_score 

def predict_causal_direction(X, Y, K=1, G=10, B=10, n_iter=10, n_inner=5, n_init=5, fix_g=False): 

    X_std = (X - np.mean(X)) / (np.std(X) + 1e-8)
    Y_std = (Y - np.mean(Y)) / (np.std(Y) + 1e-8)
    
    score_xy = fit_pnl_lrqs(X_std, Y_std, K=K, G=G, B=B, n_iter=n_iter, n_inner=n_inner, n_init=n_init, fix_g=fix_g)
    score_yx = fit_pnl_lrqs(Y_std, X_std, K=K, G=G, B=B, n_iter=n_iter, n_inner=n_inner, n_init=n_init, fix_g=fix_g)
    
    if score_xy < score_yx:
        return 1 
    else:
        return -1 